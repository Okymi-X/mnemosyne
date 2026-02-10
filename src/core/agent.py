"""
Mnemosyne v3.0 -- Agent Orchestrator
Autonomous tool-calling loop that lets the LLM decide when to
read files, search the codebase, run commands, and gather information
before synthesising a final answer.

The agent uses an event callback to report progress to the UI layer.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.core.brain import (
    _boost_results,
    _build_context_block,
    _estimate_complexity,
    _load_episodic_memory,
    _rewrite_query_for_retrieval,
    summarise_history,
)
from src.core.config import resolve_config
from src.core.providers import get_llm
from src.core.tools import (
    ToolCall,
    ToolResult,
    build_tools_prompt,
    execute_tool,
    parse_tool_calls,
)
from src.core.vector_store import query as vector_query

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_AGENT_STEPS = 15
MAX_TOOL_OUTPUT_LEN = 4_000  # Truncate tool output in messages to save context
MAX_TOOL_CALLS_PER_STEP = 5  # Cap tool calls per step to control context growth
MAX_STEP_OUTPUT_LEN = 10_000  # Max total tool output injected per step


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class AgentStep:
    """One iteration of the agent loop."""

    step_num: int
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    duration: float = 0.0
    is_final: bool = False


@dataclass
class AgentResponse:
    """Final result from the agent after all steps."""

    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    total_duration: float = 0.0
    total_tool_calls: int = 0
    provider: str = ""


# ---------------------------------------------------------------------------
# Agent system prompt (extends the base brain prompt)
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """\
You are **Mnemosyne v3.1**, an autonomous agentic coding assistant living in \
the developer's terminal.  You combine the precision of a compiler with the \
creativity of a senior architect.

You have full agency: you can read files, search code, run commands, and \
browse the web to gather information before answering.  Use your tools \
strategically -- verify assumptions, explore the codebase, and build a \
thorough understanding before responding.

{tools}

APPROACH:
1. **Assess** the request -- what do you need to know?
2. **Gather** information using tools (don't guess -- verify).
3. **Synthesise** a thorough, accurate answer using what you found.
4. For code changes: always read the target file first, then provide complete content.

CRITICAL FILE FORMAT RULE:
When writing code meant for files, you MUST use this exact fence format:

```language:relative/path/to/file.ext
<complete file content>
```

Examples:
```python:src/main.py
print("hello")
```
```html:public/index.html
<!DOCTYPE html>...
```

RULES:
1. **Mirror the user's language** -- French -> French, English -> English.
2. When writing/modifying files, provide COMPLETE file contents in the fenced format above.
3. Do NOT output file blocks for explanations, greetings, or analysis.
4. For project scaffolding: directory tree first, then ALL files.
5. Be concise, technical, direct.  Zero fluff.  Density over length.
6. Answer general questions freely.
7. Each file gets ONE fenced block.  NEVER duplicate.
8. When modifying existing files: output the ENTIRE new content.
9. Proactively flag issues: security holes, perf traps, missing edge cases.
10. When you don't need tools, just answer directly -- don't force tool usage.

DEEP REASONING:
For complex tasks, use a <thinking> block to plan step-by-step before acting.

CODE QUALITY:
- Prefer composition over inheritance.
- Descriptive names; no cryptic abbreviations.
- Handle errors explicitly.
- Follow existing codebase conventions.
"""


def _build_agent_system_prompt(memory: str = "") -> str:
    """Build the complete agent system prompt with tools and memory."""
    prompt = AGENT_SYSTEM_PROMPT.format(tools=build_tools_prompt())
    if memory:
        prompt += f"\n\n## Project Memory\n{memory}"
    return prompt


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------


# Use shared resolve_config from config module


# ---------------------------------------------------------------------------
# Response cleaning
# ---------------------------------------------------------------------------

_THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)


def extract_thinking(text: str) -> tuple[str, str]:
    """
    Extract <thinking> blocks from text.
    Returns (clean_text, thinking_content).
    """
    thinking_parts: list[str] = []
    for m in _THINKING_RE.finditer(text):
        thinking_parts.append(m.group(1).strip())
    clean = _THINKING_RE.sub("", text).strip()
    return clean, "\n".join(thinking_parts)


# ---------------------------------------------------------------------------
# Native tool-call translation
# ---------------------------------------------------------------------------

# Maps native/external tool names to Mnemosyne tool equivalents
_NATIVE_TOOL_MAP: dict[str, tuple[str, dict[str, str]]] = {
    # (mnemosyne_tool_name, {native_param -> mnemosyne_param})
    "repo_browser.print_tree": ("list_directory", {"path": "path"}),
    "repo_browser.list_dir": ("list_directory", {"path": "path"}),
    "repo_browser.list_directory": ("list_directory", {"path": "path"}),
    "repo_browser.read_file": ("read_file", {"path": "path", "file_path": "path"}),
    "file_reader": ("read_file", {"path": "path", "file_path": "path"}),
    "file_search": ("find_files", {"query": "pattern", "pattern": "pattern"}),
    "code_search": ("search_codebase", {"query": "query"}),
    "grep": ("grep_search", {"pattern": "pattern", "query": "pattern"}),
    "web_search": ("search_web", {"query": "query"}),
    "terminal": ("run_command", {"command": "command"}),
    "shell": ("run_command", {"command": "command"}),
    "execute_command": ("run_command", {"command": "command"}),
    "run_terminal_command": ("run_command", {"command": "command"}),
}

# Keywords in native tool names → best Mnemosyne tool fallback
_TOOL_KEYWORD_FALLBACK: dict[str, str] = {
    "read": "read_file",
    "file": "read_file",
    "list": "list_directory",
    "tree": "list_directory",
    "dir": "list_directory",
    "browse": "list_directory",
    "search": "search_codebase",
    "find": "find_files",
    "grep": "grep_search",
    "web": "search_web",
    "shell": "run_command",
    "terminal": "run_command",
    "command": "run_command",
    "exec": "run_command",
}


def _translate_native_tool_call(failed_gen: str) -> str:
    """
    Convert a native/function-calling tool invocation (from failed_generation)
    into Mnemosyne's text-based <tool_call> format so the agent loop can
    process it like any normal response.
    """
    import json as _json

    try:
        data = _json.loads(failed_gen.strip())
    except _json.JSONDecodeError:
        # Not valid JSON -- return as plain text so the loop treats it
        # as a final answer (no tool calls detected).
        return failed_gen

    native_name: str = data.get("name", "")
    native_args: dict = data.get("arguments", data.get("params", {}))

    # Try to map to a known Mnemosyne tool
    mapped = _NATIVE_TOOL_MAP.get(native_name)
    if mapped:
        mn_name, param_map = mapped
        mn_params = {}
        for nk, nv in native_args.items():
            mk = param_map.get(nk, nk)
            mn_params[mk] = nv
        return (
            f'<tool_call>\n'
            f'{_json.dumps({"name": mn_name, "params": mn_params})}\n'
            f'</tool_call>'
        )

    # Fuzzy fallback: match keywords in the native tool name
    name_lower = native_name.lower()
    for keyword, mn_tool in _TOOL_KEYWORD_FALLBACK.items():
        if keyword in name_lower:
            return (
                f'<tool_call>\n'
                f'{_json.dumps({"name": mn_tool, "params": native_args})}\n'
                f'</tool_call>'
            )

    # Unknown tool -- direct pass-through
    # (execute_tool will return an "unknown tool" error, which is fine)
    return (
        f'<tool_call>\n'
        f'{_json.dumps({"name": native_name, "params": native_args})}\n'
        f'</tool_call>'
    )


# ---------------------------------------------------------------------------
# Context-size management
# ---------------------------------------------------------------------------


def _estimate_tokens(msgs: list[BaseMessage]) -> int:
    """Rough token estimate for messages (4 chars ≈ 1 token)."""
    total = 0
    for m in msgs:
        c = m.content if isinstance(m.content, str) else str(m.content)
        total += len(c) // 4
    return total


def _trim_messages_to_budget(msgs: list[BaseMessage], budget: int = 6000) -> list[BaseMessage]:
    """
    If estimated token count exceeds ``budget``, aggressively trim
    tool-result messages (the biggest contributor) from the middle,
    keeping the system prompt and the most recent exchanges.
    """
    if _estimate_tokens(msgs) <= budget:
        return msgs

    # Keep system prompt (index 0) and last 4 messages always
    keep_head = 1
    keep_tail = 4
    if len(msgs) <= keep_head + keep_tail:
        return msgs

    head = msgs[:keep_head]
    tail = msgs[-keep_tail:]
    middle = msgs[keep_head:-keep_tail]

    # Trim middle messages from oldest first
    trimmed_middle: list[BaseMessage] = []
    remaining_budget = budget - _estimate_tokens(head) - _estimate_tokens(tail)

    for m in reversed(middle):  # keep newest middle messages
        cost = len(m.content if isinstance(m.content, str) else str(m.content)) // 4
        if remaining_budget - cost > 0:
            trimmed_middle.insert(0, m)
            remaining_budget -= cost
        # else: drop this message

    return head + trimmed_middle + tail


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

# Event types for the callback:
#   ("thinking",   {"step": int})
#   ("reasoning",  {"text": str})
#   ("thinking_block", {"text": str})
#   ("tool_start", {"call": ToolCall})
#   ("tool_end",   {"call": ToolCall, "result": ToolResult})
#   ("step_done",  {"step": AgentStep})
#   ("error",      {"error": str})

EventCallback = Callable[[str, dict[str, Any]], None]


def run_agent(
    question: str,
    *,
    history: list[BaseMessage] | None = None,
    n_results: int = 8,
    provider_override: str | None = None,
    model_override: str | None = None,
    filter_meta: dict[str, Any] | None = None,
    on_event: EventCallback | None = None,
) -> AgentResponse:
    """
    Run the autonomous agent loop.

    The agent invokes the LLM, parses tool calls from the response,
    executes them, feeds results back, and repeats until the LLM
    produces a final answer (no tool calls).

    ``on_event`` is called for each significant event so the UI can
    show real-time progress.
    """
    cfg = resolve_config(provider_override, model_override)
    llm = get_llm(cfg)

    # -- RAG context -------------------------------------------------------
    effective_n = max(n_results, _estimate_complexity(question))
    search_query = _rewrite_query_for_retrieval(question)
    rag_results = vector_query(search_query, n_results=effective_n, filter_meta=filter_meta)
    rag_results = _boost_results(rag_results, question)
    ctx = _build_context_block(rag_results)
    mem = _load_episodic_memory()

    sources: list[str] = list(dict.fromkeys(r.source for r in rag_results))

    # -- Build messages ----------------------------------------------------
    sys_prompt = _build_agent_system_prompt(mem)
    msgs: list[BaseMessage] = [SystemMessage(content=sys_prompt)]

    # Conversation history (compacted if needed)
    if history:
        if len(history) > 12:
            msgs.extend(summarise_history(history, keep_last=6))
        else:
            msgs.extend(history)

    # User message with RAG context
    user_msg = f"[Codebase Context]\n{ctx}\n\n{question}" if ctx else question
    msgs.append(HumanMessage(content=user_msg))

    # -- Emit helper -------------------------------------------------------
    def emit(event: str, data: dict[str, Any] | None = None) -> None:
        if on_event:
            on_event(event, data or {})

    # -- Agent loop --------------------------------------------------------
    steps: list[AgentStep] = []
    total_tool_calls = 0
    t_total = time.perf_counter()

    for step_num in range(1, MAX_AGENT_STEPS + 1):
        emit("thinking", {"step": step_num})

        t0 = time.perf_counter()
        try:
            # Trim context if it exceeds budget (prevents 413 errors)
            msgs = _trim_messages_to_budget(msgs)
            response = llm.invoke(msgs)
            raw_content = response.content
            response_text: str = raw_content if isinstance(raw_content, str) else str(raw_content)
        except Exception as e:
            err_str = str(e)
            # Handle models that attempt native tool calls when tool_choice is none.
            # Instead of retrying (which often fails again), translate the native
            # tool call into Mnemosyne's text-based <tool_call> format so the
            # agent loop can process it normally.
            if "tool_use_failed" in err_str or "Tool choice is none" in err_str:
                import json as _json

                failed_gen = ""
                try:
                    # The error format is: "Error code: 400 - {json_payload}"
                    # but some providers return Python dict repr with single quotes.
                    raw_payload = err_str.split(" - ", 1)[1]
                    # Try standard JSON first
                    try:
                        payload = _json.loads(raw_payload)
                    except _json.JSONDecodeError:
                        # Fallback: convert Python repr (single quotes) to JSON
                        import ast as _ast
                        payload = _ast.literal_eval(raw_payload)
                    failed_gen = payload.get("error", {}).get("failed_generation", "")
                except Exception:
                    # Last resort: regex extract from the raw error string
                    import re as _re
                    m = _re.search(r"'failed_generation':\s*'(.*?)'(?:}|,)", err_str)
                    if not m:
                        m = _re.search(r'"failed_generation":\s*"(.*?)"(?:}|,)', err_str)
                    if m:
                        failed_gen = m.group(1)

                if failed_gen:
                    response_text = _translate_native_tool_call(failed_gen)
                else:
                    emit("error", {"error": err_str})
                    return AgentResponse(
                        answer=f"Error in agent step {step_num}: {e}",
                        steps=steps,
                        sources=sources,
                        total_duration=time.perf_counter() - t_total,
                        total_tool_calls=total_tool_calls,
                        provider=cfg.llm_provider,
                    )
            # Handle context/token limit errors (413, rate_limit_exceeded)
            elif "rate_limit" in err_str or "413" in err_str or "too large" in err_str.lower() or "Request too large" in err_str:
                emit("error", {"error": "Context limit reached — summarising collected information."})
                # Try to salvage: gather reasoning from previous steps
                gathered: list[str] = []
                for prev in steps:
                    if prev.reasoning:
                        gathered.append(prev.reasoning)
                    for tr in prev.tool_results:
                        if tr.success and tr.output:
                            # Include just the first 200 chars of each tool result
                            gathered.append(f"[{tr.name}]: {tr.output[:200]}")
                fallback_answer = (
                    "I gathered some information but hit the model's token limit before I could synthesise a full answer.\n\n"
                    + "\n".join(gathered[:10])
                ) if gathered else f"Error: token limit exceeded at step {step_num}. Try a simpler question or a model with a larger context window."
                return AgentResponse(
                    answer=fallback_answer,
                    steps=steps,
                    sources=sources,
                    total_duration=time.perf_counter() - t_total,
                    total_tool_calls=total_tool_calls,
                    provider=cfg.llm_provider,
                )
            else:
                emit("error", {"error": err_str})
                return AgentResponse(
                    answer=f"Error in agent step {step_num}: {e}",
                    steps=steps,
                    sources=sources,
                    total_duration=time.perf_counter() - t_total,
                    total_tool_calls=total_tool_calls,
                    provider=cfg.llm_provider,
                )
        step_duration = time.perf_counter() - t0

        # Parse tool calls
        reasoning, tool_calls = parse_tool_calls(response_text)

        # Extract <thinking> blocks from reasoning
        clean_reasoning, thinking_text = extract_thinking(reasoning)
        if thinking_text:
            emit("thinking_block", {"text": thinking_text})

        # -- No tool calls → final answer ----------------------------------
        if not tool_calls:
            step = AgentStep(
                step_num=step_num,
                reasoning=response_text,
                duration=step_duration,
                is_final=True,
            )
            steps.append(step)
            emit("step_done", {"step": step})

            # Clean the final answer (remove thinking tags, tool call remnants)
            final_answer, _ = extract_thinking(response_text)

            return AgentResponse(
                answer=final_answer,
                steps=steps,
                sources=sources,
                total_duration=time.perf_counter() - t_total,
                total_tool_calls=total_tool_calls,
                provider=cfg.llm_provider,
            )

        # -- Tool calls found → execute and loop ---------------------------
        if clean_reasoning:
            emit("reasoning", {"text": clean_reasoning})

        # Cap tool calls per step to control context growth
        if len(tool_calls) > MAX_TOOL_CALLS_PER_STEP:
            tool_calls = tool_calls[:MAX_TOOL_CALLS_PER_STEP]

        step = AgentStep(
            step_num=step_num,
            reasoning=clean_reasoning,
            tool_calls=tool_calls,
            duration=step_duration,
        )

        # Add assistant message to history
        msgs.append(AIMessage(content=response_text))

        # Execute each tool, tracking total output size
        tool_output_parts: list[str] = []
        total_step_output = 0
        for tc in tool_calls:
            total_tool_calls += 1
            emit("tool_start", {"call": tc})

            result = execute_tool(tc)
            step.tool_results.append(result)

            emit("tool_end", {"call": tc, "result": result})

            # Track sources from codebase search
            if tc.name == "search_codebase" and result.success:
                for m in re.finditer(r"### (\S+)", result.output):
                    src = m.group(1)
                    if src not in sources:
                        sources.append(src)

            # Truncate individual output
            output = result.output
            if len(output) > MAX_TOOL_OUTPUT_LEN:
                output = output[:MAX_TOOL_OUTPUT_LEN] + "\n... (truncated)"

            # Check if adding this would exceed the per-step budget
            entry = f"[Tool: {tc.name}] [{'SUCCESS' if result.success else 'FAILED'}]\n{output}"
            if total_step_output + len(entry) > MAX_STEP_OUTPUT_LEN and tool_output_parts:
                tool_output_parts.append(
                    f"[Tool: {tc.name}] [SKIPPED] (output omitted to stay within context budget)"
                )
                total_step_output += 80
                continue

            tool_output_parts.append(entry)
            total_step_output += len(entry)

        # Inject tool results as a user message
        tool_results_text = "\n\n---\n\n".join(tool_output_parts)
        msgs.append(HumanMessage(content=f"[Tool Results — step {step_num}]\n\n{tool_results_text}"))

        steps.append(step)
        emit("step_done", {"step": step})

    # -- Max steps reached -------------------------------------------------
    last_reasoning = steps[-1].reasoning if steps else ""
    return AgentResponse(
        answer=(
            "I've reached the maximum number of reasoning steps. "
            "Here is what I've gathered so far:\n\n" + last_reasoning
        ),
        steps=steps,
        sources=sources,
        total_duration=time.perf_counter() - t_total,
        total_tool_calls=total_tool_calls,
        provider=cfg.llm_provider,
    )
