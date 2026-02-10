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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.core.config import get_config, MnemosyneConfig
from src.core.providers import get_llm
from src.core.tools import (
    ToolCall,
    ToolResult,
    parse_tool_calls,
    execute_tool,
    build_tools_prompt,
    strip_tool_calls,
    TOOL_REGISTRY,
)
from src.core.brain import (
    _load_episodic_memory,
    _rewrite_query_for_retrieval,
    _boost_results,
    _build_context_block,
    _estimate_complexity,
    summarise_history,
)
from src.core.vector_store import query as vector_query

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_AGENT_STEPS = 15
MAX_TOOL_OUTPUT_LEN = 8_000  # Truncate tool output in messages to save context


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
You are **Mnemosyne v3.0**, an autonomous agentic coding assistant living in \
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

def _resolve_config(
    provider_override: str | None = None,
    model_override: str | None = None,
) -> MnemosyneConfig:
    cfg = get_config()
    needs_rebuild = False
    d = cfg.model_dump()
    if provider_override:
        d["llm_provider"] = provider_override
        needs_rebuild = True
    if model_override:
        d["llm_model"] = model_override
        needs_rebuild = True
    return MnemosyneConfig(**d) if needs_rebuild else cfg


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
    cfg = _resolve_config(provider_override, model_override)
    llm = get_llm(cfg)

    # -- RAG context -------------------------------------------------------
    effective_n = max(n_results, _estimate_complexity(question))
    search_query = _rewrite_query_for_retrieval(question)
    rag_results = vector_query(
        search_query, n_results=effective_n, filter_meta=filter_meta
    )
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
            response = llm.invoke(msgs)
            response_text: str = response.content  # type: ignore
        except Exception as e:
            emit("error", {"error": str(e)})
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

        step = AgentStep(
            step_num=step_num,
            reasoning=clean_reasoning,
            tool_calls=tool_calls,
            duration=step_duration,
        )

        # Add assistant message to history
        msgs.append(AIMessage(content=response_text))

        # Execute each tool
        tool_output_parts: list[str] = []
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

            # Truncate output for context management
            output = result.output
            if len(output) > MAX_TOOL_OUTPUT_LEN:
                output = output[:MAX_TOOL_OUTPUT_LEN] + "\n... (truncated)"

            status = "SUCCESS" if result.success else "FAILED"
            tool_output_parts.append(
                f"[Tool: {tc.name}] [{status}]\n{output}"
            )

        # Inject tool results as a user message
        tool_results_text = "\n\n---\n\n".join(tool_output_parts)
        msgs.append(HumanMessage(
            content=f"[Tool Results — step {step_num}]\n\n{tool_results_text}"
        ))

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
