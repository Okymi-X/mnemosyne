"""
Mnemosyne v3.0 -- Agent Tool System
Defines the tool registry, implementations, and parsing for the
autonomous agent loop. Each tool is a self-contained function that
the LLM can invoke via structured <tool_call> tags.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.cli.theme import IGNORED


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    """A tool available to the agent."""
    name: str
    description: str
    parameters: dict[str, str]   # param_name -> description
    handler: Callable[..., str]


@dataclass
class ToolCall:
    """A parsed tool invocation from LLM output."""
    name: str
    params: dict[str, Any]


@dataclass
class ToolResult:
    """Result of executing a tool."""
    name: str
    output: str
    success: bool
    duration: float = 0.0


# ---------------------------------------------------------------------------
# Parse tool calls from LLM output
# ---------------------------------------------------------------------------

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def parse_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """
    Extract tool calls and reasoning text from LLM output.
    Returns (reasoning_text, list_of_tool_calls).
    """
    calls: list[ToolCall] = []
    for m in TOOL_CALL_RE.finditer(text):
        try:
            data = json.loads(m.group(1))
            calls.append(ToolCall(
                name=data.get("name", "unknown"),
                params=data.get("params", {}),
            ))
        except json.JSONDecodeError:
            continue
    reasoning = TOOL_CALL_RE.sub("", text).strip()
    return reasoning, calls


def strip_tool_calls(text: str) -> str:
    """Remove all <tool_call> blocks from text."""
    return TOOL_CALL_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _read_file(path: str = ".", **_: Any) -> str:
    """Read the full contents of a file."""
    p = Path(path)
    if not p.is_file():
        return f"Error: file not found: {path}"
    size = p.stat().st_size
    if size > 500_000:
        return f"Error: file too large ({size:,} bytes). Try a specific section."
    try:
        content = p.read_text(encoding="utf-8", errors="ignore")
        lines = content.count("\n") + 1
        return f"[{path} — {lines} lines, {size:,}B]\n{content}"
    except Exception as e:
        return f"Error reading {path}: {e}"


def _list_directory(path: str = ".", **_: Any) -> str:
    """List files and subdirectories."""
    p = Path(path)
    if not p.is_dir():
        return f"Error: not a directory: {path}"
    items: list[str] = []
    dirs = files = 0
    try:
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        return f"Error: permission denied: {path}"
    for item in entries:
        if item.name in IGNORED:
            continue
        if item.is_dir():
            items.append(f"  {item.name}/")
            dirs += 1
        else:
            try:
                sz = item.stat().st_size
                items.append(f"  {item.name}  ({sz:,}B)")
            except OSError:
                items.append(f"  {item.name}")
            files += 1
        if dirs + files >= 60:
            items.append("  ... (truncated)")
            break
    header = f"{p.name}/  ({dirs} dirs, {files} files)"
    return header + "\n" + "\n".join(items)


def _find_files(pattern: str = "", **_: Any) -> str:
    """Find files matching a name pattern recursively."""
    if not pattern:
        return "Error: pattern required"
    cwd = Path.cwd()
    hits: list[str] = []
    for p in cwd.rglob(f"*{pattern}*"):
        rel = str(p.relative_to(cwd))
        if set(rel.replace("\\", "/").split("/")) & IGNORED:
            continue
        suffix = "/" if p.is_dir() else ""
        hits.append(f"  {rel}{suffix}")
        if len(hits) >= 40:
            hits.append("  ... (truncated)")
            break
    return "\n".join(hits) if hits else f"No files matching: {pattern}"


def _grep_search(pattern: str = "", **_: Any) -> str:
    """Search inside file contents for a text pattern."""
    if not pattern:
        return "Error: pattern required"
    # Try git-grep first (faster)
    try:
        r = subprocess.run(
            ["git", "grep", "-n", "--color=never", "-I", pattern],
            capture_output=True, text=True, timeout=10, cwd=Path.cwd(),
        )
        if r.stdout:
            lines = r.stdout.strip().split("\n")
            result = "\n".join(lines[:35])
            if len(lines) > 35:
                result += f"\n... ({len(lines)} total matches)"
            return result
    except Exception:
        pass
    # Fallback: manual search
    cwd = Path.cwd()
    hits: list[str] = []
    for p in cwd.rglob("*"):
        if not p.is_file() or p.stat().st_size > 500_000:
            continue
        if set(str(p.relative_to(cwd)).replace("\\", "/").split("/")) & IGNORED:
            continue
        try:
            for i, ln in enumerate(
                p.read_text("utf-8", errors="ignore").split("\n"), 1
            ):
                if pattern.lower() in ln.lower():
                    hits.append(f"{p.relative_to(cwd)}:{i}: {ln.strip()}")
                    if len(hits) >= 35:
                        break
        except Exception:
            continue
        if len(hits) >= 35:
            break
    return "\n".join(hits) if hits else f"No matches for: {pattern}"


def _search_codebase(query: str = "", n_results: int = 10, **_: Any) -> str:
    """Semantic search across the indexed codebase via vector store."""
    if not query:
        return "Error: query required"
    try:
        from src.core.vector_store import query as vq
        from src.core.brain import _rewrite_query_for_retrieval, _boost_results

        search_q = _rewrite_query_for_retrieval(query)
        results = vq(search_q, n_results=n_results)
        results = _boost_results(results, query)

        if not results:
            return "No relevant code found in the index."

        parts: list[str] = []
        for r in results:
            parts.append(
                f"### {r.source} ({r.score:.0%})\n```\n{r.content}\n```"
            )
        return "\n\n".join(parts)
    except Exception as e:
        return f"Search failed: {e}"


def _search_web(query: str = "", **_: Any) -> str:
    """Search the web via DuckDuckGo."""
    if not query:
        return "Error: query required"
    try:
        from src.core.web import search_web
        return search_web(query, max_results=6)
    except Exception as e:
        return f"Web search failed: {e}"


def _run_command(command: str = "", **_: Any) -> str:
    """Execute a shell command and return output."""
    if not command:
        return "Error: command required"
    # Block dangerous commands
    dangerous = [
        "rm -rf /", "rm -rf ~", "format c:", "del /f /s /q c:",
        "mkfs", ":(){:|:&};:", "dd if=/dev/zero",
    ]
    cmd_lower = command.lower().strip()
    if any(d in cmd_lower for d in dangerous):
        return "Error: blocked potentially destructive command."
    try:
        r = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=45, cwd=Path.cwd(),
        )
        output = ""
        if r.stdout:
            output += r.stdout.strip()
        if r.stderr:
            if output:
                output += "\n"
            output += r.stderr.strip()
        if r.returncode != 0:
            output += f"\n[exit code: {r.returncode}]"
        # Truncate very long output
        if len(output) > 12_000:
            output = output[:12_000] + "\n... (output truncated)"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out (45s limit)"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Tool] = {
    "read_file": Tool(
        name="read_file",
        description="Read the full contents of a file from disk.",
        parameters={"path": "Relative or absolute file path to read"},
        handler=_read_file,
    ),
    "list_directory": Tool(
        name="list_directory",
        description="List files and subdirectories at a given path.",
        parameters={"path": "Directory path (default: current directory)"},
        handler=_list_directory,
    ),
    "find_files": Tool(
        name="find_files",
        description="Find files by name pattern (glob-style). Searches recursively.",
        parameters={"pattern": "Filename pattern to match (e.g. 'auth', '.py', 'config')"},
        handler=_find_files,
    ),
    "grep_search": Tool(
        name="grep_search",
        description="Search for a text pattern inside file contents. Returns file:line:match.",
        parameters={"pattern": "Text pattern to search for in file contents"},
        handler=_grep_search,
    ),
    "search_codebase": Tool(
        name="search_codebase",
        description="Semantic vector search across the indexed codebase. Best for conceptual or architectural queries.",
        parameters={"query": "Natural language search query about the codebase"},
        handler=_search_codebase,
    ),
    "search_web": Tool(
        name="search_web",
        description="Search the web via DuckDuckGo. Use for external docs, APIs, or current information.",
        parameters={"query": "Web search query"},
        handler=_search_web,
    ),
    "run_command": Tool(
        name="run_command",
        description="Execute a shell command and return stdout+stderr. Use for builds, tests, git, linters, etc.",
        parameters={"command": "Shell command to execute"},
        handler=_run_command,
    ),
}


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def execute_tool(call: ToolCall) -> ToolResult:
    """Execute a single tool call and return the result."""
    tool = TOOL_REGISTRY.get(call.name)
    if not tool:
        return ToolResult(
            name=call.name,
            output=f"Error: unknown tool '{call.name}'. Available: {', '.join(TOOL_REGISTRY)}",
            success=False,
        )
    t0 = time.perf_counter()
    try:
        output = tool.handler(**call.params)
        elapsed = time.perf_counter() - t0
        return ToolResult(name=call.name, output=output, success=True, duration=elapsed)
    except TypeError as e:
        elapsed = time.perf_counter() - t0
        return ToolResult(name=call.name, output=f"Bad params: {e}", success=False, duration=elapsed)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return ToolResult(name=call.name, output=f"Error: {e}", success=False, duration=elapsed)


# ---------------------------------------------------------------------------
# Prompt builder -- generates tool docs for the system prompt
# ---------------------------------------------------------------------------

def build_tools_prompt() -> str:
    """Build the tools documentation section for the agent system prompt."""
    lines = [
        "## Agent Tools",
        "",
        "You have access to tools that let you explore, understand, and modify the codebase.",
        "To invoke a tool, include a <tool_call> block in your response:",
        "",
        "<tool_call>",
        '{"name": "tool_name", "params": {"param1": "value1"}}',
        "</tool_call>",
        "",
        "RULES:",
        "- You can make MULTIPLE tool calls in a single response.",
        "- After tool execution, you receive results and can continue reasoning.",
        "- When you have enough information, provide your final answer WITHOUT any <tool_call> blocks.",
        "- Only use tools when you need information you DON'T already have.",
        "- If codebase context is already provided, use it before calling tools.",
        "- Prefer search_codebase for conceptual queries, grep_search for exact text matches.",
        "- Always read a file before modifying it to understand its current state.",
        "",
        "Available tools:",
        "",
    ]
    for name, tool in TOOL_REGISTRY.items():
        params_str = ", ".join(f"{k}: {v}" for k, v in tool.parameters.items())
        lines.append(f"  **{name}**({params_str})")
        lines.append(f"    {tool.description}")
        lines.append("")

    return "\n".join(lines)
