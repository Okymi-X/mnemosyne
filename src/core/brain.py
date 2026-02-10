"""
Mnemosyne -- Brain (RAG Engine) v2.0
Agentic coding assistant with adaptive context, query rewriting,
smart summarisation, and priority-based retrieval.
"""

from __future__ import annotations

import re
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from rich.console import Console

from src.core.config import resolve_config
from src.core.providers import get_llm
from src.core.vector_store import QueryResult
from src.core.vector_store import query as vector_query

console = Console(highlight=False)

# ---------------------------------------------------------------------------
# Query intelligence helpers
# ---------------------------------------------------------------------------

_COMPLEX_SIGNALS = re.compile(
    r"\b(refactor|architect|debug|optimise|optimize|redesign|migrate|"
    r"security|performance|scale|deploy|integrate|compare|trade-?off|"
    r"explain.*entire|full.*overview|how.*everything)\b",
    re.IGNORECASE,
)

_SIMPLE_SIGNALS = re.compile(
    r"\b(hello|hi|hey|thanks|merci|bonjour|what is|who are|version)\b",
    re.IGNORECASE,
)


def _estimate_complexity(query: str) -> int:
    """Return an adaptive n_results based on query complexity."""
    if _SIMPLE_SIGNALS.search(query):
        return 3
    if _COMPLEX_SIGNALS.search(query):
        return 14
    words = len(query.split())
    if words > 40:
        return 12
    if words > 20:
        return 10
    return 8


def _rewrite_query_for_retrieval(query: str) -> str:
    """
    Expand the user query with likely code-related keywords to
    improve vector-search recall.  Cheap heuristic -- no LLM call.
    """
    expansions: list[str] = [query]

    keyword_map: dict[str, list[str]] = {
        "auth": ["authentication", "login", "jwt", "token", "session"],
        "db": ["database", "query", "model", "migration", "schema"],
        "api": ["endpoint", "route", "handler", "controller", "request"],
        "test": ["unittest", "pytest", "spec", "mock", "fixture"],
        "deploy": ["dockerfile", "ci", "pipeline", "build", "release"],
        "style": ["css", "tailwind", "theme", "component", "layout"],
        "error": ["exception", "handle", "catch", "raise", "try"],
        "config": ["settings", "env", "environment", "dotenv", "toml"],
        "perf": ["performance", "cache", "optimise", "benchmark", "profil"],
    }
    q_lower = query.lower()
    for trigger, extras in keyword_map.items():
        if trigger in q_lower:
            expansions.extend(extras[:3])

    return " ".join(expansions)


def _boost_results(results: list[QueryResult], query: str) -> list[QueryResult]:
    """Re-rank results: boost files explicitly mentioned in the query."""
    q_lower = query.lower()
    boosted: list[QueryResult] = []
    for r in results:
        bonus = 0.0
        src = r.source.lower()
        if src in q_lower or Path(src).stem in q_lower:
            bonus = 0.15
        if "memory" in src:
            bonus = max(bonus, 0.10)
        boosted.append(
            QueryResult(
                content=r.content,
                source=r.source,
                score=min(r.score + bonus, 1.0),
                metadata=r.metadata,
            )
        )
    boosted.sort(key=lambda r: r.score, reverse=True)
    return boosted


def summarise_history(history: list[BaseMessage], keep_last: int = 4) -> list[BaseMessage]:
    """
    Compact conversation history: keep the last *keep_last* messages verbatim,
    summarise older ones into a single condensed message.
    Preserves auto-loaded file references so the model retains file context.
    """
    if len(history) <= keep_last:
        return history

    old = history[:-keep_last]
    recent = history[-keep_last:]

    summary_parts: list[str] = []
    file_refs: list[str] = []

    for m in old:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        content = m.content if isinstance(m.content, str) else str(m.content)

        # Keep track of file references for continuity
        if content.startswith("[auto-loaded:") or content.startswith("[File:"):
            end = content.find("]")
            if end > 0:
                file_refs.append(content[1:end])
            continue  # Don't summarise file loads, just track them

        preview = content[:300].replace("\n", " ")
        if len(content) > 300:
            preview += "..."
        summary_parts.append(f"[{role}]: {preview}")

    pieces: list[str] = ["[Conversation summary -- older turns condensed]"]
    if file_refs:
        pieces.append("Files referenced: " + ", ".join(file_refs))
    pieces.extend(summary_parts)

    summary_msg = HumanMessage(content="\n".join(pieces))
    return [summary_msg] + recent


SYSTEM_PROMPT = """\
You are **Mnemosyne v3.1**, an autonomous agentic coding assistant living in the \
developer's terminal.  You combine the precision of a compiler with the \
creativity of a senior architect.

You can create projects, write production-grade code, analyze codebases, \
debug subtle issues, scaffold complete apps, refactor at scale, and answer \
any technical question -- all while adapting to the developer's style.

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
1. **Mirror the user's language** -- French -> French, English -> English, mix -> match.
2. IF AND ONLY IF you are writing/modifying a file, provide COMPLETE file contents.
3. DO NOT output file blocks for explanations, greetings, analysis, or when told to stop.
4. For project scaffolding: present the directory tree first, then ALL files in order.
5. Reference source files from the codebase context when they informed your answer.
6. Be concise, technical, and direct.  Zero fluff.  Density over length.
7. Answer general questions freely -- never force codebase context where it's irrelevant.
8. Each file gets ONE fenced block.  NEVER duplicate a file.
9. When modifying an existing file, output the ENTIRE new file content -- no partial patches.
10. Proactively flag potential issues: security holes, performance traps, missing edge cases.

DEEP REASONING:
For complex tasks (debugging, architecture, refactoring, multi-file changes):
  - Open a <thinking> block.
  - Analyse the request, enumerate constraints, plan your approach step-by-step.
  - Verify assumptions against the provided codebase context.
  - Close </thinking>, then deliver the clean final answer.

CODE QUALITY PRINCIPLES:
- Prefer composition over inheritance.
- Use descriptive names; avoid abbreviations outside well-known idioms.
- Handle errors explicitly -- no silent swallows.
- Add brief inline comments only where the *why* isn't obvious.
- Follow the conventions already present in the codebase (detected from context).
"""


@dataclass
class BrainResponse:
    answer: str
    sources: list[str] = field(default_factory=list)
    provider: str = ""


def _load_episodic_memory(project_root: Path | None = None) -> str:
    root = project_root or Path.cwd()
    mem = root / "MEMORY.md"
    if mem.is_file():
        try:
            return mem.read_text(encoding="utf-8", errors="ignore").strip()
        except (OSError, PermissionError):
            return ""
    return ""


def _build_context_block(results: list[QueryResult]) -> str:
    """Build a ranked context block with relevance indicators."""
    if not results:
        return ""
    parts: list[str] = []
    for i, r in enumerate(results):
        if r.score >= 0.75:
            relevance = "[high]"
        elif r.score >= 0.5:
            relevance = "[mid]"
        else:
            relevance = "[low]"
        parts.append(f"### {relevance} `{r.source}` (relevance: {r.score:.0%})\n```\n{r.content}\n```")
    return "\n\n".join(parts)


def _build_messages(q: str, ctx: str, mem: str) -> list[BaseMessage]:
    parts: list[str] = []
    if mem:
        parts.append(f"## Project Memory\n{mem}")
    if ctx:
        parts.append(f"## Codebase Context\n{ctx}")
    parts.append(q)
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content="\n\n---\n\n".join(parts))]


def _build_chat_messages(
    q: str,
    ctx: str,
    mem: str,
    history: list[BaseMessage],
) -> list[BaseMessage]:
    sys = SYSTEM_PROMPT
    if mem:
        sys += f"\n\n## Project Memory\n{mem}"
    msgs: list[BaseMessage] = [SystemMessage(content=sys)]
    msgs.extend(history)
    user_msg = f"[Context]\n{ctx}\n\n{q}" if ctx else q
    msgs.append(HumanMessage(content=user_msg))
    return msgs


def _extract_sources(results: list[QueryResult]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for r in results:
        if r.source not in seen:
            seen.add(r.source)
            out.append(r.source)
    return out


# Public aliases for use by other modules
load_episodic_memory = _load_episodic_memory
rewrite_query = _rewrite_query_for_retrieval
build_context_block = _build_context_block
boost_results = _boost_results


# -- Single-shot -------------------------------------------------------------


def ask(
    question: str,
    *,
    n_results: int = 8,
    project_root: Path | None = None,
    provider_override: str | None = None,
    filter_meta: dict[str, Any] | None = None,
) -> BrainResponse:
    cfg = resolve_config(provider_override)
    # Adaptive retrieval -- adjust chunk count by query complexity
    effective_n = max(n_results, _estimate_complexity(question))
    search_query = _rewrite_query_for_retrieval(question)
    res = vector_query(search_query, n_results=effective_n, filter_meta=filter_meta)
    res = _boost_results(res, question)
    msgs = _build_messages(question, _build_context_block(res), _load_episodic_memory(project_root))
    try:
        raw_answer = get_llm(cfg).invoke(msgs).content
        answer: str = raw_answer if isinstance(raw_answer, str) else str(raw_answer)
    except Exception as exc:
        return BrainResponse(answer=f"Error: {exc}", provider=cfg.llm_provider)
    return BrainResponse(answer=answer, sources=_extract_sources(res), provider=cfg.llm_provider)


# -- Streaming ---------------------------------------------------------------


def ask_streaming(
    question: str,
    *,
    history: list[BaseMessage] | None = None,
    n_results: int = 8,
    project_root: Path | None = None,
    provider_override: str | None = None,
    model_override: str | None = None,
    filter_meta: dict[str, Any] | None = None,
) -> Generator[str, None, tuple[str, list[str]]]:
    """Yields tokens. Returns (full_answer, sources) on StopIteration."""
    cfg = resolve_config(provider_override, model_override)

    # Adaptive retrieval: adjust chunk count by query complexity
    effective_n = max(n_results, _estimate_complexity(question))
    search_query = _rewrite_query_for_retrieval(question)
    res = vector_query(search_query, n_results=effective_n, filter_meta=filter_meta)
    res = _boost_results(res, question)

    ctx = _build_context_block(res)
    mem = _load_episodic_memory(project_root)

    # If history is very long, auto-compact older turns
    compact_history = history
    if history and len(history) > 12:
        compact_history = summarise_history(history, keep_last=6)

    msgs = (
        _build_chat_messages(question, ctx, mem, compact_history)
        if compact_history
        else _build_messages(question, ctx, mem)
    )

    tokens: list[str] = []
    for chunk in get_llm(cfg).stream(msgs):
        t: str = chunk.content if isinstance(chunk.content, str) else str(chunk.content)  # type: ignore[assignment]
        if t:
            tokens.append(t)
            yield t
    return "".join(tokens), _extract_sources(res)
