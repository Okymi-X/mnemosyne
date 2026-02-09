"""
Mnemosyne -- Brain (RAG Engine)
Agentic coding assistant with codebase context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

from typing import Generator, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from rich.console import Console

from src.core.config import get_config, MnemosyneConfig
from src.core.providers import get_llm
from src.core.vector_store import query as vector_query, QueryResult

console = Console(highlight=False)

SYSTEM_PROMPT = """\
You are **Mnemosyne**, a powerful agentic coding assistant running in the \
developer's terminal. Think of yourself as a senior developer pair-programmer.

You can help create projects, write code, analyze codebases, debug issues, \
scaffold apps, and answer technical questions.

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
1. Match the user's language (French → French, English → English).
2. IF AND ONLY IF you are writing a file, give COMPLETE file contents.
3. DO NOT generate file blocks if the user just asks for explanation, analysis, or says "hello"/"don't touch".
4. If the user says "no", "stop", or "cancel", do NOT output any code blocks.
5. For project scaffolding: list the structure first, then ALL files.
6. When codebase context is provided, reference the source files you used.
7. Be concise, technical, direct. No fluff.
8. You can answer general questions — don't force codebase context.
9. When creating multiple files, use a SEPARATE fenced block per file.
10. NEVER repeat the same file block twice.

DEEP REASONING:
For complex tasks (debugging, architecture, refactoring), start your response with a <thinking> block.
Inside <thinking>, analyze the request, plan your approach, and verify assumptions.
Close the </thinking> block before generating the final response.
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
    if not results:
        return ""
    parts = [
        f"### `{r.source}` (score: {r.score})\n```\n{r.content}\n```"
        for r in results
    ]
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
    q: str, ctx: str, mem: str, history: list[BaseMessage],
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
    return [r.source for r in results if r.source not in seen and not seen.add(r.source)]  # type: ignore


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
    if needs_rebuild:
        cfg = MnemosyneConfig(**d)
    return cfg


# -- Single-shot -------------------------------------------------------------

def ask(
    question: str, *, n_results: int = 8,
    project_root: Path | None = None, provider_override: str | None = None,
    filter_meta: dict[str, Any] | None = None,
) -> BrainResponse:
    cfg = _resolve_config(provider_override)
    res = vector_query(question, n_results=n_results, filter_meta=filter_meta)
    msgs = _build_messages(question, _build_context_block(res), _load_episodic_memory(project_root))
    try:
        answer: str = get_llm(cfg).invoke(msgs).content  # type: ignore
    except Exception as exc:
        return BrainResponse(answer=f"Error: {exc}", provider=cfg.llm_provider)
    return BrainResponse(answer=answer, sources=_extract_sources(res), provider=cfg.llm_provider)


# -- Streaming ---------------------------------------------------------------

def ask_streaming(
    question: str, *, history: list[BaseMessage] | None = None,
    n_results: int = 8, project_root: Path | None = None,
    provider_override: str | None = None,
    model_override: str | None = None,
    filter_meta: dict[str, Any] | None = None,
) -> Generator[str, None, tuple[str, list[str]]]:
    """Yields tokens. Returns (full_answer, sources) on StopIteration."""
    cfg = _resolve_config(provider_override, model_override)
    res = vector_query(question, n_results=n_results, filter_meta=filter_meta)
    ctx = _build_context_block(res)
    mem = _load_episodic_memory(project_root)
    msgs = _build_chat_messages(question, ctx, mem, history) if history else _build_messages(question, ctx, mem)

    tokens: list[str] = []
    for chunk in get_llm(cfg).stream(msgs):
        t = chunk.content  # type: ignore
        if t:
            tokens.append(t)
            yield t
    return "".join(tokens), _extract_sources(res)
