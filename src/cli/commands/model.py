"""
Model/provider switching, filtering, web search, gemini delegation.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from rich import box
from rich.rule import Rule
from rich.table import Table

from src.cli.commands.files import (
    dedup_files,
    extract_blocks,
)
from src.cli.theme import FAIL, OK, WARN, C, D, G, M, R, Y, console

# ---------------------------------------------------------------------------
# /model, /provider, /filter
# ---------------------------------------------------------------------------


def cmd_set_model(model: str, model_holder: list[str | None]) -> None:
    if not model:
        console.print(f"  {WARN} usage: /model <model-name>\n")
        return
    model_holder[0] = model
    console.print(f"  {OK} model -> {C}{model}{R}\n")


def cmd_set_provider(prov: str, provider_holder: list[str | None], model_holder: list[str | None]) -> None:
    if not prov:
        console.print(f"  {WARN} usage: /provider <name>\n")
        return
    from src.core.providers import PROVIDERS

    if prov.lower() not in PROVIDERS:
        avail = ", ".join(sorted(PROVIDERS.keys()))
        console.print(f"  {FAIL} unknown: {prov}  ({D}available: {avail}{R})\n")
        return
    provider_holder[0] = prov.lower()
    model_holder[0] = None
    spec = PROVIDERS[prov.lower()]
    console.print(f"  {OK} provider -> {C}{spec.label}{R} ({D}{spec.default_model}{R})\n")


def cmd_set_filter(exts: str, filter_holder: list[set[str] | None]) -> None:
    if not exts:
        filter_holder[0] = None
        console.print(f"  {OK} filters cleared\n")
        return
    parts = {e.strip() if e.strip().startswith(".") else f".{e.strip()}" for e in exts.split(",")}
    clean = {e for e in parts if len(e) >= 2 and e.startswith(".")}
    if clean:
        filter_holder[0] = clean
        console.print(f"  {OK} filter set: {C}{', '.join(sorted(clean))}{R}\n")
    else:
        filter_holder[0] = None
        console.print(f"  {WARN} no valid extensions found (try: .py, .ts, .md)\n")


def cmd_toggle_readonly(readonly_holder: list[bool]) -> None:
    readonly_holder[0] = not readonly_holder[0]
    state = f"{G}ON{R}" if readonly_holder[0] else f"{D}OFF{R}"
    console.print(f"  {OK} readonly mode {state}\n")


# ---------------------------------------------------------------------------
# /web, /news
# ---------------------------------------------------------------------------


def cmd_web(query: str, history: list[BaseMessage]) -> None:
    if not query:
        console.print(f"  {WARN} usage: /web <query>\n")
        return
    from src.core.web import search_web

    console.print(f"  {D}searching web for '{query}'...{R}")
    res = search_web(query)
    history.append(HumanMessage(content=f"[Web Search: {query}]\n\n{res}"))
    console.print(f"  {OK} search results added to context\n")


def cmd_news(query: str, history: list[BaseMessage]) -> None:
    if not query:
        console.print(f"  {WARN} usage: /news <query>\n")
        return
    from src.core.web import search_web_news

    console.print(f"  {D}searching news for '{query}'...{R}")
    res = search_web_news(query)
    history.append(HumanMessage(content=f"[News Search: {query}]\n\n{res}"))
    console.print(f"  {OK} news results added to context\n")


# ---------------------------------------------------------------------------
# /save
# ---------------------------------------------------------------------------


def cmd_save(path: str, history: list[BaseMessage]) -> None:
    out = Path(path) if path else Path(f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    lines: list[str] = [f"# Mnemosyne Chat -- {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"]
    for m in history:
        if isinstance(m, HumanMessage):
            lines.append(f"## User\n\n{m.content}\n\n---\n\n")
        else:
            lines.append(f"## Assistant\n\n{m.content}\n\n---\n\n")
    try:
        out.write_text("".join(lines), encoding="utf-8")
        console.print(f"  {OK} saved to {C}{out}{R}\n")
    except Exception as exc:
        console.print(f"  {FAIL} {exc}\n")


# ---------------------------------------------------------------------------
# /gemini, /gemini-interactive
# ---------------------------------------------------------------------------


def cmd_gemini(
    query: str,
    history: list[BaseMessage],
    filter_ext: set[str] | None,
    last_holder: list[str],
    readonly: bool,
    confirm_fn,
) -> None:
    """Delegate a query to Gemini CLI with full RAG context."""
    if not query:
        console.print(f"  {WARN} usage: /gemini <query>\n")
        console.print(f"  {D}  /gemini explain the auth flow{R}")
        console.print(f"  {D}  /gemini-interactive  (full session){R}\n")
        return

    from src.core.gemini_cli import (
        build_context_prompt,
        get_install_instructions,
        is_gemini_cli_installed,
        query_streaming,
    )

    if not is_gemini_cli_installed():
        console.print(f"  {FAIL} Gemini CLI not found on PATH\n")
        console.print(f"  {D}{get_install_instructions()}{R}\n")
        return

    console.print(f"  {M}preparing context for Gemini CLI...{R}")

    codebase_ctx = ""
    memory = ""
    history_summary = ""

    try:
        from src.core.brain import load_episodic_memory, rewrite_query
        from src.core.vector_store import query as vq

        fmeta: dict[str, Any] | None = None
        if filter_ext:
            fmeta = {"extension": {"$in": list(filter_ext)}}

        search_q = rewrite_query(query)
        results = vq(search_q, n_results=10, filter_meta=fmeta)
        if results:
            parts = [f"### `{r.source}`\n```\n{r.content}\n```" for r in results[:8]]
            codebase_ctx = "\n\n".join(parts)
        memory = load_episodic_memory()
    except Exception:
        pass

    if history:
        summary_parts: list[str] = []
        for m in history[-6:]:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            preview = m.content[:150].replace("\n", " ")
            summary_parts.append(f"[{role}]: {preview}...")
        history_summary = "\n".join(summary_parts)

    full_prompt = build_context_prompt(
        query,
        codebase_context=codebase_ctx,
        memory=memory,
        history_summary=history_summary,
    )

    console.print(Rule(f"{M}gemini{R}", style="magenta"))
    console.print()

    tokens: list[str] = []
    t0 = time.perf_counter()

    try:
        gen = query_streaming(full_prompt, cwd=Path.cwd())
        for chunk in gen:
            tokens.append(chunk)
            sys.stdout.write(chunk)
            sys.stdout.flush()
    except Exception as exc:
        console.print(f"\n  {FAIL} Gemini CLI error: {exc}\n")
        return

    elapsed = time.perf_counter() - t0
    full_output = "".join(tokens)

    sys.stdout.write("\n")
    console.print()
    console.print(Rule(style="magenta"))
    console.print(f"  {D}gemini  |  {elapsed:.1f}s  |  {len(full_output)} chars{R}")
    console.print()

    history.append(HumanMessage(content=f"[Delegated to Gemini CLI]: {query}"))
    history.append(AIMessage(content=f"[Gemini CLI response]:\n{full_output}"))
    last_holder[0] = full_output

    # Detect files in output
    files = dedup_files(extract_blocks(full_output))
    if files and not readonly:
        ft = Table(box=box.SIMPLE, show_header=False, padding=(0, 1), show_edge=False)
        ft.add_column("s", style="bold", width=3)
        ft.add_column("path", style="bold cyan")
        for f in files:
            exists = Path(f.path).exists()
            icon = f"{Y}>{R}" if exists else f"{G}+{R}"
            ft.add_row(icon, f.path)
        console.print(ft)
        confirm_fn(files)


def cmd_gemini_interactive() -> None:
    """Launch full interactive Gemini CLI session."""
    from src.core.gemini_cli import (
        get_install_instructions,
        is_gemini_cli_installed,
        launch_interactive,
    )

    if not is_gemini_cli_installed():
        console.print(f"  {FAIL} Gemini CLI not found on PATH\n")
        console.print(f"  {D}{get_install_instructions()}{R}\n")
        return

    console.print(f"\n  {M}launching Gemini CLI interactive session...{R}")
    console.print(f"  {D}type 'exit' or Ctrl-C in Gemini CLI to return to Mnemosyne{R}\n")

    exit_code = launch_interactive(cwd=Path.cwd())
    console.print(f"\n  {OK} returned to Mnemosyne {D}(gemini-cli exit: {exit_code}){R}\n")
