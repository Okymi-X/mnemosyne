"""
Mnemosyne v3.0 -- CLI Entry Point
Typer interface built with Rich.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from src.cli.theme import BRAND, FAIL, OK, TABLE_BOX, WARN, console

# Force UTF-8 output on Windows to avoid CP1252 encoding errors
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass


def _version_callback(value: bool) -> None:
    if value:
        from importlib.metadata import version

        try:
            v = version("mnemosyne")
        except Exception:
            v = "dev"
        console.print(f"mnemosyne {v}")
        raise typer.Exit()


app = typer.Typer(
    name="mnemosyne",
    help="Mnemosyne -- Autonomous agentic coding assistant with infinite context.",
    add_completion=False,
    no_args_is_help=True,
)


@app.callback()
def _main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Mnemosyne -- Autonomous agentic coding assistant with infinite context."""


# -- Valid providers for CLI help --------------------------------------------
PROVIDER_HELP = "LLM provider override (google|anthropic|groq|openrouter|openai|ollama)."


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@app.command()
def init(
    path: str = typer.Argument(
        ".",
        help="Root directory for the project (default: current dir).",
    ),
) -> None:
    """Initialise Mnemosyne in a project directory."""
    root = Path(path).resolve()
    mnemosyne_dir = root / ".mnemosyne"
    mnemosyne_dir.mkdir(parents=True, exist_ok=True)

    # Scaffold MEMORY.md if it doesn't exist
    memory_file = root / "MEMORY.md"
    if not memory_file.exists():
        memory_file.write_text(
            "# Project Memory -- Mnemosyne\n\n"
            "<!-- Write architectural decisions, conventions, and notes here.\n"
            "     Mnemosyne treats this file as HIGH-PRIORITY context. -->\n",
            encoding="utf-8",
        )
        console.print(f"{OK} Created {memory_file}")
    else:
        console.print(f"{WARN} MEMORY.md already exists -- skipping.")

    console.print(
        Panel(
            f"{OK} {BRAND} initialised at [bold cyan]{root}[/bold cyan]\n   db: [dim]{mnemosyne_dir / 'chroma'}[/dim]",
            title="[bright_green]init[/bright_green]",
            border_style="bright_green",
            padding=(0, 1),
        )
    )


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@app.command()
def ingest(
    path: str = typer.Argument(
        ...,
        help="Directory to scan and ingest into the knowledge base.",
    ),
    ext: list[str] = typer.Option(
        [],
        "--ext",
        "-e",
        help="Extra file extensions to include (e.g. --ext .log --ext .csv).",
    ),
) -> None:
    """Scan a directory and ingest code files into the vector store."""
    from src.core.ingester import chunk_documents, scan_directory
    from src.core.vector_store import add_documents

    root = Path(path).resolve()
    if not root.is_dir():
        console.print(f"{FAIL} Not a valid directory: [cyan]{root}[/cyan]")
        raise typer.Exit(code=1)

    # Normalise extra extensions (ensure leading dot)
    extra: set[str] | None = None
    if ext:
        extra = {e if e.startswith(".") else f".{e}" for e in ext}
        console.print(f"{OK} Extra extensions: [cyan]{', '.join(sorted(extra))}[/cyan]")

    console.print(f"\n{OK} Scanning [cyan]{root}[/cyan] for code files ...\n")

    with console.status("[bright_green]Scanning files...[/bright_green]"):
        try:
            docs = list(scan_directory(root, extra_extensions=extra))
        except Exception as e:
            console.print(f"{FAIL} Scan failed: {e}")
            raise typer.Exit(code=1)

    if not docs:
        console.print(f"{WARN} No code files found.")
        raise typer.Exit(code=0)

    console.print(f"{OK} Found [bold]{len(docs)}[/bold] files.")

    with console.status("[bright_green]Chunking documents...[/bright_green]"):
        chunks = chunk_documents(docs)

    console.print(f"{OK} Split into [bold]{len(chunks)}[/bold] chunks.")

    with console.status("[bright_green]Upserting into ChromaDB...[/bright_green]"):
        written = add_documents(chunks)

    console.print(
        Panel(
            f"{OK} [bold green]{written}[/bold green] chunks from [bold]{len(docs)}[/bold] files",
            title="[bright_green]ingested[/bright_green]",
            border_style="bright_green",
            padding=(0, 1),
        )
    )


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


@app.command()
def ask(
    question: str = typer.Argument(
        ...,
        help="The question to ask about your codebase.",
    ),
    n_results: int = typer.Option(
        8,
        "--top",
        "-n",
        help="Number of context chunks to retrieve.",
    ),
    provider: str = typer.Option(
        "",
        "--provider",
        "-p",
        help=PROVIDER_HELP,
    ),
) -> None:
    """Ask a question about your ingested codebase."""
    from src.core.brain import ask as brain_ask

    console.print(f"\n{OK} {BRAND} is thinking ...\n")

    with console.status("[bright_green]Retrieving & generating...[/bright_green]"):
        result = brain_ask(
            question,
            n_results=n_results,
            provider_override=provider or None,
        )

    # -- Provider badge ---------------------------------------------------
    if result.provider:
        console.print(f"   [dim]provider:[/dim] [cyan]{result.provider}[/cyan]\n")

    # -- Answer -----------------------------------------------------------
    console.print(
        Panel(
            Markdown(result.answer),
            title="[bright_green]answer[/bright_green]",
            border_style="bright_green",
            padding=(1, 2),
        )
    )

    # -- Source Attribution ------------------------------------------------
    if result.sources:
        table = Table(
            title="[dim]sources[/dim]",
            box=TABLE_BOX,
            title_style="",
            header_style="bold cyan",
            show_edge=False,
        )
        table.add_column("#", style="dim", width=4)
        table.add_column("file", style="bold cyan")

        for idx, src in enumerate(result.sources, 1):
            table.add_row(str(idx), src)

        console.print(table)
    else:
        console.print(f"\n{WARN} No sources were referenced.")


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------


@app.command()
def forget() -> None:
    """Wipe the entire knowledge base (irreversible)."""
    from src.core.vector_store import reset_collection

    confirm = typer.confirm("WARNING: This will erase ALL ingested data. Continue?")
    if not confirm:
        console.print("Aborted.")
        raise typer.Exit(code=0)

    reset_collection()

    console.print(
        Panel(
            f"{OK} Knowledge base wiped.",
            title="[bright_yellow]forget[/bright_yellow]",
            border_style="bright_yellow",
            padding=(0, 1),
        )
    )


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@app.command()
def status() -> None:
    """Show current Mnemosyne configuration and collection stats."""
    from src.core.config import get_config
    from src.core.providers import PROVIDERS, get_provider_display
    from src.core.vector_store import get_or_create_collection

    config = get_config()
    pinfo = get_provider_display(config)

    try:
        collection = get_or_create_collection()
        doc_count: int | str = collection.count()
    except Exception:
        doc_count = "N/A"

    table = Table(
        title=f"{BRAND} [dim]status[/dim]",
        box=TABLE_BOX,
        title_style="",
        header_style="bold cyan",
        show_edge=False,
    )
    table.add_column("parameter", style="dim")
    table.add_column("value", style="bold cyan")

    table.add_row("LLM Provider", f"{pinfo['label']} [dim]({pinfo['provider']})[/dim]")
    table.add_row("Model", pinfo["model"])
    table.add_row("API Key", pinfo["key_status"])
    table.add_row("ChromaDB", config.chroma_db_path)
    table.add_row("Collection", config.collection_name)
    table.add_row("Indexed", str(doc_count))

    console.print(table)

    # -- Available providers list ------------------------------------------
    console.print("\n[dim]Available providers:[/dim]", end=" ")
    names = [f"[green]{n}[/green]" if n == pinfo["provider"] else f"[dim]{n}[/dim]" for n in sorted(PROVIDERS.keys())]
    console.print(" | ".join(names))


# ---------------------------------------------------------------------------
# chat  (interactive REPL -- Claude Code-style)
# ---------------------------------------------------------------------------


@app.command()
def chat(
    provider: str = typer.Option(
        "",
        "--provider",
        "-p",
        help=PROVIDER_HELP,
    ),
    n_results: int = typer.Option(
        8,
        "--top",
        "-n",
        help="Number of context chunks to retrieve per turn.",
    ),
) -> None:
    """Launch an interactive agent session (autonomous tool-calling)."""
    from src.cli.chat import ChatSession

    session = ChatSession(
        provider_override=provider or None,
        n_results=n_results,
    )
    session.run()


# ---------------------------------------------------------------------------
# gemini  (Gemini CLI integration)
# ---------------------------------------------------------------------------


@app.command()
def gemini(
    query: str = typer.Argument(
        "",
        help="Query to send to Gemini CLI with RAG context. Leave empty for interactive mode.",
    ),
    model: str = typer.Option(
        "",
        "--model",
        "-m",
        help="Gemini model to use (e.g. gemini-2.5-flash).",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Launch Gemini CLI in full interactive mode.",
    ),
    n_results: int = typer.Option(
        10,
        "--top",
        "-n",
        help="Number of context chunks to include.",
    ),
) -> None:
    """Delegate to Gemini CLI with Mnemosyne's RAG context."""
    from src.core.gemini_cli import (
        build_context_prompt,
        get_install_instructions,
        is_gemini_cli_installed,
        launch_interactive,
        query_headless,
    )

    if not is_gemini_cli_installed():
        console.print(f"\n{FAIL} Gemini CLI not found on PATH.\n")
        console.print(get_install_instructions())
        raise typer.Exit(code=1)

    # Interactive mode: just launch gemini cli
    if interactive or not query:
        if not query:
            console.print(
                f"\n{OK} Launching Gemini CLI interactive session...\n"
                f"   {BRAND} context is available in your project's MEMORY.md.\n"
            )
        exit_code = launch_interactive(model=model, cwd=None)
        raise typer.Exit(code=exit_code)

    # Headless mode: enrich the query with RAG context
    console.print(f"\n{OK} Preparing RAG context for Gemini CLI...\n")

    codebase_ctx = ""
    memory = ""

    with console.status("[bright_green]Retrieving context...[/bright_green]"):
        try:
            from src.core.brain import load_episodic_memory, rewrite_query
            from src.core.vector_store import query as vq

            search_q = rewrite_query(query)
            results = vq(search_q, n_results=n_results)
            if results:
                parts = [f"### `{r.source}`\n```\n{r.content}\n```" for r in results]
                codebase_ctx = "\n\n".join(parts)
            memory = load_episodic_memory()
        except Exception as exc:
            console.print(f"{WARN} Context retrieval failed: {exc}")

    full_prompt = build_context_prompt(query, codebase_context=codebase_ctx, memory=memory)

    console.print(f"{OK} Sending to Gemini CLI...\n")

    result = query_headless(full_prompt, model=model, timeout=180)

    if result.error:
        console.print(f"{FAIL} {result.error}")
        raise typer.Exit(code=1)

    console.print(
        Panel(
            Markdown(result.output),
            title="[magenta]gemini[/magenta]",
            border_style="magenta",
            padding=(1, 2),
        )
    )


# ---------------------------------------------------------------------------
# Entry point guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
