"""
Mnemosyne -- CLI Entry Point
Beautiful, hacker-style terminal interface built with Typer + Rich.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import box

# Force UTF-8 output on Windows to avoid CP1252 encoding errors
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass

app = typer.Typer(
    name="mnemosyne",
    help="Mnemosyne -- The Infinite Context Engine",
    add_completion=False,
    no_args_is_help=True,
)

console = Console(highlight=False)

# -- Styling constants -------------------------------------------------------
BRAND = "[bold bright_green]Mnemosyne[/bold bright_green]"
OK    = "[green][+][/green]"
WARN  = "[yellow][!][/yellow]"
FAIL  = "[red][-][/red]"

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
            f"{OK} {BRAND} initialised at [cyan]{root}[/cyan]\n"
            f"   Database path: [dim]{mnemosyne_dir / 'chroma'}[/dim]",
            title="[bright_green]>> Init Complete[/bright_green]",
            border_style="bright_green",
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
    from src.core.ingester import scan_directory, chunk_documents
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

    console.print(
        f"\n{OK} Scanning [cyan]{root}[/cyan] for code files ...\n"
    )

    with console.status("[bright_green]Scanning files...[/bright_green]"):
        docs = list(scan_directory(root, extra_extensions=extra))

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
            f"{OK} Ingested [bold green]{written}[/bold green] chunks "
            f"from [bold]{len(docs)}[/bold] files.",
            title="[bright_green]>> Ingestion Complete[/bright_green]",
            border_style="bright_green",
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

    console.print(
        f"\n{OK} {BRAND} is thinking ...\n"
    )

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
            title="[bright_green]>> Answer[/bright_green]",
            border_style="bright_green",
            padding=(1, 2),
        )
    )

    # -- Source Attribution ------------------------------------------------
    if result.sources:
        table = Table(
            title="[bright_green]>> Sources Referenced[/bright_green]",
            box=box.SIMPLE_HEAVY,
            title_style="bold bright_green",
            header_style="bold cyan",
        )
        table.add_column("#", style="dim", width=4)
        table.add_column("File", style="green")

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

    confirm = typer.confirm(
        "WARNING: This will erase ALL ingested data. Continue?"
    )
    if not confirm:
        console.print("Aborted.")
        raise typer.Exit(code=0)

    reset_collection()

    console.print(
        Panel(
            f"{OK} Knowledge base has been wiped.",
            title="[bright_yellow]>> Forget Complete[/bright_yellow]",
            border_style="bright_yellow",
        )
    )


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@app.command()
def status() -> None:
    """Show current Mnemosyne configuration and collection stats."""
    from src.core.config import get_config
    from src.core.providers import get_provider_display, PROVIDERS
    from src.core.vector_store import get_or_create_collection

    config = get_config()
    pinfo = get_provider_display(config)

    try:
        collection = get_or_create_collection()
        doc_count = collection.count()
    except Exception:
        doc_count = "N/A"

    table = Table(
        title=f">> {BRAND} Status",
        box=box.ROUNDED,
        title_style="bold bright_green",
        header_style="bold cyan",
    )
    table.add_column("Parameter", style="dim")
    table.add_column("Value", style="green")

    table.add_row("LLM Provider", f"{pinfo['label']} [dim]({pinfo['provider']})[/dim]")
    table.add_row("Model", pinfo["model"])
    table.add_row("API Key", pinfo["key_status"])
    table.add_row("ChromaDB Path", config.chroma_db_path)
    table.add_row("Collection", config.collection_name)
    table.add_row("Documents Indexed", str(doc_count))

    console.print(table)

    # -- Available providers list ------------------------------------------
    console.print("\n[dim]Available providers:[/dim]", end=" ")
    names = [
        f"[green]{n}[/green]" if n == pinfo["provider"] else f"[dim]{n}[/dim]"
        for n in sorted(PROVIDERS.keys())
    ]
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
    """Launch an interactive chat session (Claude Code-style)."""
    from src.cli.chat import ChatSession

    session = ChatSession(
        provider_override=provider or None,
        n_results=n_results,
    )
    session.run()


# ---------------------------------------------------------------------------
# Entry point guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
