"""
Shell, git, lint, cd, ingest commands.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from rich.panel import Panel

from src.cli.theme import FAIL, OK, WARN, C, D, G, M, R, console

# ---------------------------------------------------------------------------
# /run  (with smart error hints)
# ---------------------------------------------------------------------------

_CMD_HINTS: dict[str, str] = {
    "python3": "try: python",
    "py": "try: python",
    "pip3": "try: pip",
    "node": "install Node.js",
    "npm": "install Node.js",
    "cargo": "install Rust",
    "go": "install Go",
    "docker": "install Docker",
}


def cmd_run(cmd: str) -> None:
    """Execute an arbitrary shell command with smart error recovery."""
    if not cmd:
        console.print(f"  {WARN} usage: /run <command>\n")
        return

    console.print(f"\n  {D}${R} {C}{cmd}{R}\n")
    try:
        r = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=Path.cwd(),
        )
        if r.stdout:
            console.print(r.stdout.rstrip())
        if r.stderr:
            err = r.stderr.rstrip()
            console.print(f"[yellow]{err}[/yellow]")
            _suggest_fix(err, cmd)
        if r.returncode != 0:
            console.print(f"\n  {D}exit {r.returncode}{R}")
    except subprocess.TimeoutExpired:
        console.print(f"  {FAIL} timeout (60s) -- try running in background")
    except Exception as exc:
        console.print(f"  {FAIL} {exc}")
    console.print()


def _suggest_fix(err: str, cmd: str) -> None:
    """Print helpful hints based on the error message."""
    low = err.lower()
    first = cmd.split()[0]

    if "command not found" in low or "is not recognized" in low:
        hint = _CMD_HINTS.get(first, "check spelling or install the package")
        console.print(f"  {WARN} command not found: {first} -> {hint}")
    elif "permission denied" in low:
        console.print(f"  {WARN} try with elevated permissions (sudo/admin)")
    elif "no such file" in low:
        console.print(f"  {WARN} file/directory not found -- check the path")
    elif "modulenotfounderror" in low or "no module named" in low:
        mod = re.search(r"no module named ['\"]?(\w+)", low)
        pkg = mod.group(1) if mod else "package"
        console.print(f"  {WARN} missing module -> pip install {pkg}")
    elif "enoent" in low or "errno 2" in low:
        console.print(f"  {WARN} file not found -- verify the path exists")


# ---------------------------------------------------------------------------
# /git  (with smart commit messages)
# ---------------------------------------------------------------------------


def cmd_git(args: str, provider_override: str | None = None, model_override: str | None = None) -> None:
    """Git wrapper with LLM-powered commit message generation."""
    if not args:
        console.print(f"  {WARN} usage: /git <args>\n")
        return

    # Smart commit: auto-generate message from staged diff
    if "commit" in args and "-m" not in args:
        args = _smart_commit(args, provider_override, model_override)
        if args is None:
            return  # aborted

    console.print(f"\n  {D}${R} {C}git {args}{R}\n")
    try:
        r = subprocess.run(
            f"git {args}",
            shell=True,
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )
        if r.stdout:
            console.print(r.stdout.rstrip())
        if r.stderr:
            console.print(f"[yellow]{r.stderr.rstrip()}[/yellow]")
        if r.returncode == 0:
            console.print(f"  {OK} success\n")
        else:
            console.print(f"  {FAIL} exit {r.returncode}\n")
    except Exception as exc:
        console.print(f"  {FAIL} {exc}\n")


def _smart_commit(args: str, provider_override: str | None, model_override: str | None) -> str | None:
    """Generate a commit message via LLM. Returns updated args or None."""
    console.print(f"  {M}generating commit message...{R}")
    try:
        diff_stat = subprocess.check_output(
            ["git", "diff", "--cached", "--stat"],
            text=True,
            cwd=Path.cwd(),
        ).strip()
        if not diff_stat:
            console.print(f"  {WARN} no staged changes. Stage with: git add <files>\n")
            return None

        full_diff = subprocess.check_output(
            ["git", "diff", "--cached"],
            text=True,
            cwd=Path.cwd(),
        ).strip()

        from langchain_core.messages import HumanMessage as HM
        from langchain_core.messages import SystemMessage as SM

        from src.core.brain import _resolve_config
        from src.core.providers import get_llm

        cfg = _resolve_config(provider_override, model_override)
        llm = get_llm(cfg)

        msgs = [
            SM(
                content=(
                    "You are a git commit message generator. "
                    "Write a single conventional commit message for the given diff. "
                    "Format: type(scope): description\n"
                    "Types: feat, fix, refactor, docs, style, test, chore, perf\n"
                    "Keep it under 72 chars. No quotes. Just the message. Nothing else."
                )
            ),
            HM(content=f"Files changed:\n{diff_stat}\n\nDiff:\n{full_diff[:3000]}"),
        ]
        msg = llm.invoke(msgs).content.strip().strip("\"'")  # type: ignore
        console.print(f"\n  {G}suggested:{R} {C}{msg}{R}")

        try:
            ans = input("\n  use this message? [Y/n/edit] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print(f"  {D}aborted{R}\n")
            return None

        if ans in ("", "y", "yes", "o", "oui"):
            return f'commit -m "{msg}"'
        elif ans in ("e", "edit"):
            try:
                custom = input("  message: ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print(f"  {D}aborted{R}\n")
                return None
            return f'commit -m "{custom}"' if custom else None
        else:
            console.print(f"  {D}aborted{R}\n")
            return None

    except subprocess.CalledProcessError:
        return args  # fall through to normal git
    except Exception as exc:
        console.print(f"  {WARN} auto-message failed ({exc}), opening editor...")
        return args


# ---------------------------------------------------------------------------
# /lint
# ---------------------------------------------------------------------------


def cmd_lint(path: str) -> None:
    """Run available linters on the given path."""
    target = path or "."
    linters = ["ruff", "flake8", "black --check", "pylint"]
    found = False
    console.print()

    for linter in linters:
        base = linter.split()[0]
        if not shutil.which(base):
            continue
        found = True
        console.print(f"  {D}{linter}...{R}")
        try:
            cmd = f"{linter} {target}"
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if r.returncode == 0:
                console.print(f"    {OK} {base} passed")
            else:
                console.print(f"    {FAIL} {base} found issues")
                output = (r.stdout + r.stderr).strip()
                if output:
                    console.print(
                        Panel(
                            output,
                            title=f"[dim]{base}[/dim]",
                            border_style="red",
                            padding=(0, 1),
                        )
                    )
        except Exception:
            pass

    if not found:
        console.print(f"  {WARN} no linters found (install ruff, flake8, etc.)\n")
    console.print()


# ---------------------------------------------------------------------------
# /cd
# ---------------------------------------------------------------------------


def cmd_cd(path: str, check_index_fn=None) -> None:
    """Change working directory."""
    if not path:
        console.print(f"  {D}{Path.cwd()}{R}\n")
        return
    try:
        t = Path(path).resolve()
        os.chdir(t)
        console.print(f"  {OK} {C}{t}{R}\n")
        if check_index_fn:
            check_index_fn()
    except Exception as exc:
        console.print(f"  {FAIL} {exc}\n")


# ---------------------------------------------------------------------------
# /ingest
# ---------------------------------------------------------------------------


def cmd_ingest(path: str) -> None:
    """Re-index codebase into vector store."""
    target = Path(path).resolve() if path else Path.cwd()
    if not target.is_dir():
        console.print(f"  {FAIL} not a directory: {target}\n")
        return
    console.print(f"  {D}indexing {target}...{R}")
    try:
        from src.core.ingester import chunk_documents, scan_directory
        from src.core.vector_store import add_documents

        docs = list(scan_directory(target))
        chunks = chunk_documents(docs)
        written = add_documents(chunks)
        console.print(f"  {OK} indexed {G}{written}{R} chunks from {len(docs)} files\n")
    except Exception as exc:
        console.print(f"  {FAIL} {exc}\n")
