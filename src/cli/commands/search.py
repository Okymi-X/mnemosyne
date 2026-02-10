"""
Search & navigation commands: ls, find, grep, diff, copy.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree
from rich import box

from src.cli.theme import (
    console, IGNORED,
    C, D, R, OK, FAIL, WARN, fsize,
)
from src.cli.commands.files import extract_blocks


# ---------------------------------------------------------------------------
# /ls
# ---------------------------------------------------------------------------

def cmd_ls(path: str) -> None:
    """List directory tree with sizes."""
    t = Path(path) if path else Path.cwd()
    if not t.is_dir():
        console.print(f"  {FAIL} not a directory: {t}\n")
        return

    tree = Tree(f"{C}{t.name}/{R}", guide_style="dim")
    items = sorted(t.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    dirs = files = 0

    for item in items:
        if item.name.startswith(".") and item.name not in (".env", ".gitignore"):
            continue
        if item.name in IGNORED:
            continue
        if item.is_dir():
            tree.add(f"{C}{item.name}/{R}")
            dirs += 1
        else:
            sz = fsize(item.stat().st_size)
            tree.add(f"{item.name}  {D}{sz}{R}")
            files += 1
        if dirs + files >= 50:
            tree.add(f"{D}...{R}")
            break

    console.print()
    console.print(tree)
    console.print(f"  {D}{dirs}d {files}f{R}\n")


# ---------------------------------------------------------------------------
# /find
# ---------------------------------------------------------------------------

def cmd_find(pat: str) -> None:
    """Find files by name pattern."""
    if not pat:
        console.print(f"  {WARN} usage: /find <pattern>\n")
        return

    cwd = Path.cwd()
    hits: list[tuple[str, bool]] = []
    for p in cwd.rglob(f"*{pat}*"):
        rel = str(p.relative_to(cwd))
        if set(rel.replace("\\", "/").split("/")) & IGNORED:
            continue
        hits.append((rel, p.is_dir()))
        if len(hits) >= 30:
            break

    if not hits:
        console.print(f"  {D}no matches{R}\n")
        return
    console.print()
    for h, is_d in hits:
        if is_d:
            console.print(f"  {C}{h}/{R}")
        else:
            console.print(f"  {h}")
    console.print(f"  {D}{len(hits)} result(s){R}\n")


# ---------------------------------------------------------------------------
# /grep
# ---------------------------------------------------------------------------

def cmd_grep(pat: str) -> None:
    """Search inside file contents using git-grep or manual fallback."""
    if not pat:
        console.print(f"  {WARN} usage: /grep <pattern>\n")
        return

    hits: list[str] = []
    try:
        r = subprocess.run(
            ["git", "grep", "-n", "--color=never", "-I", pat],
            capture_output=True, text=True, timeout=10, cwd=Path.cwd(),
        )
        if r.stdout:
            hits = r.stdout.strip().split("\n")[:25]
    except Exception:
        cwd = Path.cwd()
        for p in cwd.rglob("*"):
            if not p.is_file() or p.stat().st_size > 500_000:
                continue
            if set(str(p.relative_to(cwd)).replace("\\", "/").split("/")) & IGNORED:
                continue
            try:
                for i, ln in enumerate(p.read_text("utf-8", errors="ignore").split("\n"), 1):
                    if pat.lower() in ln.lower():
                        hits.append(f"{p.relative_to(cwd)}:{i}: {ln.strip()}")
                        if len(hits) >= 25:
                            break
            except Exception:
                continue
            if len(hits) >= 25:
                break

    if not hits:
        console.print(f"  {D}no matches for '{pat}'{R}\n")
        return

    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1), show_edge=False)
    t.add_column("match", style="dim")
    for h in hits:
        if ":" in h:
            parts = h.split(":", 2)
            if len(parts) >= 3:
                t.add_row(f"{C}{parts[0]}{R}:{D}{parts[1]}{R}: {parts[2]}")
            else:
                t.add_row(h)
        else:
            t.add_row(h)
    console.print()
    console.print(t)
    console.print(f"\n  {D}{len(hits)} match(es){R}\n")


# ---------------------------------------------------------------------------
# /diff
# ---------------------------------------------------------------------------

def cmd_diff(path: str) -> None:
    """Show git diff."""
    try:
        cmd = ["git", "diff", "--color=never"] + ([path] if path else [])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if r.stdout:
            console.print()
            console.print(Syntax(r.stdout, "diff", theme="monokai", line_numbers=False))
            console.print()
        else:
            console.print(f"  {D}no changes{' in ' + path if path else ''}{R}\n")
    except Exception as exc:
        console.print(f"  {FAIL} {exc}\n")


# ---------------------------------------------------------------------------
# /copy
# ---------------------------------------------------------------------------

def cmd_copy(last_response: str) -> None:
    """Copy the last code block to clipboard."""
    if not last_response:
        console.print(f"  {WARN} no response yet\n")
        return
    blocks = extract_blocks(last_response)
    if not blocks:
        console.print(f"  {WARN} no code blocks in last response\n")
        return
    content = blocks[-1].content
    try:
        if sys.platform == "win32":
            subprocess.run("clip", input=content.encode("utf-8"), check=True)
        elif sys.platform == "darwin":
            subprocess.run("pbcopy", input=content.encode("utf-8"), check=True)
        else:
            subprocess.run(["xclip", "-selection", "clipboard"],
                           input=content.encode("utf-8"), check=True)
        console.print(f"  {OK} copied {len(content)} chars to clipboard\n")
    except Exception:
        console.print(f"  {WARN} clipboard not available\n")
