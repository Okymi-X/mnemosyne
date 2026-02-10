"""
File I/O commands: read, write, writeall, undo, create.
Also includes code-block extraction and diff viewing.
"""

from __future__ import annotations

import difflib
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage
from rich.syntax import Syntax

from src.cli.theme import (
    BLOCK_RE,
    FAIL,
    OK,
    WARN,
    C,
    D,
    R,
    console,
    fsize,
)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class CodeBlock:
    """A fenced code block parsed from LLM output."""

    lang: str
    path: str
    content: str


@dataclass
class WriteRecord:
    """Tracks a single file write for undo support."""

    path: str
    had_backup: bool
    backup_path: str


# ---------------------------------------------------------------------------
# Code-block extraction
# ---------------------------------------------------------------------------


def extract_blocks(text: str) -> list[CodeBlock]:
    """Parse all fenced code blocks from markdown text."""
    return [CodeBlock(m.group(1) or "", (m.group(2) or "").strip(), m.group(3)) for m in BLOCK_RE.finditer(text)]


def dedup_files(blocks: list[CodeBlock]) -> list[CodeBlock]:
    """Keep only the last occurrence of each file path."""
    od: OrderedDict[str, CodeBlock] = OrderedDict()
    for b in blocks:
        if b.path:
            od[b.path] = b
    return list(od.values())


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def write_file(path: str, content: str, backup: bool = True) -> WriteRecord | None:
    """Write content to *path*, optionally creating a .bak backup.

    Validates that the resolved path stays within the current working directory
    to prevent path-traversal attacks from LLM-generated file paths.
    """
    try:
        p = Path(path).resolve()
        cwd = Path.cwd().resolve()
        # Security: prevent writing outside the project directory
        if not str(p).startswith(str(cwd)):
            console.print(f"  {FAIL} path traversal blocked: {path}")
            return None
        p.parent.mkdir(parents=True, exist_ok=True)
        bak_path = ""
        had_bak = False
        if backup and p.exists():
            bak_path = str(p) + ".bak"
            shutil.copy2(str(p), bak_path)
            had_bak = True
        p.write_text(content, encoding="utf-8")
        return WriteRecord(path=path, had_backup=had_bak, backup_path=bak_path)
    except Exception as exc:
        console.print(f"  {FAIL} write failed: {path} -- {exc}")
        return None


def read_file(path: str) -> str | None:
    """Read a text file, returning None on failure."""
    try:
        p = Path(path)
        if not p.is_file():
            console.print(f"  {FAIL} not found: {path}")
            return None
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        console.print(f"  {FAIL} {exc}")
        return None


def is_identical(path: str, new_content: str) -> bool:
    """Check whether a file already contains *new_content*."""
    try:
        p = Path(path)
        if not p.exists():
            return False
        return p.read_text(encoding="utf-8", errors="ignore") == new_content
    except Exception:
        return False


def view_diff(path: str, new_content: str) -> None:
    """Print a unified diff between the existing file and *new_content*."""
    try:
        p = Path(path)
        if not p.exists():
            return
        old = p.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
        new = new_content.splitlines(keepends=True)
        diff = difflib.unified_diff(old, new, fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="")
        diff_text = "".join(diff)
        if diff_text:
            console.print()
            console.print(Syntax(diff_text, "diff", theme="monokai", line_numbers=False))
            console.print()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Session-level command handlers
#
# Each function receives the ChatSession (or relevant state) so it can
# mutate history / write tracking.  Keeping logic here keeps chat.py slim.
# ---------------------------------------------------------------------------


def cmd_read(path: str, history: list) -> None:
    """Load a file into conversation context."""
    if not path:
        console.print(f"  {WARN} usage: /read <file>\n")
        return
    content = read_file(path)
    if content is None:
        return
    lc = content.count("\n") + 1
    history.append(HumanMessage(content=f"[File: {path}]\n```\n{content}\n```"))
    console.print(f"  {OK} loaded {C}{path}{R} {D}({lc} lines, {fsize(len(content))}){R}\n")


def cmd_write(
    path: str, last_response: str, last_writes: list[WriteRecord], files_written_counter: list[int], readonly: bool
) -> None:
    """Write one code block to disk."""
    if readonly:
        console.print(f"  {FAIL} readonly mode is active\n")
        return
    if not last_response:
        console.print(f"  {WARN} no response yet\n")
        return
    blocks = extract_blocks(last_response)
    if path:
        b = next((x for x in blocks if x.path == path), blocks[0] if blocks else None)
        if b:
            b.path = b.path or path
            rec = write_file(b.path, b.content)
            if rec:
                last_writes.clear()
                last_writes.append(rec)
                files_written_counter[0] += 1
                console.print(f"  {OK} wrote {C}{b.path}{R}\n")
        else:
            console.print(f"  {FAIL} no code blocks\n")
    else:
        fb = dedup_files(blocks)
        if fb:
            rec = write_file(fb[0].path, fb[0].content)
            if rec:
                last_writes.clear()
                last_writes.append(rec)
                files_written_counter[0] += 1
                console.print(f"  {OK} wrote {C}{fb[0].path}{R}\n")
        else:
            console.print(f"  {WARN} usage: /write <path>\n")


def cmd_writeall(
    last_response: str, last_writes: list[WriteRecord], files_written_counter: list[int], readonly: bool
) -> None:
    """Write all detected file blocks to disk."""
    if readonly:
        console.print(f"  {FAIL} readonly mode is active\n")
        return
    if not last_response:
        console.print(f"  {WARN} no response yet\n")
        return
    fb = dedup_files(extract_blocks(last_response))
    if not fb:
        console.print(f"  {WARN} no file blocks detected\n")
        return
    last_writes.clear()
    for b in fb:
        rec = write_file(b.path, b.content)
        if rec:
            console.print(f"    {OK} {b.path}")
            last_writes.append(rec)
            files_written_counter[0] += 1
    console.print(f"\n  {OK} {len(last_writes)}/{len(fb)} written\n")


def cmd_undo(last_writes: list[WriteRecord]) -> None:
    """Revert previously written files."""
    if not last_writes:
        console.print(f"  {WARN} nothing to undo\n")
        return
    for rec in last_writes:
        p = Path(rec.path)
        if rec.had_backup and Path(rec.backup_path).exists():
            shutil.move(rec.backup_path, rec.path)
            console.print(f"    {OK} restored {C}{rec.path}{R}")
        elif p.exists():
            p.unlink()
            console.print(f"    {OK} deleted {C}{rec.path}{R}")
        else:
            console.print(f"    {WARN} already gone: {rec.path}")
    console.print(f"\n  {OK} undo ({len(last_writes)} files)\n")
    last_writes.clear()


def cmd_create(path: str, readonly: bool) -> None:
    """Create an empty file or directory."""
    if readonly:
        console.print(f"  {FAIL} readonly mode is active\n")
        return
    if not path:
        console.print(f"  {WARN} usage: /create <path>\n")
        return
    p = Path(path)
    try:
        if path.endswith(("/", "\\")):
            p.mkdir(parents=True, exist_ok=True)
            console.print(f"  {OK} dir: {C}{p}{R}\n")
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch(exist_ok=True)
            console.print(f"  {OK} file: {C}{p}{R}\n")
    except Exception as exc:
        console.print(f"  {FAIL} {exc}\n")
