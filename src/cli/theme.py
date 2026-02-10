"""
Shared visual constants, styles, and formatting helpers.
Single source of truth for all CLI styling across the project.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from prompt_toolkit.styles import Style as PTStyle
from rich import box
from rich.console import Console
from rich.text import Text

# ---------------------------------------------------------------------------
# Console singleton
# ---------------------------------------------------------------------------

console = Console(highlight=False)

# ---------------------------------------------------------------------------
# Rich markup tokens
# ---------------------------------------------------------------------------

G = "[bold bright_green]"
C = "[bold cyan]"
D = "[dim]"
Y = "[bold yellow]"
M = "[bold magenta]"
W = "[bold white]"
R = "[/]"
A = "[bold #e5a00d]"  # Amber -- agent / tool calls

OK = "[bold green]✓[/]"
FAIL = "[bold red]✕[/]"
WARN = "[bold yellow]![/]"
INFO = "[bold cyan]>[/]"

BRAND = f"{G}mnemosyne{R}"
VERSION = "3.1"

# ---------------------------------------------------------------------------
# prompt-toolkit theme
# ---------------------------------------------------------------------------

PT_STYLE = PTStyle.from_dict(
    {
        "prompt": "#a8e6cf bold",
        "path": "#888888",
        "filter": "#56c8d8",
        "readonly": "#ff6b6b bold",
        "agent": "#e5a00d bold",
        "bottom-toolbar": "#777777 bg:#0d0d1a",
        "completion-menu": "bg:#1a1a2e #e0e0e0",
        "completion-menu.completion.current": "bg:#3a3a5e #ffffff bold",
    }
)

# ---------------------------------------------------------------------------
# Preferred table style
# ---------------------------------------------------------------------------

TABLE_BOX = box.SIMPLE_HEAD

# ---------------------------------------------------------------------------
# Ignored directories (shared between cli and ingester)
# ---------------------------------------------------------------------------

IGNORED = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".mnemosyne",
        "dist",
        "build",
        ".next",
        ".nuxt",
        ".output",
        ".svelte-kit",
        "target",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".egg-info",
        ".eggs",
        "out",
        "vendor",
        ".gradle",
        ".idea",
        ".vs",
        "coverage",
        ".nyc_output",
        ".terraform",
        ".cache",
        ".parcel-cache",
        "bin",
        "obj",
    }
)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

PATH_RE = re.compile(r"(?:^|\s)((?:[\w.-]+/)+[\w.-]+\.[\w]+)", re.MULTILINE)
BLOCK_RE = re.compile(r"```(\w[\w+#.-]*)?(?::([^\n]+))?\s*\n(.*?)\n```", re.DOTALL)

# ---------------------------------------------------------------------------
# Extension -> short tag map
# ---------------------------------------------------------------------------

EXT_MAP: dict[str, str] = {
    ".py": "py",
    ".js": "js",
    ".ts": "ts",
    ".jsx": "jsx",
    ".tsx": "tsx",
    ".html": "htm",
    ".css": "css",
    ".json": "json",
    ".yaml": "yml",
    ".yml": "yml",
    ".toml": "toml",
    ".md": "md",
    ".rs": "rs",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".sh": "sh",
    ".sql": "sql",
    ".vue": "vue",
    ".svelte": "sv",
}

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fsize(n: int) -> str:
    """Human-readable file size."""
    if n < 1024:
        return f"{n}B"
    if n < 1048576:
        return f"{n // 1024}K"
    return f"{n // 1048576}M"


def tool_icon(name: str) -> str:
    """Return a short icon/prefix for a tool name."""
    icons = {
        "read_file": ">>",
        "list_directory": "ls",
        "find_files": "??",
        "grep_search": "//",
        "search_codebase": "::",
        "search_web": "~~",
        "run_command": "$ ",
    }
    return icons.get(name, "->")


def ext_tag(name: str) -> str:
    """Return a dim extension tag like 'py', 'ts', etc."""
    ext = Path(name).suffix.lower()
    tag = EXT_MAP.get(ext, ext.lstrip(".") if ext else "")
    return f"{D}{tag}{R}" if tag else ""


# ---------------------------------------------------------------------------
# Gradient banner system
# ---------------------------------------------------------------------------

_LETTERS: dict[str, list[str]] = {
    "M": ["█▄ ▄█", "█ ▀ █", "█   █"],
    "N": ["█▄  █", "█ █ █", "█  ▀█"],
    "E": ["█▀▀▀ ", "█▀▀  ", "█▄▄▄ "],
    "O": ["▄▀▀▀▄", "█   █", "▀▄▄▄▀"],
    "S": ["▄▀▀▀ ", " ▀█▄ ", " ▄▄▄▀"],
    "Y": ["▀▄ ▄▀", "  █  ", "  █  "],
}

GRADIENT_COLORS: list[str] = [
    "#c084fc",  # purple
    "#b57af8",
    "#a78bfa",  # violet
    "#9994f8",
    "#818cf8",  # indigo
    "#7285f6",
    "#6366f1",  # blue-indigo
    "#619df8",
    "#60a5fa",  # blue
    "#4cb2fc",
    "#38bdf8",  # sky
    "#2dc8ee",
    "#22d3ee",  # cyan
    "#28d4d6",
    "#2dd4bf",  # teal
    "#30d4ac",
    "#34d399",  # emerald
]

TIPS: list[str] = [
    "Ask questions, edit files, or run commands.",
    "Be specific for the best results.",
    "[bold bright_green]/help[/] for more information.",
]


def render_banner() -> Text:
    """Render the MNEMOSYNE banner with a purple → cyan → green gradient."""
    word = "MNEMOSYNE"
    rows: list[str] = []
    for r in range(3):
        rows.append(" ".join(_LETTERS[ch][r] for ch in word))

    text = Text()
    max_width = max(len(row) for row in rows)

    for i, row in enumerate(rows):
        padded = row.ljust(max_width)
        for j, ch in enumerate(padded):
            ci = int(j / max(max_width - 1, 1) * (len(GRADIENT_COLORS) - 1))
            if ch.strip():
                text.append(ch, style=f"bold {GRADIENT_COLORS[ci]}")
            else:
                text.append(ch)
        if i < len(rows) - 1:
            text.append("\n")

    return text


# ---------------------------------------------------------------------------
# Git helpers (for toolbar / status)
# ---------------------------------------------------------------------------


def get_git_branch() -> str:
    """Get the current git branch name, or empty string."""
    try:
        r = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=3,
            cwd=Path.cwd(),
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def get_git_dirty() -> bool:
    """Check if the working tree has uncommitted changes."""
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=3,
            cwd=Path.cwd(),
        )
        return bool(r.stdout.strip()) if r.returncode == 0 else False
    except Exception:
        return False
