"""
Shared visual constants, styles, and formatting helpers.
Single source of truth for all CLI styling across the project.
"""

from __future__ import annotations

import re
from pathlib import Path

from prompt_toolkit.styles import Style as PTStyle
from rich import box
from rich.console import Console

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

OK = "[bold green]>[/]"
FAIL = "[bold red]x[/]"
WARN = "[bold yellow]![/]"
INFO = "[bold cyan]~[/]"

BRAND = f"{G}mnemosyne{R}"
VERSION = "3.0"

# ---------------------------------------------------------------------------
# prompt-toolkit theme
# ---------------------------------------------------------------------------

PT_STYLE = PTStyle.from_dict(
    {
        "prompt": "#a8e6cf bold",
        "path": "#666666",
        "filter": "#56c8d8",
        "readonly": "#ff6b6b bold",
        "agent": "#e5a00d bold",
        "bottom-toolbar": "#555555 bg:#111122",
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
