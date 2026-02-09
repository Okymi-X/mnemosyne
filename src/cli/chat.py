"""
Mnemosyne v0.6 -- Intelligent Coding Agent
Polished terminal coding agent with Smart Editing, Pre-emptive Ingestion, and Robust Error Recovery.
"""

from __future__ import annotations

import difflib
import os
import re
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter, PathCompleter, WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style as PTStyle
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree
from rich import box

console = Console(highlight=False)

# ---------------------------------------------------------------------------
# Visual tokens & Theme
# ---------------------------------------------------------------------------
G  = "[bold bright_green]"
C  = "[cyan]"
D  = "[dim]"
Y  = "[bold yellow]"
M  = "[magenta]"
R  = "[/]"

OK   = f"{G}+{R}"
FAIL = "[bold red]-[/]"
WARN = f"{Y}!{R}"
INFO = f"{G}>{R}"

PT_STYLE = PTStyle.from_dict({
    "prompt":          "#00ff00 bold",
    "path":            "#888888",
    "filter":          "#00ffff",
    "readonly":        "#ff0000 bold",
    "bottom-toolbar":  "#333333 bg:#dddddd",
    "completion-menu": "bg:#333333 #ffffff",
})

IGNORED = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".mnemosyne", "dist", "build", ".next", ".nuxt", "target",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".egg-info",
})

_PATH_RE = re.compile(r"(?:^|\s)((?:[\w.-]+/)+[\w.-]+\.[\w]+)", re.MULTILINE)
_BLOCK_RE = re.compile(r"```(\w[\w+#.-]*)?(?::([^\n]+))?\s*\n(.*?)\n```", re.DOTALL)

_EXT_MAP: dict[str, str] = {
    ".py": "py", ".js": "js", ".ts": "ts", ".jsx": "jsx", ".tsx": "tsx",
    ".html": "htm", ".css": "css", ".json": "json", ".yaml": "yml",
    ".yml": "yml", ".toml": "toml", ".md": "md", ".rs": "rs",
    ".go": "go", ".java": "java", ".c": "c", ".cpp": "cpp",
    ".sh": "sh", ".sql": "sql", ".vue": "vue", ".svelte": "sv",
}


# ---------------------------------------------------------------------------
# Prompt & Toolbar
# ---------------------------------------------------------------------------

def _prompt(readonly: bool, filter_set: set[str] | None) -> HTML:
    parts = ["<path>" + Path.cwd().name + "</path>"]
    if readonly:
        parts.append("<readonly>(ro)</readonly>")
    if filter_set:
        f = ",".join(sorted(filter_set))
        parts.append(f"<filter>[{f}]</filter>")
    parts.append("<prompt>&gt; </prompt>")
    return HTML(" ".join(parts))


def _make_completer() -> NestedCompleter:
    """Build the intelligent autocompleter."""
    path_comp = PathCompleter(only_directories=False, expanduser=True)
    dir_comp = PathCompleter(only_directories=True, expanduser=True)
    
    provs = WordCompleter(["groq", "anthropic", "google", "openai", "openrouter", "ollama"])
    
    return NestedCompleter.from_nested_dict({
        "/help": None,
        "/clear": None,
        "/compact": None,
        "/context": None,
        "/status": None,
        "/quit": None, "/exit": None,
        "/undo": None,
        "/writeall": None,
        "/readonly": None,
        "/copy": None,
        "/model": None,
        "/provider": provs,
        "/read": path_comp,
        "/write": path_comp,
        "/create": path_comp,
        "/ls": dir_comp,
        "/cd": dir_comp,
        "/diff": path_comp,
        "/find": None,
        "/grep": None,
        "/run": None,
        "/ingest": dir_comp,
        "/save": path_comp,
        "/web": None,
        "/git": None,
        "/lint": path_comp,
        "/filter": WordCompleter([".py", ".js", ".ts", ".md", ".html", ".css", ".rs", ".go"]),
    })


# ---------------------------------------------------------------------------
# Welcome & Help
# ---------------------------------------------------------------------------

def _welcome(provider: str, model: str, chunks: int | str) -> None:
    console.print()
    console.print(Rule(style="bright_green"))
    console.print(
        f"  {G}mnemosyne{R}  {D}v1.1 (Smart Shell){R}\n"
        f"\n"
        f"  {D}provider{R}  {C}{provider}{R}\n"
        f"  {D}model{R}     {C}{model}{R}\n"
        f"  {D}indexed{R}   {C}{chunks}{R} chunks\n"
        f"  {D}cwd{R}       {C}{Path.cwd()}{R}\n"
    )
    console.print(Rule(style="bright_green"))
    console.print(f"  {D}Type / for commands  |  Ctrl-C to exit{R}")
    console.print()


def _help() -> None:
    t = Table(
        box=box.SIMPLE, show_header=True, header_style="bold bright_green",
        padding=(0, 2), title=f"{G}mnemosyne{R} {D}commands{R}", title_style="",
    )
    t.add_column("command", style="green", min_width=16)
    t.add_column("description", style="dim")

    cmds = [
        ("", "[bold]conversation[/bold]"),
        ("/clear", "reset history"),
        ("/compact", "trim to last 2 turns"),
        ("/save [path]", "export chat to markdown"),
        ("/status", "show config and stats"),
        ("", "[bold]model & retrieval[/bold]"),
        ("/model <name>", "switch model mid-session"),
        ("/provider <name>", "switch provider mid-session"),
        ("/filter <exts>", "filter context (e.g. .py)"),
        ("/web <query>", "search the web"),
        ("", "[bold]files[/bold]"),
        ("/read <path>", "load file into context"),
        ("/write <path>", "write last code block"),
        ("/writeall", "write all detected files"),
        ("/undo", "revert last written files"),
        ("/readonly", "toggle read-only mode"),
        ("/create <path>", "create file or directory"),
        ("/ls [path]", "directory tree listing"),
        ("/find <pattern>", "find files by name"),
        ("/grep <pattern>", "search file contents"),
        ("/diff [path]", "git diff"),
        ("/copy", "copy last code to clipboard"),
        ("", "[bold]system[/bold]"),
        ("/run <cmd>", "execute shell command"),
        ("/git <args>", "run git commands"),
        ("/lint [path]", "lint code (ruff/flake8)"),
        ("/ingest [path]", "re-index codebase"),
        ("/cd <path>", "change directory"),
        ("/quit", "exit"),
    ]
    for c, d in cmds:
        t.add_row(c, d)

    console.print()
    console.print(t)
    console.print(f"\n  {D}tip: wrap multi-line input in triple quotes (\"\"\"){R}\n")


# ---------------------------------------------------------------------------
# Code block extraction
# ---------------------------------------------------------------------------

@dataclass
class CodeBlock:
    lang: str
    path: str
    content: str


def _extract(text: str) -> list[CodeBlock]:
    return [
        CodeBlock(m.group(1) or "", (m.group(2) or "").strip(), m.group(3))
        for m in _BLOCK_RE.finditer(text)
    ]


def _dedup_files(blocks: list[CodeBlock]) -> list[CodeBlock]:
    od: OrderedDict[str, CodeBlock] = OrderedDict()
    for b in blocks:
        if b.path:
            od[b.path] = b
    return list(od.values())


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

@dataclass
class WriteRecord:
    path: str
    had_backup: bool
    backup_path: str


def _write(path: str, content: str, backup: bool = True) -> WriteRecord | None:
    try:
        p = Path(path)
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


def _read(path: str) -> str | None:
    try:
        p = Path(path)
        if not p.is_file():
            console.print(f"  {FAIL} not found: {path}")
            return None
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        console.print(f"  {FAIL} {exc}")
        return None


def _is_identical(path: str, new_content: str) -> bool:
    try:
        p = Path(path)
        if not p.exists(): return False
        return p.read_text(encoding="utf-8", errors="ignore") == new_content
    except Exception:
        return False


def _view_diff(path: str, new_content: str) -> None:
    """Show a unified diff between existing file and new content."""
    try:
        p = Path(path)
        if not p.exists():
            return
        old_lines = p.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            old_lines, new_lines, fromfile=f"a/{path}", tofile=f"b/{path}", lineterm=""
        )
        diff_text = "".join(diff)
        if diff_text:
            console.print()
            console.print(Syntax(diff_text, "diff", theme="monokai", line_numbers=False))
            console.print()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Auto-detect file paths
# ---------------------------------------------------------------------------

def _auto_read_paths(query: str, history: list[BaseMessage]) -> list[str]:
    found = _PATH_RE.findall(query)
    injected: list[str] = []
    for fp in found:
        p = Path(fp)
        if p.is_file() and p.stat().st_size < 200_000:
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                history.append(HumanMessage(content=f"[auto-loaded: {fp}]\n```\n{content}\n```"))
                injected.append(fp)
            except Exception:
                pass
    return injected


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _fsize(n: int) -> str:
    if n < 1024: return f"{n}B"
    if n < 1048576: return f"{n // 1024}K"
    return f"{n // 1048576}M"


def _ext_tag(name: str) -> str:
    ext = Path(name).suffix.lower()
    tag = _EXT_MAP.get(ext, ext.lstrip(".") if ext else "")
    return f"{D}{tag}{R}" if tag else ""


# ===========================================================================
# Session
# ===========================================================================

class ChatSession:

    def __init__(self, provider_override: str | None = None, n_results: int = 8):
        self.provider_override = provider_override
        self.n_results = n_results
        self.history: list[BaseMessage] = []
        self.turns: int = 0
        self.last: str = ""
        self.last_writes: list[WriteRecord] = []
        self._model_override: str | None = None
        self._files_written: int = 0
        self.readonly: bool = False
        self.filter_ext: set[str] | None = None
        
        # Pre-emptive check
        self._check_index()

    def _check_index(self) -> None:
        """Check if index is empty or might need refresh."""
        try:
            from src.core.vector_store import get_or_create_collection
            cnt = get_or_create_collection().count()
            if cnt == 0:
                console.print(f"  {WARN} Index is empty. Auto-running /ingest .")
                self._cmd_ingest(".")
        except Exception:
            pass

    def _get_toolbar(self) -> HTML:
        """Dynamic bottom toolbar."""
        from src.core.providers import get_provider_display
        from src.core.config import get_config
        
        cfg = get_config()
        if self.provider_override:
             from src.core.config import MnemosyneConfig
             d = cfg.model_dump(); d["llm_provider"] = self.provider_override
             cfg = MnemosyneConfig(**d)
             
        p = get_provider_display(cfg)
        model = self._model_override or p['model']
        prov = p['provider']
        
        parts = [
            f" <b>{prov}</b> ",
            f" {model} ",
            f" <b>ctx:</b> {self.n_results} ",
        ]
        
        if self.filter_ext:
            f = ",".join(sorted(self.filter_ext))
            parts.append(f" <b>filter:</b> [{f}] ")
            
        if self.readonly:
            parts.append(" <style bg='red' fg='white'> READONLY </style> ")
            
        return HTML(" | ".join(parts))

    # -- Core: stream + respond ---------------------------------------------

    def _ask(self, q: str) -> None:
        from src.core.brain import ask_streaming

        injected = _auto_read_paths(q, self.history)
        if injected:
            for fp in injected:
                console.print(f"  {D}auto-loaded: {fp}{R}")
            console.print()

        console.print(Rule(f"{D}turn {self.turns + 1}{R}", style="dim"))
        console.print()

        fmeta: dict[str, Any] | None = None
        if self.filter_ext:
            fmeta = {"extension": {"$in": list(self.filter_ext)}}

        sys.stdout.write(f"  \033[2mthinking...\033[0m\r")
        sys.stdout.flush()

        tokens: list[str] = []
        sources: list[str] = []
        first_tok = True
        t0 = time.perf_counter()
        
        in_thinking = False

        try:
            gen = ask_streaming(
                q, history=self.history or None, n_results=self.n_results,
                provider_override=self.provider_override,
                model_override=self._model_override,
                filter_meta=fmeta,
            )
            for tok in gen:
                if first_tok:
                    sys.stdout.write("\r\033[K")
                    sys.stdout.flush()
                    first_tok = False

                if "<thinking>" in tok:
                    in_thinking = True
                    tok = tok.replace("<thinking>", "")
                    console.print(f"  {M} reasoning...{R}")
                
                if "</thinking>" in tok:
                    in_thinking = False
                    tok = tok.replace("</thinking>", "")

                tokens.append(tok)
                
                if not in_thinking:
                    sys.stdout.write(tok)
                    sys.stdout.flush()

        except StopIteration as e:
            if e.value:
                _, sources = e.value
        except Exception as exc:
            sys.stdout.write("\r\033[K")
            console.print(f"\n  {FAIL} {exc}\n")
            return

        sys.stdout.write("\n")
        sys.stdout.flush()

        elapsed = time.perf_counter() - t0
        full = "".join(tokens)
        self.last = full

        if not sources:
            try:
                from src.core.vector_store import query as vq
                seen: set[str] = set()
                for r in vq(q, n_results=self.n_results, filter_meta=fmeta):
                    if r.source not in seen:
                        seen.add(r.source)
                        sources.append(r.source)
            except Exception:
                pass

        tps = len(tokens) / elapsed if elapsed > 0 else 0
        console.print(
            f"\n  {D}{len(tokens)} tokens  |  {elapsed:.1f}s  |  {tps:.0f} tok/s{R}"
        )
        if sources:
            console.print(f"  {D}sources: {', '.join(sources[:6])}{R}")

        # Detect files
        files = _dedup_files(_extract(full))
        if files:
            console.print()
            ft = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
            ft.add_column("icon", style="bold bright_green", width=3)
            ft.add_column("path", style="cyan")
            ft.add_column("info", style="dim")
            
            pending_writes: list[CodeBlock] = []
            
            for f in files:
                lc = f.content.count("\n") + 1
                exists = Path(f.path).exists()
                identical = _is_identical(f.path, f.content)
                
                info_parts = [f"{lc}L"]
                if identical:
                    info_parts.append("identical (skip)")
                    icon = f"{D}={R}"
                elif exists:
                    info_parts.append("overwrite (diff available)")
                    icon = f"{Y}>{R}"
                    pending_writes.append(f)
                else:
                    icon = f"{G}+{R}"
                    pending_writes.append(f)
                    
                ft.add_row(icon, f.path, " | ".join(info_parts))
            
            console.print(ft)
            
            if self.readonly:
                console.print(f"\n  {D}readonly mode: skipping writes{R}")
            elif pending_writes:
                self._confirm(pending_writes)
            else:
                console.print(f"\n  {OK} all files identical -- nothing to write")

        console.print()
        self.history.append(HumanMessage(content=q))
        self.history.append(AIMessage(content=full))
        self.turns += 1

    def _confirm(self, files: list[CodeBlock]) -> None:
        while True:
            try:
                ans = input(f"\n  write {len(files)} file(s)? [Y/n/diff] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return
            
            if ans == "diff" or ans == "d":
                for f in files:
                    if Path(f.path).exists():
                         _view_diff(f.path, f.content)
                continue
            
            if ans in ("", "y", "yes", "o", "oui"):
                self.last_writes.clear()
                for f in files:
                    rec = _write(f.path, f.content, backup=True)
                    if rec:
                        bak = f" {D}(backup saved){R}" if rec.had_backup else ""
                        console.print(f"    {OK} {f.path}{bak}")
                        self.last_writes.append(rec)
                        self._files_written += 1
                console.print(f"\n  {OK} {len(self.last_writes)}/{len(files)} written")
                break
            else:
                console.print(f"  {D}skipped -- use /writeall later{R}")
                break

    # -- Slash router -------------------------------------------------------

    def _slash(self, raw: str) -> bool:
        parts = raw.strip().split(None, 1)
        c = parts[0].lower()
        a = parts[1].strip() if len(parts) > 1 else ""

        cmds: dict[str, object] = {
            "/quit": None, "/exit": None, "/q": None,
            "/help":      lambda: _help(),
            "/clear":     lambda: self._clear(),
            "/compact":   lambda: self._compact(),
            "/context":   lambda: self._context(),
            "/status":    lambda: self._status(),
            "/model":     lambda: self._set_model(a),
            "/provider":  lambda: self._set_provider(a),
            "/read":      lambda: self._cmd_read(a),
            "/write":     lambda: self._cmd_write(a),
            "/writeall":  lambda: self._cmd_writeall(),
            "/undo":      lambda: self._cmd_undo(),
            "/readonly":  lambda: self._toggle_readonly(),
            "/filter":    lambda: self._set_filter(a),
            "/create":    lambda: self._cmd_create(a),
            "/ls":        lambda: self._cmd_ls(a),
            "/find":      lambda: self._cmd_find(a),
            "/grep":      lambda: self._cmd_grep(a),
            "/diff":      lambda: self._cmd_diff(a),
            "/run":       lambda: self._cmd_run(a),
            "/ingest":    lambda: self._cmd_ingest(a),
            "/cd":        lambda: self._cmd_cd(a),
            "/save":      lambda: self._cmd_save(a),
            "/web":       lambda: self._cmd_web(a),
            "/git":       lambda: self._cmd_git(a),
            "/lint":      lambda: self._cmd_lint(a),
            "/copy":      lambda: self._cmd_copy(),
        }

        handler = cmds.get(c, "UNKNOWN")
        if handler is None:
            self._exit_summary()
            return False
        if handler == "UNKNOWN":
            console.print(f"  {WARN} unknown: {c} -- try /help\n")
            return True
        handler()  # type: ignore
        return True

    # -- Exit summary -------------------------------------------------------

    def _exit_summary(self) -> None:
        console.print()
        console.print(Rule(style="dim"))
        parts: list[str] = [f"{self.turns} turns"]
        parts.append(f"{len(self.history)} messages")
        if self._files_written:
            parts.append(f"{self._files_written} files written")
        console.print(f"  {D}session ended  --  {' | '.join(parts)}{R}")
        console.print()

    # -- Session commands ----------------------------------------------------

    def _clear(self) -> None:
        self.history.clear(); self.turns = 0; self.last = ""; self.last_writes.clear()
        console.print(f"  {OK} cleared\n")

    def _compact(self) -> None:
        if len(self.history) < 4:
            console.print(f"  {WARN} not enough history\n"); return
        n = len(self.history) - 4
        self.history = self.history[-4:]
        console.print(f"  {OK} removed {n} messages\n")

    def _context(self) -> None:
        t = Table(
            box=box.ROUNDED, show_header=True, header_style="bold cyan",
            title=f"{D}conversation context{R}", title_style="",
            padding=(0, 1),
        )
        t.add_column("#", style="dim", width=4, justify="right")
        t.add_column("role", width=10)
        t.add_column("preview", style="dim", max_width=55)
        t.add_column("chars", style="dim", justify="right", width=7)
        for i, m in enumerate(self.history):
            role = f"{G}user{R}" if isinstance(m, HumanMessage) else f"{C}assistant{R}"
            preview = m.content[:60].replace("\n", " ") + ("..." if len(m.content) > 60 else "")
            t.add_row(str(i), role, preview, str(len(m.content)))
        console.print(); console.print(t)
        total_chars = sum(len(m.content) for m in self.history)
        est_tokens = total_chars // 4
        console.print(f"\n  {D}{len(self.history)} messages | ~{est_tokens:,} tokens | {self.turns} turns{R}\n")

    def _status(self) -> None:
        from src.core.providers import get_provider_display, PROVIDERS
        from src.core.config import get_config
        from src.core.vector_store import get_or_create_collection
        cfg = get_config(); p = get_provider_display(cfg)
        model_display = self._model_override or p['model']
        prov_display = self.provider_override or p['provider']
        try: dc = get_or_create_collection().count()
        except Exception: dc = "?"

        t = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
        t.add_column("key", style="dim", width=12)
        t.add_column("value", style="cyan")
        t.add_row("provider", str(prov_display))
        t.add_row("model", str(model_display))
        t.add_row("indexed", f"{dc} chunks")
        t.add_row("turns", str(self.turns))
        t.add_row("history", f"{len(self.history)} messages")
        t.add_row("files out", str(self._files_written))
        t.add_row("readonly", str(self.readonly))
        t.add_row("filters", str(list(self.filter_ext)) if self.filter_ext else "None")
        t.add_row("cwd", str(Path.cwd()))
        console.print(); console.print(t)

    def _set_model(self, model: str) -> None:
        if not model: console.print(f"  {WARN} usage: /model <model-name>\n"); return
        self._model_override = model
        console.print(f"  {OK} model -> {C}{model}{R}\n")

    def _set_provider(self, prov: str) -> None:
        if not prov: console.print(f"  {WARN} usage: /provider <name>\n"); return
        from src.core.providers import PROVIDERS
        if prov.lower() not in PROVIDERS:
            avail = ", ".join(sorted(PROVIDERS.keys()))
            console.print(f"  {FAIL} unknown: {prov}  ({D}available: {avail}{R})\n"); return
        self.provider_override = prov.lower()
        self._model_override = None
        spec = PROVIDERS[self.provider_override]
        console.print(f"  {OK} provider -> {C}{spec.label}{R} ({D}{spec.default_model}{R})\n")

    def _toggle_readonly(self) -> None:
        self.readonly = not self.readonly
        state = f"{G}ON{R}" if self.readonly else f"{D}OFF{R}"
        console.print(f"  {OK} readonly mode {state}\n")

    def _set_filter(self, exts: str) -> None:
        if not exts:
            self.filter_ext = None
            console.print(f"  {OK} filters cleared\n")
            return
        parts = {e.strip() if e.strip().startswith(".") else f".{e.strip()}" for e in exts.split(",")}
        valid = {k for k in _EXT_MAP.keys()}
        clean = {e for e in parts if e in valid}
        if clean:
            self.filter_ext = clean
            console.print(f"  {OK} filter set: {C}{', '.join(clean)}{R}\n")
        else:
            self.filter_ext = None
            console.print(f"  {WARN} no valid extensions found (try: .py, .ts, .md)\n")

    # -- File commands -------------------------------------------------------

    def _cmd_read(self, path: str) -> None:
        if not path: console.print(f"  {WARN} usage: /read <file>\n"); return
        content = _read(path)
        if content is None: return
        lc = content.count("\n") + 1
        self.history.append(HumanMessage(content=f"[File: {path}]\n```\n{content}\n```"))
        console.print(f"  {OK} loaded {C}{path}{R} {D}({lc} lines, {_fsize(len(content))}){R}\n")

    def _cmd_write(self, path: str) -> None:
        if self.readonly: console.print(f"  {FAIL} readonly mode is active\n"); return
        if not self.last: console.print(f"  {WARN} no response yet\n"); return
        blocks = _extract(self.last)
        if path:
            b = next((x for x in blocks if x.path == path), blocks[0] if blocks else None)
            if b:
                b.path = b.path or path
                rec = _write(b.path, b.content)
                if rec:
                    self.last_writes = [rec]; self._files_written += 1
                    console.print(f"  {OK} wrote {C}{b.path}{R}\n")
            else:
                console.print(f"  {FAIL} no code blocks\n")
        else:
            fb = _dedup_files(blocks)
            if fb:
                rec = _write(fb[0].path, fb[0].content)
                if rec:
                    self.last_writes = [rec]; self._files_written += 1
                    console.print(f"  {OK} wrote {C}{fb[0].path}{R}\n")
            else:
                console.print(f"  {WARN} usage: /write <path>\n")

    def _cmd_writeall(self) -> None:
        if self.readonly: console.print(f"  {FAIL} readonly mode is active\n"); return
        if not self.last: console.print(f"  {WARN} no response yet\n"); return
        fb = _dedup_files(_extract(self.last))
        if not fb: console.print(f"  {WARN} no file blocks detected\n"); return
        self.last_writes.clear()
        for b in fb:
            rec = _write(b.path, b.content)
            if rec:
                console.print(f"    {OK} {b.path}"); self.last_writes.append(rec)
                self._files_written += 1
        console.print(f"\n  {OK} {len(self.last_writes)}/{len(fb)} written\n")

    def _cmd_undo(self) -> None:
        if not self.last_writes:
            console.print(f"  {WARN} nothing to undo\n"); return
        for rec in self.last_writes:
            p = Path(rec.path)
            if rec.had_backup and Path(rec.backup_path).exists():
                shutil.move(rec.backup_path, rec.path)
                console.print(f"    {OK} restored {C}{rec.path}{R}")
            elif p.exists():
                p.unlink()
                console.print(f"    {OK} deleted {C}{rec.path}{R}")
            else:
                console.print(f"    {WARN} already gone: {rec.path}")
        console.print(f"\n  {OK} undo ({len(self.last_writes)} files)\n")
        self.last_writes.clear()

    def _cmd_create(self, path: str) -> None:
        if self.readonly: console.print(f"  {FAIL} readonly mode is active\n"); return
        if not path: console.print(f"  {WARN} usage: /create <path>\n"); return
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

    def _cmd_ls(self, path: str) -> None:
        t = Path(path) if path else Path.cwd()
        if not t.is_dir(): console.print(f"  {FAIL} not a directory: {t}\n"); return
        tree = Tree(f"{G}{t.name}/{R}")
        items = sorted(t.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        dirs = files = 0
        for item in items:
            if item.name.startswith(".") and item.name not in (".env", ".gitignore"): continue
            if item.name in IGNORED: continue
            if item.is_dir():
                tree.add(f"{C}{item.name}/{R}")
                dirs += 1
            else:
                tag = _ext_tag(item.name)
                sz = _fsize(item.stat().st_size)
                label = f"{item.name}  {D}{sz}{R}"
                if tag: label += f"  {tag}"
                tree.add(label)
                files += 1
            if dirs + files >= 50: tree.add(f"{D}...{R}"); break
        console.print(); console.print(tree)
        console.print(f"\n  {D}{dirs} dirs, {files} files{R}\n")

    def _cmd_find(self, pat: str) -> None:
        if not pat: console.print(f"  {WARN} usage: /find <pattern>\n"); return
        cwd = Path.cwd(); hits: list[tuple[str, bool]] = []
        for p in cwd.rglob(f"*{pat}*"):
            rel = str(p.relative_to(cwd))
            if set(rel.replace("\\", "/").split("/")) & IGNORED: continue
            hits.append((rel, p.is_dir()))
            if len(hits) >= 30: break
        if not hits: console.print(f"  {WARN} no matches\n"); return
        console.print()
        for h, is_d in hits:
            if is_d:
                console.print(f"  {C}>{R} {C}{h}/{R}")
            else:
                console.print(f"  {G}>{R} {h}")
        console.print(f"\n  {D}{len(hits)} result(s){R}\n")

    def _cmd_grep(self, pat: str) -> None:
        if not pat: console.print(f"  {WARN} usage: /grep <pattern>\n"); return
        hits: list[str] = []
        try:
            r = subprocess.run(
                ["git", "grep", "-n", "--color=never", "-I", pat],
                capture_output=True, text=True, timeout=10, cwd=Path.cwd(),
            )
            if r.stdout: hits = r.stdout.strip().split("\n")[:25]
        except Exception:
            cwd = Path.cwd()
            for p in cwd.rglob("*"):
                if not p.is_file() or p.stat().st_size > 500_000: continue
                if set(str(p.relative_to(cwd)).replace("\\", "/").split("/")) & IGNORED: continue
                try:
                    for i, ln in enumerate(p.read_text("utf-8", errors="ignore").split("\n"), 1):
                        if pat.lower() in ln.lower():
                            hits.append(f"{p.relative_to(cwd)}:{i}: {ln.strip()}")
                            if len(hits) >= 25: break
                except Exception: continue
                if len(hits) >= 25: break
        if not hits: console.print(f"  {WARN} no matches for '{pat}'\n"); return

        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
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
        console.print(); console.print(t)
        console.print(f"\n  {D}{len(hits)} match(es){R}\n")

    def _cmd_diff(self, path: str) -> None:
        try:
            cmd = ["git", "diff", "--color=never"] + ([path] if path else [])
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if r.stdout:
                console.print()
                console.print(Syntax(r.stdout, "diff", theme="monokai", line_numbers=False))
                console.print()
            else:
                console.print(f"  {D}no changes{' in '+path if path else ''}{R}\n")
        except Exception as exc:
            console.print(f"  {FAIL} {exc}\n")

    def _cmd_web(self, query: str) -> None:
        if not query: console.print(f"  {WARN} usage: /web <query>\n"); return
        from src.core.web import search_web
        console.print(f"  {D}searching web for '{query}'...{R}")
        res = search_web(query)
        self.history.append(HumanMessage(content=f"[Web Search: {query}]\n\n{res}"))
        console.print(f"  {OK} search results added to context\n")

    def _cmd_git(self, args: str) -> None:
        if not args: console.print(f"  {WARN} usage: /git <args>\n"); return
        
        # Auto-commit feature
        if "commit" in args and "-m" not in args:
             console.print(f"  {M} generating commit message...{R}")
             try:
                 diff = subprocess.check_output(["git", "diff", "--cached"], text=True)
                 if not diff:
                     console.print(f"  {WARN} no staged changes to commit\n"); return
                 
                 # Simple heuristic generation for v1.0
                 # In a real agent, we'd call the LLM here.
                 # For now, let's ask the USER to provide it or just run it interactively?
                 # Actually, let's try to grab a quick summary if we can, 
                 # but since we can't easily query the LLM synchronously without refactoring _ask needed
                 # Let's just warn for now, or just run git which will open editor.
                 pass 
             except Exception:
                 pass
        
        console.print(f"\n  {D}${R} {C}git {args}{R}\n")
        try:
            # We use shell=True to allow complex args, but git is safe-ish
            r = subprocess.run(f"git {args}", shell=True, capture_output=True, text=True, cwd=Path.cwd())
            if r.stdout: console.print(r.stdout.rstrip())
            if r.stderr: console.print(f"[yellow]{r.stderr.rstrip()}[/yellow]")
            if r.returncode == 0:
                 console.print(f"  {OK} success\n")
            else:
                 console.print(f"  {FAIL} exit {r.returncode}\n")
        except Exception as exc:
            console.print(f"  {FAIL} {exc}\n")

    def _cmd_lint(self, path: str) -> None:
        target = path or "."
        linters = ["ruff", "flake8", "black --check", "pylint"]
        found = False
        console.print()
        for linter in linters:
            base = linter.split()[0]
            if shutil.which(base):
                found = True
                console.print(f"  {D}running {linter}...{R}")
                try:
                    cmd = f"{linter} {target}"
                    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if r.returncode == 0:
                        console.print(f"    {OK} {base}: pass")
                    else:
                        console.print(f"    {FAIL} {base}: issues found")
                        console.print(Panel(r.stdout + r.stderr, title=base, border_style="red"))
                except Exception:
                    pass
        if not found:
            console.print(f"  {WARN} no linters found (install ruff, flake8, etc.)\n")
        console.print()

    # -- System commands -----------------------------------------------------

    def _cmd_run(self, cmd: str) -> None:
        if not cmd: console.print(f"  {WARN} usage: /run <command>\n"); return
        console.print(f"\n  {D}${R} {C}{cmd}{R}\n")
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60, cwd=Path.cwd())
            if r.stdout: console.print(r.stdout.rstrip())
            if r.stderr: 
                err = r.stderr.rstrip()
                console.print(f"[yellow]{err}[/yellow]")
                
                # Robust Error Recovery
                if "command not found" in err or "is not recognized" in err:
                    console.print(f"  {WARN} suggest: typo or missing package?")
                elif "pip" in cmd and "install" in cmd:
                    console.print(f"  {WARN} ensure virtualenv is active")

            if r.returncode != 0: console.print(f"\n  {D}exit {r.returncode}{R}")
        except subprocess.TimeoutExpired: console.print(f"  {FAIL} timeout (60s)")
        except Exception as exc: console.print(f"  {FAIL} {exc}")
        console.print()

    def _cmd_ingest(self, path: str) -> None:
        target = Path(path).resolve() if path else Path.cwd()
        if not target.is_dir():
            console.print(f"  {FAIL} not a directory: {target}\n"); return
        console.print(f"  {D}indexing {target}...{R}")
        try:
            from src.core.ingester import scan_directory, chunk_documents
            from src.core.vector_store import add_documents
            docs = list(scan_directory(target))
            chunks = chunk_documents(docs)
            written = add_documents(chunks)
            console.print(f"  {OK} indexed {G}{written}{R} chunks from {len(docs)} files\n")
        except Exception as exc:
            console.print(f"  {FAIL} {exc}\n")

    def _cmd_cd(self, path: str) -> None:
        if not path: console.print(f"  {D}{Path.cwd()}{R}\n"); return
        try:
            t = Path(path).resolve(); os.chdir(t)
            console.print(f"  {OK} {C}{t}{R}\n")
            # Proactive check
            self._check_index()
        except Exception as exc:
            console.print(f"  {FAIL} {exc}\n")

    def _cmd_save(self, path: str) -> None:
        out = Path(path) if path else Path(f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        lines: list[str] = [f"# Mnemosyne Chat -- {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"]
        for m in self.history:
            if isinstance(m, HumanMessage):
                lines.append(f"## User\n\n{m.content}\n\n---\n\n")
            else:
                lines.append(f"## Assistant\n\n{m.content}\n\n---\n\n")
        try:
            out.write_text("".join(lines), encoding="utf-8")
            console.print(f"  {OK} saved to {C}{out}{R}\n")
        except Exception as exc:
            console.print(f"  {FAIL} {exc}\n")

    def _cmd_copy(self) -> None:
        if not self.last:
            console.print(f"  {WARN} no response yet\n"); return
        blocks = _extract(self.last)
        if not blocks:
            console.print(f"  {WARN} no code blocks in last response\n"); return
        content = blocks[-1].content
        try:
            if sys.platform == "win32":
                subprocess.run("clip", input=content.encode("utf-8"), check=True)
            elif sys.platform == "darwin":
                subprocess.run("pbcopy", input=content.encode("utf-8"), check=True)
            else:
                subprocess.run(["xclip", "-selection", "clipboard"], input=content.encode("utf-8"), check=True)
            console.print(f"  {OK} copied {len(content)} chars to clipboard\n")
        except Exception:
            console.print(f"  {WARN} clipboard not available\n")

    # -- Multi-line ----------------------------------------------------------

    def _multiline(self, session: PromptSession[str]) -> str:
        lines: list[str] = []
        while True:
            try:
                ln = session.prompt(HTML('<path>... </path>'))
                if '"""' in ln:
                    lines.append(ln.split('"""')[0]); break
                lines.append(ln)
            except (KeyboardInterrupt, EOFError): break
        return "\n".join(lines)

    # -- Main loop -----------------------------------------------------------

    def run(self) -> None:
        from src.core.config import get_config
        from src.core.providers import get_provider_display
        from src.core.vector_store import get_or_create_collection

        cfg = get_config()
        if self.provider_override:
            from src.core.config import MnemosyneConfig
            d = cfg.model_dump(); d["llm_provider"] = self.provider_override
            cfg = MnemosyneConfig(**d)

        pinfo = get_provider_display(cfg)
        try: chunks = get_or_create_collection().count()
        except Exception: chunks = "?"

        _welcome(pinfo["label"], pinfo["model"], chunks)

        session: PromptSession[str] = PromptSession(
            history=InMemoryHistory(),
            style=PT_STYLE,
            completer=_make_completer(),
            bottom_toolbar=self._get_toolbar,
        )

        while True:
            try:
                raw = session.prompt(_prompt(self.readonly, self.filter_ext)).strip()
            except (KeyboardInterrupt, EOFError):
                self._exit_summary()
                break

            if not raw: continue
            
            if raw.startswith('"""'):
                rest = raw[3:]
                raw = (rest[:-3] if rest.endswith('"""') else rest + "\n" + self._multiline(session))

            if raw.startswith("/"):
                if not self._slash(raw): break
                continue

            # Smart Shell Direct Execution (v1.1)
            # If input starts with a known shell command, treat it as /run or /git
            first_word = raw.split()[0].lower()
            if first_word in ("ls", "dir", "cd", "pwd", "cls", "clear"):
                # Safe navigation commands
                if first_word == "cd":
                    self._cmd_cd(raw[2:].strip())
                elif first_word in ("ls", "dir"):
                     self._cmd_ls(raw[2:].strip() or ".")
                elif first_word == "pwd":
                     console.print(f"  {D}{Path.cwd()}{R}\n")
                elif first_word in ("cls", "clear"):
                     self._clear()
                continue
            
            if first_word == "git":
                self._cmd_git(raw[3:].strip())
                continue
                
            if first_word in ("curl", "wget", "npm", "uv", "pip", "python", "node", "docker", "grep", "cat", "echo", "mkdir", "rm", "rmdir"):
                console.print(f"  {D}executing shell command...{R}")
                self._cmd_run(raw)
                continue

            self._ask(raw)
