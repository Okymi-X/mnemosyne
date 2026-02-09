"""
Mnemosyne v0.3 -- Agentic Chat REPL
Clean, minimal terminal coding agent.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style as PTStyle
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree

console = Console(highlight=False)

# ---------------------------------------------------------------------------
# Visual tokens -- clean, no emojis
# ---------------------------------------------------------------------------
G  = "[bold bright_green]"   # green open
C  = "[cyan]"                # cyan open
D  = "[dim]"                 # dim open
R  = "[/]"                   # reset (close any)

OK   = f"{G}+{R}"
FAIL = "[bold red]-[/]"
WARN = "[bold yellow]![/]"
INFO = f"{G}>{R}"

PT_STYLE = PTStyle.from_dict({
    "prompt": "#00ff00 bold",
    "path":   "#555555",
})

IGNORED = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".mnemosyne", "dist", "build", ".next", ".nuxt", "target",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".egg-info",
})


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _prompt() -> HTML:
    return HTML(f"<path>{Path.cwd().name}</path> <prompt>&gt; </prompt>")


# ---------------------------------------------------------------------------
# Welcome
# ---------------------------------------------------------------------------

def _welcome(provider: str, model: str, chunks: int | str) -> None:
    console.print()
    console.print(Panel(
        f"  {G}mnemosyne{R}  {D}v0.3{R}\n"
        f"\n"
        f"  {D}provider{R}  {C}{provider}{R}\n"
        f"  {D}model{R}     {C}{model}{R}\n"
        f"  {D}indexed{R}   {C}{chunks}{R} chunks\n"
        f"  {D}cwd{R}       {C}{Path.cwd()}{R}\n"
        f"\n"
        f"  {D}/help for commands  |  Ctrl-C to exit{R}",
        border_style="bright_green",
        padding=(0, 1),
    ))
    console.print()


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

HELP = f"""
  {G}mnemosyne{R} {D}commands{R}

  {D}conversation{R}
    {G}/clear{R}              reset history
    {G}/compact{R}            trim to last 2 turns
    {G}/status{R}             show provider, model, stats

  {D}files{R}
    {G}/read{R} {D}<path>{R}         load file into context
    {G}/write{R} {D}<path>{R}        write last code block to file
    {G}/writeall{R}           write all detected files
    {G}/create{R} {D}<path>{R}       create empty file (dir if ends with /)
    {G}/ls{R} {D}[path]{R}           list directory tree
    {G}/find{R} {D}<pattern>{R}      find files by name
    {G}/grep{R} {D}<pattern>{R}      search file contents
    {G}/diff{R} {D}[path]{R}         git diff

  {D}system{R}
    {G}/run{R} {D}<cmd>{R}           execute shell command
    {G}/cd{R} {D}<path>{R}           change directory
    {G}/quit{R}               exit

  {D}tip: wrap multi-line input in triple quotes (\"\"\"){R}
"""


# ---------------------------------------------------------------------------
# Code block extraction
# ---------------------------------------------------------------------------

_BLOCK_RE = re.compile(
    r"```(\w[\w+#.-]*)?(?::([^\n]+))?\s*\n(.*?)\n```",
    re.DOTALL,
)


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
    """Deduplicate by path -- last occurrence wins."""
    od: OrderedDict[str, CodeBlock] = OrderedDict()
    for b in blocks:
        if b.path:
            od[b.path] = b
    return list(od.values())


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _write(path: str, content: str) -> bool:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return True
    except Exception as exc:
        console.print(f"  {FAIL} write failed: {path} -- {exc}")
        return False


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


# ---------------------------------------------------------------------------
# Stat bar
# ---------------------------------------------------------------------------

def _stat_bar(elapsed: float, n: int) -> None:
    tps = n / elapsed if elapsed > 0 else 0
    console.print(f"\n  {D}{n} tokens  |  {elapsed:.1f}s  |  {tps:.0f} tok/s{R}")


def _source_bar(srcs: list[str]) -> None:
    if srcs:
        console.print(f"  {D}sources: {', '.join(srcs[:6])}{R}")


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

    # -- Core: stream + auto-write ------------------------------------------

    def _ask(self, q: str) -> None:
        from src.core.brain import ask_streaming

        console.print()
        tokens: list[str] = []
        sources: list[str] = []
        t0 = time.perf_counter()

        try:
            gen = ask_streaming(
                q, history=self.history or None,
                n_results=self.n_results,
                provider_override=self.provider_override,
            )
            for tok in gen:
                tokens.append(tok)
                sys.stdout.write(tok)
                sys.stdout.flush()
        except StopIteration as e:
            if e.value:
                _, sources = e.value
        except Exception as exc:
            console.print(f"\n\n  {FAIL} {exc}\n")
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
                for r in vq(q, n_results=self.n_results):
                    if r.source not in seen:
                        seen.add(r.source)
                        sources.append(r.source)
            except Exception:
                pass

        _stat_bar(elapsed, len(tokens))
        _source_bar(sources)

        # Detect files
        files = _dedup_files(_extract(full))
        if files:
            console.print()
            for f in files:
                console.print(f"    {INFO} {C}{f.path}{R}")
            console.print()
            self._confirm(files)

        console.print()
        self.history.append(HumanMessage(content=q))
        self.history.append(AIMessage(content=full))
        self.turns += 1

    def _confirm(self, files: list[CodeBlock]) -> None:
        try:
            ans = input(f"  write {len(files)} file(s)? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if ans in ("", "y", "yes", "o", "oui"):
            ok = sum(1 for f in files if _write(f.path, f.content) and not console.print(f"    {OK} {f.path}"))
            console.print(f"\n  {OK} {ok}/{len(files)} written")
        else:
            console.print(f"  {D}skipped -- use /writeall later{R}")

    # -- Slash router -------------------------------------------------------

    def _slash(self, raw: str) -> bool:
        parts = raw.strip().split(None, 1)
        c = parts[0].lower()
        a = parts[1].strip() if len(parts) > 1 else ""

        cmds: dict[str, object] = {
            "/quit": None, "/exit": None, "/q": None,
            "/help":     lambda: console.print(HELP),
            "/clear":    lambda: self._clear(),
            "/compact":  lambda: self._compact(),
            "/status":   lambda: self._status(),
            "/read":     lambda: self._cmd_read(a),
            "/write":    lambda: self._cmd_write(a),
            "/writeall": lambda: self._cmd_writeall(),
            "/create":   lambda: self._cmd_create(a),
            "/ls":       lambda: self._cmd_ls(a),
            "/find":     lambda: self._cmd_find(a),
            "/grep":     lambda: self._cmd_grep(a),
            "/diff":     lambda: self._cmd_diff(a),
            "/run":      lambda: self._cmd_run(a),
            "/cd":       lambda: self._cmd_cd(a),
        }

        handler = cmds.get(c, "UNKNOWN")
        if handler is None:
            console.print(f"\n  {D}session ended -- {self.turns} turns{R}\n")
            return False
        if handler == "UNKNOWN":
            console.print(f"  {WARN} unknown: {c} -- try /help\n")
            return True
        handler()  # type: ignore
        return True

    # -- Slash implementations -----------------------------------------------

    def _clear(self) -> None:
        self.history.clear(); self.turns = 0; self.last = ""
        console.print(f"  {OK} cleared\n")

    def _compact(self) -> None:
        if len(self.history) < 4:
            console.print(f"  {WARN} not enough history\n"); return
        n = len(self.history) - 4
        self.history = self.history[-4:]
        console.print(f"  {OK} removed {n} messages\n")

    def _status(self) -> None:
        from src.core.providers import get_provider_display
        from src.core.config import get_config
        from src.core.vector_store import get_or_create_collection
        cfg = get_config(); p = get_provider_display(cfg)
        try: dc = get_or_create_collection().count()
        except Exception: dc = "?"
        console.print(
            f"\n  {D}provider{R}  {C}{p['label']}{R}"
            f"\n  {D}model{R}     {C}{p['model']}{R}"
            f"\n  {D}indexed{R}   {C}{dc}{R} chunks"
            f"\n  {D}turns{R}     {C}{self.turns}{R}"
            f"\n  {D}history{R}   {C}{len(self.history)}{R} messages"
            f"\n  {D}cwd{R}       {C}{Path.cwd()}{R}\n"
        )

    def _cmd_read(self, path: str) -> None:
        if not path: console.print(f"  {WARN} usage: /read <file>\n"); return
        content = _read(path)
        if content is None: return
        self.history.append(HumanMessage(content=f"[File: {path}]\n```\n{content}\n```"))
        console.print(f"  {OK} loaded {C}{path}{R} ({content.count(chr(10))+1} lines)\n")

    def _cmd_write(self, path: str) -> None:
        if not self.last: console.print(f"  {WARN} no response yet\n"); return
        blocks = _extract(self.last)
        if path:
            b = next((x for x in blocks if x.path == path), blocks[0] if blocks else None)
            if b:
                b.path = b.path or path
                if _write(b.path, b.content): console.print(f"  {OK} wrote {C}{b.path}{R}\n")
            else:
                console.print(f"  {FAIL} no code blocks\n")
        else:
            fb = _dedup_files(blocks)
            if fb:
                if _write(fb[0].path, fb[0].content): console.print(f"  {OK} wrote {C}{fb[0].path}{R}\n")
            else:
                console.print(f"  {WARN} usage: /write <path>\n")

    def _cmd_writeall(self) -> None:
        if not self.last: console.print(f"  {WARN} no response yet\n"); return
        fb = _dedup_files(_extract(self.last))
        if not fb: console.print(f"  {WARN} no file blocks detected\n"); return
        ok = 0
        for b in fb:
            if _write(b.path, b.content):
                console.print(f"    {OK} {b.path}"); ok += 1
        console.print(f"\n  {OK} {ok}/{len(fb)} written\n")

    def _cmd_create(self, path: str) -> None:
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
        tree = Tree(f"{C}{t.name}/{R}")
        items = sorted(t.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        n = 0
        for item in items:
            if item.name.startswith(".") and item.name not in (".env", ".gitignore"): continue
            if item.name in IGNORED: continue
            if item.is_dir():
                tree.add(f"{C}{item.name}/{R}")
            else:
                sz = item.stat().st_size
                h = f"{sz}B" if sz < 1024 else f"{sz//1024}K" if sz < 1048576 else f"{sz//1048576}M"
                tree.add(f"{item.name} {D}{h}{R}")
            n += 1
            if n >= 40: tree.add(f"{D}...{R}"); break
        console.print(); console.print(tree); console.print()

    def _cmd_find(self, pat: str) -> None:
        if not pat: console.print(f"  {WARN} usage: /find <pattern>\n"); return
        cwd = Path.cwd(); hits: list[str] = []
        for p in cwd.rglob(f"*{pat}*"):
            rel = str(p.relative_to(cwd))
            if set(rel.replace("\\", "/").split("/")) & IGNORED: continue
            hits.append(rel)
            if len(hits) >= 25: break
        if not hits: console.print(f"  {WARN} no matches\n"); return
        console.print()
        for h in hits:
            pre = f"{C}>{R}" if Path(cwd / h).is_dir() else f"{G}>{R}"
            console.print(f"  {pre} {h}")
        console.print(f"\n  {D}{len(hits)} result(s){R}\n")

    def _cmd_grep(self, pat: str) -> None:
        if not pat: console.print(f"  {WARN} usage: /grep <pattern>\n"); return
        hits: list[str] = []
        try:
            r = subprocess.run(
                ["git", "grep", "-n", "--color=never", "-I", pat],
                capture_output=True, text=True, timeout=10, cwd=Path.cwd(),
            )
            if r.stdout:
                hits = r.stdout.strip().split("\n")[:20]
        except Exception:
            cwd = Path.cwd()
            for p in cwd.rglob("*"):
                if not p.is_file() or p.stat().st_size > 500_000: continue
                if set(str(p.relative_to(cwd)).replace("\\","/").split("/")) & IGNORED: continue
                try:
                    for i, ln in enumerate(p.read_text("utf-8", errors="ignore").split("\n"), 1):
                        if pat.lower() in ln.lower():
                            hits.append(f"{p.relative_to(cwd)}:{i}: {ln.strip()}")
                            if len(hits) >= 20: break
                except Exception: continue
                if len(hits) >= 20: break
        if not hits: console.print(f"  {WARN} no matches for '{pat}'\n"); return
        console.print()
        for h in hits: console.print(f"  {D}{h}{R}")
        console.print(f"\n  {D}{len(hits)} match(es){R}\n")

    def _cmd_diff(self, path: str) -> None:
        try:
            cmd = ["git", "diff", "--color=never"] + ([path] if path else [])
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if r.stdout:
                console.print(Syntax(r.stdout, "diff", theme="monokai", line_numbers=False))
            else:
                console.print(f"  {D}no changes{' in '+path if path else ''}{R}\n")
        except Exception as exc:
            console.print(f"  {FAIL} {exc}\n")

    def _cmd_run(self, cmd: str) -> None:
        if not cmd: console.print(f"  {WARN} usage: /run <command>\n"); return
        console.print(f"\n  {D}$ {cmd}{R}\n")
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60, cwd=Path.cwd())
            if r.stdout: console.print(r.stdout.rstrip())
            if r.stderr: console.print(f"[yellow]{r.stderr.rstrip()}[/yellow]")
            if r.returncode != 0: console.print(f"  {D}exit {r.returncode}{R}")
        except subprocess.TimeoutExpired: console.print(f"  {FAIL} timeout (60s)")
        except Exception as exc: console.print(f"  {FAIL} {exc}")
        console.print()

    def _cmd_cd(self, path: str) -> None:
        if not path: console.print(f"  {D}{Path.cwd()}{R}\n"); return
        try:
            t = Path(path).resolve(); os.chdir(t)
            console.print(f"  {OK} {C}{t}{R}\n")
        except Exception as exc:
            console.print(f"  {FAIL} {exc}\n")

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
            history=InMemoryHistory(), style=PT_STYLE,
        )

        while True:
            try:
                raw = session.prompt(_prompt()).strip()
            except (KeyboardInterrupt, EOFError):
                console.print(f"\n\n  {D}session ended -- {self.turns} turns{R}\n")
                break

            if not raw: continue

            if raw.startswith('"""'):
                rest = raw[3:]
                raw = (rest[:-3] if rest.endswith('"""') else rest + "\n" + self._multiline(session))

            if raw.startswith("/"):
                if not self._slash(raw): break
                continue

            self._ask(raw)
