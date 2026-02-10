"""
Mnemosyne v3.1 -- Interactive Agent Session
REPL with autonomous tool-calling agent and modular command handlers.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter, PathCompleter, WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from rich import box
from rich.align import Align
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# Command modules
from src.cli.commands.files import (
    CodeBlock,
    WriteRecord,
    cmd_create,
    cmd_read,
    cmd_undo,
    cmd_write,
    cmd_writeall,
    dedup_files,
    extract_blocks,
    is_identical,
    view_diff,
    write_file,
)
from src.cli.commands.model import (
    cmd_gemini,
    cmd_gemini_interactive,
    cmd_news,
    cmd_save,
    cmd_set_filter,
    cmd_set_model,
    cmd_set_provider,
    cmd_toggle_readonly,
    cmd_web,
)
from src.cli.commands.search import (
    cmd_copy,
    cmd_diff,
    cmd_find,
    cmd_grep,
    cmd_ls,
)
from src.cli.commands.shell import (
    cmd_cd,
    cmd_git,
    cmd_ingest,
    cmd_lint,
    cmd_run,
)

# Shared theme -- single source of truth
from src.cli.theme import (
    FAIL,
    OK,
    PATH_RE,
    PT_STYLE,
    TABLE_BOX,
    TIPS,
    VERSION,
    WARN,
    A,
    C,
    D,
    G,
    M,
    R,
    W,
    Y,
    console,
    get_git_branch,
    get_git_dirty,
    render_banner,
    tool_icon,
)

# ---------------------------------------------------------------------------
# Prompt & Completion
# ---------------------------------------------------------------------------


def _prompt(readonly: bool, filter_set: set[str] | None) -> HTML:
    parts = ["<path>" + Path.cwd().name + "</path>"]
    branch = get_git_branch()
    if branch:
        dirty = "*" if get_git_dirty() else ""
        parts.append(f"<path>({branch}{dirty})</path>")
    if readonly:
        parts.append("<readonly> ro </readonly>")
    if filter_set:
        f = ",".join(sorted(filter_set))
        parts.append(f"<filter>{f}</filter>")
    parts.append("<prompt> > </prompt>")
    return HTML(" ".join(parts))


def _make_completer() -> NestedCompleter:
    path_comp = PathCompleter(only_directories=False, expanduser=True)
    dir_comp = PathCompleter(only_directories=True, expanduser=True)
    provs = WordCompleter(["groq", "anthropic", "google", "openai", "openrouter", "ollama"])

    return NestedCompleter.from_nested_dict(
        {
            "/help": None,
            "/clear": None,
            "/compact": None,
            "/context": None,
            "/status": None,
            "/tools": None,
            "/quit": None,
            "/exit": None,
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
            "/news": None,
            "/gemini": None,
            "/gemini-interactive": None,
            "/git": None,
            "/lint": path_comp,
            "/filter": WordCompleter([".py", ".js", ".ts", ".md", ".html", ".css", ".rs", ".go"]),
        }
    )


# ---------------------------------------------------------------------------
# Welcome & Help (standalone display functions)
# ---------------------------------------------------------------------------


def _detect_project() -> str:
    cwd = Path.cwd()
    indicators: dict[str, str] = {
        "package.json": "Node.js",
        "pyproject.toml": "Python",
        "Cargo.toml": "Rust",
        "go.mod": "Go",
        "pom.xml": "Java (Maven)",
        "build.gradle": "Java (Gradle)",
        "Gemfile": "Ruby",
        "composer.json": "PHP",
        "tsconfig.json": "TypeScript",
        "next.config.js": "Next.js",
        "next.config.mjs": "Next.js",
        "next.config.ts": "Next.js",
        "vite.config.ts": "Vite",
        "vite.config.js": "Vite",
        "svelte.config.js": "SvelteKit",
        "astro.config.mjs": "Astro",
        "nuxt.config.ts": "Nuxt",
        "Dockerfile": "Docker",
        "flake.nix": "Nix",
        "CMakeLists.txt": "C/C++ (CMake)",
        "Makefile": "Make",
    }
    detected = [label for fn, label in indicators.items() if (cwd / fn).exists()]
    return " + ".join(detected[:3]) if detected else "Unknown"


def _welcome(provider: str, model: str, chunks: int | str) -> None:
    console.print()

    # Gradient ASCII banner
    banner = render_banner()
    console.print(Align.center(banner))
    console.print()
    console.print(Align.center(Text(f"v{VERSION} · agentic coding assistant", style="dim")))
    console.print()

    project_type = _detect_project()

    from src.core.tools import TOOL_REGISTRY

    tools_count = len(TOOL_REGISTRY)

    from src.core.gemini_cli import get_gemini_cli_version, is_gemini_cli_installed

    gemini_line = ""
    if is_gemini_cli_installed():
        ver = get_gemini_cli_version() or "installed"
        gemini_line = f"\n   {D}gemini{R}     {C}{ver}{R}"

    info = (
        f"   {D}provider{R}   {C}{provider}{R}\n"
        f"   {D}model{R}      {C}{model}{R}\n"
        f"   {D}indexed{R}    {W}{chunks}{R} {D}chunks{R}\n"
        f"   {D}tools{R}      {A}{tools_count} available{R}\n"
        f"   {D}project{R}    {C}{project_type}{R}\n"
        f"   {D}cwd{R}        {D}{Path.cwd()}{R}"
        f"{gemini_line}"
    )
    console.print(
        Panel(
            info,
            border_style="bright_green",
            padding=(0, 1),
        )
    )
    console.print()

    # Tips for getting started
    console.print(f"  {D}Tips for getting started:{R}")
    for i, tip in enumerate(TIPS, 1):
        console.print(f"  {D}{i}.{R} {tip}")
    console.print()


def _help() -> None:
    t = Table(
        box=TABLE_BOX,
        show_header=True,
        header_style="bold bright_green",
        padding=(0, 2),
        title=f"{G}commands{R}",
        title_style="",
        show_edge=False,
    )
    t.add_column("command", style="bright_green", min_width=22, no_wrap=True)
    t.add_column("description", style="dim white")

    sections = [
        (
            f"{W}💬 conversation{R}",
            [
                ("/clear", "reset conversation history"),
                ("/compact", "smart-trim old turns to save context"),
                ("/save [path]", "export chat transcript to markdown"),
                ("/status", "show config and session stats"),
                ("/context", "inspect conversation messages"),
            ],
        ),
        (
            f"{W}🤖 model{R}",
            [
                ("/model <name>", "switch model mid-session"),
                ("/provider <name>", "switch LLM provider"),
                ("/filter <exts>", "restrict context by extension (.py,.ts)"),
                ("/web <query>", "search the web, inject results"),
            ],
        ),
        (
            f"{W}📁 files{R}",
            [
                ("/read <path>", "load file into context"),
                ("/write <path>", "write last code block to disk"),
                ("/writeall", "write all detected files"),
                ("/undo", "revert last written files"),
                ("/readonly", "toggle read-only safety lock"),
                ("/create <path>", "create empty file or directory"),
            ],
        ),
        (
            f"{W}🔍 search{R}",
            [
                ("/ls [path]", "list directory tree"),
                ("/find <pattern>", "find files by name pattern"),
                ("/grep <pattern>", "search inside file contents"),
                ("/diff [path]", "show git diff"),
                ("/copy", "copy last code block to clipboard"),
            ],
        ),
        (
            f"{W}⚡ system{R}",
            [
                ("/run <cmd>", "execute shell command"),
                ("/git <args>", "git with smart commit messages"),
                ("/lint [path]", "run available linters"),
                ("/ingest [path]", "re-index codebase into vector store"),
                ("/cd <path>", "change working directory"),
            ],
        ),
        (
            f"{W}🧠 agent{R}",
            [
                ("/tools", "show available agent tools"),
                ("/gemini <query>", "delegate to Gemini CLI with RAG context"),
                ("/gemini-interactive", "launch full Gemini CLI session"),
            ],
        ),
    ]

    for header, cmds in sections:
        t.add_row("", header)
        for c, d in cmds:
            t.add_row(c, d)
        t.add_row("", "")

    t.add_row("/quit", "exit session")

    console.print()
    console.print(t)
    console.print(f"\n  {A}agent mode{R} {D}── ask naturally, tools run automatically{R}")
    console.print(f"  {D}shell commands work directly:{R} ls, git, python, npm …")
    console.print(f"  {D}multi-line input: wrap in triple-quotes{R}\n")


# ---------------------------------------------------------------------------
# Auto-detect file paths in user queries
# ---------------------------------------------------------------------------


def _auto_read_paths(query: str, history: list[BaseMessage]) -> list[str]:
    already_loaded: set[str] = set()
    for m in history:
        if isinstance(m, HumanMessage):
            c = m.content if isinstance(m.content, str) else str(m.content)
            if c.startswith("[auto-loaded:"):
                end = c.find("]")
                if end > 0:
                    already_loaded.add(c[14:end].strip())
            elif c.startswith("[File:"):
                end = c.find("]")
                if end > 0:
                    already_loaded.add(c[7:end].strip())

    found = PATH_RE.findall(query)
    injected: list[str] = []
    for fp in found:
        if fp in already_loaded:
            continue
        p = Path(fp)
        if p.is_file() and p.stat().st_size < 200_000:
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                history.append(HumanMessage(content=f"[auto-loaded: {fp}]\n```\n{content}\n```"))
                injected.append(fp)
            except Exception:
                pass
    return injected


# ===========================================================================
# Session
# ===========================================================================


class ChatSession:
    def __init__(self, provider_override: str | None = None, n_results: int = 8):
        self.n_results = n_results
        self.history: list[BaseMessage] = []
        self.turns: int = 0
        self.last: str = ""
        self.last_writes: list[WriteRecord] = []
        self.readonly: bool = False
        self.filter_ext: set[str] | None = None
        self._files_written: int = 0
        self._last_tool_count: int = 0

        # Mutable holders so command modules can update these
        self._provider: list[str | None] = [provider_override]
        self._model: list[str | None] = [None]

        self._check_index()

    # -- Properties for cleaner access -------------------------------------

    @property
    def provider_override(self) -> str | None:
        return self._provider[0]

    @property
    def model_override(self) -> str | None:
        return self._model[0]

    # -- Helpers -----------------------------------------------------------

    def _check_index(self) -> None:
        try:
            from src.core.vector_store import get_or_create_collection

            if get_or_create_collection().count() == 0:
                console.print(f"  {WARN} Index is empty. Auto-running /ingest .")
                cmd_ingest(".")
        except Exception:
            pass

    def _get_toolbar(self) -> HTML:
        from src.core.config import MnemosyneConfig, get_config
        from src.core.providers import get_provider_display

        cfg = get_config()
        if self.provider_override:
            d = cfg.model_dump()
            d["llm_provider"] = self.provider_override
            cfg = MnemosyneConfig(**d)

        p = get_provider_display(cfg)
        model = self.model_override or p["model"]
        prov = p["provider"]

        total_chars = sum(len(str(m.content)) for m in self.history)
        est_tokens = total_chars // 4
        tok_display = f"{est_tokens:,}" if est_tokens > 0 else "0"

        # Left side: project + git
        left_parts: list[str] = [f" <b>{Path.cwd().name}</b>"]
        branch = get_git_branch()
        if branch:
            dirty = "*" if get_git_dirty() else ""
            left_parts.append(f"<style fg='#888888'>({branch}{dirty})</style>")

        # Right side: provider, model, stats
        right_parts: list[str] = [
            "<style fg='#e5a00d'>agent</style>",
            f"<style bg='#2a2a4e' fg='#aaaaaa'> {prov} ({model}) </style>",
            f"<style fg='#666666'>t:{self.turns} ~{tok_display}tok</style>",
        ]
        if self._last_tool_count > 0:
            right_parts.append(f"<style fg='#e5a00d'>{self._last_tool_count} tools</style>")
        if self.filter_ext:
            f = ",".join(sorted(self.filter_ext))
            right_parts.append(f"<style fg='#56c8d8'>{f}</style>")
        if self.readonly:
            right_parts.append("<style bg='#ff4444' fg='white'> RO </style>")
        if self._files_written:
            right_parts.append(f"<style fg='#a8e6cf'>{self._files_written} written</style>")

        left = " ".join(left_parts)
        right = "  ".join(right_parts)
        return HTML(f"{left}    {right}")

    # -- Core: agent-powered respond ----------------------------------------

    def _ask(self, q: str) -> None:
        from src.core.agent import run_agent

        injected = _auto_read_paths(q, self.history)
        if injected:
            for fp in injected:
                console.print(f"  {D}auto-loaded: {fp}{R}")
            console.print()

        console.print(Rule(style="dim"))
        console.print()

        fmeta: dict[str, Any] | None = None
        if self.filter_ext:
            fmeta = {"extension": {"$in": list(self.filter_ext)}}

        # -- Agent event handler (renders tool calls in real-time) ---------
        step_count = 0
        tool_count = 0

        def on_event(event: str, data: dict) -> None:
            nonlocal step_count, tool_count

            if event == "thinking":
                step = data.get("step", 1)
                label = "reasoning" if step == 1 else f"step {step}"
                sys.stdout.write(f"  \033[2m⟳ {label}…\033[0m\r")
                sys.stdout.flush()

            elif event == "thinking_block":
                sys.stdout.write("\r\033[K")
                text = data.get("text", "")
                if text:
                    preview = text[:300].replace("\n", " ").strip()
                    console.print(f"  {M}── reasoning ──{R}")
                    console.print(f"  {D}{preview}{R}")
                    console.print(f"  {M}───────────────{R}")
                    console.print()

            elif event == "reasoning":
                sys.stdout.write("\r\033[K")
                text = data.get("text", "")
                if text:
                    preview = text[:250].replace("\n", " ").strip()
                    if preview:
                        console.print(f"  {D}{preview}{R}")
                        console.print()

            elif event == "tool_start":
                sys.stdout.write("\r\033[K")

            elif event == "tool_end":
                tc = data.get("call")
                tr = data.get("result")
                if tc and tr:
                    tool_count += 1
                    icon = tool_icon(tc.name)
                    params_preview = " ".join(str(v)[:50] for v in tc.params.values())
                    if tr.success:
                        # Show first meaningful line of output
                        out_lines = [l.strip() for l in tr.output.split("\n") if l.strip() and not l.startswith("[")]
                        out_preview = out_lines[0][:70] if out_lines else ""
                        console.print(f"  {A}{icon}{R} {C}{tc.name}{R}  {D}{params_preview}{R}")
                        if out_preview:
                            console.print(
                                f"     {D}{out_preview}"
                                f"{'...' if len(out_preview) >= 70 else ''}{R}"
                                f"  {D}{tr.duration:.1f}s{R}"
                            )
                        else:
                            console.print(f"     {D}{tr.duration:.1f}s{R}")
                    else:
                        console.print(f"  [bold red]x[/] {C}{tc.name}{R}  {D}{params_preview}{R}")
                        console.print(f"     {D}{tr.output[:60]}{R}")
                    console.print()

            elif event == "step_done":
                step_obj = data.get("step")
                if step_obj and not step_obj.is_final:
                    step_count += 1

            elif event == "error":
                sys.stdout.write("\r\033[K")
                console.print(f"  {FAIL} {data.get('error', 'unknown error')}")

        # -- Run agent -----------------------------------------------------
        t0 = time.perf_counter()
        try:
            response = run_agent(
                q,
                history=self.history or None,
                n_results=self.n_results,
                provider_override=self.provider_override,
                model_override=self.model_override,
                filter_meta=fmeta,
                on_event=on_event,
            )
        except Exception as exc:
            sys.stdout.write("\r\033[K")
            console.print(f"\n  {FAIL} {exc}\n")
            return

        elapsed = time.perf_counter() - t0
        sys.stdout.write("\r\033[K")  # Clear any residual spinner

        full = response.answer
        self.last = full
        self._last_tool_count = tool_count

        # -- Display final answer ------------------------------------------
        if step_count > 0:
            console.print(Rule(f"{D}answer{R}", style="dim"))
            console.print()

        sys.stdout.write(full)
        sys.stdout.write("\n")
        sys.stdout.flush()

        # -- Stats ---------------------------------------------------------
        tok_est = len(full) // 4
        parts_s: list[str] = [f"~{tok_est} tok", f"{elapsed:.1f}s"]
        if step_count > 0:
            parts_s.append(f"{step_count + 1} steps")
        if tool_count > 0:
            parts_s.append(f"{tool_count} tools")
        console.print(f"\n  {D}{' │ '.join(parts_s)}{R}")

        if response.sources:
            src_list = ", ".join(response.sources[:6])
            if len(response.sources) > 6:
                src_list += f" +{len(response.sources) - 6}"
            console.print(f"  {D}from: {src_list}{R}")

        # -- Detect files in response --------------------------------------
        files = dedup_files(extract_blocks(full))
        if files:
            console.print()
            ft = Table(box=box.SIMPLE, show_header=False, padding=(0, 1), show_edge=False)
            ft.add_column("s", style="bold", width=3)
            ft.add_column("path", style="bold cyan")
            ft.add_column("info", style="dim")

            pending: list[CodeBlock] = []
            for f in files:
                lc = f.content.count("\n") + 1
                exists = Path(f.path).exists()
                identical = is_identical(f.path, f.content)
                if identical:
                    ft.add_row(f"{D}={R}", f"{D}{f.path}{R}", f"{lc}L -- identical")
                elif exists:
                    ft.add_row(f"{Y}~{R}", f.path, f"{lc}L -- overwrite")
                    pending.append(f)
                else:
                    ft.add_row(f"{G}+{R}", f.path, f"{lc}L -- new")
                    pending.append(f)
            console.print(ft)

            if self.readonly:
                console.print(f"\n  {D}readonly mode: skipping writes{R}")
            elif pending:
                self._confirm(pending)
            else:
                console.print(f"\n  {OK} all files identical -- nothing to write")

        console.print()
        self.history.append(HumanMessage(content=q))
        self.history.append(AIMessage(content=full))
        self.turns += 1

    def _confirm(self, files: list[CodeBlock]) -> None:
        while True:
            try:
                ans = input(f"\n  write {len(files)} file(s)? [Y/n/d] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return
            if ans in ("diff", "d"):
                for f in files:
                    if Path(f.path).exists():
                        view_diff(f.path, f.content)
                continue
            if ans in ("", "y", "yes", "o", "oui"):
                self.last_writes.clear()
                ok = 0
                for f in files:
                    rec = write_file(f.path, f.content, backup=True)
                    if rec:
                        tag = f" {D}(backup){R}" if rec.had_backup else ""
                        console.print(f"    {OK} {C}{f.path}{R}{tag}")
                        self.last_writes.append(rec)
                        self._files_written += 1
                        ok += 1
                console.print(f"\n  {D}{ok}/{len(files)} written{R}")
                break
            else:
                console.print(f"  {D}skipped{R}")
                break

    # -- Slash command router ----------------------------------------------

    def _slash(self, raw: str) -> bool:
        """Route /commands. Returns False to exit."""
        parts = raw.strip().split(None, 1)
        c = parts[0].lower()
        a = parts[1].strip() if len(parts) > 1 else ""

        # Mutable wrappers so commands can modify session state
        fw = [self._files_written]
        ro = [self.readonly]
        flt = [self.filter_ext]

        cmds: dict[str, object] = {
            "/quit": None,
            "/exit": None,
            "/q": None,
            "/help": lambda: _help(),
            "/clear": lambda: self._clear(),
            "/compact": lambda: self._compact(),
            "/context": lambda: self._context(),
            "/status": lambda: self._status(),
            "/model": lambda: cmd_set_model(a, self._model),
            "/provider": lambda: cmd_set_provider(a, self._provider, self._model),
            "/read": lambda: cmd_read(a, self.history),
            "/write": lambda: cmd_write(a, self.last, self.last_writes, fw, self.readonly),
            "/writeall": lambda: cmd_writeall(self.last, self.last_writes, fw, self.readonly),
            "/undo": lambda: cmd_undo(self.last_writes),
            "/readonly": lambda: cmd_toggle_readonly(ro),
            "/filter": lambda: cmd_set_filter(a, flt),
            "/create": lambda: cmd_create(a, self.readonly),
            "/ls": lambda: cmd_ls(a),
            "/find": lambda: cmd_find(a),
            "/grep": lambda: cmd_grep(a),
            "/diff": lambda: cmd_diff(a),
            "/run": lambda: cmd_run(a),
            "/ingest": lambda: cmd_ingest(a),
            "/cd": lambda: cmd_cd(a, self._check_index),
            "/save": lambda: cmd_save(a, self.history),
            "/web": lambda: cmd_web(a, self.history),
            "/news": lambda: cmd_news(a, self.history),
            "/git": lambda: cmd_git(a, self.provider_override, self.model_override),
            "/gemini": lambda: cmd_gemini(
                a,
                self.history,
                self.filter_ext,
                [self.last],
                self.readonly,
                self._confirm,
            ),
            "/gemini-interactive": lambda: cmd_gemini_interactive(),
            "/lint": lambda: cmd_lint(a),
            "/copy": lambda: cmd_copy(self.last),
            "/tools": lambda: self._show_tools(),
        }

        handler = cmds.get(c, "UNKNOWN")
        if handler is None:
            self._exit_summary()
            return False
        if handler == "UNKNOWN":
            console.print(f"  {WARN} unknown: {c} -- try /help\n")
            return True

        handler()  # type: ignore

        # Sync mutable wrappers back to session
        self._files_written = fw[0]
        self.readonly = ro[0]
        self.filter_ext = flt[0]

        return True

    # -- Session-level commands (small, kept inline) -----------------------

    def _exit_summary(self) -> None:
        console.print()
        total_chars = sum(len(str(m.content)) for m in self.history)
        est_tokens = total_chars // 4
        parts: list[str] = [f"{self.turns} turns"]
        parts.append(f"~{est_tokens:,} tokens")
        if self._last_tool_count:
            parts.append(f"{self._last_tool_count} tool calls")
        if self._files_written:
            parts.append(f"{self._files_written} files")
        console.print(Rule(f"{D}{' │ '.join(parts)}{R}", style="bright_green"))
        console.print(f"  {D}goodbye.{R}")
        console.print()

    def _clear(self) -> None:
        self.history.clear()
        self.turns = 0
        self.last = ""
        self.last_writes.clear()
        self._last_tool_count = 0
        console.print(f"  {OK} cleared\n")

    def _compact(self) -> None:
        if len(self.history) < 4:
            console.print(f"  {WARN} not enough history\n")
            return
        from src.core.brain import summarise_history

        old_len = len(self.history)
        self.history = summarise_history(self.history, keep_last=4)
        removed = old_len - len(self.history)
        console.print(f"  {OK} compacted {removed} messages -> {len(self.history)} remaining\n")

    def _show_tools(self) -> None:
        """Display available agent tools."""
        from src.core.tools import TOOL_REGISTRY

        t = Table(
            box=TABLE_BOX,
            show_header=True,
            header_style="bold #e5a00d",
            title=f"{A}agent tools{R}",
            title_style="",
            padding=(0, 2),
            show_edge=False,
        )
        t.add_column("tool", style="bold cyan", min_width=18, no_wrap=True)
        t.add_column("params", style="dim", min_width=20)
        t.add_column("description", style="dim white")
        for name, tool in TOOL_REGISTRY.items():
            icon = tool_icon(name)
            params = ", ".join(tool.parameters.keys())
            t.add_row(f"{A}{icon}{R} {name}", params, tool.description)
        console.print()
        console.print(t)
        console.print(f"\n  {D}tools are called automatically by the agent during conversations{R}\n")

    def _context(self) -> None:
        t = Table(
            box=TABLE_BOX,
            show_header=True,
            header_style="bold cyan",
            title=f"{D}conversation context{R}",
            title_style="",
            padding=(0, 1),
            show_edge=False,
        )
        t.add_column("#", style="dim", width=4, justify="right")
        t.add_column("role", width=10)
        t.add_column("preview", style="dim", max_width=50)
        t.add_column("size", style="dim", justify="right", width=7)
        for i, m in enumerate(self.history):
            role = f"{G}user{R}" if isinstance(m, HumanMessage) else f"{C}assistant{R}"
            preview = (m.content[:60] if isinstance(m.content, str) else str(m.content)[:60]).replace("\n", " ") + (
                "…" if len(str(m.content)) > 60 else ""
            )
            t.add_row(str(i), role, preview, str(len(str(m.content))))
        console.print()
        console.print(t)
        total_chars = sum(len(str(m.content)) for m in self.history)
        est_tokens = total_chars // 4
        console.print(f"\n  {D}{len(self.history)} messages │ ~{est_tokens:,} tokens │ {self.turns} turns{R}\n")

    def _status(self) -> None:
        from src.core.config import get_config
        from src.core.providers import get_provider_display
        from src.core.vector_store import get_or_create_collection

        cfg = get_config()
        p = get_provider_display(cfg)
        model_display = self.model_override or p["model"]
        prov_display = self.provider_override or p["provider"]
        try:
            dc = get_or_create_collection().count()
        except Exception:
            dc = "?"

        t = Table(box=TABLE_BOX, show_header=False, padding=(0, 2), show_edge=False)
        t.add_column("key", style="dim", width=12)
        t.add_column("value", style="bold cyan")
        t.add_row("provider", str(prov_display))
        t.add_row("model", str(model_display))
        t.add_row("indexed", f"{dc} chunks")
        t.add_row("turns", str(self.turns))
        t.add_row("history", f"{len(self.history)} messages")
        t.add_row("files out", str(self._files_written))
        t.add_row("readonly", str(self.readonly))
        t.add_row("filters", str(list(self.filter_ext)) if self.filter_ext else "None")
        t.add_row("cwd", str(Path.cwd()))

        from src.core.gemini_cli import get_gemini_cli_version, is_gemini_cli_installed

        if is_gemini_cli_installed():
            ver = get_gemini_cli_version() or "yes"
            t.add_row("gemini-cli", f"[green]{ver}[/green]")
        else:
            t.add_row("gemini-cli", "[dim]not installed[/dim]")

        console.print()
        console.print(t)

    # -- Multi-line input --------------------------------------------------

    def _multiline(self, session: PromptSession[str]) -> str:
        lines: list[str] = []
        while True:
            try:
                ln = session.prompt(HTML("<path>... </path>"))
                if '"""' in ln:
                    lines.append(ln.split('"""')[0])
                    break
                lines.append(ln)
            except (KeyboardInterrupt, EOFError):
                break
        return "\n".join(lines)

    # -- Main loop ---------------------------------------------------------

    def run(self) -> None:
        from src.core.config import MnemosyneConfig, get_config
        from src.core.providers import get_provider_display
        from src.core.vector_store import get_or_create_collection

        cfg = get_config()
        if self.provider_override:
            d = cfg.model_dump()
            d["llm_provider"] = self.provider_override
            cfg = MnemosyneConfig(**d)

        pinfo = get_provider_display(cfg)
        try:
            chunks: int | str = get_or_create_collection().count()
        except Exception:
            chunks = "?"

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

            if not raw:
                continue

            if raw.startswith('"""'):
                rest = raw[3:]
                raw = rest[:-3] if rest.endswith('"""') else rest + "\n" + self._multiline(session)

            if raw.startswith("/"):
                if not self._slash(raw):
                    break
                continue

            # Smart shell detection
            first = raw.split()[0].lower()
            if first in ("ls", "dir", "cd", "pwd", "cls", "clear"):
                rest_arg = raw[len(first) :].strip()
                if first == "cd":
                    cmd_cd(rest_arg, self._check_index)
                elif first in ("ls", "dir"):
                    cmd_ls(rest_arg or ".")
                elif first == "pwd":
                    console.print(f"  {D}{Path.cwd()}{R}\n")
                elif first in ("cls", "clear"):
                    self._clear()
                continue

            if first == "git":
                cmd_git(raw[3:].strip(), self.provider_override, self.model_override)
                continue

            if first == "gemini":
                cmd_gemini(
                    raw[6:].strip(),
                    self.history,
                    self.filter_ext,
                    [self.last],
                    self.readonly,
                    self._confirm,
                )
                continue

            if first in (
                "curl",
                "wget",
                "npm",
                "uv",
                "pip",
                "python",
                "node",
                "docker",
                "grep",
                "cat",
                "echo",
                "mkdir",
                "rm",
                "rmdir",
            ):
                console.print(f"  {D}executing shell command...{R}")
                cmd_run(raw)
                continue

            self._ask(raw)
