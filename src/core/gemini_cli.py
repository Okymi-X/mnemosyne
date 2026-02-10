"""
Mnemosyne -- Gemini CLI Integration
Bridge between Mnemosyne's RAG context and Google's Gemini CLI agent.
Supports headless queries, interactive delegation, streaming, and context piping.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

console = Console(highlight=False)

# On Windows, npm-installed CLIs are .cmd wrappers that need shell=True
# or the fully-resolved path.  We cache the lookup once.
_GEMINI_PATH: str | None = shutil.which("gemini")
_USE_SHELL = sys.platform == "win32" and (_GEMINI_PATH or "").lower().endswith(".cmd")


def _gemini_cmd() -> str:
    """Return the resolved gemini executable path (or just 'gemini')."""
    return _GEMINI_PATH or "gemini"


# ---------------------------------------------------------------------------
# Detection & status
# ---------------------------------------------------------------------------


def is_gemini_cli_installed() -> bool:
    """Check if Gemini CLI (gemini) is available on PATH."""
    return _GEMINI_PATH is not None


def get_gemini_cli_version() -> str | None:
    """Return the installed Gemini CLI version, or None."""
    try:
        r = subprocess.run(
            [_gemini_cmd(), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            shell=_USE_SHELL,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None


def get_install_instructions() -> str:
    """Return install instructions for the user's platform."""
    if sys.platform == "win32":
        return (
            "Install Gemini CLI:\n"
            "  npm install -g @google/gemini-cli\n"
            "  # or run without installing:\n"
            "  npx @google/gemini-cli"
        )
    elif sys.platform == "darwin":
        return "Install Gemini CLI:\n  brew install gemini-cli\n  # or via npm:\n  npm install -g @google/gemini-cli"
    else:
        return "Install Gemini CLI:\n  npm install -g @google/gemini-cli"


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


def build_context_prompt(
    query: str,
    codebase_context: str = "",
    memory: str = "",
    history_summary: str = "",
) -> str:
    """
    Build a rich prompt for Gemini CLI that includes Mnemosyne's
    RAG context, episodic memory, and conversation history.
    """
    parts: list[str] = []

    parts.append(
        "You are being invoked by Mnemosyne, a RAG-powered coding assistant. "
        "The developer has provided codebase context below. "
        "Use it to give an accurate, grounded answer."
    )

    if memory:
        parts.append(f"\n## Project Memory\n{memory}")

    if codebase_context:
        parts.append(f"\n## Codebase Context (from vector search)\n{codebase_context}")

    if history_summary:
        parts.append(f"\n## Conversation so far\n{history_summary}")

    parts.append(f"\n## User Request\n{query}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Execution modes
# ---------------------------------------------------------------------------


@dataclass
class GeminiCLIResult:
    """Result from a Gemini CLI invocation."""

    output: str
    exit_code: int
    model: str = ""
    error: str = ""


# Windows command-line length limit (8191 chars).  When the prompt
# exceeds this we write it to a temp file and pass the path instead.
_CMD_LINE_LIMIT = 7_500


def _prompt_args_and_stdin(
    prompt: str,
) -> tuple[list[str], str | None, "_PromptTempFile | None"]:
    """
    Decide how to pass *prompt* to Gemini CLI.

    * Short prompts  -> ``["-p", prompt]`` as CLI args.
    * Long prompts   -> write to a temp file and pass
      ``["-p", "@<path>"]`` (Gemini CLI's file reference syntax),
      or fall back to piping via stdin with ``["--stdin"]``.

    Returns ``(extra_args, stdin_text, temp_file_handle)``.
    The caller must keep the temp-file handle alive until the
    subprocess completes, then delete it.
    """
    import tempfile

    if len(prompt) <= _CMD_LINE_LIMIT:
        return (["-p", prompt], None, None)

    # Write to a temp file so the command line stays short
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".md",
        prefix="mnemosyne_gemini_",
        delete=False,
        encoding="utf-8",
    )
    tmp.write(prompt)
    tmp.flush()
    tmp.close()  # close so Gemini can read it on Windows
    return (["-p", f"@{tmp.name}"], None, tmp)


class _PromptTempFile:
    """Thin wrapper so callers can clean up the temp file."""

    def __init__(self, path: str) -> None:
        self.path = path

    def cleanup(self) -> None:
        try:
            os.unlink(self.path)
        except OSError:
            pass


def query_headless(
    prompt: str,
    *,
    model: str = "",
    cwd: Path | None = None,
    timeout: int = 120,
    output_format: str = "text",
    include_dirs: list[str] | None = None,
) -> GeminiCLIResult:
    """
    Run Gemini CLI in headless (non-interactive) mode with `-p`.
    Returns structured result.
    """
    if not is_gemini_cli_installed():
        return GeminiCLIResult(
            output="",
            exit_code=-1,
            error="Gemini CLI is not installed. " + get_install_instructions(),
        )

    prompt_args, stdin_text, tmp = _prompt_args_and_stdin(prompt)
    cmd: list[str] = [_gemini_cmd()] + prompt_args

    if model:
        cmd.extend(["-m", model])

    if output_format == "json":
        cmd.extend(["--output-format", "json"])

    if include_dirs:
        cmd.extend(["--include-directories", ",".join(include_dirs)])

    work_dir = str(cwd) if cwd else None

    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            input=stdin_text,
            timeout=timeout,
            cwd=work_dir,
            shell=_USE_SHELL,
        )
        return GeminiCLIResult(
            output=r.stdout.strip(),
            exit_code=r.returncode,
            model=model or "default",
            error=r.stderr.strip() if r.returncode != 0 else "",
        )
    except subprocess.TimeoutExpired:
        return GeminiCLIResult(
            output="",
            exit_code=-2,
            error=f"Gemini CLI timed out after {timeout}s",
        )
    except Exception as exc:
        return GeminiCLIResult(
            output="",
            exit_code=-3,
            error=str(exc),
        )
    finally:
        if tmp:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


def query_streaming(
    prompt: str,
    *,
    model: str = "",
    cwd: Path | None = None,
    include_dirs: list[str] | None = None,
) -> Generator[str, None, int]:
    """
    Run Gemini CLI and yield output chunks as they arrive.
    Returns exit code on StopIteration.
    """
    if not is_gemini_cli_installed():
        yield f"[Error] Gemini CLI not installed.\n{get_install_instructions()}\n"
        return -1

    prompt_args, stdin_text, tmp = _prompt_args_and_stdin(prompt)
    cmd: list[str] = [_gemini_cmd()] + prompt_args

    if model:
        cmd.extend(["-m", model])

    if include_dirs:
        cmd.extend(["--include-directories", ",".join(include_dirs)])

    work_dir = str(cwd) if cwd else None

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE if stdin_text else None,
            text=True,
            cwd=work_dir,
            env={**os.environ},
            bufsize=1,
            shell=_USE_SHELL,
        )

        # If we piped the prompt via stdin, send it and close
        if stdin_text and proc.stdin:
            proc.stdin.write(stdin_text)
            proc.stdin.close()

        if proc.stdout:
            for line in proc.stdout:
                yield line

        proc.wait()

        if proc.returncode != 0 and proc.stderr:
            err = proc.stderr.read()
            if err.strip():
                yield f"\n[stderr] {err.strip()}\n"

        return proc.returncode or 0

    except Exception as exc:
        yield f"\n[Error] {exc}\n"
        return -3
    finally:
        if tmp:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


def launch_interactive(
    *,
    model: str = "",
    cwd: Path | None = None,
    include_dirs: list[str] | None = None,
) -> int:
    """
    Launch Gemini CLI in full interactive mode (takes over the terminal).
    Returns exit code when the user quits Gemini CLI.
    """
    if not is_gemini_cli_installed():
        console.print(f"[red][-][/red] Gemini CLI not installed.\n[dim]{get_install_instructions()}[/dim]")
        return -1

    cmd: list[str] = [_gemini_cmd()]

    if model:
        cmd.extend(["-m", model])

    if include_dirs:
        cmd.extend(["--include-directories", ",".join(include_dirs)])

    work_dir = str(cwd) if cwd else None

    try:
        r = subprocess.run(cmd, cwd=work_dir, shell=_USE_SHELL)
        return r.returncode
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        console.print(f"[red][-][/red] Failed to launch Gemini CLI: {exc}")
        return -3


def query_with_context(
    query: str,
    *,
    codebase_context: str = "",
    memory: str = "",
    history_summary: str = "",
    model: str = "",
    cwd: Path | None = None,
    stream: bool = True,
) -> GeminiCLIResult | Generator[str, None, int]:
    """
    High-level helper: build a context-enriched prompt and send it to Gemini CLI.
    If stream=True, returns a generator; otherwise returns a GeminiCLIResult.
    """
    full_prompt = build_context_prompt(
        query,
        codebase_context=codebase_context,
        memory=memory,
        history_summary=history_summary,
    )

    if stream:
        return query_streaming(full_prompt, model=model, cwd=cwd)
    else:
        return query_headless(full_prompt, model=model, cwd=cwd)
