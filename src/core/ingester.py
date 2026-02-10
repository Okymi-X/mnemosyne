"""
Mnemosyne -- Smart Ingester v2.0
Recursively walks a directory, respects .gitignore, filters for code files,
and chunks documents using LangChain text splitters.
Supports 80+ file extensions with priority-based file scoring.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import pathspec
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from rich.console import Console

console = Console(highlight=False)

# ---------------------------------------------------------------------------
# Supported file extensions -- grouped by category
# ---------------------------------------------------------------------------

CODE_EXTENSIONS: set[str] = {
    # -- Systems & compiled languages -----------------------------------
    ".py", ".pyx", ".pyi", ".pyw",              # Python
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx",   # C / C++
    ".rs",                                        # Rust
    ".go",                                        # Go
    ".java", ".kt", ".kts",                       # Java / Kotlin
    ".cs",                                        # C#
    ".swift",                                     # Swift
    ".m", ".mm",                                  # Objective-C / C++
    ".scala", ".sc",                              # Scala
    ".zig",                                       # Zig
    ".nim",                                       # Nim
    ".ex", ".exs",                                # Elixir
    ".erl", ".hrl",                               # Erlang
    ".hs", ".lhs",                                # Haskell
    ".ml", ".mli",                                # OCaml
    ".clj", ".cljs", ".cljc", ".edn",            # Clojure
    ".dart",                                      # Dart
    ".r", ".R",                                   # R
    ".jl",                                        # Julia
    ".lua",                                       # Lua
    ".v",                                         # V / Verilog
    ".sol",                                       # Solidity

    # -- Web & scripting ------------------------------------------------
    ".js", ".mjs", ".cjs",                        # JavaScript
    ".ts", ".mts", ".cts",                        # TypeScript
    ".jsx", ".tsx",                               # React
    ".vue",                                       # Vue
    ".svelte",                                    # Svelte
    ".astro",                                     # Astro
    ".php",                                       # PHP
    ".rb", ".erb", ".rake",                       # Ruby
    ".pl", ".pm",                                 # Perl

    # -- Shells & scripts -----------------------------------------------
    ".sh", ".bash", ".zsh", ".fish",              # Shell
    ".ps1", ".psm1", ".psd1",                     # PowerShell
    ".bat", ".cmd",                               # Windows batch

    # -- Markup & styling -----------------------------------------------
    ".html", ".htm", ".xhtml",                    # HTML
    ".css", ".scss", ".sass", ".less", ".styl",   # CSS & preprocessors
    ".xml", ".xsl", ".xslt", ".svg",              # XML

    # -- Data & config ---------------------------------------------------
    ".json", ".jsonc", ".json5", ".jsonl",        # JSON
    ".yaml", ".yml",                              # YAML
    ".toml",                                      # TOML
    ".ini", ".cfg", ".conf",                      # INI / config
    ".env",                                       # Dotenv
    ".properties",                                # Java properties
    ".hcl", ".tf", ".tfvars",                     # Terraform / HCL

    # -- Documentation & text -------------------------------------------
    ".md", ".mdx", ".rst", ".txt",                # Markdown / text
    ".tex", ".bib",                               # LaTeX
    ".adoc",                                      # AsciiDoc
    ".org",                                       # Org-mode

    # -- Database & query -----------------------------------------------
    ".sql", ".prisma", ".graphql", ".gql",        # SQL / GraphQL

    # -- DevOps & CI/CD -------------------------------------------------
    ".dockerfile",                                # Docker (explicit ext)
    ".nginx", ".htaccess",                        # Web server configs

    # -- Editor & IDE config --------------------------------------------
    ".editorconfig",                              # EditorConfig
    ".prettierrc",                                # Prettier
    ".eslintrc",                                  # ESLint
    ".babelrc",                                   # Babel
}

# Files matched by exact name (no extension filter)
EXACT_FILENAMES: set[str] = {
    "Dockerfile", "Containerfile",
    "Makefile", "GNUmakefile",
    "Justfile",
    "Vagrantfile",
    "Gemfile", "Rakefile",
    "CMakeLists.txt",
    "BUILD", "WORKSPACE",
    ".gitignore", ".dockerignore", ".eslintignore",
    "tsconfig.json", "jsconfig.json",
    "pyproject.toml", "setup.cfg", "setup.py",
    "Cargo.toml", "Cargo.lock",
    "go.mod", "go.sum",
    "package.json", "package-lock.json",
    "composer.json",
    "requirements.txt", "Pipfile",
    "flake.nix", "shell.nix",
    ".env.example", ".env.local", ".env.development", ".env.production",
}

# ---------------------------------------------------------------------------
# Directories that are always skipped
# ---------------------------------------------------------------------------

IGNORED_DIRS: set[str] = {
    ".git", "__pycache__", "node_modules",
    "venv", ".venv", ".mnemosyne",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".next", ".nuxt", ".output", ".svelte-kit",
    "dist", "build", ".eggs", "out",
    "target",                      # Rust / Java / Scala
    "vendor",                      # Go / PHP / Ruby
    ".gradle", ".idea", ".vs",     # IDE dirs
    "coverage", ".nyc_output",     # Test coverage
    ".terraform",                  # Terraform
    ".cache", ".parcel-cache",     # Misc caches
    "bin", "obj",                  # .NET
}


@dataclass
class IngestedDocument:
    """Represents a single file read from disk."""
    source: str
    content: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """A chunk of text with provenance metadata."""
    chunk_id: str
    content: str
    metadata: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# File priority scoring -- higher = more important for context
# ---------------------------------------------------------------------------

_HIGH_PRIORITY_NAMES: set[str] = {
    "README.md", "MEMORY.md", "CHANGELOG.md",
    "pyproject.toml", "package.json", "Cargo.toml", "go.mod",
    "Makefile", "Dockerfile", "docker-compose.yml",
    ".env.example", "tsconfig.json",
}

_HIGH_PRIORITY_PATTERNS: list[str] = [
    "main", "app", "index", "server", "config", "routes", "schema",
    "models", "auth", "api", "handler", "controller", "service",
]


def _compute_priority(filepath: Path, root: Path) -> str:
    """Compute a priority tag for a file: 'high', 'medium', or 'low'."""
    name = filepath.name
    rel = str(filepath.relative_to(root)).lower()
    
    # Explicit high-priority files
    if name in _HIGH_PRIORITY_NAMES:
        return "high"
    
    # Pattern-based priority
    stem = filepath.stem.lower()
    if any(pat in stem for pat in _HIGH_PRIORITY_PATTERNS):
        return "high"
    
    # Tests and generated files are lower priority
    if "test" in rel or "spec" in rel or "__" in name:
        return "low"
    if "generated" in rel or "vendor" in rel or "migrations" in rel:
        return "low"
    
    return "medium"


# ---------------------------------------------------------------------------
# Gitignore helper
# ---------------------------------------------------------------------------

def _load_gitignore(root: Path) -> pathspec.PathSpec | None:
    """Load .gitignore patterns from the project root, if it exists."""
    gitignore_path = root / ".gitignore"
    if not gitignore_path.is_file():
        return None
    try:
        with gitignore_path.open("r", encoding="utf-8", errors="ignore") as fh:
            return pathspec.PathSpec.from_lines("gitwildmatch", fh)
    except Exception:
        return None


def _is_ignored(path: Path, root: Path, gitignore: pathspec.PathSpec | None) -> bool:
    """Check whether a path should be skipped."""
    for part in path.relative_to(root).parts:
        if part in IGNORED_DIRS:
            return True

    if gitignore is not None:
        rel = str(path.relative_to(root)).replace("\\", "/")
        if gitignore.match_file(rel):
            return True

    return False


def _should_include(filepath: Path, extra_extensions: set[str] | None = None) -> bool:
    """Check if a file matches supported extensions or exact filenames."""
    # Exact filename match
    if filepath.name in EXACT_FILENAMES:
        return True

    # Extension match
    allowed = CODE_EXTENSIONS
    if extra_extensions:
        allowed = allowed | extra_extensions

    return filepath.suffix.lower() in allowed


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def scan_directory(
    root: Path,
    extra_extensions: set[str] | None = None,
) -> Generator[IngestedDocument, None, None]:
    """
    Recursively walk *root* and yield an ``IngestedDocument`` for every
    code file that passes the extension filter and ignore rules.

    *extra_extensions* can be used to add additional extensions (e.g. {".log"}).
    """
    root = root.resolve()
    if not root.is_dir():
        console.print(f"[red][-][/red] Path is not a directory: {root}")
        return

    gitignore = _load_gitignore(root)
    file_count = 0

    for filepath in sorted(root.rglob("*")):
        if not filepath.is_file():
            continue

        if _is_ignored(filepath, root, gitignore):
            continue

        if not _should_include(filepath, extra_extensions):
            continue

        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
        except (OSError, PermissionError) as exc:
            console.print(f"[yellow][!][/yellow] Skipping {filepath}: {exc}")
            continue

        if not content.strip():
            continue

        file_count += 1
        rel = str(filepath.relative_to(root))
        priority = _compute_priority(filepath, root)
        yield IngestedDocument(
            source=rel,
            content=content,
            metadata={
                "source": rel,
                "extension": filepath.suffix or filepath.name,
                "priority": priority,
            },
        )

    if file_count == 0:
        console.print("[yellow][!][/yellow] No code files found to ingest.")


# ---------------------------------------------------------------------------
# Language-aware splitter selection
# ---------------------------------------------------------------------------

_EXT_TO_LANGUAGE: dict[str, Language] = {
    ".py":    Language.PYTHON,
    ".pyx":   Language.PYTHON,
    ".pyi":   Language.PYTHON,
    ".js":    Language.JS,
    ".mjs":   Language.JS,
    ".cjs":   Language.JS,
    ".jsx":   Language.JS,
    ".ts":    Language.TS,
    ".mts":   Language.TS,
    ".tsx":   Language.TS,
    ".go":    Language.GO,
    ".rs":    Language.RUST,
    ".java":  Language.JAVA,
    ".kt":    Language.KOTLIN,
    ".kts":   Language.KOTLIN,
    ".scala": Language.SCALA,
    ".swift": Language.SWIFT,
    ".rb":    Language.RUBY,
    ".md":    Language.MARKDOWN,
    ".mdx":   Language.MARKDOWN,
    ".rst":   Language.RST,
    ".html":  Language.HTML,
    ".htm":   Language.HTML,
    ".sol":   Language.SOL,
    ".hs":    Language.HASKELL,
    ".php":   Language.PHP,
    ".lua":   Language.LUA,
    ".pl":    Language.PERL,
    ".cpp":   Language.CPP,
    ".cc":    Language.CPP,
    ".cxx":   Language.CPP,
    ".hpp":   Language.CPP,
    ".c":     Language.C,
    ".h":     Language.C,
    ".cs":    Language.CSHARP,
    ".ex":    Language.ELIXIR,
    ".exs":   Language.ELIXIR,
}


def _get_splitter(extension: str, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Return a language-aware text splitter when possible."""
    lang = _EXT_TO_LANGUAGE.get(extension.lower())
    if lang:
        try:
            return RecursiveCharacterTextSplitter.from_language(
                language=lang,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except Exception:
            pass  # fall through to generic

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\nclass ", "\ndef ", "\nfunction ", "\n\n", "\n", " ", ""],
    )


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

def chunk_documents(
    documents: list[IngestedDocument],
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> list[DocumentChunk]:
    """
    Split a list of ``IngestedDocument`` into smaller
    ``DocumentChunk`` objects using language-aware LangChain splitters.
    """
    chunks: list[DocumentChunk] = []

    for doc in documents:
        ext = doc.metadata.get("extension", "")
        splitter = _get_splitter(ext, chunk_size, chunk_overlap)
        splits = splitter.split_text(doc.content)

        for idx, text in enumerate(splits):
            raw_id = f"{doc.source}::chunk_{idx}"
            chunk_id = hashlib.sha256(raw_id.encode()).hexdigest()[:16]

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    content=text,
                    metadata={
                        "source": doc.source,
                        "chunk_index": str(idx),
                        "extension": ext,
                        "priority": doc.metadata.get("priority", "medium"),
                    },
                )
            )

    return chunks