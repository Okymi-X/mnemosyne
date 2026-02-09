<p align="center">
  <h1 align="center">mnemosyne</h1>
  <p align="center">
    <strong>Local, agentic coding assistant with infinite context.</strong>
  </p>
  <p align="center">
    RAG-powered CLI that indexes your codebase, remembers everything, and acts on it.
  </p>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#providers">Providers</a> •
  <a href="#commands">Commands</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#license">License</a>
</p>

---

## What is Mnemosyne?

Mnemosyne is a **local-first coding assistant** that runs entirely in your terminal. It ingests your codebase into a local vector database (ChromaDB), then uses RAG to give any LLM deep, accurate knowledge about your project — no cloud uploads, no token limits.

It includes an **interactive agentic REPL** (v0.4) with **Deep Reasoning** capabilities, **Adaptive Context Filtering**, and a full suite of file operations. It can create files, run commands, search your codebase, and scaffold entire projects — all from a single terminal session.

```
╭──────────────────────────────────────────────╮
│   mnemosyne  v0.3                            │
│                                              │
│   provider  Groq                             │
│   model     llama-3.3-70b-versatile          │
│   indexed   46 chunks                        │
│   cwd       ~/myproject                      │
│                                              │
│   /help for commands  |  Ctrl-C to exit      │
╰──────────────────────────────────────────────╯

myproject > explain how auth works in this codebase
```

---

## Quick Start

### Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip
- An API key from any [supported provider](#providers)

### Install

```bash
git clone https://github.com/Okymi-X/mnemosyne.git
cd mnemosyne
uv sync
```

### Configure

```bash
cp .env.example .env
```

Edit `.env` and set your provider + API key:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
```

### Run

```bash
# Index your codebase
mnemosyne ingest .

# Interactive chat (recommended)
mnemosyne chat

# Or single-shot query
mnemosyne ask "How does the auth module work?"
```

---

## Features

### Multi-Provider LLM Support

Switch between 6 providers with a single env variable. No code changes needed.

| Provider | Default Model | Env Variable |
|----------|--------------|--------------|
| **Google Gemini** | `gemini-2.0-flash` | `GOOGLE_API_KEY` |
| **Anthropic Claude** | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| **Groq** | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| **OpenRouter** | `google/gemini-2.0-flash-exp:free` | `OPENROUTER_API_KEY` |
| **OpenAI** | `gpt-4o-mini` | `OPENAI_API_KEY` |
| **Ollama** | `llama3.2` | No key needed |

Override per-command:

```bash
mnemosyne ask "explain this" --provider anthropic
mnemosyne chat -p groq
```

### Interactive Agentic REPL

The `chat` command launches a full terminal coding session:

- **Web Search** — (v1.0) `/web <query>` access to live internet data
- **Git Commander** — (v1.0) `/git` integration with auto-commit suggestions
- **Code Linter** — (v1.0) `/lint` checks code quality (ruff, flake8, etc.)
- **Smart Diff** — (v0.6) view changes before applying (`d` or `diff`)
- **Auto-Ingestion** — (v0.6) detects new projects and offers to index
- **Robust Error Recovery** — (v0.6) suggests fixes for failed commands
- **Autocompletion** — (v0.5) smart suggestions for commands and files
- **Bottom Toolbar** — (v0.5) persistent status bar
- **Streaming output** — tokens appear in real-time
- **Conversation memory** — multi-turn follow-ups
- **Auto file creation** — detects files in responses, asks to write them
- **Shell execution** — run commands without leaving chat
- **Codebase search** — grep, find, read files from chat
- **Multi-line input** — wrap in triple quotes (`"""`)

```
myproject > create a flask API with user auth

  ... LLM streams response with code blocks ...

    > src/app.py
    > src/auth.py
    > requirements.txt

  write 3 file(s)? [Y/n] y
    + src/app.py
    + src/auth.py
    + requirements.txt

  + 3/3 written
```

### Slash Commands

| Command | Description |
|---------|-------------|
| `/filter <exts>` | Filter context by file extension (e.g. .py) |
| `/readonly` | Toggle read-only mode (safety) |
| `/read <path>` | Load file into conversation context |
| `/write <path>` | Write last code block to file |
| `/writeall` | Write all detected files |
| `/create <path>` | Create empty file or directory |
| `/ls [path]` | Tree directory listing |
| `/find <pattern>` | Find files by name |
| `/grep <pattern>` | Search file contents |
| `/diff [path]` | Show git diff |
| `/run <cmd>` | Execute shell command |
| `/cd <path>` | Change directory |
| `/clear` | Reset conversation |
| `/compact` | Trim history to save context |
| `/status` | Show config and stats |
| `/quit` | Exit |

### Smart Ingestion

Supports **80+ file types** across all major languages and frameworks:

- **Languages** — Python, JavaScript/TypeScript, Rust, Go, C/C++, Java, C#, Ruby, PHP, Swift, Kotlin, Scala, and more
- **Web** — HTML, CSS, SCSS, Vue, Svelte, JSX, TSX
- **Config** — YAML, TOML, JSON, Dockerfiles, Makefiles, CI configs
- **Docs** — Markdown, RST, plain text

Features:
- Language-aware chunking (splits on function/class boundaries)
- `.gitignore`-aware (respects your ignore rules)
- Exact filename matching (Dockerfile, Makefile, pyproject.toml, etc.)
- Custom extensions via `--ext` flag

```bash
mnemosyne ingest . --ext .log --ext .csv
```

### Episodic Memory (MEMORY.md)

Create a `MEMORY.md` file in your project root to give Mnemosyne high-priority architectural context:

```markdown
# MEMORY.md

- Auth uses JWT tokens stored in httpOnly cookies
- Database is PostgreSQL with Prisma ORM
- All API routes require authentication except /health
- Frontend uses React 19 with server components
```

This file is treated as **high-priority context** in every query — like giving your assistant permanent project knowledge.

---

## Commands

```bash
mnemosyne init              # Initialize in current project
mnemosyne ingest <path>     # Scan and index a directory
mnemosyne ask "<question>"  # Single-shot RAG query
mnemosyne chat              # Interactive agentic REPL
mnemosyne status            # Show config and collection stats
mnemosyne forget            # Wipe the knowledge base
```

---

## Architecture

```
src/
├── cli/
│   ├── main.py          # Typer CLI (init, ingest, ask, chat, status, forget)
│   └── chat.py          # Interactive agentic REPL
└── core/
    ├── config.py        # Pydantic-settings configuration
    ├── providers.py     # LLM provider factory (6 providers)
    ├── ingester.py      # Smart file scanner + language-aware chunker
    ├── vector_store.py  # ChromaDB persistence layer
    └── brain.py         # RAG engine (streaming + conversation history)
```

**Stack:**

| Component | Technology |
|-----------|-----------|
| CLI | Typer + Rich |
| LLM | LangChain (multi-provider) |
| Vector DB | ChromaDB (local, persistent) |
| Config | pydantic-settings + dotenv |
| Input | prompt-toolkit |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |

---

## Configuration

All configuration is done via environment variables (`.env` file):

```env
# Required: pick one provider
LLM_PROVIDER=groq              # google | anthropic | groq | openrouter | openai | ollama

# Required: API key for your provider
GROQ_API_KEY=gsk_...

# Optional: override default model
LLM_MODEL=llama-3.1-8b-instant

# Optional: ChromaDB settings
CHROMA_DB_PATH=.mnemosyne/chroma
COLLECTION_NAME=mnemosyne
```

For Ollama (fully local, no API key):

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.2
```

---

## Development

```bash
# Clone and install
git clone https://github.com/Okymi-X/mnemosyne.git
cd mnemosyne
uv sync

# Run in dev mode
uv run mnemosyne --help
uv run mnemosyne chat
```

---

## License

MIT

---

<p align="center">
  <sub>Built with LangChain, ChromaDB, and too much coffee.</sub>
</p>
