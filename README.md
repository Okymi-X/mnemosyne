<p align="center">
  <h1 align="center">mnemosyne</h1>
  <p align="center">
    <strong>Local, agentic coding assistant with infinite context.</strong>
  </p>
  <p align="center">
    RAG-powered CLI that indexes your codebase, remembers everything, and acts on it -- with adaptive retrieval, deep reasoning, and smart context.
  </p>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> |
  <a href="#features">Features</a> |
  <a href="#providers">Providers</a> |
  <a href="#commands">Commands</a> |
  <a href="#architecture">Architecture</a> |
  <a href="#license">License</a>
</p>

---

## What is Mnemosyne?

Mnemosyne is a **local-first coding assistant** that runs entirely in your terminal. It ingests your codebase into a local vector database (ChromaDB), then uses RAG to give any LLM deep, accurate knowledge about your project -- no cloud uploads, no token limits.

It includes an **interactive agentic REPL** (v2.0) with **Deep Reasoning**, **Adaptive Context Retrieval**, **Query Rewriting**, **Smart Conversation Compaction**, **Git Intelligence** and a full suite of file operations. It can create files, run commands, search your codebase, generate commit messages, and scaffold entire projects -- all from a single terminal session.

```
  ┏┳┓┏┓┏┓┏┳┓┏┓┏━┓┓ ┏┏┓┏┓
  ┃┃┃┃┃┣ ┃┃┃┃┃┗━┓┗┳┛┃┃┣
  ┛ ┗┛┗┗┛┛ ┗┗┛┗━┛ ┻ ┛┗┗┛

╭─ v2.0 ───────────────────────────────────────╮
│   provider   Google Gemini                   │
│   model      gemini-2.0-flash                │
│   indexed    46 chunks                       │
│   project    Python                          │
│   cwd        ~/myproject                     │
╰──────── Type / for commands │ Ctrl-C to exit ╯

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

# Delegate to Gemini CLI with RAG context
mnemosyne gemini "explain the architecture"

# Launch Gemini CLI interactive with context
mnemosyne gemini --interactive
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

- **Adaptive Context** -- auto-adjusts retrieval depth based on query complexity
- **Query Rewriting** -- expands queries for better vector search recall
- **Smart Compaction** -- summarises old turns instead of dropping them
- **Git Intelligence** -- LLM-generated conventional commit messages
- **Project Detection** -- auto-detects framework (Next.js, Python, Rust...)
- **Priority Scoring** -- high-priority files (main, config, API) ranked first
- **Smart Shell** -- run commands directly (`ls`, `curl`, `git`, etc.) without `/run`
- **Web Search** -- `/web <query>` access to live internet data
- **Smart Error Recovery** -- context-aware suggestions for failed commands
- **Smart Diff** -- view changes before applying (`d` or `diff`)
- **Auto-Ingestion** -- detects new projects and offers to index
- **Autocompletion** -- smart suggestions for commands and files
- **Bottom Toolbar** -- persistent status bar with token count
- **Streaming output** -- tokens appear in real-time with speed indicator
- **Conversation memory** -- multi-turn follow-ups with auto-compaction
- **Auto file creation** -- detects files in responses, asks to write them
- **Codebase search** -- grep, find, read files from chat
- **Multi-line input** -- wrap in triple quotes (`"""`)

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
| `/gemini <query>` | Delegate to Gemini CLI with RAG context |
| `/gemini-interactive` | Launch full Gemini CLI session |
| `/news <query>` | Search recent news |
| `/clear` | Reset conversation |
| `/compact` | Trim history to save context |
| `/status` | Show config and stats |
| `/quit` | Exit |

### Gemini CLI Integration

Mnemosyne can delegate tasks to [Gemini CLI](https://github.com/google-gemini/gemini-cli) — Google's open-source terminal agent with 1M token context, Google Search grounding, and built-in tools.

**How it works:** Mnemosyne retrieves relevant codebase context via RAG, injects your MEMORY.md and conversation history, then sends the enriched prompt to Gemini CLI. The response flows back into your Mnemosyne session.

```bash
# From the chat session
myproject > /gemini refactor the auth module to use JWT

  preparing context for Gemini CLI...
  ──────────── gemini cli ────────────

  ... Gemini CLI streams response ...

  ────────────────────────────────────
  gemini-cli  |  12.3s  |  2847 chars

# Or as a standalone command
mnemosyne gemini "explain the API architecture"

# Full interactive Gemini CLI session
mnemosyne gemini -i
```

**Prerequisites:** Node.js 20+ and Gemini CLI installed:

```bash
npm install -g @google/gemini-cli
```

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
mnemosyne gemini "<query>"  # Delegate to Gemini CLI with RAG context
mnemosyne gemini -i         # Launch Gemini CLI interactive
mnemosyne status            # Show config and collection stats
mnemosyne forget            # Wipe the knowledge base
```

---

## Architecture

```
src/
├── cli/
│   ├── main.py          # Typer CLI (init, ingest, ask, chat, gemini, status, forget)
│   └── chat.py          # Interactive agentic REPL v2.0
└── core/
    ├── config.py        # Pydantic-settings configuration
    ├── providers.py     # LLM provider factory (6 providers)
    ├── ingester.py      # Smart file scanner + language-aware chunker + priority scoring
    ├── vector_store.py  # ChromaDB persistence layer
    ├── brain.py         # RAG engine (adaptive retrieval + query rewriting + streaming)
    ├── web.py           # Web search (DuckDuckGo + news)
    └── gemini_cli.py    # Gemini CLI bridge (headless, streaming, interactive)
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
  <sub>Built with LangChain, ChromaDB, and an unreasonable amount of terminal obsession.</sub>
</p>
