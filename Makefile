# ── Mnemosyne Development Makefile ────────────────────────────────────────
#
# Usage:
#   make install      Install all dependencies (including dev)
#   make lint         Run ruff linter
#   make format       Auto-format code with ruff
#   make typecheck    Run mypy type checker
#   make check        Run all checks (format + lint + typecheck)
#   make build        Build wheel and sdist
#   make audit        Run dependency security audit
#   make clean        Remove build artifacts and caches
#   make run          Start interactive chat session
#   make help         Show this help
#
# ──────────────────────────────────────────────────────────────────────────

.DEFAULT_GOAL := help
.PHONY: install lint format typecheck check build audit clean run help

# ── Setup ─────────────────────────────────────────────────────────────────

install:  ## Install all dependencies (including dev tools)
	uv sync
	uv pip install ruff mypy pip-audit

# ── Code Quality ──────────────────────────────────────────────────────────

format:  ## Auto-format code with ruff
	uv run ruff format src/

lint:  ## Run ruff linter
	uv run ruff check src/

lint-fix:  ## Run ruff linter with auto-fix
	uv run ruff check src/ --fix

typecheck:  ## Run mypy type checker
	uv run mypy src/ --ignore-missing-imports

check: format lint typecheck  ## Run all checks (format + lint + types)
	@echo ""
	@echo "  All checks passed."

# ── Build & Audit ─────────────────────────────────────────────────────────

build:  ## Build wheel and sdist
	uv build

audit:  ## Run dependency security audit
	uv run pip-audit

# ── Run ───────────────────────────────────────────────────────────────────

run:  ## Start interactive chat session
	uv run mnemosyne chat

ingest:  ## Index current directory
	uv run mnemosyne ingest .

status:  ## Show Mnemosyne status
	uv run mnemosyne status

# ── Clean ─────────────────────────────────────────────────────────────────

clean:  ## Remove build artifacts and caches
	rm -rf dist/ build/ *.egg-info .mypy_cache .ruff_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "  Cleaned."

# ── Help ──────────────────────────────────────────────────────────────────

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
