"""
Mnemosyne -- Configuration Module
Loads settings from .env via pydantic-settings.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class MnemosyneConfig(BaseSettings):
    """Central configuration for the Mnemosyne engine."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- Provider selection ------------------------------------------------
    llm_provider: str = "google"  # google | anthropic | groq | openrouter | openai | ollama

    # -- API keys (set whichever matches your provider) --------------------
    google_api_key: str = ""
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    openai_api_key: str = ""

    # -- Ollama (local) ----------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"

    # -- Model override (empty = use provider default) ---------------------
    llm_model: str = ""

    # -- ChromaDB ----------------------------------------------------------
    chroma_db_path: str = ".mnemosyne/chroma"
    collection_name: str = "mnemosyne"
    embedding_model: str = "models/text-embedding-004"


@lru_cache(maxsize=1)
def get_config() -> MnemosyneConfig:
    """Return a cached singleton of the configuration."""
    cfg = MnemosyneConfig()
    # Resolve relative path to absolute based on INITIAL cwd
    # This prevents DB lock errors if the chat session changes directory.
    import os

    if not os.path.isabs(cfg.chroma_db_path):
        cfg.chroma_db_path = os.path.abspath(cfg.chroma_db_path)
    return cfg
