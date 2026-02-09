"""
Mnemosyne -- Provider Factory
Maps provider names to LangChain chat model classes.
Supports: google, anthropic, groq, openrouter, openai, ollama.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from rich.console import Console

from src.core.config import MnemosyneConfig, get_config

console = Console(highlight=False)


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderSpec:
    """Describes how to instantiate a specific LLM provider."""
    label: str                      # Human-friendly name
    default_model: str              # Sensible default model id
    key_field: str                   # Config attribute that holds the API key ("" for keyless)
    factory: str                    # Dotted import path  module:class


PROVIDERS: dict[str, ProviderSpec] = {
    "google": ProviderSpec(
        label="Google Gemini",
        default_model="gemini-2.0-flash",
        key_field="google_api_key",
        factory="langchain_google_genai:ChatGoogleGenerativeAI",
    ),
    "anthropic": ProviderSpec(
        label="Anthropic Claude",
        default_model="claude-sonnet-4-20250514",
        key_field="anthropic_api_key",
        factory="langchain_anthropic:ChatAnthropic",
    ),
    "groq": ProviderSpec(
        label="Groq",
        default_model="llama-3.3-70b-versatile",
        key_field="groq_api_key",
        factory="langchain_groq:ChatGroq",
    ),
    "openrouter": ProviderSpec(
        label="OpenRouter",
        default_model="google/gemini-2.0-flash-exp:free",
        key_field="openrouter_api_key",
        factory="langchain_openai:ChatOpenAI",
    ),
    "openai": ProviderSpec(
        label="OpenAI",
        default_model="gpt-4o-mini",
        key_field="openai_api_key",
        factory="langchain_openai:ChatOpenAI",
    ),
    "ollama": ProviderSpec(
        label="Ollama (local)",
        default_model="llama3.2",
        key_field="",
        factory="langchain_ollama:ChatOllama",
    ),
}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _import_class(dotted: str) -> type:
    """Dynamically import 'module:ClassName'."""
    module_path, class_name = dotted.rsplit(":", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def get_llm(config: MnemosyneConfig | None = None) -> BaseChatModel:
    """
    Build and return the appropriate LangChain chat model based on
    ``config.llm_provider``.
    """
    if config is None:
        config = get_config()

    provider_name = config.llm_provider.lower().strip()
    spec = PROVIDERS.get(provider_name)

    if spec is None:
        supported = ", ".join(sorted(PROVIDERS.keys()))
        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Supported: {supported}"
        )

    # Resolve API key
    api_key: str = ""
    if spec.key_field:
        api_key = getattr(config, spec.key_field, "")
        if not api_key:
            raise ValueError(
                f"Provider '{provider_name}' requires "
                f"{spec.key_field.upper()} to be set in .env"
            )

    # Choose model: user override or provider default
    model = config.llm_model or spec.default_model

    # Build kwargs
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": 0.2,
    }

    # Provider-specific wiring
    if provider_name == "google":
        kwargs["google_api_key"] = api_key
        kwargs["convert_system_message_to_human"] = True

    elif provider_name == "anthropic":
        kwargs["anthropic_api_key"] = api_key
        kwargs["max_tokens"] = 4096

    elif provider_name == "groq":
        kwargs["groq_api_key"] = api_key

    elif provider_name == "openrouter":
        kwargs["openai_api_key"] = api_key
        kwargs["openai_api_base"] = "https://openrouter.ai/api/v1"

    elif provider_name == "openai":
        kwargs["openai_api_key"] = api_key

    elif provider_name == "ollama":
        kwargs["base_url"] = config.ollama_base_url
        # Ollama doesn't need an API key
        kwargs.pop("temperature", None)

    # Import and instantiate
    cls = _import_class(spec.factory)
    return cls(**kwargs)


def get_provider_display(config: MnemosyneConfig | None = None) -> dict[str, str]:
    """Return display-friendly info about the active provider."""
    if config is None:
        config = get_config()

    provider_name = config.llm_provider.lower().strip()
    spec = PROVIDERS.get(provider_name)

    if spec is None:
        return {"provider": provider_name, "label": "UNKNOWN", "model": "N/A", "key_status": "N/A"}

    key_status = "NOT REQUIRED"
    if spec.key_field:
        key_val = getattr(config, spec.key_field, "")
        key_status = (key_val[:8] + "...") if key_val else "NOT SET"

    model = config.llm_model or spec.default_model

    return {
        "provider": provider_name,
        "label": spec.label,
        "model": model,
        "key_status": key_status,
    }
