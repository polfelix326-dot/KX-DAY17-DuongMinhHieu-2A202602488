from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Normalize provider name and handle common aliases."""
    normalized = (value or "").strip().lower().replace("_", "-")
    alias_map = {
        "openai": "openai",
        "chatgpt": "openai",
        "custom": "custom",
        "compatible": "custom",
        "openai-compatible": "custom",
        "gemini": "gemini",
        "google": "gemini",
        "google-genai": "gemini",
        "anthropic": "anthropic",
        "anthorpic": "anthropic",
        "claude": "anthropic",
        "ollama": "ollama",
        "openrouter": "openrouter",
        "open-router": "openrouter",
    }
    return alias_map.get(normalized, normalized)


def build_chat_model(config: ProviderConfig) -> Any:
    """Instantiate the real chat model for the selected provider."""
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatOpenAI(**kwargs)

    elif provider == "custom":
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": config.model_name,
            "temperature": config.temperature,
            "base_url": config.base_url,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        else:
            kwargs["api_key"] = "custom-key"
        return ChatOpenAI(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs: dict[str, Any] = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**kwargs)

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs: dict[str, Any] = {
            "model_name": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatAnthropic(**kwargs)

    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs: dict[str, Any] = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    elif provider == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter

            kwargs: dict[str, Any] = {
                "model": config.model_name,
                "temperature": config.temperature,
            }
            if config.api_key:
                kwargs["api_key"] = config.api_key
            return ChatOpenRouter(**kwargs)
        except Exception:
            # Fallback to ChatOpenAI with openrouter base url
            from langchain_openai import ChatOpenAI

            kwargs: dict[str, Any] = {
                "model": config.model_name,
                "temperature": config.temperature,
                "base_url": config.base_url or "https://openrouter.ai/api/v1",
            }
            if config.api_key:
                kwargs["api_key"] = config.api_key
            return ChatOpenAI(**kwargs)

    else:
        raise ValueError(f"Unsupported provider: '{config.provider}' (normalized: '{provider}')")
