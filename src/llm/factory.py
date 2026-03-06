"""Provider/model factory for creating LLM chat clients."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from src.config.settings import LlmConfig, LlmMode, LlmProvider, LlmProviderConfig

logger = logging.getLogger(__name__)

def create_llm(config: LlmConfig, mode: LlmMode | None = None) -> Runnable:
    """Create an LLM client based on provider configuration.

    Args:
        config: LlmConfig with provider and model settings.
        mode: Optional LlmMode for mode-specific model/options override.

    Returns:
        A configured BaseChatModel instance.

    Raises:
        ValueError: If the provider is not supported.
    """
    provider = config.provider
    provider_config = config.providers.get(provider)
    if provider_config is None:
        msg = f"Unsupported LLM provider: {provider}"
        raise ValueError(msg)

    model, options = _resolve_model(provider_config, mode)

    logger.info(
        "Creating LLM: provider=%s, model=%s, option_keys=%s",
        provider,
        model,
        sorted(options.keys()),
    )
    if provider == LlmProvider.CLAUDE:
        llm = _create_claude(model, **options)
    elif provider == LlmProvider.GEMINI:
        llm = _create_gemini(model, **options)
    elif provider == LlmProvider.CODEX:
        llm = _create_codex(model, **options)
    elif provider == LlmProvider.OLLAMA:
        llm = _create_ollama(model, **options)
    else:
        msg = f"Unsupported LLM provider: {provider}"
        raise ValueError(msg)

    return llm


def _resolve_model(provider_config: LlmProviderConfig, mode: LlmMode | None) -> tuple[str, dict[str, Any]]:
    """Resolve model name and options, using mode-specific config if available."""
    if mode:
        mode_config = provider_config.modes.get(mode)
        if mode_config:
            model = mode_config.model or provider_config.model
            options = dict(mode_config.options) if mode_config.options else dict(provider_config.options)
            return model, options
    return provider_config.model, dict(provider_config.options)


def _create_claude(model: str, **kwargs: Any) -> BaseChatModel:
    """Create a Claude (Anthropic) chat model."""
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model, **kwargs)


def _create_gemini(model: str, **kwargs: Any) -> BaseChatModel:
    """Create a Gemini (Google) chat model."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=model, **kwargs)


def _create_codex(model: str, **kwargs: Any) -> BaseChatModel:
    """Create an OpenAI chat model."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, **kwargs)


def _create_ollama(model: str, **kwargs: Any) -> BaseChatModel:
    """Create an Ollama chat model."""
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, **kwargs)
