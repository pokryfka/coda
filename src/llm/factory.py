"""Provider/model factory for creating LLM chat clients."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

if TYPE_CHECKING:
    from src.config.settings import AppConfig


def create_llm(config: AppConfig, task: str | None = None) -> BaseChatModel:
    """Create an LLM client based on provider configuration.

    Args:
        config: Application configuration.
        task: Optional task name (plan, implement, fix) for model override.

    Returns:
        A configured BaseChatModel instance.

    Raises:
        ValueError: If the provider is not supported.
    """
    provider = config.llm.provider
    provider_config = getattr(config.llm, provider, None)
    if provider_config is None:
        msg = f"Unsupported LLM provider: {provider}"
        raise ValueError(msg)

    model = _resolve_model(provider_config.model, provider_config, task)

    if provider == "claude":
        return _create_claude(model)
    if provider == "gemini":
        return _create_gemini(model)
    if provider == "codex":
        return _create_codex(model)
    if provider == "ollama":
        return _create_ollama(model, provider_config.base_url)

    msg = f"Unsupported LLM provider: {provider}"
    raise ValueError(msg)


def _resolve_model(default_model: str, provider_config: object, task: str | None) -> str:
    """Resolve the model name, applying task-specific overrides if set."""
    if task:
        override = getattr(provider_config, f"{task}_model", "")
        if override:
            return override
    return default_model


def _create_claude(model: str) -> BaseChatModel:
    """Create a Claude (Anthropic) chat model."""
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=model, temperature=0)


def _create_gemini(model: str) -> BaseChatModel:
    """Create a Gemini (Google) chat model."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=model, temperature=0)


def _create_codex(model: str) -> BaseChatModel:
    """Create an OpenAI chat model."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, temperature=0)


def _create_ollama(model: str, base_url: str) -> BaseChatModel:
    """Create an Ollama chat model."""
    from langchain_ollama import ChatOllama

    kwargs: dict = {"model": model, "temperature": 0}
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOllama(**kwargs)
