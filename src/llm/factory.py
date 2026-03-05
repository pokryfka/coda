"""Provider/model factory for creating LLM chat clients."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from src.config.settings import LlmConfig, LlmMode, LlmProvider, LlmProviderConfig


def _get_tools() -> list:
    """Import and return all coding agent tools."""
    from src.agent.coding.tools import list_files, read_file, run_command, write_file

    return [read_file, write_file, list_files, run_command]


def create_llm(config: LlmConfig, task: LlmMode | None = None) -> Runnable:
    """Create an LLM client based on provider configuration.

    Args:
        config: LLM configuration.
        task: Optional LLM mode for model override.

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

    model = _resolve_model(provider_config.model, provider_config, task)
    kwargs = _merge_options(provider_config, task)

    if provider == LlmProvider.CLAUDE:
        llm = _create_claude(model, **kwargs)
    elif provider == LlmProvider.GEMINI:
        llm = _create_gemini(model, **kwargs)
    elif provider == LlmProvider.CODEX:
        llm = _create_codex(model, **kwargs)
    elif provider == LlmProvider.OLLAMA:
        llm = _create_ollama(model, **kwargs)
    else:
        msg = f"Unsupported LLM provider: {provider}"
        raise ValueError(msg)

    return llm.bind_tools(_get_tools())


def _resolve_model(default_model: str, provider_config: LlmProviderConfig, task: LlmMode | None) -> str:
    """Resolve the model name, applying task-specific overrides if set."""
    if task:
        mode_config = provider_config.modes.get(task)
        if mode_config and mode_config.model:
            return mode_config.model
    return default_model


def _merge_options(provider_config: LlmProviderConfig, task: LlmMode | None) -> dict[str, Any]:
    """Merge provider-level options with mode-level overrides."""
    merged = dict(provider_config.options)
    if task:
        mode_config = provider_config.modes.get(task)
        if mode_config:
            merged.update(mode_config.options)
    return merged


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
