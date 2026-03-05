"""Provider/model factory for creating LLM chat clients."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from src.config.settings import LlmMode, LlmProvider, LlmProviderConfig

if TYPE_CHECKING:
    from src.config.settings import AppConfig


def _get_tools() -> list:
    """Import and return all coding agent tools."""
    from src.agent.coding.tools import list_files, read_file, run_command, write_file

    return [read_file, write_file, list_files, run_command]


def create_llm(config: AppConfig, task: LlmMode | None = None) -> Runnable:
    """Create an LLM client based on provider configuration.

    Args:
        config: Application configuration.
        task: Optional LLM mode for model override.

    Returns:
        A configured BaseChatModel instance.

    Raises:
        ValueError: If the provider is not supported.
    """
    provider = config.llm.provider
    provider_config = config.llm.providers.get(provider)
    if provider_config is None:
        msg = f"Unsupported LLM provider: {provider}"
        raise ValueError(msg)

    model = _resolve_model(provider_config.model, provider_config, task)

    if provider == LlmProvider.CLAUDE:
        llm = _create_claude(model)
    elif provider == LlmProvider.GEMINI:
        llm = _create_gemini(model)
    elif provider == LlmProvider.CODEX:
        llm = _create_codex(model)
    elif provider == LlmProvider.OLLAMA:
        llm = _create_ollama(model, provider_config.base_url)
    else:
        msg = f"Unsupported LLM provider: {provider}"
        raise ValueError(msg)

    return llm.bind_tools(_get_tools())


def _resolve_model(default_model: str, provider_config: LlmProviderConfig, task: LlmMode | None) -> str:
    """Resolve the model name, applying task-specific overrides if set."""
    if task:
        override = provider_config.model_overrides.get(task, "")
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
