"""Integration tests for LLM factory and client creation."""

from __future__ import annotations

import os

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from src.config.settings import LlmConfig, LlmMode, LlmModeConfig, LlmProvider, LlmProviderConfig
from src.llm.factory import create_llm

SIMPLE_PROMPT = [HumanMessage(content="Reply with exactly one word: hello")]


class TestLlmFactory:
    """Tests for LLM factory creation."""

    def test_create_claude_client(self) -> None:
        """Factory creates a Claude client when configured."""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
        config = LlmConfig(provider="claude")
        llm = create_llm(config)
        assert isinstance(llm, BaseChatModel)

    def test_create_gemini_client(self) -> None:
        """Factory creates a Gemini client when configured."""
        if not os.environ.get("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")
        config = LlmConfig(provider="gemini")
        llm = create_llm(config)
        assert isinstance(llm, BaseChatModel)

    def test_create_openai_client(self) -> None:
        """Factory creates an OpenAI client when configured."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        config = LlmConfig(provider="codex")
        llm = create_llm(config)
        assert isinstance(llm, BaseChatModel)

    def test_create_ollama_client(self) -> None:
        """Factory creates an Ollama client when configured."""
        config = LlmConfig(
            provider="ollama",
            providers={
                LlmProvider.OLLAMA: LlmProviderConfig(
                    model="qwen2.5-coder:14b",
                    options={"base_url": "http://localhost:11434"},
                ),
            },
        )
        llm = create_llm(config)
        assert isinstance(llm, BaseChatModel)

    def test_invalid_provider_raises(self) -> None:
        """Factory raises ValueError for unsupported provider."""
        config = LlmConfig(provider="invalid")
        with pytest.raises(ValueError, match="Unsupported"):
            create_llm(config)

    def test_task_model_override(self) -> None:
        """Factory uses task-specific model when set."""
        config = LlmConfig(
            provider="ollama",
            providers={
                LlmProvider.OLLAMA: LlmProviderConfig(
                    model="default-model",
                    options={"base_url": "http://localhost:11434"},
                    modes={LlmMode.PLAN: LlmModeConfig(model="plan-model")},
                ),
            },
        )
        llm = create_llm(config, task=LlmMode.PLAN)
        assert llm.model == "plan-model"

    def test_mode_options_override(self) -> None:
        """Factory uses mode-specific options when mode has its own model."""
        config = LlmConfig(
            provider="ollama",
            providers={
                LlmProvider.OLLAMA: LlmProviderConfig(
                    model="test-model",
                    options={"base_url": "http://localhost:11434", "temperature": 0},
                    modes={LlmMode.PLAN: LlmModeConfig(
                        model="plan-model",
                        options={"base_url": "http://localhost:11434", "temperature": 0.5},
                    )},
                ),
            },
        )
        llm = create_llm(config, task=LlmMode.PLAN)
        assert llm.temperature == 0.5

    def test_mode_options_only_without_model_are_ignored(self) -> None:
        """Mode with options but no model falls back to provider, ignoring mode options."""
        config = LlmConfig(
            provider="ollama",
            providers={
                LlmProvider.OLLAMA: LlmProviderConfig(
                    model="default-model",
                    options={"base_url": "http://localhost:11434", "temperature": 0},
                    modes={LlmMode.PLAN: LlmModeConfig(options={"temperature": 0.7})},
                ),
            },
        )
        llm = create_llm(config, task=LlmMode.PLAN)
        # BUG: mode options are ignored because mode has no model set
        assert llm.temperature == 0  # would expect 0.7

    def test_mode_with_model_drops_provider_options(self) -> None:
        """Mode with model loses provider-level options like base_url."""
        config = LlmConfig(
            provider="ollama",
            providers={
                LlmProvider.OLLAMA: LlmProviderConfig(
                    model="default-model",
                    options={"base_url": "http://localhost:11434", "temperature": 0},
                    modes={LlmMode.PLAN: LlmModeConfig(
                        model="plan-model",
                        options={"temperature": 0.5},
                    )},
                ),
            },
        )
        llm = create_llm(config, task=LlmMode.PLAN)
        # BUG: base_url from provider options is dropped
        assert not hasattr(llm, "base_url") or llm.base_url != "http://localhost:11434"


class TestLlmInvoke:
    """Tests for invoking LLM clients with a simple prompt."""

    def test_invoke_claude(self) -> None:
        """Claude client responds with a valid AIMessage."""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
        config = LlmConfig(provider="claude")
        llm = create_llm(config)
        response = llm.invoke(SIMPLE_PROMPT)
        assert isinstance(response, AIMessage)
        assert len(response.content) > 0

    def test_invoke_gemini(self) -> None:
        """Gemini client responds with a valid AIMessage."""
        if not os.environ.get("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")
        config = LlmConfig(provider="gemini")
        llm = create_llm(config)
        response = llm.invoke(SIMPLE_PROMPT)
        assert isinstance(response, AIMessage)
        assert len(response.content) > 0

    def test_invoke_codex(self) -> None:
        """OpenAI client responds with a valid AIMessage."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        config = LlmConfig(provider="codex")
        llm = create_llm(config)
        response = llm.invoke(SIMPLE_PROMPT)
        assert isinstance(response, AIMessage)
        assert len(response.content) > 0

    def test_invoke_ollama(self) -> None:
        """Ollama client responds with a valid AIMessage."""
        if not os.environ.get("OLLAMA_BASE_URL"):
            pytest.skip("OLLAMA_BASE_URL not set")
        config = LlmConfig(
            provider="ollama",
            providers={
                LlmProvider.OLLAMA: LlmProviderConfig(
                    model="qwen2.5-coder:14b",
                    options={"base_url": os.environ["OLLAMA_BASE_URL"]},
                ),
            },
        )
        llm = create_llm(config)
        response = llm.invoke(SIMPLE_PROMPT)
        assert isinstance(response, AIMessage)
        assert len(response.content) > 0
