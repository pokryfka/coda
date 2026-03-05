"""Integration tests for LLM factory and client creation."""

from __future__ import annotations

import os

import pytest
from langchain_core.runnables import RunnableBinding

from src.config.settings import AppConfig, LlmConfig, LlmMode, LlmModeConfig, LlmProvider, LlmProviderConfig
from src.llm.factory import create_llm


class TestLlmFactory:
    """Tests for LLM factory creation."""

    def test_create_claude_client(self) -> None:
        """Factory creates a Claude client when configured."""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
        config = AppConfig(llm=LlmConfig(provider="claude"))
        llm = create_llm(config)
        assert isinstance(llm, RunnableBinding)

    def test_create_gemini_client(self) -> None:
        """Factory creates a Gemini client when configured."""
        if not os.environ.get("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")
        config = AppConfig(llm=LlmConfig(provider="gemini"))
        llm = create_llm(config)
        assert isinstance(llm, RunnableBinding)

    def test_create_openai_client(self) -> None:
        """Factory creates an OpenAI client when configured."""
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        config = AppConfig(llm=LlmConfig(provider="codex"))
        llm = create_llm(config)
        assert isinstance(llm, RunnableBinding)

    def test_create_ollama_client(self) -> None:
        """Factory creates an Ollama client when configured."""
        config = AppConfig(
            llm=LlmConfig(
                provider="ollama",
                providers={
                    LlmProvider.OLLAMA: LlmProviderConfig(
                        model="qwen2.5-coder:14b",
                        options={"base_url": "http://localhost:11434"},
                    ),
                },
            )
        )
        llm = create_llm(config)
        assert isinstance(llm, RunnableBinding)

    def test_invalid_provider_raises(self) -> None:
        """Factory raises ValueError for unsupported provider."""
        config = AppConfig(llm=LlmConfig(provider="invalid"))
        with pytest.raises(ValueError, match="Unsupported"):
            create_llm(config)

    def test_task_model_override(self) -> None:
        """Factory uses task-specific model when set."""
        config = AppConfig(
            llm=LlmConfig(
                provider="ollama",
                providers={
                    LlmProvider.OLLAMA: LlmProviderConfig(
                        model="default-model",
                        options={"base_url": "http://localhost:11434"},
                        modes={LlmMode.PLAN: LlmModeConfig(model="plan-model")},
                    ),
                },
            )
        )
        llm = create_llm(config, task=LlmMode.PLAN)
        assert llm.bound.model == "plan-model"

    def test_mode_options_override(self) -> None:
        """Factory merges mode-level options over provider-level options."""
        config = AppConfig(
            llm=LlmConfig(
                provider="ollama",
                providers={
                    LlmProvider.OLLAMA: LlmProviderConfig(
                        model="test-model",
                        options={"base_url": "http://localhost:11434", "temperature": 0},
                        modes={LlmMode.PLAN: LlmModeConfig(options={"temperature": 0.5})},
                    ),
                },
            )
        )
        llm = create_llm(config, task=LlmMode.PLAN)
        assert llm.bound.temperature == 0.5
