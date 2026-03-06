"""Unit tests for LLM factory _resolve_model logic."""

from __future__ import annotations

from src.config.settings import LlmMode, LlmModeConfig, LlmProviderConfig
from src.llm.factory import _resolve_model


class TestResolveModel:
    """Tests for _resolve_model fallback behavior."""

    def test_no_task_returns_provider_defaults(self) -> None:
        pc = LlmProviderConfig(model="default-model", options={"temperature": 0})
        model, options = _resolve_model(pc, task=None)
        assert model == "default-model"
        assert options == {"temperature": 0}

    def test_task_with_model_override(self) -> None:
        pc = LlmProviderConfig(
            model="default-model",
            options={"temperature": 0},
            modes={LlmMode.PLAN: LlmModeConfig(model="plan-model", options={"temperature": 0.5})},
        )
        model, options = _resolve_model(pc, task=LlmMode.PLAN)
        assert model == "plan-model"
        assert options == {"temperature": 0.5}

    def test_task_with_options_only_inherits_provider_model(self) -> None:
        """Mode with options but no model should inherit provider model."""
        pc = LlmProviderConfig(
            model="default-model",
            options={"temperature": 0},
            modes={LlmMode.PLAN: LlmModeConfig(options={"temperature": 0.7})},
        )
        model, options = _resolve_model(pc, task=LlmMode.PLAN)
        assert model == "default-model"
        assert options == {"temperature": 0.7}

    def test_task_without_mode_config_returns_provider(self) -> None:
        pc = LlmProviderConfig(model="default-model", options={"temperature": 0})
        model, options = _resolve_model(pc, task=LlmMode.FIX)
        assert model == "default-model"
        assert options == {"temperature": 0}

    def test_mode_with_model_and_no_options_uses_provider_options(self) -> None:
        """Mode with model but empty options falls back to provider options."""
        pc = LlmProviderConfig(
            model="default-model",
            options={"temperature": 0, "base_url": "http://localhost"},
            modes={LlmMode.PLAN: LlmModeConfig(model="plan-model")},
        )
        model, options = _resolve_model(pc, task=LlmMode.PLAN)
        assert model == "plan-model"
        assert options == {"temperature": 0, "base_url": "http://localhost"}
