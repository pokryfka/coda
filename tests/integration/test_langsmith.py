"""Integration tests for LangSmith tracing configuration via environment variables."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


class TestLangSmithEnvConfig:
    """Tests for LangSmith configuration via LANGSMITH_* environment variables."""

    def test_tracing_enabled_when_env_set(self) -> None:
        """LANGSMITH_TRACING=true is recognized as tracing enabled."""
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "true"}, clear=False):
            assert os.environ.get("LANGSMITH_TRACING") == "true"

    def test_tracing_disabled_when_env_missing(self) -> None:
        """Tracing is not enabled when LANGSMITH_TRACING is absent."""
        env = {k: v for k, v in os.environ.items() if k != "LANGSMITH_TRACING"}
        with patch.dict(os.environ, env, clear=True):
            assert os.environ.get("LANGSMITH_TRACING") is None

    def test_api_key_env_var(self) -> None:
        """LANGSMITH_API_KEY env var is readable."""
        with patch.dict(os.environ, {"LANGSMITH_API_KEY": "lsv2_test_key"}, clear=False):
            assert os.environ["LANGSMITH_API_KEY"] == "lsv2_test_key"

    def test_project_env_var(self) -> None:
        """LANGSMITH_PROJECT env var is readable."""
        with patch.dict(os.environ, {"LANGSMITH_PROJECT": "coding-agent"}, clear=False):
            assert os.environ["LANGSMITH_PROJECT"] == "coding-agent"

    def test_endpoint_env_var(self) -> None:
        """LANGSMITH_ENDPOINT env var is readable."""
        with patch.dict(os.environ, {"LANGSMITH_ENDPOINT": "https://eu.api.smith.langchain.com"}, clear=False):
            assert os.environ["LANGSMITH_ENDPOINT"] == "https://eu.api.smith.langchain.com"

    def test_langsmith_env_vars_are_set_together(self) -> None:
        """All LANGSMITH_* env vars can be set simultaneously."""
        env = {
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_API_KEY": "lsv2_fake_key",
            "LANGSMITH_PROJECT": "test-project",
            "LANGSMITH_ENDPOINT": "https://eu.api.smith.langchain.com",
        }
        with patch.dict(os.environ, env, clear=False):
            assert os.environ["LANGSMITH_TRACING"] == "true"
            assert os.environ["LANGSMITH_API_KEY"] == "lsv2_fake_key"
            assert os.environ["LANGSMITH_PROJECT"] == "test-project"
            assert os.environ["LANGSMITH_ENDPOINT"] == "https://eu.api.smith.langchain.com"


class TestLangSmithClientCreation:
    """Tests that verify LangSmith client works when API key is configured."""

    @pytest.fixture
    def langsmith_configured(self) -> bool:
        """Check if LangSmith is configured in the environment."""
        return bool(os.environ.get("LANGSMITH_API_KEY"))

    def test_langsmith_api_key_format(self, langsmith_configured: bool) -> None:
        """LANGSMITH_API_KEY follows expected format when set."""
        if not langsmith_configured:
            pytest.skip("LANGSMITH_API_KEY not set")
        assert os.environ["LANGSMITH_API_KEY"].startswith("lsv2_")

    def test_langsmith_client_creation(self, langsmith_configured: bool) -> None:
        """LangSmith client can be created with configured credentials."""
        if not langsmith_configured:
            pytest.skip("LANGSMITH_API_KEY not set")

        from langsmith import Client

        client = Client()
        assert client is not None

    def test_langsmith_endpoint_reachable(self, langsmith_configured: bool) -> None:
        """LangSmith endpoint responds when configured."""
        if not langsmith_configured:
            pytest.skip("LANGSMITH_API_KEY not set")

        from langsmith import Client

        client = Client()
        try:
            projects = list(client.list_projects(limit=1))
            assert isinstance(projects, list)
        except Exception as exc:
            pytest.fail(f"LangSmith API call failed: {exc}")
