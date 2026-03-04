"""Integration tests for LangSmith tracing configuration via environment variables."""

from __future__ import annotations

import os

import pytest


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
