"""Unit tests for config loading and default behavior."""

from __future__ import annotations

import textwrap
from pathlib import Path

from src.config.settings import AppConfig, LlmMode, LlmProvider, find_repo, load_config


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_defaults_when_no_file(self, tmp_path: Path) -> None:
        """Loading a non-existent config file returns defaults."""
        config = load_config(tmp_path / "nonexistent.yaml")
        assert config.llm.provider == "claude"
        assert config.git.default_branch == "main"
        assert config.agent.max_fix_attempts == 5

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        """Loading a valid YAML file populates config correctly."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(textwrap.dedent("""\
            llm:
              provider: gemini
              gemini:
                model: gemini-pro
            git:
              default_branch: develop
              branch_prefix: "bot/"
            repositories:
              - name: test-repo
                url: https://github.com/org/test-repo.git
                language: python
                test_command: pytest
            agent:
              max_fix_attempts: 3
              workspace_dir: /tmp/work
        """))
        config = load_config(config_file)
        assert config.llm.provider == "gemini"
        assert config.llm.providers[LlmProvider.GEMINI].model == "gemini-pro"
        assert config.git.default_branch == "develop"
        assert config.git.branch_prefix == "bot/"
        assert len(config.repositories) == 1
        assert config.repositories[0].name == "test-repo"
        assert config.agent.max_fix_attempts == 3

    def test_env_var_override_provider(self, tmp_path: Path, monkeypatch: object) -> None:
        """LLM_PROVIDER env var overrides YAML config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("llm:\n  provider: claude\n")
        monkeypatch.setenv("LLM_PROVIDER", "ollama")  # type: ignore[attr-defined]
        config = load_config(config_file)
        assert config.llm.provider == "ollama"

    def test_env_var_override_ollama_url(self, tmp_path: Path, monkeypatch: object) -> None:
        """OLLAMA_BASE_URL env var overrides config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("llm:\n  provider: ollama\n")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")  # type: ignore[attr-defined]
        config = load_config(config_file)
        assert config.llm.providers[LlmProvider.OLLAMA].base_url == "http://localhost:11434"

    def test_provider_model_overrides(self, tmp_path: Path) -> None:
        """Per-task model overrides are loaded from config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(textwrap.dedent("""\
            llm:
              provider: claude
              claude:
                model: claude-sonnet-4-6
                plan_model: claude-opus-4-6
                implement_model: claude-sonnet-4-6
        """))
        config = load_config(config_file)
        claude = config.llm.providers[LlmProvider.CLAUDE]
        assert claude.model_overrides[LlmMode.PLAN] == "claude-opus-4-6"
        assert claude.model_overrides[LlmMode.IMPLEMENT] == "claude-sonnet-4-6"
        assert LlmMode.FIX not in claude.model_overrides


class TestFindRepo:
    """Tests for find_repo function."""

    def test_find_existing_repo(self) -> None:
        """Finding an existing repo returns its config."""
        config = AppConfig()
        from src.config.settings import RepoConfig

        config.repositories = [RepoConfig(name="my-repo", url="https://example.com")]
        result = find_repo(config, "my-repo")
        assert result is not None
        assert result.name == "my-repo"

    def test_find_missing_repo(self) -> None:
        """Finding a missing repo returns None."""
        config = AppConfig()
        result = find_repo(config, "nonexistent")
        assert result is None
