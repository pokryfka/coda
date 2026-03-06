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

    def test_provider_mode_overrides(self, tmp_path: Path) -> None:
        """Per-mode model and options overrides are loaded from config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(textwrap.dedent("""\
            llm:
              provider: claude
              claude:
                model: claude-sonnet-4-6
                options:
                  temperature: 0
                plan:
                  model: claude-opus-4-6
                  temperature: 0.2
                implement:
                  model: claude-sonnet-4-6
        """))
        config = load_config(config_file)
        claude = config.llm.providers[LlmProvider.CLAUDE]
        assert claude.options == {"temperature": 0}
        assert claude.modes[LlmMode.PLAN].model == "claude-opus-4-6"
        assert claude.modes[LlmMode.PLAN].options == {"temperature": 0.2}
        assert claude.modes[LlmMode.IMPLEMENT].model == "claude-sonnet-4-6"
        assert LlmMode.FIX not in claude.modes

    def test_provider_options_loaded(self, tmp_path: Path) -> None:
        """Provider-level options are loaded from config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(textwrap.dedent("""\
            llm:
              provider: ollama
              ollama:
                model: qwen2.5-coder:14b
                options:
                  base_url: "http://ollama:11434"
                  temperature: 0.5
        """))
        config = load_config(config_file)
        ollama = config.llm.providers[LlmProvider.OLLAMA]
        assert ollama.options["base_url"] == "http://ollama:11434"
        assert ollama.options["temperature"] == 0.5


    def test_provider_model_only(self, tmp_path: Path) -> None:
        """Provider config with only model set has empty options and no modes."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(textwrap.dedent("""\
            llm:
              provider: claude
              claude:
                model: "claude-sonnet-4-6"
        """))
        config = load_config(config_file)
        claude = config.llm.providers[LlmProvider.CLAUDE]
        assert claude.model == "claude-sonnet-4-6"
        assert claude.options == {}
        assert claude.modes == {}

    def test_mode_options_without_model_override(self, tmp_path: Path) -> None:
        """Mode with options but no model should inherit provider model."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(textwrap.dedent("""\
            llm:
              provider: claude
              claude:
                model: claude-sonnet-4-6
                options:
                  temperature: 0
                plan:
                  temperature: 0.5
        """))
        config = load_config(config_file)
        claude = config.llm.providers[LlmProvider.CLAUDE]
        # Mode should have options but no model override
        assert claude.modes[LlmMode.PLAN].model == ""
        assert claude.modes[LlmMode.PLAN].options == {"temperature": 0.5}

    def test_mode_with_model_only_inherits_provider_options(self, tmp_path: Path) -> None:
        """Mode with model but no options should inherit provider-level options."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(textwrap.dedent("""\
            llm:
              provider: claude
              claude:
                model: claude-sonnet-4-6
                options:
                  temperature: 1.0
                plan:
                  model: claude-opus-4-6
                implement:
                  options:
                    temperature: 0.9
                fix:
                  model: claude-sonnet-4-6
        """))
        config = load_config(config_file)
        claude = config.llm.providers[LlmProvider.CLAUDE]
        # fix has model override but no options -> should inherit provider options
        assert claude.modes[LlmMode.FIX].model == "claude-sonnet-4-6"
        assert claude.modes[LlmMode.FIX].options == {}
        # plan has model override but no options -> same
        assert claude.modes[LlmMode.PLAN].model == "claude-opus-4-6"
        assert claude.modes[LlmMode.PLAN].options == {}
        # implement has options but no model
        assert claude.modes[LlmMode.IMPLEMENT].model == ""
        assert claude.modes[LlmMode.IMPLEMENT].options == {"temperature": 0.9}

    def test_mode_options_not_merged_with_provider(self, tmp_path: Path) -> None:
        """Mode-specific options should not be merged with provider-level options."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(textwrap.dedent("""\
            llm:
              provider: claude
              claude:
                model: claude-sonnet-4-6
                options:
                  temperature: 0
                  max_tokens: 4096
                plan:
                  model: claude-opus-4-6
                  temperature: 0.2
        """))
        config = load_config(config_file)
        claude = config.llm.providers[LlmProvider.CLAUDE]
        # Provider options should have both keys
        assert claude.options == {"temperature": 0, "max_tokens": 4096}
        # Mode options should only have its own keys, not inherited from provider
        assert claude.modes[LlmMode.PLAN].options == {"temperature": 0.2}
        assert "max_tokens" not in claude.modes[LlmMode.PLAN].options


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
