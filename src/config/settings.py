"""Dataclass-based configuration models and YAML/env loading."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from typing import Any

import yaml


@dataclass
class LlmModeConfig:
    """Per-mode model and options overrides."""

    model: str = ""
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class LlmProviderConfig:
    """Configuration for a specific LLM provider."""

    model: str = ""
    readme: str = ""
    options: dict[str, Any] = field(default_factory=dict)
    modes: dict[LlmMode, LlmModeConfig] = field(default_factory=dict)


class LlmMode(StrEnum):
    """LLM task modes."""

    PLAN = "plan"
    IMPLEMENT = "implement"
    FIX = "fix"


class LlmProvider(StrEnum):
    """Supported LLM providers."""

    CLAUDE = "claude"
    GEMINI = "gemini"
    CODEX = "codex"
    OLLAMA = "ollama"


DEFAULT_PROVIDERS: dict[LlmProvider, LlmProviderConfig] = {
    LlmProvider.CLAUDE: LlmProviderConfig(model="claude-sonnet-4-6", readme="CLAUDE.md"),
    LlmProvider.GEMINI: LlmProviderConfig(model="gemini-3-flash-preview", readme="GEMINI.md"),
    LlmProvider.CODEX: LlmProviderConfig(model="gpt-4o"),
    LlmProvider.OLLAMA: LlmProviderConfig(model="qwen2.5-coder:14b"),
}


@dataclass
class LlmConfig:
    """Top-level LLM configuration."""

    provider: LlmProvider = LlmProvider.CLAUDE
    readme: str = "AGENTS.md"
    providers: dict[LlmProvider, LlmProviderConfig] = field(
        default_factory=lambda: dict(DEFAULT_PROVIDERS)
    )


@dataclass
class GitConfig:
    """Git and GitHub configuration."""

    default_branch: str = "main"
    branch_prefix: str = "agent/"
    commit_author: str = "Coding Agent <coding-agent@automated.dev>"


@dataclass
class RepoConfig:
    """Configuration for a single repository."""

    name: str = ""
    url: str = ""
    default_branch: str = "main"
    language: str = "python"
    test_command: str = "pytest"
    lint_command: str = ""
    setup_command: str = ""
    private: bool = False
    auth_method: str = "token"
    token_env: str = ""


@dataclass
class AgentConfig:
    """Agent behavior configuration."""

    max_fix_attempts: int = 5
    request_review_from: list[str] = field(default_factory=list)
    auto_push: bool = True
    auto_create_pr: bool = True
    workspace_dir: str = "/workspace"


@dataclass
class AppConfig:
    """Root application configuration."""

    llm: LlmConfig = field(default_factory=LlmConfig)
    git: GitConfig = field(default_factory=GitConfig)
    repositories: list[RepoConfig] = field(default_factory=list)
    agent: AgentConfig = field(default_factory=AgentConfig)


def _build_provider_config(data: dict) -> LlmProviderConfig:
    """Build a provider config from a dictionary."""
    modes: dict[LlmMode, LlmModeConfig] = {}
    for mode in LlmMode:
        if mode in data and isinstance(data[mode], dict):
            mode_data = data[mode]
            opts = dict(mode_data.get("options", {}))
            opts.update({k: v for k, v in mode_data.items() if k not in ("model", "options")})
            modes[mode] = LlmModeConfig(
                model=mode_data.get("model", ""),
                options=opts,
            )
    return LlmProviderConfig(
        model=data.get("model", ""),
        readme=data.get("readme", ""),
        options=data.get("options", {}),
        modes=modes,
    )


def _build_llm_config(data: dict) -> LlmConfig:
    """Build LLM config from a dictionary."""
    config = LlmConfig(
        provider=LlmProvider(data.get("provider", "claude")),
        readme=data.get("readme", "AGENTS.md"),
    )
    for provider in LlmProvider:
        if provider in data:
            config.providers[provider] = _build_provider_config(data[provider])
    return config


def _build_repo_config(data: dict) -> RepoConfig:
    """Build a repo config from a dictionary."""
    return RepoConfig(
        name=data.get("name", ""),
        url=data.get("url", ""),
        default_branch=data.get("default_branch", "main"),
        language=data.get("language", "python"),
        test_command=data.get("test_command", "pytest"),
        lint_command=data.get("lint_command", ""),
        setup_command=data.get("setup_command", ""),
        private=data.get("private", False),
        auth_method=data.get("auth_method", "token"),
        token_env=data.get("token_env", ""),
    )


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load configuration from YAML file, merging environment variable overrides."""
    path = Path(path)
    if not path.exists():
        return _apply_env_overrides(AppConfig())

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    config = AppConfig()

    if "llm" in raw:
        config.llm = _build_llm_config(raw["llm"])

    if "git" in raw:
        g = raw["git"]
        config.git = GitConfig(
            default_branch=g.get("default_branch", "main"),
            branch_prefix=g.get("branch_prefix", "agent/"),
            commit_author=g.get("commit_author", "Coding Agent <coding-agent@automated.dev>"),
        )

    if "repositories" in raw:
        config.repositories = [_build_repo_config(r) for r in raw["repositories"]]

    if "agent" in raw:
        a = raw["agent"]
        config.agent = AgentConfig(
            max_fix_attempts=a.get("max_fix_attempts", 5),
            request_review_from=a.get("request_review_from", []),
            auto_push=a.get("auto_push", True),
            auto_create_pr=a.get("auto_create_pr", True),
            workspace_dir=a.get("workspace_dir", "/workspace"),
        )

    return _apply_env_overrides(config)


def _apply_env_overrides(config: AppConfig) -> AppConfig:
    """Apply environment variable overrides to config."""
    provider = os.environ.get("LLM_PROVIDER")
    if provider:
        config.llm.provider = LlmProvider(provider)

    ollama_url = os.environ.get("OLLAMA_BASE_URL")
    if ollama_url:
        config.llm.providers[LlmProvider.OLLAMA].options["base_url"] = ollama_url

    return config


def find_repo(config: AppConfig, name: str) -> RepoConfig | None:
    """Find a repository configuration by name."""
    for repo in config.repositories:
        if repo.name == name:
            return repo
    return None
