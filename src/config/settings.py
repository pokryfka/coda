"""Dataclass-based configuration models and YAML/env loading."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml


@dataclass
class LlmProviderConfig:
    """Configuration for a specific LLM provider."""

    model: str = ""
    base_url: str = ""
    readme: str = ""
    plan_model: str = ""
    implement_model: str = ""
    fix_model: str = ""


class LlmProvider(StrEnum):
    """Supported LLM providers."""

    CLAUDE = "claude"
    GEMINI = "gemini"
    CODEX = "codex"
    OLLAMA = "ollama"


@dataclass
class LlmConfig:
    """Top-level LLM configuration."""

    provider: LlmProvider = LlmProvider.CLAUDE
    readme: str = "AGENTS.md"
    ollama: LlmProviderConfig = field(default_factory=lambda: LlmProviderConfig(model="qwen2.5-coder:14b"))
    claude: LlmProviderConfig = field(
        default_factory=lambda: LlmProviderConfig(model="claude-sonnet-4-6", readme="CLAUDE.md")
    )
    gemini: LlmProviderConfig = field(
        default_factory=lambda: LlmProviderConfig(model="gemini-3-flash-preview", readme="GEMINI.md")
    )
    codex: LlmProviderConfig = field(default_factory=lambda: LlmProviderConfig(model="gpt-4o"))


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
    return LlmProviderConfig(
        model=data.get("model", ""),
        base_url=data.get("base_url", ""),
        readme=data.get("readme", ""),
        plan_model=data.get("plan_model", ""),
        implement_model=data.get("implement_model", ""),
        fix_model=data.get("fix_model", ""),
    )


def _build_llm_config(data: dict) -> LlmConfig:
    """Build LLM config from a dictionary."""
    config = LlmConfig(
        provider=LlmProvider(data.get("provider", "claude")),
        readme=data.get("readme", "AGENTS.md"),
    )
    for provider_name in ("ollama", "claude", "gemini", "codex"):
        if provider_name in data:
            setattr(config, provider_name, _build_provider_config(data[provider_name]))
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
        config.llm.ollama.base_url = ollama_url

    return config


def find_repo(config: AppConfig, name: str) -> RepoConfig | None:
    """Find a repository configuration by name."""
    for repo in config.repositories:
        if repo.name == name:
            return repo
    return None
