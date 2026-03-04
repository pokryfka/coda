"""LangGraph state definition for the coding agent."""

from __future__ import annotations

from enum import Enum
from typing import TypedDict

from src.config.settings import AppConfig, RepoConfig


class Status(str, Enum):
    """Agent execution status."""

    PLANNING = "planning"
    IMPLEMENTING = "implementing"
    TESTING = "testing"
    FIXING = "fixing"
    DONE = "done"
    FAILED = "failed"


class AgentState(TypedDict, total=False):
    """State passed through the LangGraph coding agent."""

    repo_path: str
    repo_config: RepoConfig
    branch: str
    task: str
    config: AppConfig
    commits: list[str]
    pr_info: dict | None
    plan: str
    implementation: list[dict]
    test_result: dict
    fix_attempts: int
    error: str | None
    status: Status
