"""High-level orchestration wrapper around graph execution."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from src.agent.coding.graph import build_graph
from src.agent.coding.state import AgentState, Status
from src.config.settings import AppConfig, RepoConfig, find_repo

logger = logging.getLogger(__name__)


def _local_repo_config(config: AppConfig) -> RepoConfig:
    """Build a RepoConfig for the current working directory."""
    cwd = Path.cwd()
    name = cwd.name or "local"

    # Detect default branch from git
    default_branch = config.git.default_branch
    try:
        result = subprocess.run(
            ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
            capture_output=True, text=True, cwd=cwd,
        )
        if result.returncode == 0:
            # e.g. "refs/remotes/origin/main" -> "main"
            default_branch = result.stdout.strip().rsplit("/", 1)[-1]
    except OSError:
        pass

    return RepoConfig(
        name=name,
        url=".",
        default_branch=default_branch,
    )


async def run_agent(
    repo_name: str,
    task: str,
    config: AppConfig,
    branch: str = "",
) -> AgentState:
    """Run the coding agent for a given repository and task.

    Args:
        repo_name: Name of the repository from config.
        task: Description of the task to perform.
        config: Application configuration.
        branch: Optional branch name to work on.

    Returns:
        Final agent state after execution.

    Raises:
        ValueError: If the repository is not found in config.
    """
    if repo_name == ".":
        repo_config = _local_repo_config(config)
        config.agent.auto_push = False
        config.agent.auto_create_pr = False
    else:
        repo_config = find_repo(config, repo_name)
        if repo_config is None:
            msg = f"Repository '{repo_name}' not found in configuration"
            raise ValueError(msg)

    graph = build_graph(config)

    initial_state: AgentState = {
        "repo_config": repo_config,
        "task": task,
        "config": config,
        "branch": branch,
        "repo_path": "",
        "commits": [],
        "pr_info": None,
        "plan": "",
        "implementation": [],
        "test_result": {},
        "fix_attempts": 0,
        "error": None,
        "status": Status.PLANNING,
    }

    logger.info("Starting coding agent for repo=%s task=%s", repo_name, task[:80])

    try:
        result = await graph.ainvoke(initial_state)
    except KeyboardInterrupt:
        logger.warning("Agent interrupted by user")
        return {**initial_state, "status": Status.FAILED, "error": "Interrupted"}
    except Exception as exc:
        logger.exception("Agent failed with error")
        return {**initial_state, "status": Status.FAILED, "error": str(exc)}

    logger.info("Agent finished with status=%s", result.get("status"))
    return result
