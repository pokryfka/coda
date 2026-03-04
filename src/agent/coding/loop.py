"""High-level orchestration wrapper around graph execution."""

from __future__ import annotations

import logging

from src.agent.coding.graph import build_graph
from src.agent.coding.state import AgentState, Status
from src.config.settings import AppConfig, find_repo

logger = logging.getLogger(__name__)


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
