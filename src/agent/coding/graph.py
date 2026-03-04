"""LangGraph graph construction and routing functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph

from src.agent.coding.nodes import (
    check_pr,
    clone_repo,
    create_pr,
    fix_code,
    implement,
    plan,
    push_changes,
    run_tests,
    setup_branch,
)
from src.agent.coding.state import AgentState

if TYPE_CHECKING:
    from src.config.settings import AppConfig


def _should_fix_or_finish(state: AgentState) -> str:
    """Route after tests: fix if failed and under limit, else finish."""
    test_result = state.get("test_result", {})
    if test_result.get("passed", False):
        return "push"

    config = state.get("config")
    max_attempts = 3
    if config:
        max_attempts = config.agent.max_fix_attempts

    if state.get("fix_attempts", 0) >= max_attempts:
        return "end_failed"

    return "fix"


def _should_push(state: AgentState) -> str:
    """Route after push: create PR if configured, else done."""
    config = state.get("config")
    if config and config.agent.auto_create_pr:
        return "create_pr"
    return "end_done"


def build_graph(config: AppConfig | None = None) -> StateGraph:
    """Build the coding agent state graph.

    Args:
        config: Optional app config for customizing routing behavior.

    Returns:
        A compiled LangGraph StateGraph.
    """
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("clone", clone_repo)
    builder.add_node("branch", setup_branch)
    builder.add_node("check_pr", check_pr)
    builder.add_node("plan", plan)
    builder.add_node("implement", implement)
    builder.add_node("test", run_tests)
    builder.add_node("fix", fix_code)
    builder.add_node("push", push_changes)
    builder.add_node("create_pr", create_pr)

    # Wire edges
    builder.set_entry_point("clone")
    builder.add_edge("clone", "branch")
    builder.add_edge("branch", "check_pr")
    builder.add_edge("check_pr", "plan")
    builder.add_edge("plan", "implement")
    builder.add_edge("implement", "test")

    # Conditional: after test
    builder.add_conditional_edges(
        "test",
        _should_fix_or_finish,
        {"push": "push", "fix": "fix", "end_failed": END},
    )

    # Fix loops back to test
    builder.add_edge("fix", "test")

    # After push: optionally create PR
    builder.add_conditional_edges(
        "push",
        _should_push,
        {"create_pr": "create_pr", "end_done": END},
    )

    builder.add_edge("create_pr", END)

    return builder.compile()


# Exported for LangGraph Studio
graph = build_graph()
