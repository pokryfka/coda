"""Unit tests for LangGraph graph wiring and routing behavior."""

from __future__ import annotations

from src.agent.coding.graph import _should_fix_or_finish, _should_push, _should_test_or_cleanup, build_graph
from src.agent.coding.state import AgentState
from src.config.settings import AgentConfig, AppConfig


class TestRouting:
    """Tests for conditional routing functions."""

    def test_should_finish_when_tests_pass(self) -> None:
        """Route to push when tests pass."""
        state: AgentState = {"test_result": {"passed": True, "output": ""}}  # type: ignore[typeddict-item]
        assert _should_fix_or_finish(state) == "push"

    def test_should_fix_when_tests_fail(self) -> None:
        """Route to fix when tests fail and under limit."""
        config = AppConfig(agent=AgentConfig(max_fix_attempts=3))
        state: AgentState = {  # type: ignore[typeddict-item]
            "test_result": {"passed": False, "output": "error"},
            "fix_attempts": 0,
            "config": config,
        }
        assert _should_fix_or_finish(state) == "fix"

    def test_should_end_failed_when_max_attempts(self) -> None:
        """Route to end_failed when fix attempts exhausted."""
        config = AppConfig(agent=AgentConfig(max_fix_attempts=3))
        state: AgentState = {  # type: ignore[typeddict-item]
            "test_result": {"passed": False, "output": "error"},
            "fix_attempts": 3,
            "config": config,
        }
        assert _should_fix_or_finish(state) == "end_failed"

    def test_should_push_creates_pr_when_configured(self) -> None:
        """Route to create_pr when auto_create_pr is True."""
        config = AppConfig(agent=AgentConfig(auto_create_pr=True))
        state: AgentState = {"config": config}  # type: ignore[typeddict-item]
        assert _should_push(state) == "create_pr"

    def test_should_push_ends_when_no_pr(self) -> None:
        """Route to end when auto_create_pr is False."""
        config = AppConfig(agent=AgentConfig(auto_create_pr=False))
        state: AgentState = {"config": config}  # type: ignore[typeddict-item]
        assert _should_push(state) == "end_done"

    def test_should_test_when_changes_made(self) -> None:
        """Route to test when implementation has changes."""
        state: AgentState = {  # type: ignore[typeddict-item]
            "implementation": [{"path": "foo.py", "content": "x", "action": "write"}],
        }
        assert _should_test_or_cleanup(state) == "test"

    def test_should_cleanup_when_no_changes(self) -> None:
        """Route to cleanup when implementation is empty."""
        state: AgentState = {"implementation": []}  # type: ignore[typeddict-item]
        assert _should_test_or_cleanup(state) == "cleanup"

    def test_should_cleanup_when_implementation_missing(self) -> None:
        """Route to cleanup when implementation key is absent."""
        state: AgentState = {}  # type: ignore[typeddict-item]
        assert _should_test_or_cleanup(state) == "cleanup"


class TestGraphConstruction:
    """Tests for graph building."""

    def test_build_graph_returns_compiled(self) -> None:
        """build_graph returns a compiled graph object."""
        graph = build_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self) -> None:
        """Graph contains all expected node names."""
        graph = build_graph()
        node_names = set(graph.nodes.keys())
        expected = {"clone", "branch", "check_pr", "plan", "implement", "test", "fix", "push", "create_pr", "cleanup"}
        # LangGraph adds __start__ and __end__ nodes
        assert expected.issubset(node_names)
