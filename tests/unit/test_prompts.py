"""Unit tests for prompt construction and formatting."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.coding.prompts import (
    FIX_SCHEMA,
    IMPLEMENT_SCHEMA,
    PLAN_SCHEMA,
    build_fix_prompt,
    build_implement_prompt,
    build_plan_prompt,
)


class TestBuildPlanPrompt:
    """Tests for build_plan_prompt."""

    def test_returns_system_and_human_messages(self) -> None:
        """Plan prompt returns system + human message pair."""
        messages = build_plan_prompt("fix bug", "src/\n  main.py", [], None, "")
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)

    def test_includes_task_in_prompt(self) -> None:
        """Task description appears in the human message."""
        messages = build_plan_prompt("add login", "", [], None, "")
        assert "add login" in messages[1].content

    def test_includes_commits(self) -> None:
        """Commit history is included when provided."""
        messages = build_plan_prompt("task", "", ["abc123 initial", "def456 fix"], None, "")
        assert "abc123 initial" in messages[1].content

    def test_includes_pr_info(self) -> None:
        """PR info and reviews are included when provided."""
        pr = {"title": "My PR", "body": "Description", "reviews": [{"author": {"login": "alice"}, "body": "LGTM"}]}
        messages = build_plan_prompt("task", "", [], pr, "")
        assert "My PR" in messages[1].content
        assert "alice" in messages[1].content

    def test_includes_readme(self) -> None:
        """Project readme is included when provided."""
        messages = build_plan_prompt("task", "", [], None, "# Project\nUse pytest")
        assert "Use pytest" in messages[1].content


class TestBuildImplementPrompt:
    """Tests for build_implement_prompt."""

    def test_returns_messages(self) -> None:
        """Implement prompt returns system + human messages."""
        messages = build_implement_prompt("task", "step 1\nstep 2", "")
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)

    def test_includes_plan(self) -> None:
        """Plan text appears in the prompt."""
        messages = build_implement_prompt("task", "modify src/foo.py", "")
        assert "modify src/foo.py" in messages[1].content


class TestBuildFixPrompt:
    """Tests for build_fix_prompt."""

    def test_returns_messages(self) -> None:
        """Fix prompt returns system + human messages."""
        messages = build_fix_prompt("task", "plan", "FAILED test_foo", ["src/foo.py"])
        assert len(messages) == 2

    def test_includes_test_output(self) -> None:
        """Test output appears in the fix prompt."""
        messages = build_fix_prompt("task", "plan", "AssertionError: 1 != 2", ["src/foo.py"])
        assert "AssertionError" in messages[1].content

    def test_includes_files_changed(self) -> None:
        """Changed files are listed in the prompt."""
        messages = build_fix_prompt("task", "plan", "error", ["src/a.py", "src/b.py"])
        assert "src/a.py" in messages[1].content
        assert "src/b.py" in messages[1].content


class TestSchemas:
    """Tests for JSON schemas."""

    def test_plan_schema_has_required_fields(self) -> None:
        """Plan schema requires plan, files_to_modify, approach."""
        assert "plan" in PLAN_SCHEMA["properties"]
        assert set(PLAN_SCHEMA["required"]) == {"plan", "files_to_modify", "approach"}

    def test_implement_schema_has_required_fields(self) -> None:
        """Implement schema requires changes and commit_message."""
        assert set(IMPLEMENT_SCHEMA["required"]) == {"changes", "commit_message"}

    def test_fix_schema_has_required_fields(self) -> None:
        """Fix schema requires changes, commit_message, explanation."""
        assert set(FIX_SCHEMA["required"]) == {"changes", "commit_message", "explanation"}
