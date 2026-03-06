"""Prompt templates and response schemas for LLM interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage


@dataclass
class PlanResponse:
    """Expected response from the planning LLM call."""

    plan: str = ""
    files_to_modify: list[str] = field(default_factory=list)
    approach: str = ""


@dataclass
class FileChange:
    """A single file change in an implementation."""

    path: str = ""
    content: str = ""
    action: str = "write"  # write, delete


@dataclass
class ImplementResponse:
    """Expected response from the implementation LLM call."""

    changes: list[FileChange] = field(default_factory=list)
    commit_message: str = ""


@dataclass
class FixResponse:
    """Expected response from the fix LLM call."""

    changes: list[FileChange] = field(default_factory=list)
    commit_message: str = ""
    explanation: str = ""


PLAN_SCHEMA: dict[str, Any] = {
    "title": "Plan",
    "type": "object",
    "properties": {
        "plan": {"type": "string", "description": "Detailed implementation plan"},
        "files_to_modify": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of file paths to modify",
        },
        "approach": {"type": "string", "description": "High-level approach summary"},
    },
    "required": ["plan", "files_to_modify", "approach"],
}

IMPLEMENT_SCHEMA: dict[str, Any] = {
    "title": "Implement",
    "type": "object",
    "properties": {
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "action": {"type": "string", "enum": ["write", "delete"]},
                },
                "required": ["path", "content", "action"],
            },
        },
        "commit_message": {"type": "string", "description": "Conventional commit message"},
    },
    "required": ["changes", "commit_message"],
}

FIX_SCHEMA: dict[str, Any] = {
    "title": "Fix",
    "type": "object",
    "properties": {
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "action": {"type": "string", "enum": ["write", "delete"]},
                },
                "required": ["path", "content", "action"],
            },
        },
        "commit_message": {"type": "string", "description": "Conventional commit message for the fix"},
        "explanation": {"type": "string", "description": "What was wrong and how this fixes it"},
    },
    "required": ["changes", "commit_message", "explanation"],
}

PLAN_SYSTEM = """You are an expert software engineer planning changes to a codebase.
Analyze the task, repository context, and any existing PR feedback.
Produce a detailed, step-by-step implementation plan.
Respond with JSON matching the provided schema."""

IMPLEMENT_SYSTEM = """You are an expert software engineer implementing code changes.
Follow the provided plan exactly. Write complete file contents for each change.
Use conventional commit format for the commit message.
Respond with JSON matching the provided schema."""

FIX_SYSTEM = """You are an expert software engineer fixing test/lint failures.
Analyze the test output, identify the root cause, and produce targeted fixes.
Use conventional commit format for the commit message.
Respond with JSON matching the provided schema."""


def build_plan_prompt(
    task: str,
    repo_context: str,
    commits: list[str],
    pr_info: dict | None,
    readme: str,
) -> list:
    """Build messages for the planning LLM call."""
    user_parts = [f"## Task\n{task}"]

    if readme:
        user_parts.append(f"## Project Instructions\n{readme}")

    if repo_context:
        user_parts.append(f"## Repository Structure\n{repo_context}")

    if commits:
        user_parts.append("## Recent Commits on Branch\n" + "\n".join(f"- {c}" for c in commits))

    if pr_info:
        user_parts.append(f"## Open PR\nTitle: {pr_info.get('title', '')}\nBody: {pr_info.get('body', '')}")
        reviews = pr_info.get("reviews", [])
        if reviews:
            review_text = "\n".join(
                f"- {r.get('author', {}).get('login', 'unknown')}: {r.get('body', '')}" for r in reviews
            )
            user_parts.append(f"## PR Reviews\n{review_text}")

    user_parts.append(f"## Response Schema\n```json\n{PLAN_SCHEMA}\n```")

    return [
        SystemMessage(content=PLAN_SYSTEM),
        HumanMessage(content="\n\n".join(user_parts)),
    ]


def build_implement_prompt(
    task: str,
    plan: str,
    repo_context: str,
) -> list:
    """Build messages for the implementation LLM call."""
    user_parts = [
        f"## Task\n{task}",
        f"## Implementation Plan\n{plan}",
    ]

    if repo_context:
        user_parts.append(f"## Repository Structure\n{repo_context}")

    user_parts.append(f"## Response Schema\n```json\n{IMPLEMENT_SCHEMA}\n```")

    return [
        SystemMessage(content=IMPLEMENT_SYSTEM),
        HumanMessage(content="\n\n".join(user_parts)),
    ]


def build_fix_prompt(
    task: str,
    plan: str,
    test_output: str,
    files_changed: list[str],
) -> list:
    """Build messages for the fix LLM call."""
    user_parts = [
        f"## Task\n{task}",
        f"## Implementation Plan\n{plan}",
        f"## Test/Lint Output (FAILING)\n```\n{test_output}\n```",
        "## Files Changed\n" + "\n".join(f"- {f}" for f in files_changed),
        f"## Response Schema\n```json\n{FIX_SCHEMA}\n```",
    ]

    return [
        SystemMessage(content=FIX_SYSTEM),
        HumanMessage(content="\n\n".join(user_parts)),
    ]
