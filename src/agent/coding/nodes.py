"""LangGraph node implementations for the coding agent."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from src.agent.coding.prompts import (
    FIX_SCHEMA,
    IMPLEMENT_SCHEMA,
    PLAN_SCHEMA,
    build_fix_prompt,
    build_implement_prompt,
    build_plan_prompt,
)
from src.agent.coding.state import AgentState, Status
from src.git_ops.repo import GitRepo
from src.llm.factory import create_llm

logger = logging.getLogger(__name__)


def _get_repo_context(repo_path: str, max_depth: int = 3) -> str:
    """Build a tree-like listing of the repository structure."""
    lines = []
    root = Path(repo_path)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", "node_modules", ".venv")]
        depth = len(Path(dirpath).relative_to(root).parts)
        if depth > max_depth:
            dirnames.clear()
            continue
        indent = "  " * depth
        lines.append(f"{indent}{Path(dirpath).name}/")
        for f in sorted(filenames):
            lines.append(f"{indent}  {f}")
    return "\n".join(lines)


def _read_readme(repo_path: str, readme_name: str) -> str:
    """Read the readme/agent instructions file if it exists."""
    readme_path = Path(repo_path) / readme_name
    if readme_path.exists():
        return readme_path.read_text(errors="replace")[:5000]
    return ""


async def clone_repo(state: AgentState) -> dict:
    """Clone the repository and run setup command."""
    config = state["config"]
    repo_config = state["repo_config"]

    workspace = Path(config.agent.workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    dest = workspace / repo_config.name

    git = GitRepo(path=workspace)

    # Set up auth token
    token = os.environ.get("GH_TOKEN", "")
    if repo_config.private and repo_config.token_env:
        token = os.environ.get(repo_config.token_env, token)
    if token:
        git.setup_auth(token)

    # Clone
    if dest.exists():
        logger.info("Repository already exists at %s, pulling latest", dest)
        git.path = dest
        await git._run("git", "fetch", "--all")
    else:
        await git.clone(repo_config.url, dest)

    # Configure git user
    git.path = dest
    await git.configure_user(config.git.commit_author)

    # Run setup command
    if repo_config.setup_command:
        logger.info("Running setup: %s", repo_config.setup_command)
        proc = await __import__("asyncio").create_subprocess_shell(
            repo_config.setup_command,
            cwd=str(dest),
            stdout=__import__("asyncio").subprocess.PIPE,
            stderr=__import__("asyncio").subprocess.PIPE,
        )
        await proc.communicate()

    return {"repo_path": str(dest)}


async def setup_branch(state: AgentState) -> dict:
    """Create or check out the working branch and gather commits."""
    config = state["config"]
    repo_path = state["repo_path"]
    branch = state.get("branch", "")

    git = GitRepo(path=Path(repo_path))
    token = os.environ.get("GH_TOKEN", "")
    if token:
        git.setup_auth(token)

    if branch:
        # Try to check out existing branch, create if not found
        try:
            await git.checkout_branch(branch)
        except RuntimeError:
            await git.checkout_branch(branch, create=True)
        commits = await git.get_commits(branch)
    else:
        # Generate branch name from task
        safe_task = state["task"][:40].lower().replace(" ", "-")
        safe_task = "".join(c for c in safe_task if c.isalnum() or c == "-")
        branch = f"{config.git.branch_prefix}{safe_task}"
        await git.checkout_branch(branch, create=True)
        commits = []

    return {"branch": branch, "commits": commits}


async def check_pr(state: AgentState) -> dict:
    """Check for an existing open PR and fetch reviews."""
    repo_path = state["repo_path"]
    branch = state["branch"]

    git = GitRepo(path=Path(repo_path))
    token = os.environ.get("GH_TOKEN", "")
    if token:
        git.setup_auth(token)

    pr = await git.get_open_pr(branch)
    if pr:
        reviews = await git.get_pr_reviews(pr["number"])
        pr["reviews"] = reviews
        return {"pr_info": pr}

    return {"pr_info": None}


async def plan(state: AgentState) -> dict:
    """Call LLM to create an implementation plan."""
    config = state["config"]
    repo_path = state["repo_path"]

    llm = create_llm(config, task="plan")
    provider_config = getattr(config.llm, config.llm.provider)
    readme_name = provider_config.readme or config.llm.readme
    readme = _read_readme(repo_path, readme_name)
    repo_context = _get_repo_context(repo_path)

    messages = build_plan_prompt(
        task=state["task"],
        repo_context=repo_context,
        commits=state.get("commits", []),
        pr_info=state.get("pr_info"),
        readme=readme,
    )

    llm_with_schema = llm.with_structured_output(PLAN_SCHEMA)
    result = await llm_with_schema.ainvoke(messages)

    plan_text = result.get("plan", "") if isinstance(result, dict) else str(result)
    return {"plan": plan_text, "status": Status.PLANNING}


async def implement(state: AgentState) -> dict:
    """Call LLM to implement the plan and write files."""
    config = state["config"]
    repo_path = state["repo_path"]

    llm = create_llm(config, task="implement")
    repo_context = _get_repo_context(repo_path)

    messages = build_implement_prompt(
        task=state["task"],
        plan=state["plan"],
        repo_context=repo_context,
    )

    llm_with_schema = llm.with_structured_output(IMPLEMENT_SCHEMA)
    result = await llm_with_schema.ainvoke(messages)

    changes = result.get("changes", []) if isinstance(result, dict) else []
    default_msg = "feat: implement changes"
    commit_msg = result.get("commit_message", default_msg) if isinstance(result, dict) else default_msg

    # Apply changes
    files_changed = []
    for change in changes:
        file_path = Path(repo_path) / change["path"]
        if change.get("action") == "delete":
            if file_path.exists():
                file_path.unlink()
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(change["content"])
        files_changed.append(change["path"])

    # Commit
    git = GitRepo(path=Path(repo_path))
    if await git.has_changes():
        await git.add_all()
        await git.commit(commit_msg, author=config.git.commit_author)

    return {"implementation": changes, "status": Status.IMPLEMENTING}


async def run_tests(state: AgentState) -> dict:
    """Run test and lint commands, capture output."""
    import asyncio as aio

    repo_config = state["repo_config"]
    repo_path = state["repo_path"]
    outputs = []
    passed = True

    for cmd_name, cmd in [("test", repo_config.test_command), ("lint", repo_config.lint_command)]:
        if not cmd:
            continue
        logger.info("Running %s: %s", cmd_name, cmd)
        proc = await aio.create_subprocess_shell(
            cmd,
            cwd=repo_path,
            stdout=aio.subprocess.PIPE,
            stderr=aio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        output = stdout.decode() + "\n" + stderr.decode()
        outputs.append(f"=== {cmd_name} ===\n{output}")
        if proc.returncode != 0:
            passed = False

    return {
        "test_result": {"passed": passed, "output": "\n".join(outputs)},
        "status": Status.TESTING,
    }


async def fix_code(state: AgentState) -> dict:
    """Call LLM to fix test/lint failures."""
    config = state["config"]
    repo_path = state["repo_path"]

    llm = create_llm(config, task="fix")
    test_output = state.get("test_result", {}).get("output", "")
    files_changed = [c.get("path", "") for c in state.get("implementation", [])]

    messages = build_fix_prompt(
        task=state["task"],
        plan=state["plan"],
        test_output=test_output,
        files_changed=files_changed,
    )

    llm_with_schema = llm.with_structured_output(FIX_SCHEMA)
    result = await llm_with_schema.ainvoke(messages)

    changes = result.get("changes", []) if isinstance(result, dict) else []
    default_msg = "fix: address test failures"
    commit_msg = result.get("commit_message", default_msg) if isinstance(result, dict) else default_msg

    # Apply fixes
    for change in changes:
        file_path = Path(repo_path) / change["path"]
        if change.get("action") == "delete":
            if file_path.exists():
                file_path.unlink()
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(change["content"])

    # Commit
    git = GitRepo(path=Path(repo_path))
    if await git.has_changes():
        await git.add_all()
        await git.commit(commit_msg, author=config.git.commit_author)

    fix_attempts = state.get("fix_attempts", 0) + 1
    return {"fix_attempts": fix_attempts, "status": Status.FIXING}


async def push_changes(state: AgentState) -> dict:
    """Push the branch to remote."""
    repo_path = state["repo_path"]
    branch = state["branch"]

    git = GitRepo(path=Path(repo_path))
    token = os.environ.get("GH_TOKEN", "")
    if token:
        git.setup_auth(token)

    await git.push(branch)
    return {}


async def create_pr(state: AgentState) -> dict:
    """Create a pull request via gh CLI."""
    config = state["config"]
    repo_path = state["repo_path"]
    repo_config = state["repo_config"]

    git = GitRepo(path=Path(repo_path))
    token = os.environ.get("GH_TOKEN", "")
    if token:
        git.setup_auth(token)

    title = f"feat: {state['task'][:60]}"
    body = f"## Task\n{state['task']}\n\n## Plan\n{state.get('plan', 'N/A')}"

    pr = await git.create_pr(title, body, base=repo_config.default_branch)

    # Request reviewers
    reviewers = config.agent.request_review_from
    if reviewers and pr.get("number"):
        await git.request_reviewers(pr["number"], reviewers)

    return {"pr_info": pr, "status": Status.DONE}
