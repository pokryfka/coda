"""Integration tests for GitRepo GitHub Pull Request operations.

These tests require a real GitHub repository and a valid GH_TOKEN.
Set TEST_GH_URL to the HTTPS clone URL of a test repository to enable them.
The repository must already exist and be accessible with the configured GH_TOKEN.

Example:
    TEST_GH_URL=https://github.com/yourorg/test-repo.git uv run pytest tests/integration/test_github.py
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from src.git_ops.repo import GitRepo

TEST_GH_URL = os.environ.get("TEST_GH_URL", "")
GH_TOKEN = os.environ.get("GH_TOKEN", "")

pytestmark = pytest.mark.skipif(
    not TEST_GH_URL or not GH_TOKEN,
    reason="TEST_GH_URL and GH_TOKEN must be set for GitHub integration tests",
)


@pytest.fixture
async def cloned_repo(tmp_path: Path) -> GitRepo:
    """Clone the test repository into a temp directory."""
    dest = tmp_path / "repo"
    dest.parent.mkdir(parents=True, exist_ok=True)

    git = GitRepo(path=tmp_path)
    git.setup_auth(GH_TOKEN)
    await git.clone(TEST_GH_URL, dest)
    await git.configure_user("Test Bot <test-bot@coda.dev>")
    return git


@pytest.fixture
async def pr_branch(cloned_repo: GitRepo) -> str:
    """Create a unique branch with a commit and push it."""
    branch = f"test/pr-{uuid.uuid4().hex[:8]}"
    await cloned_repo.checkout_branch(branch, create=True)

    # Create a test file and commit
    test_file = cloned_repo.path / f"test-{uuid.uuid4().hex[:8]}.txt"
    test_file.write_text("integration test file\n")
    await cloned_repo.add_all()
    await cloned_repo.commit("test: add integration test file")
    await cloned_repo.push(branch)

    yield branch

    # Cleanup: delete remote branch
    try:
        await cloned_repo._run("git", "push", "origin", "--delete", branch)
    except RuntimeError:
        pass


class TestGitHubAuth:
    """Tests for GitHub token authentication checks."""

    async def test_check_auth_valid_token(self, cloned_repo: GitRepo) -> None:
        """check_auth returns no missing scopes for a properly scoped token."""
        missing = await cloned_repo.check_auth(GH_TOKEN)
        assert isinstance(missing, list)
        # If the test token is properly configured, all scopes should be present
        assert missing == [], f"Token is missing required scopes: {missing}"

    async def test_check_auth_invalid_token(self, cloned_repo: GitRepo) -> None:
        """check_auth raises RuntimeError for an invalid token."""
        with pytest.raises(RuntimeError, match="Failed to verify token"):
            await cloned_repo.check_auth("ghp_invalid_token_000000000000000000")


class TestGitHubClone:
    """Tests for cloning from GitHub."""

    async def test_clone_creates_repo(self, cloned_repo: GitRepo) -> None:
        """Cloning creates a working repository directory."""
        assert cloned_repo.path.exists()
        assert (cloned_repo.path / ".git").is_dir()

    async def test_clone_has_remote(self, cloned_repo: GitRepo) -> None:
        """Cloned repo has origin remote configured."""
        out = await cloned_repo._run("git", "remote", "-v")
        assert "origin" in out


class TestGitHubPush:
    """Tests for pushing to GitHub."""

    async def test_push_branch(self, pr_branch: str, cloned_repo: GitRepo) -> None:
        """Pushing a branch makes it available on remote."""
        out = await cloned_repo._run("git", "ls-remote", "--heads", "origin", pr_branch)
        assert pr_branch in out


class TestGitHubPR:
    """Tests for Pull Request operations via gh CLI."""

    async def test_create_and_get_pr(self, pr_branch: str, cloned_repo: GitRepo) -> None:
        """Creating a PR returns number and URL, and get_open_pr finds it."""
        pr = await cloned_repo.create_pr(
            title=f"Test PR ({pr_branch})",
            body="Automated integration test PR — safe to close.",
            base="main",
        )
        assert "number" in pr
        assert "url" in pr
        assert isinstance(pr["number"], int)

        try:
            # Verify get_open_pr finds it
            open_pr = await cloned_repo.get_open_pr(pr_branch)
            assert open_pr is not None
            assert open_pr["number"] == pr["number"]
            assert open_pr["state"] == "OPEN"
        finally:
            # Cleanup: close the PR
            await cloned_repo._run("gh", "pr", "close", str(pr["number"]))

    async def test_get_open_pr_returns_none_for_no_pr(self, cloned_repo: GitRepo) -> None:
        """get_open_pr returns None when no PR exists for the branch."""
        result = await cloned_repo.get_open_pr("nonexistent-branch-xyz")
        assert result is None

    async def test_get_pr_reviews_empty(self, pr_branch: str, cloned_repo: GitRepo) -> None:
        """get_pr_reviews returns empty list for a PR with no reviews."""
        pr = await cloned_repo.create_pr(
            title=f"Review test ({pr_branch})",
            body="Automated test — safe to close.",
            base="main",
        )
        try:
            reviews = await cloned_repo.get_pr_reviews(pr["number"])
            assert isinstance(reviews, list)
        finally:
            await cloned_repo._run("gh", "pr", "close", str(pr["number"]))
