"""Integration tests for GitRepo git operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.git_ops.repo import GitRepo


@pytest.fixture
async def git_repo(tmp_path: Path) -> GitRepo:
    """Create a temporary git repository."""
    repo_path = tmp_path / "test-repo"
    repo_path.mkdir()

    git = GitRepo(path=repo_path)
    await git._run("git", "init")
    await git._run("git", "config", "user.name", "Test")
    await git._run("git", "config", "user.email", "test@test.com")
    (repo_path / "README.md").write_text("# Test Repo")
    await git.add_all()
    await git.commit("Initial commit")
    return git


class TestGitIntegration:
    """Integration tests using real git operations."""

    async def test_create_branch(self, git_repo: GitRepo) -> None:
        """Creating and checking out a branch works."""
        await git_repo.checkout_branch("feature/test", create=True)
        branch = await git_repo.get_current_branch()
        assert branch == "feature/test"

    async def test_commit_changes(self, git_repo: GitRepo) -> None:
        """Committing changes records them in history."""
        await git_repo.checkout_branch("feature/commit-test", create=True)
        (git_repo.path / "new_file.txt").write_text("content")
        await git_repo.add_all()
        await git_repo.commit("feat: add new file")

        # Verify commit exists
        log = await git_repo._run("git", "log", "--oneline", "-1")
        assert "feat: add new file" in log

    async def test_has_changes_detects_modifications(self, git_repo: GitRepo) -> None:
        """has_changes detects uncommitted files."""
        assert not await git_repo.has_changes()
        (git_repo.path / "change.txt").write_text("modified")
        assert await git_repo.has_changes()

    async def test_configure_user(self, git_repo: GitRepo) -> None:
        """configure_user sets git user name and email."""
        await git_repo.configure_user("Bot <bot@example.com>")
        name = await git_repo._run("git", "config", "user.name")
        email = await git_repo._run("git", "config", "user.email")
        assert name == "Bot"
        assert email == "bot@example.com"

    async def test_get_commits_on_branch(self, git_repo: GitRepo) -> None:
        """get_commits returns commits unique to a branch."""
        await git_repo.checkout_branch("feature/commits", create=True)
        (git_repo.path / "file1.txt").write_text("a")
        await git_repo.add_all()
        await git_repo.commit("feat: first change")

        commits = await git_repo.get_commits("feature/commits")
        assert len(commits) >= 1
        assert any("first change" in c for c in commits)
