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
    await git._run("git", "init", "-b", "main")
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

    async def test_delete_branch_switches_to_main(self, git_repo: GitRepo) -> None:
        """Deleting the current branch switches back to main."""
        await git_repo.checkout_branch("feature/to-delete", create=True)
        assert await git_repo.get_current_branch() == "feature/to-delete"

        await git_repo.delete_branch("feature/to-delete")

        assert await git_repo.get_current_branch() == "main"
        # Branch should no longer exist
        branches = await git_repo._run("git", "branch", "--list")
        assert "feature/to-delete" not in branches

    async def test_checkout_branch_already_exists_adds_suffix(self, git_repo: GitRepo) -> None:
        """checkout_branch with create=True adds a hash suffix when the branch already exists."""
        await git_repo.checkout_branch("feature/dup", create=True)
        await git_repo.checkout_branch("main")

        # Creating the same branch name again should succeed with a suffixed name
        actual = await git_repo.checkout_branch("feature/dup", create=True)
        assert actual.startswith("feature/dup-")
        assert actual != "feature/dup"
        assert await git_repo.get_current_branch() == actual

    async def test_checkout_branch_returns_name(self, git_repo: GitRepo) -> None:
        """checkout_branch returns the actual branch name used."""
        actual = await git_repo.checkout_branch("feature/new", create=True)
        assert actual == "feature/new"

    async def test_branch_exists(self, git_repo: GitRepo) -> None:
        """branch_exists returns True for existing branches, False otherwise."""
        assert await git_repo.branch_exists("main")
        assert not await git_repo.branch_exists("nonexistent")
        await git_repo.checkout_branch("feature/check", create=True)
        assert await git_repo.branch_exists("feature/check")

    async def test_reset_switches_to_default_branch(self, git_repo: GitRepo) -> None:
        """reset checks out the default branch and discards changes."""
        await git_repo.checkout_branch("feature/wip", create=True)
        (git_repo.path / "dirty.txt").write_text("uncommitted")

        await git_repo.reset("main")

        assert await git_repo.get_current_branch() == "main"
        assert not (git_repo.path / "dirty.txt").exists()

    async def test_reset_discards_staged_changes(self, git_repo: GitRepo) -> None:
        """reset discards staged modifications on the default branch."""
        (git_repo.path / "README.md").write_text("modified")
        await git_repo.add_all()

        await git_repo.reset("main")

        content = (git_repo.path / "README.md").read_text()
        assert content == "# Test Repo"
        assert not await git_repo.has_changes()

    async def test_delete_branch_from_other_branch(self, git_repo: GitRepo) -> None:
        """Deleting a branch while on a different branch works."""
        await git_repo.checkout_branch("feature/keep", create=True)
        await git_repo.checkout_branch("main")
        await git_repo.checkout_branch("feature/remove", create=True)
        await git_repo.checkout_branch("feature/keep")

        await git_repo.delete_branch("feature/remove")

        assert await git_repo.get_current_branch() == "feature/keep"
        branches = await git_repo._run("git", "branch", "--list")
        assert "feature/remove" not in branches
        assert "feature/keep" in branches
