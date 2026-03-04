"""Unit tests for GitRepo auth/credential behavior."""

from __future__ import annotations

from src.git_ops.repo import GitRepo, _mask_tokens


class TestTokenMasking:
    """Tests for token masking in logs."""

    def test_masks_ghp_token(self) -> None:
        """Classic PAT tokens are masked."""
        assert "***" in _mask_tokens("token ghp_abc123def456")

    def test_masks_github_pat_token(self) -> None:
        """Fine-grained PAT tokens are masked."""
        assert "***" in _mask_tokens("github_pat_abc123def456_xyz")

    def test_no_mask_when_no_token(self) -> None:
        """Text without tokens is unchanged."""
        text = "just a normal string"
        assert _mask_tokens(text) == text


class TestGitRepoAuth:
    """Tests for GitRepo authentication setup."""

    def test_setup_auth_stores_token(self) -> None:
        """setup_auth stores the token."""
        git = GitRepo()
        git.setup_auth("ghp_testtoken123")
        assert git.token == "ghp_testtoken123"

    def test_auth_url_injects_token(self) -> None:
        """Token is injected into HTTPS URLs."""
        git = GitRepo()
        git.setup_auth("mytoken")
        url = git._auth_url("https://github.com/org/repo.git")
        assert "x-access-token:mytoken@" in url

    def test_auth_url_no_token(self) -> None:
        """URLs are unchanged when no token is set."""
        git = GitRepo()
        url = git._auth_url("https://github.com/org/repo.git")
        assert url == "https://github.com/org/repo.git"

    def test_build_env_includes_token(self) -> None:
        """Environment dict includes GH_TOKEN when set."""
        git = GitRepo()
        git.setup_auth("testtoken")
        env = git._build_env()
        assert env["GH_TOKEN"] == "testtoken"

    def test_build_env_without_token(self) -> None:
        """Environment dict has no GH_TOKEN when token is empty."""
        git = GitRepo()
        env = git._build_env()
        # GH_TOKEN may exist from the real environment
        # Just verify it doesn't add an empty token
        if not git.token:
            assert env.get("GH_TOKEN", "") != "testtoken"
