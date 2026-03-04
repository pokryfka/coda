"""Clone/branch/commit/push/PR operations via git and gh subprocess calls."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"(ghp_\w+|gho_\w+|github_pat_\w+)")


def _mask_tokens(text: str) -> str:
    """Mask GitHub tokens in text for safe logging."""
    return _TOKEN_PATTERN.sub("***", text)


@dataclass
class GitRepo:
    """Wrapper around git and gh CLI for repository operations."""

    path: Path = field(default_factory=lambda: Path("."))
    token: str = ""

    async def _run(self, *args: str, cwd: Path | None = None) -> str:
        """Run a subprocess command and return stdout."""
        work_dir = cwd or self.path
        logger.debug("Running: %s in %s", _mask_tokens(" ".join(args)), work_dir)

        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=work_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._build_env(),
        )
        stdout, stderr = await proc.communicate()
        out = stdout.decode().strip()
        err = stderr.decode().strip()

        if proc.returncode != 0:
            logger.error("Command failed: %s\nstderr: %s", _mask_tokens(" ".join(args)), _mask_tokens(err))
            msg = f"Command failed ({args[0]}): {_mask_tokens(err)}"
            raise RuntimeError(msg)

        return out

    def _build_env(self) -> dict[str, str]:
        """Build environment dict with auth tokens."""
        env = os.environ.copy()
        if self.token:
            env["GH_TOKEN"] = self.token
        return env

    def setup_auth(self, token: str) -> None:
        """Configure authentication token for git operations."""
        self.token = token

    def _auth_url(self, url: str) -> str:
        """Inject token into HTTPS URL for authenticated clone."""
        if self.token and url.startswith("https://"):
            return url.replace("https://", f"https://x-access-token:{self.token}@")
        return url

    async def clone(self, url: str, dest: Path) -> None:
        """Clone a repository to the destination path."""
        auth_url = self._auth_url(url)
        await self._run("git", "clone", auth_url, str(dest), cwd=dest.parent)
        self.path = dest

    async def checkout_branch(self, name: str, create: bool = False) -> None:
        """Check out a branch, optionally creating it."""
        if create:
            await self._run("git", "checkout", "-b", name)
        else:
            await self._run("git", "checkout", name)

    async def get_commits(self, branch: str) -> list[str]:
        """Get commit descriptions from a branch relative to default branch."""
        try:
            out = await self._run("git", "log", "--oneline", f"main..{branch}")
            return [line for line in out.splitlines() if line.strip()]
        except RuntimeError:
            return []

    async def add_all(self) -> None:
        """Stage all changes."""
        await self._run("git", "add", "-A")

    async def commit(self, message: str, author: str = "") -> None:
        """Create a commit with the given message."""
        cmd = ["git", "commit", "-m", message]
        if author:
            cmd.extend(["--author", author])
        await self._run(*cmd)

    async def push(self, branch: str = "") -> None:
        """Push the current branch to origin."""
        cmd = ["git", "push", "origin"]
        if branch:
            cmd.append(branch)
        else:
            cmd.append("HEAD")
        await self._run(*cmd)

    async def create_pr(self, title: str, body: str, base: str = "main") -> dict:
        """Create a pull request via gh CLI."""
        out = await self._run(
            "gh", "pr", "create",
            "--title", title,
            "--body", body,
            "--base", base,
            "--json", "number,url",
        )
        import json

        return json.loads(out)

    async def get_open_pr(self, branch: str) -> dict | None:
        """Check for an open PR on the given branch."""
        try:
            out = await self._run(
                "gh", "pr", "view", branch,
                "--json", "number,title,body,url,state",
            )
            import json

            pr = json.loads(out)
            if pr.get("state") == "OPEN":
                return pr
            return None
        except RuntimeError:
            return None

    async def get_pr_reviews(self, pr_number: int) -> list[dict]:
        """Fetch reviews and comments for a PR."""
        try:
            out = await self._run(
                "gh", "pr", "view", str(pr_number),
                "--json", "reviews,comments",
            )
            import json

            data = json.loads(out)
            return data.get("reviews", []) + data.get("comments", [])
        except RuntimeError:
            return []

    async def request_reviewers(self, pr_number: int, reviewers: list[str]) -> None:
        """Request reviewers for a PR."""
        if not reviewers:
            return
        reviewer_args = []
        for r in reviewers:
            reviewer_args.extend(["--add-reviewer", r])
        await self._run("gh", "pr", "edit", str(pr_number), *reviewer_args)

    async def get_current_branch(self) -> str:
        """Get the name of the current branch."""
        return await self._run("git", "rev-parse", "--abbrev-ref", "HEAD")

    async def has_changes(self) -> bool:
        """Check if there are uncommitted changes."""
        out = await self._run("git", "status", "--porcelain")
        return bool(out.strip())

    async def configure_user(self, author: str) -> None:
        """Set git user name and email from author string."""
        # Parse "Name <email>" format
        match = re.match(r"(.+?)\s*<(.+?)>", author)
        if match:
            name, email = match.group(1).strip(), match.group(2).strip()
            await self._run("git", "config", "user.name", name)
            await self._run("git", "config", "user.email", email)
