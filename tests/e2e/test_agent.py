"""End-to-end tests for the coding agent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.agent.coding.loop import run_agent
from src.config.settings import AgentConfig, AppConfig, GitConfig, LlmConfig, LlmProvider, LlmProviderConfig, RepoConfig  # noqa: E501


@pytest.fixture
async def temp_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository for e2e testing."""
    repo_path = tmp_path / "e2e-repo"
    repo_path.mkdir()

    from src.git_ops.repo import GitRepo

    git = GitRepo(path=repo_path)
    await git._run("git", "init")
    await git._run("git", "config", "user.name", "Test")
    await git._run("git", "config", "user.email", "test@test.com")
    (repo_path / "README.md").write_text("# E2E Test")
    await git.add_all()
    await git.commit("Initial commit")

    return repo_path


@pytest.fixture
def e2e_config(tmp_path: Path, temp_repo: Path) -> AppConfig:
    """Create config for e2e testing."""
    return AppConfig(
        llm=LlmConfig(
            provider="ollama",
            providers={
                LlmProvider.OLLAMA: LlmProviderConfig(
                    model="test-model",
                    options={"base_url": "http://localhost:11434"},
                ),
            },
        ),
        git=GitConfig(commit_author="Test Bot <test@bot.com>"),
        repositories=[
            RepoConfig(
                name="e2e-repo",
                url=str(temp_repo),
                test_command="echo 'tests pass'",
                lint_command="echo 'lint pass'",
            ),
        ],
        agent=AgentConfig(
            max_fix_attempts=1,
            auto_push=False,
            auto_create_pr=False,
            workspace_dir=str(tmp_path / "workspace"),
        ),
    )


class TestAgentE2E:
    """End-to-end agent tests with mocked LLM."""

    async def test_full_workflow_creates_branch_and_commits(
        self, e2e_config: AppConfig, temp_repo: Path
    ) -> None:
        """Full workflow creates a branch, makes commits, and runs tests."""
        mock_plan_response = {
            "plan": "Add a hello.py file",
            "files_to_modify": ["hello.py"],
            "approach": "Create a simple Python file",
        }
        mock_implement_response = {
            "changes": [{"path": "hello.py", "content": "print('hello')\n", "action": "write"}],
            "commit_message": "feat: add hello.py",
        }

        mock_llm = AsyncMock()
        mock_structured = AsyncMock()
        mock_structured.ainvoke = AsyncMock(side_effect=[mock_plan_response, mock_implement_response])
        mock_llm.with_structured_output = lambda *a, **kw: mock_structured

        async def mock_clone(state):
            """Mock clone that uses the temp repo directly."""
            from src.git_ops.repo import GitRepo

            git = GitRepo(path=temp_repo)
            await git.configure_user(e2e_config.git.commit_author)
            return {"repo_path": str(temp_repo)}

        async def mock_check_pr(state):
            """Mock check_pr that returns no PR."""
            return {"pr_info": None}

        # Patch at the graph module level since build_graph uses `from ... import`
        with (
            patch("src.agent.coding.nodes.create_llm", return_value=mock_llm),
            patch("src.agent.coding.graph.clone_repo", mock_clone),
            patch("src.agent.coding.graph.check_pr", mock_check_pr),
        ):
            await run_agent(
                repo_name="e2e-repo",
                task="Add a hello world script",
                config=e2e_config,
            )

        # Verify branch was created
        from src.git_ops.repo import GitRepo

        git = GitRepo(path=temp_repo)
        branch = await git.get_current_branch()
        assert branch != "main"

        # Verify file was created
        assert (temp_repo / "hello.py").exists()

    async def test_agent_fails_on_unknown_repo(self, e2e_config: AppConfig) -> None:
        """Agent raises ValueError for unknown repository."""
        with pytest.raises(ValueError, match="not found"):
            await run_agent(
                repo_name="nonexistent",
                task="test",
                config=e2e_config,
            )
