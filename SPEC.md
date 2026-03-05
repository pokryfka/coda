# Coding Agent

PR-driven coding agent.
 
## Project overview

Autonomous coding agent that runs in Docker, connects to an LLM (Claude, Gemini, OpenAI, or local Ollama), and delivers complete pull requests. Given a task and a repo, it clones, plans, implements, runs tests, fixes failures, pushes, and opens a PR. When a branch is specified, the system gathers the commit history from that branch to build context. It also checks for any open pull requests on the branch, including all reviews and review comments, and includes this information in the context for the agent

Pipeline: Clone → Branch → Check PR → Plan → Implement → Test/Fix loop → Push → PR (optional).

## Deliverables

Implementation must produce:

- Architecture documentation in ./docs/architecture.md
- Project structure as defined below
- Fully working coding agent working in Docker container
- Fully working coding agent working locally
- Fully working LangGraph Studio setup
- Configurable repositories
- Multi-model support
- Concurrent task execution
- Structured LLM interaction
- Test suite passing
- Docstring coverage ≥ 80%

## Project structure

```text
coda/
├── .github/
│  └─ workflows/
│     └─ ci.yml # CI workflow (lint/tests on push/PR)
├─ data/ # Shared with Docker container
│  └─ .gitkeep # Keeps empty data directory tracked in git
├─ docs/
│  ├─ architecture.md # System architecture and component relationships
├─ scripts/
│  ├─ run.sh # Main shell entrypoint to run the agent with args/config
├─ src/
│  ├─ config/
│  │  └─ settings.py # Dataclass-based config models and YAML/env loading
│  ├─ agent/
│  │  └─ coding/
│  │     ├─ graph.py # LangGraph graph construction and routing functions
│  │     ├─ loop.py # High-level orchestration wrapper around graph execution
│  │     ├─ nodes.py # LangGraph node implementations
│  │     ├─ prompts.py # Prompt templates used for LLM interactions
│  │     ├─ state.py # LangGraph state definition
│  │     └─ tools.py # LangChain tool wrappers
│  ├─ git_ops/
│  │  └─ repo.py # Clone/branch/commit/push/PR operations via git + gh
│  ├─ llm/
│  │  └─ factory.py # Provider/model factory for creating chat clients
│  └─ main.py # CLI entrypoint and runtime bootstrap
├─ tests/
│  ├─ e2e/
│  │  ├─ test_agent.py # End-to-end tests for Coding Agent 
│  ├─ integration/
│  │  ├─ test_git.py # Integration tests for GitRepo GIT operations
│  │  └─ test_llm.py # Integration tests for configured LLM models
│  └─ unit/
│     ├─ test_git.py # Unit tests for GitRepo auth/credential behavior
│     ├─ test_graph.py # Unit tests for LangGraph graph wiring/routing behavior
│     ├─ test_prompts.py # Unit tests for prompt construction/formatting
│     ├─ test_settings.py # Unit tests for config loading/default behavior
│     └─ test_tools.py # Unit tests for LangChain tool wrappers
├─ .coderabbit.yaml # CodeRabbit review/automation configuration
├─ .env.example # Example environment variables for API keys/tokens
├─ .gitignore # Git ignore rules for local/build/runtime artifacts
├─ AGENTS.md # Agent-facing project instructions and conventions
├─ config.yaml.example # Example runtime configuration file
├─ langgraph.json
├─ pyproject.toml # Python package metadata, deps, tool configs
├─ Dockerfile # Container image definition for agent runtime
├─ docker-compose.yml # Multi-service local run setup (agent + optional Ollama)
├─ README.md # Project overview, setup, and usage instructions
├─ uv.lock # Locked dependency graph for uv reproducible installs
``` 

## Agent Loop

- If a branch name is provided, the system MUST switch to that branch and retrieve the descriptions of all commits on the branch to build contextual understanding.
- The system MUST then check whether an open Pull Request exists for the branch. If one exists, it MUST retrieve all reviews and review comments associated with that Pull Request and incorporate them into the execution context.
- Each state that modifies files MUST be recorded in a separate commit.
- Git commit messages MUST follow the semantic commit conventions defined at [Conventional Commits v1.0.0](https://www.conventionalcommits.org/en/v1.0.0/).

## Setup commands

- Install deps: `uv sync`
- Install dev deps: `uv sync --extra dev`
- Run the agent in Docker: `./scripts/run.sh <repo-name> "<task>" [provider] [--branch "<branch-name>"]`
- Run the agent locally (without PR): `./scripts/run.sh . "<task>" [provider]`
- Run via docker compose: `docker compose run --rm agent --repo <repo> --task "<task>" [--branch "<branch-name>"]`
- Build Docker image ./scripts/docker_build.sh
- Start LangGraph Studio ./scripts/studio.sh

## Testing instructions

- Run all tests: `uv run pytest`
- Run a single test file: `uv run pytest tests/unit/test_tools.py`
- Run a specific test: `uv run pytest tests/unit/test_tools.py -k "test_read_existing_file"`
- Lint: `uv run ruff check .`
- Always run both `uv run pytest` and `uv run ruff check .` before committing

## Testing Requirements

Unit tests must cover:

- LangGraph state transitions
- GitRepo configuration
- LLM configuration and factory

Integration tests must:

- Clone temp repo
- Validate branch + commit creation

- Validate LLM model response (for configured models)

E2e tests must:

- Clone temp repo
- Run full workflow locally
- Validate branch + commit creation

## Code style:

- Python 3.12+, async throughout the agent loop
- `from __future__ import annotations` in every module
- Ruff for linting: line length 120, rules E/F/I/W
- pytest with `asyncio_mode = "auto"`
- Dataclasses for all config/data types (no Pydantic)
- All LLM interaction is structured JSON — prompts define schemas, responses are parsed as JSON
- Docstring coverage must stay at or above 80%: `uv run interrogate src/ -v --fail-under 80 -I -M -S -p`

## Security Requirements

- Never include secrets in prompts
- Limit file context sent to LLM
- Mask tokens in logs
- Do not expose SSH keys in logs
