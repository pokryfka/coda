# Architecture

## Overview

Coda is a PR-driven coding agent built on LangGraph. It autonomously clones repositories, plans changes, implements them, runs tests, fixes failures, and opens pull requests.

## Components

### Config (`src/config/settings.py`)
Dataclass-based configuration loaded from YAML with environment variable overrides. Supports per-provider LLM settings, repository definitions, git config, and agent behavior tuning.

### LLM Factory (`src/llm/factory.py`)
Creates LLM clients based on provider configuration. Supports Claude (Anthropic), Gemini (Google), Codex (OpenAI), and Ollama. Per-task model overrides allow using different models for planning, implementing, and fixing.

### Git Operations (`src/git_ops/repo.py`)
Async wrapper around `git` and `gh` CLI subprocess calls. Handles cloning, branching, committing, pushing, and PR creation. Supports private repos with per-repo token authentication. Masks tokens in logs.

### Agent (`src/agent/coding/`)
LangGraph StateGraph implementing the coding pipeline:

- **state.py** — TypedDict state definition with status enum
- **prompts.py** — Prompt templates with JSON schemas for structured LLM output
- **tools.py** — File read/write/list and command execution tools
- **nodes.py** — Async node functions for each pipeline stage
- **graph.py** — Graph construction and conditional routing
- **loop.py** — High-level orchestration wrapper

### CLI (`src/main.py`)
Argparse-based CLI entrypoint that loads config and runs the agent.

## Data Flow

```
START → clone → branch → check_pr → plan → implement → test
                                                          │
                                              pass? ─yes─→ push → create_pr → END
                                              pass? ─no──→ fix → test (loop, max N)
                                              exceeded? ──→ END (failure)
```

## State

The agent state flows through the graph as a `TypedDict` containing:
- Repository path and configuration
- Branch name and commit history
- PR information and reviews
- Implementation plan and file changes
- Test results and fix attempt counter
- Current status

## LLM Interaction

All LLM calls use `with_structured_output()` for typed JSON responses. Each interaction type (plan, implement, fix) has its own prompt template and JSON schema.

## Security

- Tokens are never included in LLM prompts
- File context sent to LLM is limited
- Token masking in all log output
- Path validation prevents file operations outside workspace
