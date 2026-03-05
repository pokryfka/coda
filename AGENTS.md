# Agent Instructions

## Project

This is a Python project using uv for dependency management.

## Code Style

- Python 3.12+, async throughout the agent loop
- `from __future__ import annotations` in every module
- Ruff for linting: line length 120, rules E/F/I/W
- Dataclasses for all config/data types (no Pydantic)
- All LLM interaction is structured JSON
- Docstring coverage must stay at or above 80%

## Testing

- Run all tests: `uv run pytest`
- Lint: `uv run ruff check .`
- Always run both before committing

## Commit Style

Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation
- `refactor:` for refactoring
- `test:` for test changes
- `chore:` for maintenance

## Project Structure

- `src/config/` — Configuration loading
- `src/llm/` — LLM provider factory
- `src/git_ops/` — Git/GitHub operations
- `src/agent/coding/` — LangGraph agent implementation
- `tests/` — Test suite (unit, integration, e2e)
