# Coda

PR-driven coding agent powered by LangGraph with multi-LLM support.

## Overview

Coda is an autonomous coding agent that clones repositories, plans changes, implements them, runs tests, fixes failures, and opens pull requests. It supports Claude, Gemini, OpenAI, and Ollama as LLM providers.

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- [gh](https://cli.github.com/) GitHub CLI
- Docker (optional, for containerized execution)

### Installation

```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --extra dev

# Copy and configure environment
cp .env.example .env
cp config.yaml.example config.yaml
```

### Configuration

Edit `config.yaml` to configure:
- LLM provider and model settings
- Git branch prefix and commit author
- Repository definitions (URL, language, test/lint commands)
- Agent behavior (max fix attempts, auto-push, auto-PR)

Set API keys in `.env`:
- `GH_TOKEN` — GitHub personal access token
- `ANTHROPIC_API_KEY` — for Claude
- `GEMINI_API_KEY` — for Gemini
- `OPENAI_API_KEY` — for OpenAI/Codex

## Usage

### Run locally

```bash
./scripts/run.sh . "task description" claude
```

### Run with Docker

```bash
./scripts/run.sh my-backend "implement user authentication" claude --branch "agent/auth"
```

### Run via Docker Compose

```bash
docker compose run --rm agent --repo my-backend --task "fix the login bug" --branch "agent/fix-login"
```

### Run with Ollama

```bash
docker compose --profile ollama up -d ollama
./scripts/run.sh my-backend "add rate limiting" ollama
```

### LangGraph Studio

Configure `langgraph.json` and open in LangGraph Studio for visual debugging.

## Development

```bash
# Lint
uv run ruff check .

# Test
uv run pytest

# Docstring coverage
uv run interrogate src/ -v --fail-under 80 -I -M -S -p

# Build Docker image
docker build -t coda .
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for system design details.

## License

See [LICENSE](LICENSE).
