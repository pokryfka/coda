#!/usr/bin/env bash
# Main shell entrypoint to run the coding agent.
# Usage:
#   ./scripts/run.sh . "task description" [provider]            # local
#   ./scripts/run.sh repo-name "task description" [provider] [--branch "name"]  # docker
set -euo pipefail

REPO="${1:?Usage: run.sh <repo|.> \"task\" [provider] [--branch \"name\"]}"
TASK="${2:?Usage: run.sh <repo|.> \"task\" [provider] [--branch \"name\"]}"
PROVIDER="${3:-}"
shift 2
shift || true

# Collect remaining args (e.g. --branch "name")
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --branch)
            EXTRA_ARGS+=("--branch" "$2")
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

PROVIDER_ARG=""
if [[ -n "$PROVIDER" && "$PROVIDER" != --* ]]; then
    PROVIDER_ARG="--provider $PROVIDER"
fi

if [[ "$REPO" == "." ]]; then
    echo "Running locally..."
    uv run python -m src.main \
        --repo "." \
        --task "$TASK" \
        $PROVIDER_ARG \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
else
    echo "Running in Docker..."
    docker compose run --rm agent \
        --repo "$REPO" \
        --task "$TASK" \
        $PROVIDER_ARG \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
fi
