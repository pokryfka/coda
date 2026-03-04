"""CLI entrypoint and runtime bootstrap."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv

from src.agent.coding.loop import run_agent
from src.agent.coding.state import Status
from src.config.settings import load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Coda - PR-driven coding agent")
    parser.add_argument("--repo", required=True, help="Repository name from config")
    parser.add_argument("--task", required=True, help="Task description")
    parser.add_argument("--provider", default=None, help="LLM provider override")
    parser.add_argument("--branch", default="", help="Branch name to work on")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the coding agent from command line arguments."""
    load_dotenv()

    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    if args.provider:
        config.llm.provider = args.provider

    result = asyncio.run(
        run_agent(
            repo_name=args.repo,
            task=args.task,
            config=config,
            branch=args.branch,
        )
    )

    status = result.get("status")
    if status == Status.DONE:
        print(f"Agent completed successfully. PR: {result.get('pr_info', {}).get('url', 'N/A')}")
        return 0

    print(f"Agent finished with status: {status}")
    if result.get("error"):
        print(f"Error: {result['error']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
