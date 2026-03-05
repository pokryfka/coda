"""LangChain tool wrappers for the coding agent's file operations."""

from __future__ import annotations

import glob as glob_module
import os
from pathlib import Path

from langchain_core.tools import tool


def _validate_path(path: str, workspace: str) -> Path:
    """Validate that a path is within the workspace directory."""
    resolved = Path(workspace, path).resolve()
    workspace_resolved = Path(workspace).resolve()
    if not resolved.is_relative_to(workspace_resolved):
        msg = f"Path {path} is outside workspace"
        raise ValueError(msg)
    return resolved


@tool
def read_file(path: str, workspace: str = ".") -> str:
    """Read the contents of a file within the workspace.

    Args:
        path: Relative path to the file.
        workspace: Workspace root directory.

    Returns:
        File contents as a string.
    """
    resolved = _validate_path(path, workspace)
    if not resolved.exists():
        return f"Error: File not found: {path}"
    return resolved.read_text()


@tool
def write_file(path: str, content: str, workspace: str = ".") -> str:
    """Write content to a file within the workspace.

    Args:
        path: Relative path to the file.
        content: Content to write.
        workspace: Workspace root directory.

    Returns:
        Confirmation message.
    """
    resolved = _validate_path(path, workspace)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content)
    return f"Written: {path}"


@tool
def list_files(path: str = ".", pattern: str = "**/*", workspace: str = ".") -> str:
    """List files matching a glob pattern within the workspace.

    Args:
        path: Relative path to search from.
        pattern: Glob pattern to match.
        workspace: Workspace root directory.

    Returns:
        Newline-separated list of matching file paths.
    """
    resolved = _validate_path(path, workspace)
    matches = glob_module.glob(str(resolved / pattern), recursive=True)
    workspace_root = str(Path(workspace).resolve())
    relative = [os.path.relpath(m, workspace_root) for m in matches if os.path.isfile(m)]
    return "\n".join(sorted(relative))


@tool
def run_command(cmd: str, workspace: str = ".") -> str:
    """Execute a shell command within the workspace directory.

    Args:
        cmd: Command to execute.
        workspace: Workspace root directory.

    Returns:
        Combined stdout and stderr output.
    """
    import subprocess

    result = subprocess.run(
        cmd,
        shell=True,
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout
    if result.stderr:
        output += "\n" + result.stderr
    return output.strip()
