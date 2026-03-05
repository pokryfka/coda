"""Unit tests for LangChain tool wrappers."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.coding.tools import _validate_path, list_files, read_file, write_file


class TestValidatePath:
    """Tests for path validation."""

    def test_valid_path_within_workspace(self, tmp_path: Path) -> None:
        """Paths within workspace are accepted."""
        result = _validate_path("foo/bar.py", str(tmp_path))
        assert str(result).startswith(str(tmp_path))

    def test_rejects_path_traversal(self, tmp_path: Path) -> None:
        """Paths escaping workspace are rejected."""
        with pytest.raises(ValueError, match="outside workspace"):
            _validate_path("../../etc/passwd", str(tmp_path))

    def test_rejects_absolute_path_outside_workspace(self, tmp_path: Path) -> None:
        """Absolute paths outside workspace are rejected."""
        outside = tmp_path.parent / "outside.txt"
        with pytest.raises(ValueError, match="outside workspace"):
            _validate_path(str(outside), str(tmp_path))

    def test_rejects_sibling_prefix_escape(self, tmp_path: Path) -> None:
        """Sibling paths sharing prefix with workspace are rejected."""
        sibling = tmp_path.parent / f"{tmp_path.name}-evil" / "pwned.txt"
        with pytest.raises(ValueError, match="outside workspace"):
            _validate_path(str(sibling), str(tmp_path))


class TestReadFile:
    """Tests for read_file tool."""

    def test_read_existing_file(self, tmp_path: Path) -> None:
        """Reading an existing file returns its contents."""
        (tmp_path / "test.txt").write_text("hello world")
        result = read_file.invoke({"path": "test.txt", "workspace": str(tmp_path)})
        assert result == "hello world"

    def test_read_missing_file(self, tmp_path: Path) -> None:
        """Reading a missing file returns an error message."""
        result = read_file.invoke({"path": "nope.txt", "workspace": str(tmp_path)})
        assert "Error" in result


class TestWriteFile:
    """Tests for write_file tool."""

    def test_write_creates_file(self, tmp_path: Path) -> None:
        """Writing creates a new file with content."""
        result = write_file.invoke({"path": "out.txt", "content": "data", "workspace": str(tmp_path)})
        assert "Written" in result
        assert (tmp_path / "out.txt").read_text() == "data"

    def test_write_creates_subdirs(self, tmp_path: Path) -> None:
        """Writing to nested path creates intermediate directories."""
        write_file.invoke({"path": "a/b/c.txt", "content": "nested", "workspace": str(tmp_path)})
        assert (tmp_path / "a" / "b" / "c.txt").read_text() == "nested"


class TestListFiles:
    """Tests for list_files tool."""

    def test_list_files_matches(self, tmp_path: Path) -> None:
        """Listing files returns matching paths."""
        (tmp_path / "foo.py").write_text("x")
        (tmp_path / "bar.txt").write_text("y")
        result = list_files.invoke({"path": ".", "pattern": "*.py", "workspace": str(tmp_path)})
        assert "foo.py" in result
        assert "bar.txt" not in result
