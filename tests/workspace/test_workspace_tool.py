"""Tests for WorkspaceTool — write and delete capabilities (Story 5.4)."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from akgentic.tool.workspace.tool import WorkspaceTool
from akgentic.tool.workspace.workspace import Filesystem

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_observer(
    tmp_path: Path,
    team_id: uuid.UUID | None = None,
) -> tuple[MagicMock, Filesystem]:
    """Create a mock observer and matching Filesystem for tests."""
    tid = team_id or uuid.uuid4()
    observer = MagicMock()
    observer.orchestrator = MagicMock()
    observer._team_id = tid
    fs = Filesystem(str(tmp_path), str(tid))
    return observer, fs


def make_wired_tool(tmp_path: Path) -> tuple[WorkspaceTool, Filesystem]:
    """Create a WorkspaceTool wired to a real tmp Filesystem."""
    observer, fs = make_observer(tmp_path)
    with patch("akgentic.tool.workspace.tool.get_workspace", return_value=fs):
        tool = WorkspaceTool()
        tool.observer(observer)
    return tool, fs


# ---------------------------------------------------------------------------
# Task 2: observer() delegation (AC 1)
# ---------------------------------------------------------------------------


class TestObserverDelegation:
    def test_observer_sets_workspace(self, tmp_path: Path) -> None:
        """observer() via super() sets self.workspace correctly."""
        tool, fs = make_wired_tool(tmp_path)
        assert tool.workspace is fs

    def test_observer_returns_self(self, tmp_path: Path) -> None:
        """observer() returns self typed as WorkspaceTool."""
        observer, fs = make_observer(tmp_path)
        with patch("akgentic.tool.workspace.tool.get_workspace", return_value=fs):
            tool = WorkspaceTool()
            result = tool.observer(observer)
        assert result is tool
        assert isinstance(result, WorkspaceTool)

    def test_observer_raises_when_orchestrator_none(self, tmp_path: Path) -> None:
        """Inherited guard: observer with None orchestrator raises ValueError."""
        observer = MagicMock()
        observer.orchestrator = None
        tool = WorkspaceTool()
        with pytest.raises(ValueError, match="orchestrator"):
            tool.observer(observer)


# ---------------------------------------------------------------------------
# Task 2 / Task 7: get_tools() count and names (AC 2)
# ---------------------------------------------------------------------------


class TestGetToolsDefault:
    def test_default_count_is_six(self, tmp_path: Path) -> None:
        """By default get_tools() returns 6 tools: 4 read + write + delete."""
        tool, _ = make_wired_tool(tmp_path)
        tools = tool.get_tools()
        assert len(tools) == 6

    def test_default_includes_all_read_tools(self, tmp_path: Path) -> None:
        tool, _ = make_wired_tool(tmp_path)
        names = [t.__name__ for t in tool.get_tools()]
        assert "workspace_read" in names
        assert "workspace_list" in names
        assert "workspace_glob" in names
        assert "workspace_grep" in names

    def test_default_includes_write_and_delete(self, tmp_path: Path) -> None:
        tool, _ = make_wired_tool(tmp_path)
        names = [t.__name__ for t in tool.get_tools()]
        assert "workspace_write" in names
        assert "workspace_delete" in names


# ---------------------------------------------------------------------------
# Task 3: workspace_write — new file (AC 3)
# ---------------------------------------------------------------------------


class TestWorkspaceWriteNewFile:
    def test_write_new_file_creates_it(self, tmp_path: Path) -> None:
        """workspace_write creates a new file with the given content."""
        tool, fs = make_wired_tool(tmp_path)
        write_fn = next(t for t in tool.get_tools() if t.__name__ == "workspace_write")
        result = write_fn("new_file.py", "print('hello')\n")
        assert result == "Written: new_file.py"
        assert (fs._root / "new_file.py").read_text(encoding="utf-8") == "print('hello')\n"

    def test_write_new_file_creates_parent_dirs(self, tmp_path: Path) -> None:
        """workspace_write creates missing parent directories."""
        tool, fs = make_wired_tool(tmp_path)
        write_fn = next(t for t in tool.get_tools() if t.__name__ == "workspace_write")
        result = write_fn("deep/nested/dir/file.py", "content\n")
        assert result == "Written: deep/nested/dir/file.py"
        assert (fs._root / "deep/nested/dir/file.py").exists()

    def test_write_new_file_returns_written_path(self, tmp_path: Path) -> None:
        tool, fs = make_wired_tool(tmp_path)
        write_fn = next(t for t in tool.get_tools() if t.__name__ == "workspace_write")
        result = write_fn("src/module.py", "# empty\n")
        assert result == "Written: src/module.py"


# ---------------------------------------------------------------------------
# Task 3: workspace_write — overwrite with line ending preservation (AC 4)
# ---------------------------------------------------------------------------


class TestWorkspaceWriteLineEndingPreservation:
    def test_write_preserves_crlf(self, tmp_path: Path) -> None:
        """Overwriting a CRLF file with LF content normalises to CRLF."""
        tool, fs = make_wired_tool(tmp_path)
        fs.write("existing.py", b"line1\r\nline2\r\n")
        write_fn = next(t for t in tool.get_tools() if t.__name__ == "workspace_write")
        result = write_fn("existing.py", "new_line1\nnew_line2\n")
        assert result == "Written: existing.py"
        written = fs.read("existing.py")
        assert b"\r\n" in written
        assert b"new_line1" in written

    def test_write_preserves_lf(self, tmp_path: Path) -> None:
        """Overwriting an LF file with LF content keeps LF."""
        tool, fs = make_wired_tool(tmp_path)
        fs.write("lf_file.py", b"line1\nline2\n")
        write_fn = next(t for t in tool.get_tools() if t.__name__ == "workspace_write")
        result = write_fn("lf_file.py", "new_line1\nnew_line2\n")
        assert result == "Written: lf_file.py"
        written = fs.read("lf_file.py")
        assert b"\r\n" not in written
        assert b"new_line1" in written

    def test_write_overwrite_no_crlf_difference(self, tmp_path: Path) -> None:
        """Overwrite where content already matches line endings writes correctly."""
        tool, fs = make_wired_tool(tmp_path)
        fs.write("same.py", b"a\nb\n")
        write_fn = next(t for t in tool.get_tools() if t.__name__ == "workspace_write")
        result = write_fn("same.py", "updated\n")
        assert result == "Written: same.py"
        assert b"updated" in fs.read("same.py")

    def test_write_non_utf8_existing_file_does_not_raise(self, tmp_path: Path) -> None:
        """Overwriting a non-UTF-8 binary file writes content as-is without raising."""
        tool, fs = make_wired_tool(tmp_path)
        # Write a file with non-UTF-8 bytes (Windows-1252 / Latin-1 encoded)
        fs.write("binary.dat", b"\xff\xfe binary garbage \x80\x81")
        write_fn = next(t for t in tool.get_tools() if t.__name__ == "workspace_write")
        result = write_fn("binary.dat", "replacement\n")
        assert result == "Written: binary.dat"
        assert b"replacement" in fs.read("binary.dat")


# ---------------------------------------------------------------------------
# Task 4: workspace_delete — success and not-found (AC 5, 6)
# ---------------------------------------------------------------------------


class TestWorkspaceDelete:
    def test_delete_existing_file(self, tmp_path: Path) -> None:
        """workspace_delete removes the file and returns confirmation."""
        tool, fs = make_wired_tool(tmp_path)
        fs.write("to_delete.py", b"# delete me\n")
        delete_fn = next(t for t in tool.get_tools() if t.__name__ == "workspace_delete")
        result = delete_fn("to_delete.py")
        assert result == "Deleted: to_delete.py"
        assert not (fs._root / "to_delete.py").exists()

    def test_delete_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """workspace_delete raises FileNotFoundError for missing files."""
        tool, fs = make_wired_tool(tmp_path)
        delete_fn = next(t for t in tool.get_tools() if t.__name__ == "workspace_delete")
        with pytest.raises(FileNotFoundError):
            delete_fn("nonexistent.py")

    def test_delete_returns_deleted_path(self, tmp_path: Path) -> None:
        """Returned string includes the path that was deleted."""
        tool, fs = make_wired_tool(tmp_path)
        fs.write("src/old.py", b"pass\n")
        delete_fn = next(t for t in tool.get_tools() if t.__name__ == "workspace_delete")
        result = delete_fn("src/old.py")
        assert result == "Deleted: src/old.py"


# ---------------------------------------------------------------------------
# Capability toggling (AC 7)
# ---------------------------------------------------------------------------


class TestCapabilityToggling:
    def test_workspace_delete_disabled_returns_five_tools(self, tmp_path: Path) -> None:
        """WorkspaceTool(workspace_delete=False) exposes 5 tools."""
        observer, fs = make_observer(tmp_path)
        with patch("akgentic.tool.workspace.tool.get_workspace", return_value=fs):
            tool = WorkspaceTool(workspace_delete=False)
            tool.observer(observer)
        tools = tool.get_tools()
        names = [t.__name__ for t in tools]
        assert "workspace_delete" not in names
        assert "workspace_write" in names
        assert len(tools) == 5

    def test_workspace_write_disabled_returns_five_tools(self, tmp_path: Path) -> None:
        """WorkspaceTool(workspace_write=False) exposes 5 tools."""
        observer, fs = make_observer(tmp_path)
        with patch("akgentic.tool.workspace.tool.get_workspace", return_value=fs):
            tool = WorkspaceTool(workspace_write=False)
            tool.observer(observer)
        tools = tool.get_tools()
        names = [t.__name__ for t in tools]
        assert "workspace_write" not in names
        assert "workspace_delete" in names
        assert len(tools) == 5

    def test_both_write_and_delete_disabled_returns_four_read_tools(
        self, tmp_path: Path
    ) -> None:
        """WorkspaceTool(workspace_write=False, workspace_delete=False) returns 4 read tools."""
        observer, fs = make_observer(tmp_path)
        with patch("akgentic.tool.workspace.tool.get_workspace", return_value=fs):
            tool = WorkspaceTool(workspace_write=False, workspace_delete=False)
            tool.observer(observer)
        tools = tool.get_tools()
        names = [t.__name__ for t in tools]
        assert "workspace_write" not in names
        assert "workspace_delete" not in names
        assert "workspace_read" in names
        assert len(tools) == 4

    def test_workspace_delete_false_count(self, tmp_path: Path) -> None:
        """Repeated get_tools() call count is stable."""
        observer, fs = make_observer(tmp_path)
        with patch("akgentic.tool.workspace.tool.get_workspace", return_value=fs):
            tool = WorkspaceTool(workspace_delete=False)
            tool.observer(observer)
        assert len(tool.get_tools()) == 5


# ---------------------------------------------------------------------------
# Security: path traversal raises PermissionError
# ---------------------------------------------------------------------------


class TestPathSecurity:
    def test_write_path_traversal_raises(self, tmp_path: Path) -> None:
        """workspace_write raises PermissionError for paths escaping workspace root."""
        tool, fs = make_wired_tool(tmp_path)
        write_fn = next(t for t in tool.get_tools() if t.__name__ == "workspace_write")
        with pytest.raises(PermissionError):
            write_fn("../escape.py", "malicious content\n")

    def test_delete_path_traversal_raises(self, tmp_path: Path) -> None:
        """workspace_delete raises PermissionError for paths escaping workspace root."""
        tool, fs = make_wired_tool(tmp_path)
        delete_fn = next(t for t in tool.get_tools() if t.__name__ == "workspace_delete")
        with pytest.raises(PermissionError):
            delete_fn("../escape.py")
