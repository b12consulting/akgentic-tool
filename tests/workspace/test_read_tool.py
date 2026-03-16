"""Tests for akgentic.tool.workspace.tool module (Story 5.2)."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from akgentic.tool.workspace.tool import (
    WorkspaceGlob,
    WorkspaceGrep,
    WorkspaceList,
    WorkspaceRead,
    WorkspaceReadTool,
    _grep_python,
    _grep_rg,
)
from akgentic.tool.workspace.workspace import Filesystem

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_observer(
    orchestrator_is_none: bool = False,
    team_id: uuid.UUID | None = None,
) -> MagicMock:
    """Build a minimal mock ActorToolObserver."""
    observer = MagicMock()
    observer.orchestrator = None if orchestrator_is_none else MagicMock()
    observer._team_id = team_id or uuid.uuid4()
    return observer


def make_tool(tmp_path: Path, team_id: uuid.UUID | None = None) -> WorkspaceReadTool:
    """Build a WorkspaceReadTool wired to a Filesystem rooted at tmp_path."""
    tid = team_id or uuid.uuid4()
    fs = Filesystem(str(tmp_path), str(tid))
    observer = make_observer(team_id=tid)
    tool = WorkspaceReadTool()
    with patch("akgentic.tool.workspace.tool.get_workspace", return_value=fs):
        tool.observer(observer)
    return tool


# ---------------------------------------------------------------------------
# Task 1: WorkspaceReadTool fields (AC 1)
# ---------------------------------------------------------------------------


class TestWorkspaceReadToolFields:
    def test_default_fields(self) -> None:
        tool = WorkspaceReadTool()
        assert tool.name == "WorkspaceRead"
        assert tool.workspace_id is None
        assert tool.workspace_read is True
        assert tool.workspace_list is True
        assert tool.workspace_glob is True
        assert tool.workspace_grep is True

    def test_workspace_id_set(self) -> None:
        tool = WorkspaceReadTool(workspace_id="my-workspace")
        assert tool.workspace_id == "my-workspace"

    def test_capabilities_accept_param_models(self) -> None:
        tool = WorkspaceReadTool(
            workspace_read=WorkspaceRead(default_limit=500),
            workspace_list=WorkspaceList(),
            workspace_glob=WorkspaceGlob(max_results=50),
            workspace_grep=WorkspaceGrep(max_results=50),
        )
        assert isinstance(tool.workspace_read, WorkspaceRead)
        assert tool.workspace_read.default_limit == 500


# ---------------------------------------------------------------------------
# Task 1: observer() wiring (AC 2, 3)
# ---------------------------------------------------------------------------


class TestObserverWiring:
    def test_observer_raises_when_orchestrator_is_none(self, tmp_path: Path) -> None:
        observer = make_observer(orchestrator_is_none=True)
        tool = WorkspaceReadTool()
        with pytest.raises(ValueError, match="orchestrator"):
            tool.observer(observer)

    def test_observer_sets_workspace_from_team_id(self, tmp_path: Path) -> None:
        team_id = uuid.uuid4()
        observer = make_observer(team_id=team_id)
        fs = Filesystem(str(tmp_path), str(team_id))
        tool = WorkspaceReadTool()
        with patch(
            "akgentic.tool.workspace.tool.get_workspace", return_value=fs
        ) as mock_gw:
            result = tool.observer(observer)
            mock_gw.assert_called_once_with(str(team_id))
            assert tool.workspace is fs
            assert result is tool

    def test_observer_uses_explicit_workspace_id(self, tmp_path: Path) -> None:
        observer = make_observer()
        fs = Filesystem(str(tmp_path), "explicit-ws")
        tool = WorkspaceReadTool(workspace_id="explicit-ws")
        with patch(
            "akgentic.tool.workspace.tool.get_workspace", return_value=fs
        ) as mock_gw:
            tool.observer(observer)
            mock_gw.assert_called_once_with("explicit-ws")

    def test_observer_returns_self(self, tmp_path: Path) -> None:
        team_id = uuid.uuid4()
        observer = make_observer(team_id=team_id)
        fs = Filesystem(str(tmp_path), str(team_id))
        tool = WorkspaceReadTool()
        with patch("akgentic.tool.workspace.tool.get_workspace", return_value=fs):
            result = tool.observer(observer)
        assert result is tool

    def test_observer_failed_call_does_not_set_internal_observer(self) -> None:
        """When observer() raises, _observer must NOT be set (no partial init)."""
        observer = make_observer(orchestrator_is_none=True)
        tool = WorkspaceReadTool()
        with pytest.raises(ValueError):
            tool.observer(observer)
        # _observer must remain unset — check via __pydantic_private__
        assert tool.__pydantic_private__ is None or "_observer" not in (
            tool.__pydantic_private__ or {}
        )

    def test_get_tools_raises_before_observer_called(self) -> None:
        """Calling get_tools() before observer() raises RuntimeError from workspace property."""
        tool = WorkspaceReadTool()
        with pytest.raises(RuntimeError, match="observer\\(\\) was called"):
            tool.get_tools()


# ---------------------------------------------------------------------------
# Task 2: workspace_read pagination (AC 4, 5)
# ---------------------------------------------------------------------------


class TestWorkspaceRead:
    def _make_file(self, root: Path, name: str, n_lines: int) -> None:
        content = "\n".join(f"line {i}" for i in range(1, n_lines + 1))
        (root / name).write_text(content, encoding="utf-8")

    def test_default_window_returns_first_2000_lines(self, tmp_path: Path) -> None:
        team_id = uuid.uuid4()
        fs = Filesystem(str(tmp_path), str(team_id))
        root = fs._root
        self._make_file(root, "big.txt", 3500)
        tool = WorkspaceReadTool()
        observer = make_observer(team_id=team_id)
        with patch("akgentic.tool.workspace.tool.get_workspace", return_value=fs):
            tool.observer(observer)
        fn = tool.get_tools()[0]  # workspace_read
        result = fn("big.txt")
        lines = result.split("\n")
        # first 2000 numbered lines + truncation notice
        assert lines[0].startswith("1     ")
        assert "2000" in lines[1999]
        assert "truncated" in lines[-1]
        assert "3500 lines total" in lines[-1]

    def test_offset_and_limit(self, tmp_path: Path) -> None:
        team_id = uuid.uuid4()
        fs = Filesystem(str(tmp_path), str(team_id))
        root = fs._root
        self._make_file(root, "file.txt", 200)
        tool = WorkspaceReadTool()
        observer = make_observer(team_id=team_id)
        with patch("akgentic.tool.workspace.tool.get_workspace", return_value=fs):
            tool.observer(observer)
        fn = tool.get_tools()[0]
        result = fn("file.txt", offset=100, limit=50)
        lines = [ln for ln in result.split("\n") if not ln.startswith("[")]
        assert lines[0].startswith("100   ")
        assert lines[-1].startswith("149   ")
        assert len(lines) == 50

    def test_no_truncation_when_file_fits(self, tmp_path: Path) -> None:
        team_id = uuid.uuid4()
        fs = Filesystem(str(tmp_path), str(team_id))
        root = fs._root
        self._make_file(root, "small.txt", 10)
        tool = WorkspaceReadTool()
        observer = make_observer(team_id=team_id)
        with patch("akgentic.tool.workspace.tool.get_workspace", return_value=fs):
            tool.observer(observer)
        fn = tool.get_tools()[0]
        result = fn("small.txt")
        assert "truncated" not in result
        assert result.split("\n")[0].startswith("1     ")

    def test_line_numbers_are_correct(self, tmp_path: Path) -> None:
        team_id = uuid.uuid4()
        fs = Filesystem(str(tmp_path), str(team_id))
        (fs._root / "abc.txt").write_text("alpha\nbeta\ngamma", encoding="utf-8")
        tool = WorkspaceReadTool()
        observer = make_observer(team_id=team_id)
        with patch("akgentic.tool.workspace.tool.get_workspace", return_value=fs):
            tool.observer(observer)
        fn = tool.get_tools()[0]
        result = fn("abc.txt")
        assert "1     alpha" in result
        assert "2     beta" in result
        assert "3     gamma" in result


# ---------------------------------------------------------------------------
# Task 3: workspace_list (AC 6)
# ---------------------------------------------------------------------------


class TestWorkspaceList:
    def test_list_shows_dir_and_file_entries(self, tmp_path: Path) -> None:
        tool = make_tool(tmp_path)
        root = tool.workspace._root
        (root / "src").mkdir()
        (root / "README.md").write_bytes(b"hello")
        fn = tool.get_tools()[1]  # workspace_list
        result = fn()
        assert "src/" in result
        assert "README.md (5 bytes)" in result

    def test_list_empty_directory(self, tmp_path: Path) -> None:
        tool = make_tool(tmp_path)
        fn = tool.get_tools()[1]
        assert fn() == "Empty directory."

    def test_list_subdirectory(self, tmp_path: Path) -> None:
        tool = make_tool(tmp_path)
        root = tool.workspace._root
        sub = root / "pkg"
        sub.mkdir()
        (sub / "mod.py").write_bytes(b"pass")
        fn = tool.get_tools()[1]
        result = fn("pkg")
        assert "mod.py (4 bytes)" in result

    def test_list_format_bytes(self, tmp_path: Path) -> None:
        tool = make_tool(tmp_path)
        root = tool.workspace._root
        (root / "data.bin").write_bytes(b"1234567890")
        fn = tool.get_tools()[1]
        result = fn()
        assert "10 bytes" in result


# ---------------------------------------------------------------------------
# Task 4: workspace_glob (AC 7, 8)
# ---------------------------------------------------------------------------


class TestWorkspaceGlob:
    def test_glob_returns_matching_files(self, tmp_path: Path) -> None:
        tool = make_tool(tmp_path)
        root = tool.workspace._root
        (root / "a.py").write_bytes(b"a")
        (root / "b.py").write_bytes(b"b")
        (root / "c.txt").write_bytes(b"c")
        fn = tool.get_tools()[2]  # workspace_glob
        result = fn("**/*.py")
        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result

    def test_glob_sorted_by_mtime_newest_first(self, tmp_path: Path) -> None:
        tool = make_tool(tmp_path)
        root = tool.workspace._root
        old_file = root / "old.py"
        new_file = root / "new.py"
        old_file.write_bytes(b"old")
        time.sleep(0.01)
        new_file.write_bytes(b"new")
        fn = tool.get_tools()[2]
        result = fn("**/*.py")
        lines = result.split("\n")
        assert lines[0] == "new.py"
        assert lines[1] == "old.py"

    def test_glob_cap_at_max_results_with_truncation(self, tmp_path: Path) -> None:
        tool = WorkspaceReadTool(workspace_glob=WorkspaceGlob(max_results=3))
        team_id = uuid.uuid4()
        fs = Filesystem(str(tmp_path), str(team_id))
        root = fs._root
        for i in range(5):
            (root / f"f{i}.py").write_bytes(b"x")
        observer = make_observer(team_id=team_id)
        with patch("akgentic.tool.workspace.tool.get_workspace", return_value=fs):
            tool.observer(observer)
        fn = tool.get_tools()[2]
        result = fn("**/*.py")
        lines = [ln for ln in result.split("\n") if not ln.startswith("[")]
        assert len(lines) == 3
        assert "truncated" in result
        assert "5 total" in result

    def test_glob_no_files_returns_no_files_found(self, tmp_path: Path) -> None:
        tool = make_tool(tmp_path)
        fn = tool.get_tools()[2]
        assert fn("**/*.xyz") == "No files found."

    def test_glob_path_escape_raises_permission_error(self, tmp_path: Path) -> None:
        tool = make_tool(tmp_path)
        fn = tool.get_tools()[2]
        with pytest.raises(PermissionError, match="escapes workspace root"):
            fn("**/*.py", path="../../etc")

    def test_glob_no_truncation_when_under_cap(self, tmp_path: Path) -> None:
        tool = make_tool(tmp_path)
        root = tool.workspace._root
        for i in range(3):
            (root / f"f{i}.py").write_bytes(b"x")
        fn = tool.get_tools()[2]
        result = fn("**/*.py")
        assert "truncated" not in result


# ---------------------------------------------------------------------------
# Task 5: _grep_python helper
# ---------------------------------------------------------------------------


class TestGrepPython:
    def test_finds_matching_lines(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("import os\nimport sys\n", encoding="utf-8")
        results = _grep_python(tmp_path, "import os", "", 100, 2000)
        assert len(results) == 1
        path, lineno, line = results[0]
        assert lineno == 1
        assert "import os" in line

    def test_respects_include_glob(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("import os\n", encoding="utf-8")
        (tmp_path / "b.txt").write_text("import os\n", encoding="utf-8")
        results = _grep_python(tmp_path, "import os", "*.py", 100, 2000)
        paths = [r[0].name for r in results]
        assert "a.py" in paths
        assert "b.txt" not in paths

    def test_no_matches_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / "x.py").write_text("nothing here\n", encoding="utf-8")
        results = _grep_python(tmp_path, "xyzzy_does_not_exist", "", 100, 2000)
        assert results == []

    def test_max_results_cap(self, tmp_path: Path) -> None:
        content = "\n".join(["match"] * 10)
        (tmp_path / "a.py").write_text(content, encoding="utf-8")
        results = _grep_python(tmp_path, "match", "", 3, 2000)
        assert len(results) == 3

    def test_line_truncation(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x" * 100 + "\n", encoding="utf-8")
        results = _grep_python(tmp_path, "xxx", "", 100, 10)
        assert len(results[0][2]) <= 10

    def test_skips_directories_from_rglob(self, tmp_path: Path) -> None:
        """rglob may yield directories; they must be skipped gracefully."""
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "a.py").write_text("match\n", encoding="utf-8")
        results = _grep_python(tmp_path, "match", "", 100, 2000)
        # Only the file should produce results, not the directory
        assert all(r[0].is_file() for r in results)

    def test_skips_unreadable_files(self, tmp_path: Path) -> None:
        """OSError during read_text must be swallowed and the file skipped."""
        good = tmp_path / "good.py"
        bad = tmp_path / "bad.py"
        good.write_text("match\n", encoding="utf-8")
        bad.write_text("match\n", encoding="utf-8")
        original_read_text = Path.read_text

        def patched_read_text(self: Path, **kwargs: object) -> str:  # type: ignore[misc]
            if self.name == "bad.py":
                raise OSError("permission denied")
            return original_read_text(self, **kwargs)  # type: ignore[arg-type]

        with patch.object(Path, "read_text", patched_read_text):
            results = _grep_python(tmp_path, "match", "", 100, 2000)
        result_names = [r[0].name for r in results]
        assert "good.py" in result_names
        assert "bad.py" not in result_names


# ---------------------------------------------------------------------------
# Task 5: _grep_rg helper
# ---------------------------------------------------------------------------


class TestGrepRg:
    def test_returns_none_when_rg_not_on_path(self, tmp_path: Path) -> None:
        with patch("shutil.which", return_value=None):
            result = _grep_rg(tmp_path, "pattern", "", 100)
        assert result is None

    def test_returns_none_on_subprocess_error(self, tmp_path: Path) -> None:
        with patch("shutil.which", return_value="/usr/bin/rg"), patch(
            "subprocess.run", side_effect=OSError("no rg")
        ):
            result = _grep_rg(tmp_path, "pattern", "", 100)
        assert result is None

    def test_returns_none_on_timeout(self, tmp_path: Path) -> None:
        import subprocess as _subprocess

        with patch("shutil.which", return_value="/usr/bin/rg"), patch(
            "subprocess.run",
            side_effect=_subprocess.TimeoutExpired(cmd=["rg"], timeout=15),
        ):
            result = _grep_rg(tmp_path, "pattern", "", 100)
        assert result is None

    def test_returns_none_on_nonzero_returncode(self, tmp_path: Path) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 2
        mock_result.stdout = ""
        with patch("shutil.which", return_value="/usr/bin/rg"), patch(
            "subprocess.run", return_value=mock_result
        ):
            result = _grep_rg(tmp_path, "pattern", "", 100)
        assert result is None

    def test_parses_rg_output_into_tuples(self, tmp_path: Path) -> None:
        fake_file = tmp_path / "x.py"
        fake_file.write_bytes(b"")
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = f"{fake_file}:3:import os\n"
        with patch("shutil.which", return_value="/usr/bin/rg"), patch(
            "subprocess.run", return_value=mock_result
        ):
            result = _grep_rg(tmp_path, "import", "", 100)
        assert result is not None
        assert len(result) == 1
        path, lineno, line = result[0]
        assert lineno == 3
        assert line == "import os"


# ---------------------------------------------------------------------------
# Task 5: workspace_grep integration (AC 9, 10, 11)
# ---------------------------------------------------------------------------


class TestWorkspaceGrep:
    def test_grep_python_fallback_finds_matches(self, tmp_path: Path) -> None:
        tool = make_tool(tmp_path)
        root = tool.workspace._root
        (root / "main.py").write_text("import os\npass\n", encoding="utf-8")
        fn = tool.get_tools()[3]  # workspace_grep
        with patch("akgentic.tool.workspace.tool._grep_rg", return_value=None):
            result = fn("import os")
        assert "main.py" in result
        assert "import os" in result

    def test_grep_no_matches_returns_no_matches_found(self, tmp_path: Path) -> None:
        tool = make_tool(tmp_path)
        root = tool.workspace._root
        (root / "x.py").write_text("nothing\n", encoding="utf-8")
        fn = tool.get_tools()[3]
        with patch("akgentic.tool.workspace.tool._grep_rg", return_value=None):
            result = fn("xyzzy_not_found")
        assert result == "No matches found."

    def test_grep_with_include_filter(self, tmp_path: Path) -> None:
        tool = make_tool(tmp_path)
        root = tool.workspace._root
        (root / "a.py").write_text("needle\n", encoding="utf-8")
        (root / "b.txt").write_text("needle\n", encoding="utf-8")
        fn = tool.get_tools()[3]
        with patch("akgentic.tool.workspace.tool._grep_rg", return_value=None):
            result = fn("needle", include="*.py")
        assert "a.py" in result
        assert "b.txt" not in result

    def test_grep_path_escape_raises_permission_error(self, tmp_path: Path) -> None:
        tool = make_tool(tmp_path)
        fn = tool.get_tools()[3]
        with pytest.raises(PermissionError, match="escapes workspace root"):
            fn("pattern", path="../../etc")

    def test_grep_uses_rg_when_available(self, tmp_path: Path) -> None:
        """When rg returns results, _grep_python must NOT be called."""
        tool = make_tool(tmp_path)
        root = tool.workspace._root
        fake_match = (root / "x.py", 1, "import os")
        fn = tool.get_tools()[3]
        with patch(
            "akgentic.tool.workspace.tool._grep_rg", return_value=[fake_match]
        ) as mock_rg, patch(
            "akgentic.tool.workspace.tool._grep_python"
        ) as mock_py:
            result = fn("import os")
        mock_rg.assert_called_once()
        mock_py.assert_not_called()
        assert "x.py" in result


# ---------------------------------------------------------------------------
# Capability toggling (AC 12)
# ---------------------------------------------------------------------------


class TestCapabilityToggling:
    def test_all_enabled_returns_four_tools(self, tmp_path: Path) -> None:
        tool = make_tool(tmp_path)
        assert len(tool.get_tools()) == 4

    def test_glob_and_grep_disabled_returns_two_tools(self, tmp_path: Path) -> None:
        team_id = uuid.uuid4()
        fs = Filesystem(str(tmp_path), str(team_id))
        observer = make_observer(team_id=team_id)
        tool = WorkspaceReadTool(workspace_glob=False, workspace_grep=False)
        with patch("akgentic.tool.workspace.tool.get_workspace", return_value=fs):
            tool.observer(observer)
        tools = tool.get_tools()
        assert len(tools) == 2
        names = [t.__name__ for t in tools]
        assert "workspace_read" in names
        assert "workspace_list" in names

    def test_all_disabled_returns_empty_list(self, tmp_path: Path) -> None:
        team_id = uuid.uuid4()
        fs = Filesystem(str(tmp_path), str(team_id))
        observer = make_observer(team_id=team_id)
        tool = WorkspaceReadTool(
            workspace_read=False,
            workspace_list=False,
            workspace_glob=False,
            workspace_grep=False,
        )
        with patch("akgentic.tool.workspace.tool.get_workspace", return_value=fs):
            tool.observer(observer)
        assert tool.get_tools() == []

    def test_single_capability_enabled(self, tmp_path: Path) -> None:
        team_id = uuid.uuid4()
        fs = Filesystem(str(tmp_path), str(team_id))
        observer = make_observer(team_id=team_id)
        tool = WorkspaceReadTool(
            workspace_read=False,
            workspace_list=False,
            workspace_glob=WorkspaceGlob(),
            workspace_grep=False,
        )
        with patch("akgentic.tool.workspace.tool.get_workspace", return_value=fs):
            tool.observer(observer)
        tools = tool.get_tools()
        assert len(tools) == 1
        assert tools[0].__name__ == "workspace_glob"


# ---------------------------------------------------------------------------
# Story 5.6: workspace_list depth variants
# ---------------------------------------------------------------------------


class TestWorkspaceListDepth:
    def test_depth_1_returns_flat_list_format(self, tmp_path: Path) -> None:
        tool = make_tool(tmp_path)
        root = tool.workspace._root
        (root / "src").mkdir()
        (root / "README.md").write_bytes(b"hello")
        fn = tool.get_tools()[1]  # workspace_list
        result = fn()  # default depth=1
        # Flat format: no tree connectors
        assert "src/" in result
        assert "README.md (5 bytes)" in result
        assert "├──" not in result
        assert "└──" not in result

    def test_depth_2_returns_ascii_tree(self, tmp_path: Path) -> None:
        tool = make_tool(tmp_path)
        root = tool.workspace._root
        src = root / "src"
        src.mkdir()
        (src / "main.py").write_bytes(b"pass")
        fn = tool.get_tools()[1]
        result = fn(depth=2)
        assert result.startswith(".")
        assert "src/" in result
        assert "main.py" in result
        # Tree connectors present
        assert "├──" in result or "└──" in result

    def test_depth_0_returns_unlimited_tree(self, tmp_path: Path) -> None:
        tool = make_tool(tmp_path)
        root = tool.workspace._root
        deep = root / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "file.txt").write_bytes(b"x")
        fn = tool.get_tools()[1]
        result = fn(depth=0)
        assert result.startswith(".")
        assert "file.txt" in result
        assert "a/" in result
        assert "b/" in result
        assert "c/" in result

    def test_depth_tree_ordering_dirs_before_files(self, tmp_path: Path) -> None:
        tool = make_tool(tmp_path)
        root = tool.workspace._root
        (root / "zfile.txt").write_bytes(b"z")
        (root / "adir").mkdir()
        fn = tool.get_tools()[1]
        result = fn(depth=2)
        lines = result.split("\n")
        # First non-root entry should be the directory
        entry_lines = [line for line in lines if "├──" in line or "└──" in line]
        assert "adir/" in entry_lines[0]
        assert "zfile.txt" in entry_lines[1]

    def test_empty_directory_any_depth_returns_empty(self, tmp_path: Path) -> None:
        tool = make_tool(tmp_path)
        fn = tool.get_tools()[1]
        assert fn(depth=2) == "Empty directory."
        assert fn(depth=0) == "Empty directory."
