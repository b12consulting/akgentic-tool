"""Tests for akgentic.tool.workspace.workspace module (Story 5.1)."""

from __future__ import annotations

import os
import stat
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from akgentic.tool.workspace.workspace import (
    FileEntry,
    Filesystem,
    Workspace,
    get_workspace,
)

# ---------------------------------------------------------------------------
# FileEntry model
# ---------------------------------------------------------------------------


class TestFileEntry:
    def test_file_entry_fields(self) -> None:
        entry = FileEntry(name="main.py", is_dir=False, size=42)
        assert entry.name == "main.py"
        assert entry.is_dir is False
        assert entry.size == 42

    def test_file_entry_directory(self) -> None:
        entry = FileEntry(name="src", is_dir=True, size=0)
        assert entry.is_dir is True
        assert entry.size == 0


# ---------------------------------------------------------------------------
# Workspace Protocol — runtime_checkable
# ---------------------------------------------------------------------------


class TestWorkspaceProtocol:
    def test_filesystem_satisfies_workspace_protocol(self, tmp_path: Path) -> None:
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        assert isinstance(fs, Workspace)


# ---------------------------------------------------------------------------
# Filesystem construction
# ---------------------------------------------------------------------------


class TestFilesystemConstruction:
    def test_root_is_base_path_slash_workspace_name(self, tmp_path: Path) -> None:
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        assert fs._root == tmp_path / "team-1"

    def test_root_directory_is_created(self, tmp_path: Path) -> None:
        root = tmp_path / "team-1"
        assert not root.exists()
        Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        assert root.is_dir()

    def test_construction_is_idempotent(self, tmp_path: Path) -> None:
        """Creating Filesystem twice does not raise (exist_ok=True)."""
        Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        Filesystem(base_path=str(tmp_path), workspace_name="team-1")  # should not raise


# ---------------------------------------------------------------------------
# _validate_path
# ---------------------------------------------------------------------------


class TestValidatePath:
    def test_valid_relative_path_returns_resolved_path(self, tmp_path: Path) -> None:
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        result = fs._validate_path("foo/bar.txt")
        assert result == (tmp_path / "team-1" / "foo" / "bar.txt").resolve()

    def test_traversal_raises_permission_error(self, tmp_path: Path) -> None:
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        with pytest.raises(PermissionError, match="escapes workspace root"):
            fs._validate_path("../../etc/passwd")

    def test_traversal_via_double_dot_in_middle_raises(self, tmp_path: Path) -> None:
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        with pytest.raises(PermissionError):
            fs._validate_path("src/../../../../../../etc/hosts")

    def test_sibling_workspace_name_prefix_raises(self, tmp_path: Path) -> None:
        """Sibling workspace 'team-11' must not pass validation for workspace 'team-1'.

        A naive ``str.startswith`` check would incorrectly allow this because
        ``str('/workspaces/team-11').startswith('/workspaces/team-1')`` is True.
        """
        sibling = tmp_path / "team-11"
        sibling.mkdir()
        (sibling / "secret.txt").write_bytes(b"secret")
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        # Construct a path that resolves to the sibling workspace
        sibling_relative = "../team-11/secret.txt"
        with pytest.raises(PermissionError, match="escapes workspace root"):
            fs._validate_path(sibling_relative)

    def test_absolute_path_injection_raises(self, tmp_path: Path) -> None:
        """An absolute path supplied as the path argument must be rejected."""
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        with pytest.raises(PermissionError, match="escapes workspace root"):
            fs._validate_path("/etc/passwd")


# ---------------------------------------------------------------------------
# Filesystem.read
# ---------------------------------------------------------------------------


class TestFilesystemRead:
    def test_read_returns_bytes(self, tmp_path: Path) -> None:
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        root = tmp_path / "team-1"
        (root / "hello.txt").write_bytes(b"hello world")
        assert fs.read("hello.txt") == b"hello world"

    def test_read_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        with pytest.raises(FileNotFoundError):
            fs.read("nonexistent.txt")

    def test_read_validates_path(self, tmp_path: Path) -> None:
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        with pytest.raises(PermissionError):
            fs.read("../../etc/passwd")


# ---------------------------------------------------------------------------
# Filesystem.write
# ---------------------------------------------------------------------------


class TestFilesystemWrite:
    def test_write_creates_file(self, tmp_path: Path) -> None:
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        fs.write("output.txt", b"data")
        assert (tmp_path / "team-1" / "output.txt").read_bytes() == b"data"

    def test_write_creates_missing_parent_dirs(self, tmp_path: Path) -> None:
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        fs.write("src/subdir/file.py", b"# code")
        assert (tmp_path / "team-1" / "src" / "subdir" / "file.py").read_bytes() == b"# code"

    def test_write_overwrites_existing_file(self, tmp_path: Path) -> None:
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        fs.write("a.txt", b"first")
        fs.write("a.txt", b"second")
        assert (tmp_path / "team-1" / "a.txt").read_bytes() == b"second"

    def test_write_validates_path(self, tmp_path: Path) -> None:
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        with pytest.raises(PermissionError):
            fs.write("../../evil.txt", b"x")


# ---------------------------------------------------------------------------
# Filesystem.write — atomicity
# ---------------------------------------------------------------------------

# Large enough that a non-atomic write cannot land in one syscall, so a reader
# can observe a genuine prefix and not only the empty truncate window.
_PAYLOAD_SIZE = 512 * 1024

# Bounded by iteration count, never by wall-clock sleeping.  Each iteration opens
# exactly one vulnerable window, so the count — not the payload size — is what
# decides whether a regression is caught: measured against the non-atomic
# implementation, 150 iterations let it escape one run in three.
_WRITER_ITERATIONS = 1500
_THREAD_JOIN_TIMEOUT = 60.0

# The reader must be scheduled *during* the writer's truncate-to-rewrite window,
# not merely between whole writes.  The default 5 ms switch interval lets the
# writer finish a whole cycle while holding the GIL, hiding the defect.
_SWITCH_INTERVAL = 1e-5


class TestFilesystemWriteAtomicity:
    def test_concurrent_reader_never_observes_a_partial_write(self, tmp_path: Path) -> None:
        """A reader racing a writer sees only whole payloads, never a prefix.

        Against the non-atomic ``resolved.write_bytes(data)`` implementation the
        reader observes short reads inside the window between ``O_TRUNC`` and the
        final byte; against write-temp-then-rename that window does not exist.
        """
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        target = tmp_path / "team-1" / "big.bin"
        payload_a = b"a" * _PAYLOAD_SIZE
        payload_b = b"b" * _PAYLOAD_SIZE
        fs.write("big.bin", payload_a)

        finished = threading.Event()
        writer_error: list[BaseException] = []

        def writer() -> None:
            try:
                for i in range(_WRITER_ITERATIONS):
                    fs.write("big.bin", payload_b if i % 2 else payload_a)
            except BaseException as exc:  # pragma: no cover - surfaced via assertion
                writer_error.append(exc)
            finally:
                finished.set()

        thread = threading.Thread(target=writer)
        # Summaries only — never the torn bytes themselves, which would be unbounded.
        torn: list[tuple[int, bytes]] = []
        reads = 0
        previous_interval = sys.getswitchinterval()
        sys.setswitchinterval(_SWITCH_INTERVAL)
        thread.start()
        try:
            while not finished.is_set():
                observed = target.read_bytes()
                reads += 1
                if observed != payload_a and observed != payload_b:
                    torn.append((len(observed), observed[:1]))
        finally:
            thread.join(timeout=_THREAD_JOIN_TIMEOUT)
            sys.setswitchinterval(previous_interval)

        assert not thread.is_alive(), "writer thread did not finish"
        assert not writer_error, f"writer raised: {writer_error}"
        assert reads > 0, "reader never got a chance to observe the file"
        assert not torn, f"observed {len(torn)} partial reads, e.g. {torn[:5]}"

    def test_publishing_rename_stays_inside_the_target_directory(self, tmp_path: Path) -> None:
        """``os.replace`` is atomic only within one filesystem — same dir or nothing.

        The invariant asserted is the directory, not the temporary file's name:
        a cross-directory stage degrades to copy-then-unlink behind an API that
        claims atomicity, and would pass a name-shaped assertion unnoticed.
        """
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        real_replace = os.replace
        calls: list[tuple[Path, Path]] = []

        def recording_replace(src: str | Path, dst: str | Path) -> None:
            calls.append((Path(src), Path(dst)))
            real_replace(src, dst)

        with patch("akgentic.tool.workspace.workspace.os.replace", recording_replace):
            fs.write("deep/nested/file.txt", b"payload")

        assert len(calls) == 1
        src, dst = calls[0]
        assert src.parent == dst.parent
        assert (tmp_path / "team-1" / "deep" / "nested" / "file.txt").read_bytes() == b"payload"

    def test_success_path_leaves_no_temporary_file(self, tmp_path: Path) -> None:
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        fs.write("fresh/out.txt", b"data")
        directory = tmp_path / "team-1" / "fresh"
        assert sorted(child.name for child in directory.iterdir()) == ["out.txt"]

    def test_failed_publish_removes_temp_and_leaves_original_intact(self, tmp_path: Path) -> None:
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        fs.write("keep.txt", b"original")
        directory = tmp_path / "team-1"

        with patch("akgentic.tool.workspace.workspace.os.replace", side_effect=OSError("boom")):
            with pytest.raises(OSError, match="boom"):
                fs.write("keep.txt", b"replacement")

        assert sorted(child.name for child in directory.iterdir()) == ["keep.txt"]
        assert (directory / "keep.txt").read_bytes() == b"original"

    def test_overwrite_preserves_the_existing_file_mode(self, tmp_path: Path) -> None:
        """Staging must not carry ``mkstemp``'s 0600 onto the published file.

        ``os.replace`` publishes the temporary file's inode, so its mode becomes
        the target's mode — a workspace bind-mounted into a container running as
        another uid would silently become unreadable.
        """
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        fs.write("perm.txt", b"first")
        target = tmp_path / "team-1" / "perm.txt"
        target.chmod(0o640)

        fs.write("perm.txt", b"second")

        assert stat.S_IMODE(target.stat().st_mode) == 0o640
        assert target.read_bytes() == b"second"

    def test_new_file_mode_matches_a_plain_write(self, tmp_path: Path) -> None:
        """Parity with today's behaviour: whatever the umask would have produced."""
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        fs.write("via_workspace.txt", b"x")
        control = tmp_path / "team-1" / "control.txt"
        control.write_bytes(b"x")

        written = tmp_path / "team-1" / "via_workspace.txt"
        assert stat.S_IMODE(written.stat().st_mode) == stat.S_IMODE(control.stat().st_mode)


# ---------------------------------------------------------------------------
# Filesystem.delete
# ---------------------------------------------------------------------------


class TestFilesystemDelete:
    def test_delete_removes_file(self, tmp_path: Path) -> None:
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        root = tmp_path / "team-1"
        target = root / "to_delete.txt"
        target.write_bytes(b"bye")
        fs.delete("to_delete.txt")
        assert not target.exists()

    def test_delete_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        with pytest.raises(FileNotFoundError):
            fs.delete("ghost.txt")

    def test_delete_validates_path(self, tmp_path: Path) -> None:
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        with pytest.raises(PermissionError):
            fs.delete("../../etc/passwd")


# ---------------------------------------------------------------------------
# Filesystem.list
# ---------------------------------------------------------------------------


class TestFilesystemList:
    def test_list_root_returns_file_entries(self, tmp_path: Path) -> None:
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        root = tmp_path / "team-1"
        (root / "file.txt").write_bytes(b"abc")
        (root / "subdir").mkdir()
        entries = fs.list("")
        names = [e.name for e in entries]
        assert "file.txt" in names
        assert "subdir" in names

    def test_list_dirs_come_before_files(self, tmp_path: Path) -> None:
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        root = tmp_path / "team-1"
        (root / "z_file.txt").write_bytes(b"z")
        (root / "a_dir").mkdir()
        entries = fs.list("")
        assert entries[0].is_dir is True
        assert entries[0].name == "a_dir"

    def test_list_file_size_matches_content(self, tmp_path: Path) -> None:
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        root = tmp_path / "team-1"
        (root / "data.bin").write_bytes(b"12345")
        entries = fs.list("")
        file_entry = next(e for e in entries if e.name == "data.bin")
        assert file_entry.size == 5

    def test_list_directory_size_is_zero(self, tmp_path: Path) -> None:
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        root = tmp_path / "team-1"
        sub = root / "mydir"
        sub.mkdir()
        (sub / "inner.txt").write_bytes(b"content")
        entries = fs.list("")
        dir_entry = next(e for e in entries if e.name == "mydir")
        assert dir_entry.size == 0

    def test_list_empty_directory(self, tmp_path: Path) -> None:
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        entries = fs.list("")
        assert entries == []

    def test_list_subdirectory(self, tmp_path: Path) -> None:
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        root = tmp_path / "team-1"
        sub = root / "src"
        sub.mkdir()
        (sub / "main.py").write_bytes(b"pass")
        entries = fs.list("src")
        assert len(entries) == 1
        assert entries[0].name == "main.py"

    def test_list_is_non_recursive(self, tmp_path: Path) -> None:
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        root = tmp_path / "team-1"
        nested = root / "a" / "b"
        nested.mkdir(parents=True)
        (nested / "deep.txt").write_bytes(b"deep")
        entries = fs.list("")
        # Should only see "a", not "b" or "deep.txt"
        assert len(entries) == 1
        assert entries[0].name == "a"

    def test_list_validates_path(self, tmp_path: Path) -> None:
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        with pytest.raises(PermissionError):
            fs.list("../../etc")

    def test_list_alphabetical_within_dirs_and_files(self, tmp_path: Path) -> None:
        fs = Filesystem(base_path=str(tmp_path), workspace_name="team-1")
        root = tmp_path / "team-1"
        (root / "z.txt").write_bytes(b"z")
        (root / "a.txt").write_bytes(b"a")
        (root / "m_dir").mkdir()
        (root / "b_dir").mkdir()
        entries = fs.list("")
        dir_names = [e.name for e in entries if e.is_dir]
        file_names = [e.name for e in entries if not e.is_dir]
        assert dir_names == ["b_dir", "m_dir"]
        assert file_names == ["a.txt", "z.txt"]


# ---------------------------------------------------------------------------
# get_workspace factory
# ---------------------------------------------------------------------------


class TestGetWorkspace:
    def test_get_workspace_default_root_when_env_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_workspace() uses ./workspaces when AKGENTIC_WORKSPACES_ROOT is not set."""
        monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
        with patch("akgentic.tool.workspace.workspace.Filesystem") as mock_fs_cls:
            mock_instance = MagicMock()
            mock_fs_cls.return_value = mock_instance
            result = get_workspace("team-1")
            mock_fs_cls.assert_called_once_with(base_path="./workspaces", workspace_name="team-1")
            assert result is mock_instance

    def test_get_workspace_custom_root_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_workspace() uses AKGENTIC_WORKSPACES_ROOT value when set."""
        monkeypatch.setenv("AKGENTIC_WORKSPACES_ROOT", "/workspaces")
        with patch("akgentic.tool.workspace.workspace.Filesystem") as mock_fs_cls:
            mock_instance = MagicMock()
            mock_fs_cls.return_value = mock_instance
            result = get_workspace("team-1")
            mock_fs_cls.assert_called_once_with(base_path="/workspaces", workspace_name="team-1")
            assert result is mock_instance

    def test_get_workspace_custom_tmp_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_workspace() passes any arbitrary AKGENTIC_WORKSPACES_ROOT value through."""
        monkeypatch.setenv("AKGENTIC_WORKSPACES_ROOT", "/tmp/test-workspaces")
        with patch("akgentic.tool.workspace.workspace.Filesystem") as mock_fs_cls:
            mock_instance = MagicMock()
            mock_fs_cls.return_value = mock_instance
            result = get_workspace("team-2")
            mock_fs_cls.assert_called_once_with(
                base_path="/tmp/test-workspaces", workspace_name="team-2"
            )
            assert result is mock_instance

    def test_get_workspace_workspace_name_passed_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_workspace() passes workspace_name verbatim to Filesystem."""
        monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
        with patch("akgentic.tool.workspace.workspace.Filesystem") as mock_fs_cls:
            mock_instance = MagicMock()
            mock_fs_cls.return_value = mock_instance
            get_workspace("my-special-team")
            mock_fs_cls.assert_called_once_with(
                base_path="./workspaces", workspace_name="my-special-team"
            )
