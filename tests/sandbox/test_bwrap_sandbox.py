"""Tests for ``BwrapBackend`` — Linux bubblewrap sandbox execution.

The backend is constructed directly; the ``backend`` fixture points it at a
temp directory the way ``start()`` would, without needing ``bwrap`` on PATH.

Covers Story 8.2 (AC: 1–8), through the backend:
- ``start()`` raises RuntimeError when bwrap not on PATH (AC2)
- ``start()`` creates the workspace directory and is idempotent (AC1)
- ``stop()`` spawns nothing (AC3)
- ``exec()`` builds the bwrap command with all required flags (AC4)
- ``exec()`` passes a preexec_fn and declares the process group it makes (AC5)
- ``exec()`` passes a minimal PATH-only env dict (AC5)
- ``exec()`` uses /workspace as cwd when cwd="" and appends cwd otherwise (AC4)
- ``exec()`` returns ExecResult with correct fields (AC4)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from akgentic.tool.sandbox.backend import ExecResult
from akgentic.tool.sandbox.bwrap import BwrapBackend

# The exec path runs through ``ProcessBackend._run``, so every exec-path mock
# targets ``backend.subprocess.Popen``. The start path is unchanged and still
# lives in ``bwrap.py``.
POPEN = "akgentic.tool.sandbox.backend.subprocess.Popen"


def popen_mock(
    mock_popen: MagicMock, stdout: str = "", stderr: str = "", returncode: int = 0
) -> MagicMock:
    """Shape *mock_popen* like a ``Popen``: ``communicate()`` pair plus ``returncode``."""
    proc = mock_popen.return_value
    proc.communicate.return_value = (stdout, stderr)
    proc.returncode = returncode
    proc.pid = 4242
    return proc

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backend(tmp_path: Path) -> BwrapBackend:
    """Return a BwrapBackend with its workspace resolved to a temp directory.

    Pointed at *tmp_path* directly rather than through ``start()``, so no
    ``bwrap`` binary has to be present for the ``exec()`` argv to be asserted.
    """
    b = BwrapBackend()
    b.workspace_path = tmp_path
    return b


# ---------------------------------------------------------------------------
# AC2: start() raises RuntimeError when bwrap not on PATH
# ---------------------------------------------------------------------------


def test_start_bwrap_not_on_path_raises_runtime_error() -> None:
    """AC2: start() raises RuntimeError with install hints when bwrap missing."""
    with patch("akgentic.tool.sandbox.bwrap.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="bwrap not found"):
            BwrapBackend().start("test-team")


def test_start_bwrap_not_on_path_error_contains_apt_hint() -> None:
    """AC2: RuntimeError message includes 'apt install bubblewrap'."""
    with patch("akgentic.tool.sandbox.bwrap.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="apt install bubblewrap"):
            BwrapBackend().start("test-team")


def test_start_bwrap_not_on_path_error_contains_dnf_hint() -> None:
    """AC2: RuntimeError message includes 'dnf install bubblewrap'."""
    with patch("akgentic.tool.sandbox.bwrap.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="dnf install bubblewrap"):
            BwrapBackend().start("test-team")


# ---------------------------------------------------------------------------
# AC1: start() creates workspace directory
# ---------------------------------------------------------------------------


def test_start_creates_workspace_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC1: start() creates workspace under AKGENTIC_WORKSPACES_ROOT."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)

    with patch("akgentic.tool.sandbox.bwrap.shutil.which", return_value="/usr/bin/bwrap"):
        BwrapBackend().start("test-team")

    expected = tmp_path / "workspaces" / "test-team"
    assert expected.exists()
    assert expected.is_dir()


def test_start_stores_resolved_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC1: start() stores the resolved absolute path in ``workspace_path``."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    b = BwrapBackend()

    with patch("akgentic.tool.sandbox.bwrap.shutil.which", return_value="/usr/bin/bwrap"):
        b.start("test-team")

    assert b.workspace_path is not None
    assert b.workspace_path.is_absolute()
    assert b.workspace_path == (tmp_path / "workspaces" / "test-team").resolve()


# ---------------------------------------------------------------------------
# AC1: start() is idempotent (exist_ok=True)
# ---------------------------------------------------------------------------


def test_start_idempotent_existing_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC1: Calling start() twice does not raise (idempotent mkdir)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    b = BwrapBackend()

    with patch("akgentic.tool.sandbox.bwrap.shutil.which", return_value="/usr/bin/bwrap"):
        b.start("test-team")
        b.start("test-team")  # Must not raise

    expected = tmp_path / "workspaces" / "test-team"
    assert expected.exists()


# ---------------------------------------------------------------------------
# AC3: stop() spawns nothing
# ---------------------------------------------------------------------------


def test_stop_spawns_nothing(backend: BwrapBackend) -> None:
    """AC3: stop() returns None and spawns no process.

    ``bwrap.py`` does not import ``subprocess`` at all — the exec path lives in
    ``ProcessBackend`` — so the same fact is asserted where a process could now
    actually be spawned from.
    """
    with patch(POPEN) as mock_run:
        result = backend.stop()

    assert result is None
    mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# AC4: exec() builds correct bwrap command with all required flags
# ---------------------------------------------------------------------------


def test_exec_builds_correct_bwrap_command(backend: BwrapBackend) -> None:
    """AC4: exec() passes a bwrap command list with all required flags to the process."""
    with patch(POPEN) as mock_run:
        popen_mock(mock_run)
        backend.exec("ls .", "")

        cmd_list: list[str] = mock_run.call_args[0][0]  # first positional arg

    assert cmd_list[0] == "bwrap"
    assert "--bind" in cmd_list
    assert "--ro-bind" in cmd_list
    assert "--ro-bind-try" in cmd_list
    assert "--tmpfs" in cmd_list
    assert "--dev" in cmd_list
    assert "--proc" in cmd_list
    assert "--unshare-net" in cmd_list
    assert "--unshare-pid" in cmd_list
    assert "--die-with-parent" in cmd_list
    assert "--new-session" in cmd_list
    assert "--chdir" in cmd_list
    assert "/workspace" in cmd_list
    assert "ls" in cmd_list


# ---------------------------------------------------------------------------
# AC4: --chdir argument handling for cwd
# ---------------------------------------------------------------------------


def test_exec_cwd_empty_uses_workspace_root(backend: BwrapBackend) -> None:
    """AC4: When cwd='', --chdir is followed by '/workspace'."""
    with patch(POPEN) as mock_run:
        popen_mock(mock_run)
        backend.exec("ls .", "")

        cmd_list: list[str] = mock_run.call_args[0][0]

    chdir_idx = cmd_list.index("--chdir")
    assert cmd_list[chdir_idx + 1] == "/workspace"


def test_exec_cwd_nonempty_appends_to_workspace(backend: BwrapBackend) -> None:
    """AC4: When cwd='subdir', --chdir is followed by '/workspace/subdir'."""
    with patch(POPEN) as mock_run:
        popen_mock(mock_run)
        backend.exec("ls .", "subdir")

        cmd_list: list[str] = mock_run.call_args[0][0]

    chdir_idx = cmd_list.index("--chdir")
    assert cmd_list[chdir_idx + 1] == "/workspace/subdir"


# ---------------------------------------------------------------------------
# AC5: resource limits, the process group, and env stripping
# ---------------------------------------------------------------------------


def test_exec_passes_preexec_fn_to_subprocess(backend: BwrapBackend) -> None:
    """AC5: exec() passes a non-None callable preexec_fn to the process."""
    with patch(POPEN) as mock_run:
        popen_mock(mock_run)
        backend.exec("ls .", "")

        call_kwargs = mock_run.call_args
    assert call_kwargs is not None
    preexec_fn = call_kwargs.kwargs.get("preexec_fn")
    assert preexec_fn is not None
    assert callable(preexec_fn)


def test_exec_declares_the_process_group_while_the_run_is_in_flight(
    backend: BwrapBackend,
) -> None:
    """The group ``_make_preexec`` creates is declared to ``_run``, for the whole run.

    bwrap builds its own argv and makes its own ``_run`` call, so it has its own
    flag to forget — and a forgotten flag is a group nothing ever signals.
    """
    seen: list[bool] = []

    with patch(POPEN) as mock_run:
        proc = popen_mock(mock_run)

        def communicate(timeout: float | None = None) -> tuple[str, str]:
            seen.append(backend._leads_a_group)
            return ("", "")

        proc.communicate.side_effect = communicate
        backend.exec("ls .", "")

    assert seen == [True]
    assert backend._leads_a_group is False


def test_exec_strips_env_to_minimal_path(backend: BwrapBackend) -> None:
    """AC5: exec() passes minimal PATH-only env dict to the process."""
    with patch(POPEN) as mock_run:
        popen_mock(mock_run)
        backend.exec("ls .", "")

        call_kwargs = mock_run.call_args
    assert call_kwargs is not None
    env = call_kwargs.kwargs.get("env")
    assert env == {"PATH": "/usr/bin:/bin:/usr/local/bin"}
    # The budget is an argument to communicate(), not to the constructor.
    assert mock_run.return_value.communicate.call_args.kwargs["timeout"] == 30


# ---------------------------------------------------------------------------
# AC4: exec() returns ExecResult with correct fields
# ---------------------------------------------------------------------------


def test_exec_returns_exec_result(backend: BwrapBackend) -> None:
    """AC4: exec() returns ExecResult with stdout, stderr, exit_code from the process."""
    with patch(POPEN) as mock_run:
        popen_mock(mock_run, stdout="out", stderr="err")
        result = backend.exec("ls .", "")

    assert isinstance(result, ExecResult)
    assert result.stdout == "out"
    assert result.stderr == "err"
    assert result.exit_code == 0


def test_exec_returns_exec_result_nonzero_exit(backend: BwrapBackend) -> None:
    """AC4: exec() correctly captures non-zero exit codes."""
    with patch(POPEN) as mock_run:
        popen_mock(mock_run, stderr="error output", returncode=1)
        result = backend.exec("ls .", "")

    assert result.exit_code == 1
    assert result.stderr == "error output"


# ---------------------------------------------------------------------------
# Story 46.1 — argument tokenisation (AC1)
# ---------------------------------------------------------------------------


def test_exec_keeps_a_quoted_argument_whole_after_the_bwrap_prefix(
    backend: BwrapBackend,
) -> None:
    """AC1: the tokens are shlex tokens and the whole bwrap flag list still precedes them.

    The trap this guards is a fix landing in ``local.py`` only: bwrap builds its
    own argv, so it has its own ``split`` call to forget.
    """
    with patch(POPEN) as mock_run:
        popen_mock(mock_run)
        backend.exec('echo "hello world"', "")

        cmd_list: list[str] = mock_run.call_args[0][0]

    assert cmd_list[-2:] == ["echo", "hello world"]
    assert cmd_list[-4:-2] == ["--chdir", "/workspace"]  # prefix intact, still first
    assert cmd_list[0] == "bwrap"
    assert "--unshare-net" in cmd_list
