"""Tests for ``SeatbeltBackend`` — macOS Apple Seatbelt sandbox execution.

The backend is constructed directly; the ``backend`` fixture points it at a
temp directory the way ``start()`` would, without needing ``sandbox-exec``.

Covers Story 8.3 (AC: 1–7), through the backend:
- ``start()`` raises RuntimeError when sandbox-exec not on PATH (AC2)
- ``start()`` emits DeprecationWarning with correct message (AC1)
- ``start()`` creates the workspace directory and is idempotent (AC1)
- ``stop()`` spawns nothing (AC3)
- ``exec()`` writes an SBPL policy and calls sandbox-exec -f <file> (AC4)
- the policy: (deny default), the substituted workspace path, network allowed,
  reads allowed, writes confined to the workspace (AC5)
- ``exec()`` deletes the temp file in a finally block (AC4)
- ``exec()`` returns ExecResult with correct fields (AC4)
- ``exec()`` passes no preexec_fn, makes no process group, and a kill or a
  timeout therefore signals the direct child (AC6)
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from akgentic.tool.sandbox.backend import ExecResult
from akgentic.tool.sandbox.seatbelt import SeatbeltBackend

# The exec path runs through ``ProcessBackend._run``. The start-path probe
# (``sandbox_apply``) still runs ``seatbelt.subprocess.run`` and is untouched.
POPEN = "akgentic.tool.sandbox.backend.subprocess.Popen"
KILLPG = "akgentic.tool.sandbox.backend.os.killpg"


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
def backend(tmp_path: Path) -> SeatbeltBackend:
    """Return a SeatbeltBackend with its workspace resolved to a temp directory.

    Pointed at *tmp_path* directly rather than through ``start()``, so neither
    ``sandbox-exec`` nor a working ``sandbox_apply`` has to be present for the
    ``exec()`` argv and policy to be asserted.
    """
    b = SeatbeltBackend()
    b.workspace_path = tmp_path
    return b


# ---------------------------------------------------------------------------
# AC2: start() raises RuntimeError when sandbox-exec not on PATH
# ---------------------------------------------------------------------------


def test_start_sandbox_exec_not_on_path_raises_runtime_error() -> None:
    """AC2: RuntimeError raised before DeprecationWarning when sandbox-exec is missing."""
    with patch("akgentic.tool.sandbox.seatbelt.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="sandbox-exec not found"):
            SeatbeltBackend().start("test-team")


# ---------------------------------------------------------------------------
# AC1: start() emits DeprecationWarning
# ---------------------------------------------------------------------------


def test_start_raises_runtime_error_when_probe_fails() -> None:
    """start() raises RuntimeError when sandbox_apply is blocked (macOS 15+)."""
    mock_probe = MagicMock(returncode=71)
    with (
        patch("akgentic.tool.sandbox.seatbelt.shutil.which", return_value="/usr/bin/sandbox-exec"),
        patch("akgentic.tool.sandbox.seatbelt.platform.system", return_value="Darwin"),
        patch("akgentic.tool.sandbox.seatbelt.subprocess.run", return_value=mock_probe),
    ):
        with pytest.raises(RuntimeError, match="sandbox_apply is blocked"):
            SeatbeltBackend().start("test-team")


def test_start_emits_deprecation_warning() -> None:
    """AC1: start() emits DeprecationWarning with correct message."""
    mock_probe = MagicMock(returncode=0)
    with (
        patch("akgentic.tool.sandbox.seatbelt.shutil.which", return_value="/usr/bin/sandbox-exec"),
        patch("akgentic.tool.sandbox.seatbelt.platform.system", return_value="Darwin"),
        patch("akgentic.tool.sandbox.seatbelt.subprocess.run", return_value=mock_probe),
        patch("pathlib.Path.mkdir"),
        pytest.warns(DeprecationWarning, match="sandbox-exec is deprecated"),
    ):
        SeatbeltBackend().start("test-team")


# ---------------------------------------------------------------------------
# AC1: start() creates workspace directory
# ---------------------------------------------------------------------------


def test_start_creates_workspace_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC1: start() creates workspace under AKGENTIC_WORKSPACES_ROOT."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)

    mock_probe = MagicMock(returncode=0)
    with (
        patch("akgentic.tool.sandbox.seatbelt.shutil.which", return_value="/usr/bin/sandbox-exec"),
        patch("akgentic.tool.sandbox.seatbelt.platform.system", return_value="Darwin"),
        patch("akgentic.tool.sandbox.seatbelt.subprocess.run", return_value=mock_probe),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", DeprecationWarning)
        SeatbeltBackend().start("test-team")

    expected = tmp_path / "workspaces" / "test-team"
    assert expected.exists()
    assert expected.is_dir()


# ---------------------------------------------------------------------------
# AC1: start() is idempotent
# ---------------------------------------------------------------------------


def test_start_idempotent_existing_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC1: Calling start() twice does not raise (idempotent mkdir exist_ok=True)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    b = SeatbeltBackend()

    mock_probe = MagicMock(returncode=0)
    with (
        patch("akgentic.tool.sandbox.seatbelt.shutil.which", return_value="/usr/bin/sandbox-exec"),
        patch("akgentic.tool.sandbox.seatbelt.platform.system", return_value="Darwin"),
        patch("akgentic.tool.sandbox.seatbelt.subprocess.run", return_value=mock_probe),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", DeprecationWarning)
        b.start("test-team")
        b.start("test-team")  # Must not raise

    expected = tmp_path / "workspaces" / "test-team"
    assert expected.exists()


# ---------------------------------------------------------------------------
# AC3: stop() spawns nothing
# ---------------------------------------------------------------------------


def test_stop_spawns_nothing(backend: SeatbeltBackend) -> None:
    """AC3: stop() completes without error and makes no subprocess calls.

    Both targets are asserted: ``seatbelt.subprocess.run`` still exists for the
    start-path probe, and ``backend.subprocess.Popen`` is where the exec path
    spawns from.
    """
    with (
        patch("akgentic.tool.sandbox.seatbelt.subprocess.run") as mock_run,
        patch(POPEN) as mock_popen,
    ):
        backend.stop()

    mock_run.assert_not_called()
    mock_popen.assert_not_called()


# ---------------------------------------------------------------------------
# AC4: exec() writes policy and calls sandbox-exec
# ---------------------------------------------------------------------------


def test_exec_writes_policy_and_calls_sandbox_exec(backend: SeatbeltBackend) -> None:
    """AC4: exec() starts sandbox-exec -f <policy.sb> + cmd."""
    with patch(POPEN) as mock_run:
        popen_mock(mock_run, stdout="out")
        with patch("akgentic.tool.sandbox.seatbelt.os.unlink"):
            backend.exec("ls .", "")
        cmd_list: list[str] = mock_run.call_args[0][0]
        assert cmd_list[0] == "sandbox-exec"
        assert cmd_list[1] == "-f"
        assert cmd_list[2].endswith(".sb")
        assert "ls" in cmd_list
        assert "." in cmd_list


# ---------------------------------------------------------------------------
# cwd forwarding — empty cwd defaults to workspace, explicit cwd is used
# ---------------------------------------------------------------------------


def test_exec_empty_cwd_defaults_to_workspace(backend: SeatbeltBackend, tmp_path: Path) -> None:
    """exec() passes workspace_path as cwd when cwd is empty string."""
    with patch(POPEN) as mock_run:
        popen_mock(mock_run)
        with patch("akgentic.tool.sandbox.seatbelt.os.unlink"):
            backend.exec("ls .", "")
    assert mock_run.call_args.kwargs["cwd"] == str(tmp_path)


def test_exec_explicit_cwd_is_resolved_relative_to_workspace(
    backend: SeatbeltBackend, tmp_path: Path
) -> None:
    """exec() resolves cwd as a subdirectory of the workspace path."""
    with patch(POPEN) as mock_run:
        popen_mock(mock_run)
        with patch("akgentic.tool.sandbox.seatbelt.os.unlink"):
            backend.exec("ls .", "subdir")
    assert mock_run.call_args.kwargs["cwd"] == str(tmp_path / "subdir")


def test_exec_missing_cwd_returns_error_result(backend: SeatbeltBackend, tmp_path: Path) -> None:
    """exec() returns ExecResult with exit_code=1 when cwd does not exist."""
    with patch(POPEN) as mock_run:
        mock_run.side_effect = FileNotFoundError(
            2, "No such file or directory", str(tmp_path / "nope")
        )
        with patch("akgentic.tool.sandbox.seatbelt.os.unlink"):
            result = backend.exec("ls .", "nope")
    assert result.exit_code == 1
    assert "not found" in result.stderr.lower()
    assert "nope" in result.stderr
    # Must NOT leak the absolute host path
    assert str(tmp_path) not in result.stderr


# ---------------------------------------------------------------------------
# Shared helper: capture policy content written to NamedTemporaryFile
# ---------------------------------------------------------------------------


def _capture_policy(backend: SeatbeltBackend) -> str:
    """Execute backend.exec() under mocks and return the SBPL policy string written to disk.

    Uses a real NamedTemporaryFile so file I/O is faithful, patches the process
    and os.unlink to avoid side-effects. The written policy text is read back
    from the temp file before os.unlink would remove it.
    """
    policy_path_holder: list[str] = []
    original_ntf = tempfile.NamedTemporaryFile

    def fake_ntf(
        mode: str = "w",
        suffix: str | None = None,
        delete: bool = True,
    ) -> object:
        f = original_ntf(mode=mode, suffix=suffix, delete=delete)
        policy_path_holder.append(f.name)
        return f

    with (
        patch("akgentic.tool.sandbox.seatbelt.tempfile.NamedTemporaryFile", side_effect=fake_ntf),
        patch(POPEN) as mock_run,
        patch("akgentic.tool.sandbox.seatbelt.os.unlink"),
    ):
        popen_mock(mock_run)
        backend.exec("ls .", "")

    policy_path = policy_path_holder[0]
    with open(policy_path) as fh:
        content = fh.read()
    os.unlink(policy_path)
    return content


# ---------------------------------------------------------------------------
# AC5: Policy content — (deny default)
# ---------------------------------------------------------------------------


def test_exec_policy_contains_deny_default(backend: SeatbeltBackend) -> None:
    """AC5: SBPL policy written by exec() contains '(deny default)'."""
    policy = _capture_policy(backend)
    assert "(deny default)" in policy


# ---------------------------------------------------------------------------
# AC4/AC5: Policy substitutes workspace path
# ---------------------------------------------------------------------------


def test_exec_policy_substitutes_workspace_path(backend: SeatbeltBackend, tmp_path: Path) -> None:
    """AC4: exec() substitutes the actual workspace path into the SBPL policy."""
    policy = _capture_policy(backend)
    assert str(tmp_path) in policy


# ---------------------------------------------------------------------------
# AC5: Policy allows network and reads
# ---------------------------------------------------------------------------


def test_exec_policy_allows_network(backend: SeatbeltBackend) -> None:
    """SBPL policy allows network access (needed for git clone, curl, wget, pip)."""
    policy = _capture_policy(backend)
    assert "(allow network*)" in policy


def test_exec_policy_allows_all_reads(backend: SeatbeltBackend) -> None:
    """Policy uses blanket (allow file-read*) for broad read access on macOS."""
    policy = _capture_policy(backend)
    assert "(allow file-read*)" in policy


# ---------------------------------------------------------------------------
# AC5: Policy contains file-write* for workspace
# ---------------------------------------------------------------------------


def test_exec_policy_allows_workspace_write(backend: SeatbeltBackend, tmp_path: Path) -> None:
    """AC5: SBPL policy contains file-write* subpath entry for workspace."""
    policy = _capture_policy(backend)
    ws = str(tmp_path)
    assert f'(allow file-write* (subpath "{ws}"))' in policy


# ---------------------------------------------------------------------------
# AC4: Temp file deleted in finally block
# ---------------------------------------------------------------------------


def test_exec_deletes_tempfile_in_finally(backend: SeatbeltBackend) -> None:
    """AC4: exec() deletes the temp .sb policy file in a finally block."""
    with patch(POPEN) as mock_run:
        popen_mock(mock_run)
        with patch("akgentic.tool.sandbox.seatbelt.os.unlink") as mock_unlink:
            backend.exec("ls .", "")
            mock_unlink.assert_called_once()


# ---------------------------------------------------------------------------
# AC4: exec() returns ExecResult with correct fields
# ---------------------------------------------------------------------------


def test_exec_returns_exec_result(backend: SeatbeltBackend) -> None:
    """AC4: exec() returns ExecResult with stdout, stderr, exit_code from the process."""
    with patch(POPEN) as mock_run:
        popen_mock(mock_run, stdout="out", stderr="err")
        with patch("akgentic.tool.sandbox.seatbelt.os.unlink"):
            result = backend.exec("ls .", "")

    assert isinstance(result, ExecResult)
    assert result.stdout == "out"
    assert result.stderr == "err"
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# AC6: No preexec_fn, no process group — the direct child is what gets signalled
# ---------------------------------------------------------------------------


def test_exec_no_preexec_fn_passed(backend: SeatbeltBackend) -> None:
    """AC6: exec() applies no preexec_fn (macOS: no resource.setrlimit).

    ``ProcessBackend._run`` always passes the keyword through to ``Popen``, so
    the assertion is on its **value** being ``None`` rather than on the key being
    absent. ``preexec_fn=None`` is what "no preexec_fn" means to ``Popen`` — the
    same fact, at the one place the process is now started from.
    """
    with patch(POPEN) as mock_run:
        popen_mock(mock_run)
        with patch("akgentic.tool.sandbox.seatbelt.os.unlink"):
            backend.exec("ls .", "")

    assert mock_run.call_args is not None
    assert mock_run.call_args.kwargs["preexec_fn"] is None
    # And the budget, which moved to communicate() with the call shape.
    assert mock_run.return_value.communicate.call_args.kwargs["timeout"] == 30


@patch(KILLPG)
@patch(POPEN)
def test_a_timeout_signals_the_direct_child_because_there_is_no_group(
    mock_popen: MagicMock, mock_killpg: MagicMock, backend: SeatbeltBackend
) -> None:
    """AC6, the other half: with no ``preexec_fn`` there is no group to signal.

    This is the path seatbelt takes under the group-kill rule — the direct
    ``sandbox-exec`` child, through ``Popen.kill`` — and it is the same path
    docker takes. ``os.killpg`` on a child that shares this process's group
    would signal the caller, so it must not be reached here.
    """
    proc = popen_mock(mock_popen)
    proc.communicate.side_effect = [
        subprocess.TimeoutExpired(cmd=["sandbox-exec"], timeout=30),
        ("", ""),
    ]

    with patch("akgentic.tool.sandbox.seatbelt.os.unlink"):
        with pytest.raises(subprocess.TimeoutExpired):
            backend.exec("sh -c 'sleep 999'", "")

    proc.kill.assert_called_once()
    mock_killpg.assert_not_called()


# ---------------------------------------------------------------------------
# Story 46.1 — argument tokenisation (AC1)
# ---------------------------------------------------------------------------


def test_exec_keeps_a_quoted_argument_whole_after_the_policy_flags(
    backend: SeatbeltBackend,
) -> None:
    """AC1: shlex tokens follow ``sandbox-exec -f <policy>``, which is unchanged."""
    with patch(POPEN) as mock_run:
        popen_mock(mock_run)
        backend.exec('echo "hello world"', "")

        argv: list[str] = mock_run.call_args[0][0]

    assert argv[:2] == ["sandbox-exec", "-f"]
    assert argv[2].endswith(".sb")
    assert argv[3:] == ["echo", "hello world"]
