"""Tests for SeatbeltSandboxActor — macOS Apple Seatbelt sandbox execution.

Covers Story 8.3 (AC: 1–7):
- _start_sandbox() raises RuntimeError when sandbox-exec not on PATH (AC2)
- _start_sandbox() emits DeprecationWarning with correct message (AC1)
- _start_sandbox() creates workspace directory (AC1)
- _start_sandbox() is idempotent — exist_ok=True (AC1)
- _stop_sandbox() is a no-op (AC3)
- _exec() writes SBPL policy and calls sandbox-exec -f <file> (AC4)
- _exec() policy contains (deny default) (AC5)
- _exec() policy substitutes actual workspace path (AC4, AC5)
- _exec() policy contains (deny network*) (AC5)
- _exec() policy contains file-read* and file-write* for workspace (AC5)
- _exec() deletes temp file in finally block (AC4)
- _exec() returns ExecResult with correct fields (AC4)
- _exec() does NOT pass preexec_fn to subprocess.run (AC6)
"""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from akgentic.tool.sandbox.actor import ExecResult, SandboxConfig, SandboxState
from akgentic.tool.sandbox.seatbelt import SeatbeltSandboxActor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def actor(tmp_path: Path) -> SeatbeltSandboxActor:
    """Return a SeatbeltSandboxActor with workspace resolved to a temp directory.

    Uses ``SeatbeltSandboxActor.__new__`` to bypass Pykka's actor instantiation.
    Config and state are set directly, following the same pattern as
    ``test_bwrap_sandbox.py``.
    """
    a: SeatbeltSandboxActor = SeatbeltSandboxActor.__new__(SeatbeltSandboxActor)
    a.config = SandboxConfig(name="sandbox", role="ToolActor", team_id="test-team")
    a.state = SandboxState()
    a.state.workspace_path = tmp_path
    return a


# ---------------------------------------------------------------------------
# AC2: _start_sandbox() raises RuntimeError when sandbox-exec not on PATH
# ---------------------------------------------------------------------------


def test_start_sandbox_sandbox_exec_not_on_path_raises_runtime_error() -> None:
    """AC2: _start_sandbox() raises RuntimeError before DeprecationWarning when sandbox-exec missing."""
    a: SeatbeltSandboxActor = SeatbeltSandboxActor.__new__(SeatbeltSandboxActor)
    a.config = SandboxConfig(name="sandbox", role="ToolActor", team_id="test-team")
    a.state = SandboxState()

    with patch("akgentic.tool.sandbox.seatbelt.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="sandbox-exec not found"):
            a._start_sandbox()


# ---------------------------------------------------------------------------
# AC1: _start_sandbox() emits DeprecationWarning
# ---------------------------------------------------------------------------


def test_start_sandbox_emits_deprecation_warning() -> None:
    """AC1: _start_sandbox() emits DeprecationWarning with correct message."""
    a: SeatbeltSandboxActor = SeatbeltSandboxActor.__new__(SeatbeltSandboxActor)
    a.config = SandboxConfig(name="sandbox", role="ToolActor", team_id="test-team")
    a.state = SandboxState()

    with (
        patch("akgentic.tool.sandbox.seatbelt.shutil.which", return_value="/usr/bin/sandbox-exec"),
        patch("pathlib.Path.mkdir"),
        patch("akgentic.tool.sandbox.actor.SandboxState.notify_state_change"),
        pytest.warns(DeprecationWarning, match="sandbox-exec is deprecated"),
    ):
        a._start_sandbox()


# ---------------------------------------------------------------------------
# AC1: _start_sandbox() creates workspace directory
# ---------------------------------------------------------------------------


def test_start_sandbox_creates_workspace_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC1: _start_sandbox() creates workspace under AKGENTIC_WORKSPACES_ROOT."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)

    a: SeatbeltSandboxActor = SeatbeltSandboxActor.__new__(SeatbeltSandboxActor)
    a.config = SandboxConfig(name="sandbox", role="ToolActor", team_id="test-team")
    a.state = SandboxState()

    with (
        patch("akgentic.tool.sandbox.seatbelt.shutil.which", return_value="/usr/bin/sandbox-exec"),
        patch("akgentic.tool.sandbox.actor.SandboxState.notify_state_change"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", DeprecationWarning)
        a._start_sandbox()

    expected = tmp_path / "workspaces" / "test-team"
    assert expected.exists()
    assert expected.is_dir()


# ---------------------------------------------------------------------------
# AC1: _start_sandbox() is idempotent
# ---------------------------------------------------------------------------


def test_start_sandbox_idempotent_existing_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC1: Calling _start_sandbox() twice does not raise (idempotent mkdir exist_ok=True)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)

    a: SeatbeltSandboxActor = SeatbeltSandboxActor.__new__(SeatbeltSandboxActor)
    a.config = SandboxConfig(name="sandbox", role="ToolActor", team_id="test-team")
    a.state = SandboxState()

    with (
        patch("akgentic.tool.sandbox.seatbelt.shutil.which", return_value="/usr/bin/sandbox-exec"),
        patch("akgentic.tool.sandbox.actor.SandboxState.notify_state_change"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", DeprecationWarning)
        a._start_sandbox()
        a._start_sandbox()  # Must not raise

    expected = tmp_path / "workspaces" / "test-team"
    assert expected.exists()


# ---------------------------------------------------------------------------
# AC3: _stop_sandbox() is a no-op
# ---------------------------------------------------------------------------


def test_stop_sandbox_is_noop(actor: SeatbeltSandboxActor) -> None:
    """AC3: _stop_sandbox() returns None and makes no subprocess calls."""
    with patch("akgentic.tool.sandbox.seatbelt.subprocess.run") as mock_run:
        result = actor._stop_sandbox()

    assert result is None
    mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# AC4: _exec() writes policy and calls sandbox-exec
# ---------------------------------------------------------------------------


def test_exec_writes_policy_and_calls_sandbox_exec(actor: SeatbeltSandboxActor) -> None:
    """AC4: _exec() calls subprocess.run with sandbox-exec -f <policy.sb> + cmd."""
    with patch("akgentic.tool.sandbox.seatbelt.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="out", stderr="", returncode=0)
        with patch("akgentic.tool.sandbox.seatbelt.os.unlink"):
            actor._exec("ls .", "")
        cmd_list: list[str] = mock_run.call_args[0][0]
        assert cmd_list[0] == "sandbox-exec"
        assert cmd_list[1] == "-f"
        assert cmd_list[2].endswith(".sb")
        assert "ls" in cmd_list
        assert "." in cmd_list


# ---------------------------------------------------------------------------
# AC5: Policy content — (deny default)
# ---------------------------------------------------------------------------


def test_exec_policy_contains_deny_default(actor: SeatbeltSandboxActor, tmp_path: Path) -> None:
    """AC5: SBPL policy written by _exec() contains '(deny default)'."""
    captured_policy: list[str] = []
    original_ntf = tempfile.NamedTemporaryFile

    def fake_ntf(**kwargs: object) -> object:
        f = original_ntf(**kwargs)  # type: ignore[arg-type]
        original_write = f.write

        def capturing_write(data: str) -> int:
            captured_policy.append(data)
            return original_write(data)

        f.write = capturing_write  # type: ignore[method-assign]
        return f

    with (
        patch("akgentic.tool.sandbox.seatbelt.tempfile.NamedTemporaryFile", side_effect=fake_ntf),
        patch("akgentic.tool.sandbox.seatbelt.subprocess.run") as mock_run,
        patch("akgentic.tool.sandbox.seatbelt.os.unlink"),
    ):
        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
        actor._exec("ls .", "")

    assert "(deny default)" in "".join(captured_policy)


# ---------------------------------------------------------------------------
# AC4/AC5: Policy substitutes workspace path
# ---------------------------------------------------------------------------


def test_exec_policy_substitutes_workspace_path(
    actor: SeatbeltSandboxActor, tmp_path: Path
) -> None:
    """AC4: _exec() substitutes actual workspace path into the SBPL policy."""
    captured_policy: list[str] = []
    original_ntf = tempfile.NamedTemporaryFile

    def fake_ntf(**kwargs: object) -> object:
        f = original_ntf(**kwargs)  # type: ignore[arg-type]
        original_write = f.write

        def capturing_write(data: str) -> int:
            captured_policy.append(data)
            return original_write(data)

        f.write = capturing_write  # type: ignore[method-assign]
        return f

    with (
        patch("akgentic.tool.sandbox.seatbelt.tempfile.NamedTemporaryFile", side_effect=fake_ntf),
        patch("akgentic.tool.sandbox.seatbelt.subprocess.run") as mock_run,
        patch("akgentic.tool.sandbox.seatbelt.os.unlink"),
    ):
        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
        actor._exec("ls .", "")

    assert str(tmp_path) in "".join(captured_policy)


# ---------------------------------------------------------------------------
# AC5: Policy contains (deny network*)
# ---------------------------------------------------------------------------


def test_exec_policy_denies_network(actor: SeatbeltSandboxActor, tmp_path: Path) -> None:
    """AC5: SBPL policy written by _exec() contains '(deny network*)'."""
    captured_policy: list[str] = []
    original_ntf = tempfile.NamedTemporaryFile

    def fake_ntf(**kwargs: object) -> object:
        f = original_ntf(**kwargs)  # type: ignore[arg-type]
        original_write = f.write

        def capturing_write(data: str) -> int:
            captured_policy.append(data)
            return original_write(data)

        f.write = capturing_write  # type: ignore[method-assign]
        return f

    with (
        patch("akgentic.tool.sandbox.seatbelt.tempfile.NamedTemporaryFile", side_effect=fake_ntf),
        patch("akgentic.tool.sandbox.seatbelt.subprocess.run") as mock_run,
        patch("akgentic.tool.sandbox.seatbelt.os.unlink"),
    ):
        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
        actor._exec("ls .", "")

    assert "(deny network*)" in "".join(captured_policy)


# ---------------------------------------------------------------------------
# AC5: Policy contains file-read* and file-write* for workspace
# ---------------------------------------------------------------------------


def test_exec_policy_allows_workspace_read_write(
    actor: SeatbeltSandboxActor, tmp_path: Path
) -> None:
    """AC5: SBPL policy contains file-read* and file-write* subpath entries for workspace."""
    captured_policy: list[str] = []
    original_ntf = tempfile.NamedTemporaryFile

    def fake_ntf(**kwargs: object) -> object:
        f = original_ntf(**kwargs)  # type: ignore[arg-type]
        original_write = f.write

        def capturing_write(data: str) -> int:
            captured_policy.append(data)
            return original_write(data)

        f.write = capturing_write  # type: ignore[method-assign]
        return f

    with (
        patch("akgentic.tool.sandbox.seatbelt.tempfile.NamedTemporaryFile", side_effect=fake_ntf),
        patch("akgentic.tool.sandbox.seatbelt.subprocess.run") as mock_run,
        patch("akgentic.tool.sandbox.seatbelt.os.unlink"),
    ):
        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
        actor._exec("ls .", "")

    full_policy = "".join(captured_policy)
    ws = str(tmp_path)
    assert f'(allow file-read*  (subpath "{ws}"))' in full_policy
    assert f'(allow file-write* (subpath "{ws}"))' in full_policy


# ---------------------------------------------------------------------------
# AC4: Temp file deleted in finally block
# ---------------------------------------------------------------------------


def test_exec_deletes_tempfile_in_finally(actor: SeatbeltSandboxActor, tmp_path: Path) -> None:
    """AC4: _exec() deletes the temp .sb policy file in a finally block."""
    with patch("akgentic.tool.sandbox.seatbelt.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
        with patch("akgentic.tool.sandbox.seatbelt.os.unlink") as mock_unlink:
            actor._exec("ls .", "")
            mock_unlink.assert_called_once()


# ---------------------------------------------------------------------------
# AC4: _exec() returns ExecResult with correct fields
# ---------------------------------------------------------------------------


def test_exec_returns_exec_result(actor: SeatbeltSandboxActor) -> None:
    """AC4: _exec() returns ExecResult with stdout, stderr, exit_code from subprocess.run."""
    with patch("akgentic.tool.sandbox.seatbelt.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="out", stderr="err", returncode=0)
        with patch("akgentic.tool.sandbox.seatbelt.os.unlink"):
            result = actor._exec("ls .", "")

    assert isinstance(result, ExecResult)
    assert result.stdout == "out"
    assert result.stderr == "err"
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# AC6: No preexec_fn passed to subprocess.run
# ---------------------------------------------------------------------------


def test_exec_no_preexec_fn_passed(actor: SeatbeltSandboxActor) -> None:
    """AC6: _exec() does NOT pass preexec_fn to subprocess.run (macOS: no resource.setrlimit)."""
    with patch("akgentic.tool.sandbox.seatbelt.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
        with patch("akgentic.tool.sandbox.seatbelt.os.unlink"):
            actor._exec("ls .", "")

    assert mock_run.call_args is not None
    assert "preexec_fn" not in mock_run.call_args.kwargs
