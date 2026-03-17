"""Tests for LocalSandboxActor — subprocess-based sandbox execution.

Covers AC1 through AC7 for Story 6.2 (updated for Story 6.5):
- _start_sandbox() creates workspace directory and stores absolute path
- _start_sandbox() uses AKGENTIC_WORKSPACES_ROOT (default ./workspaces)
- _start_sandbox() is idempotent (calling twice does not raise)
- _stop_sandbox() is a no-op
- _exec() with empty cwd uses state.workspace_path directly
- _exec() with non-empty cwd uses state.workspace_path / cwd
- _exec() returns ExecResult with correct stdout, stderr, exit_code
- subprocess.TimeoutExpired propagates from _exec()
- exec() public method works end-to-end through _exec()
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from akgentic.tool.sandbox.actor import (
    ExecResult,
    SandboxConfig,
    SandboxState,
)
from akgentic.tool.sandbox.local import LocalSandboxActor

# ---------------------------------------------------------------------------
# Helper factory
# ---------------------------------------------------------------------------


def make_actor(team_id: str = "team-test") -> LocalSandboxActor:
    """Create a LocalSandboxActor with config and state pre-initialized (no Pykka runtime)."""
    actor = LocalSandboxActor()
    actor.config = SandboxConfig(name="sandbox", role="ToolActor", team_id=team_id)
    actor.state = SandboxState()
    actor.state.observer(actor)
    return actor


# ---------------------------------------------------------------------------
# AC1: _start_sandbox() creates workspace and stores absolute path
# ---------------------------------------------------------------------------


def test_start_sandbox_creates_workspace_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC1: _start_sandbox() creates ./workspaces/team-1/ relative to CWD (default root)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    actor = make_actor(team_id="team-1")

    actor._start_sandbox()

    expected = tmp_path / "workspaces" / "team-1"
    assert expected.exists()
    assert expected.is_dir()


def test_start_sandbox_stores_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC1: _start_sandbox() stores resolved absolute path in state.workspace_path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    actor = make_actor(team_id="team-1")

    actor._start_sandbox()

    assert actor.state.workspace_path is not None
    assert actor.state.workspace_path.is_absolute()
    assert actor.state.workspace_path == (tmp_path / "workspaces" / "team-1").resolve()


def test_start_sandbox_uses_custom_workspaces_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Story 6.5: _start_sandbox() uses AKGENTIC_WORKSPACES_ROOT when set."""
    custom_root = tmp_path / "my-custom-root"
    monkeypatch.setenv("AKGENTIC_WORKSPACES_ROOT", str(custom_root))
    actor = make_actor(team_id="team-1")

    actor._start_sandbox()

    expected = custom_root / "team-1"
    assert expected.exists()
    assert expected.is_dir()
    assert actor.state.workspace_path == expected.resolve()


def test_start_sandbox_calls_notify_state_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC1: _start_sandbox() calls state.notify_state_change() — verified via class-level patch."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    actor = make_actor(team_id="team-1")

    with patch("akgentic.tool.sandbox.actor.SandboxState.notify_state_change") as mock_notify:
        actor._start_sandbox()
        mock_notify.assert_called_once()


# ---------------------------------------------------------------------------
# AC2: _start_sandbox() is idempotent
# ---------------------------------------------------------------------------


def test_start_sandbox_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AC2: Calling _start_sandbox() twice does not raise."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    actor = make_actor(team_id="team-1")

    actor._start_sandbox()
    actor._start_sandbox()  # Must not raise

    expected = tmp_path / "workspaces" / "team-1"
    assert expected.exists()


# ---------------------------------------------------------------------------
# AC3: _stop_sandbox() is a no-op
# ---------------------------------------------------------------------------


def test_stop_sandbox_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AC3: _stop_sandbox() returns None and has no side effects."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    actor = make_actor(team_id="team-1")
    actor._start_sandbox()

    result = actor._stop_sandbox()

    assert result is None


def test_stop_sandbox_does_not_remove_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC3: _stop_sandbox() leaves the workspace directory intact."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    actor = make_actor(team_id="team-1")
    actor._start_sandbox()

    actor._stop_sandbox()

    expected = tmp_path / "workspaces" / "team-1"
    assert expected.exists()


# ---------------------------------------------------------------------------
# AC4: _exec() with empty cwd uses state.workspace_path
# ---------------------------------------------------------------------------


@patch("akgentic.tool.sandbox.local.subprocess.run")
def test_exec_no_cwd_uses_workspace_path(
    mock_run: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC4: _exec(cmd, cwd='') passes state.workspace_path as cwd to subprocess.run."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    actor = make_actor(team_id="team-1")
    actor._start_sandbox()

    mock_run.return_value = MagicMock(stdout="output", stderr="", returncode=0)

    actor._exec("pytest tests/", "")

    mock_run.assert_called_once_with(
        ["pytest", "tests/"],
        cwd=str(actor.state.workspace_path),
        capture_output=True,
        text=True,
        timeout=30,
    )


# ---------------------------------------------------------------------------
# AC5: _exec() with non-empty cwd uses state.workspace_path / cwd
# ---------------------------------------------------------------------------


@patch("akgentic.tool.sandbox.local.subprocess.run")
def test_exec_with_cwd_appends_to_workspace_path(
    mock_run: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC5: _exec(cmd, cwd='src') passes state.workspace_path / 'src' as cwd."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    actor = make_actor(team_id="team-1")
    actor._start_sandbox()

    mock_run.return_value = MagicMock(stdout="output", stderr="", returncode=0)

    actor._exec("pytest tests/", "src")

    expected_cwd = str(actor.state.workspace_path / "src")
    mock_run.assert_called_once_with(
        ["pytest", "tests/"],
        cwd=expected_cwd,
        capture_output=True,
        text=True,
        timeout=30,
    )


# ---------------------------------------------------------------------------
# AC4 / AC5: _exec() returns ExecResult with correct fields
# ---------------------------------------------------------------------------


@patch("akgentic.tool.sandbox.local.subprocess.run")
def test_exec_returns_exec_result(
    mock_run: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_exec() returns ExecResult with stdout, stderr, exit_code from subprocess.run."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    actor = make_actor(team_id="team-1")
    actor._start_sandbox()

    mock_run.return_value = MagicMock(stdout="test passed", stderr="warning", returncode=0)

    result = actor._exec("pytest tests/", "")

    assert isinstance(result, ExecResult)
    assert result.stdout == "test passed"
    assert result.stderr == "warning"
    assert result.exit_code == 0


@patch("akgentic.tool.sandbox.local.subprocess.run")
def test_exec_captures_non_zero_exit_code(
    mock_run: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_exec() correctly captures non-zero exit codes from subprocess.run."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    actor = make_actor(team_id="team-1")
    actor._start_sandbox()

    mock_run.return_value = MagicMock(stdout="", stderr="test failed", returncode=1)

    result = actor._exec("pytest tests/", "")

    assert result.exit_code == 1
    assert result.stderr == "test failed"


# ---------------------------------------------------------------------------
# AC6: subprocess.TimeoutExpired propagates
# ---------------------------------------------------------------------------


@patch("akgentic.tool.sandbox.local.subprocess.run")
def test_exec_timeout_propagates(
    mock_run: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC6: subprocess.TimeoutExpired from subprocess.run propagates out of _exec()."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    actor = make_actor(team_id="team-1")
    actor._start_sandbox()

    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["sleep", "999"], timeout=30)

    with pytest.raises(subprocess.TimeoutExpired):
        actor._exec("sleep 999", "")


# ---------------------------------------------------------------------------
# AC7: exec() public method works end-to-end
# ---------------------------------------------------------------------------


@patch("akgentic.tool.sandbox.local.subprocess.run")
def test_public_exec_method_end_to_end(
    mock_run: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC7: actor.exec() through inherited SandboxActor.exec() delegates to _exec()."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AKGENTIC_WORKSPACES_ROOT", raising=False)
    actor = make_actor(team_id="team-1")
    actor.on_start()  # Use the full lifecycle (sets state via SandboxActor.on_start)

    mock_run.return_value = MagicMock(stdout="passed", stderr="", returncode=0)

    result = actor.exec("pytest tests/")

    assert isinstance(result, ExecResult)
    assert result.stdout == "passed"
    assert result.exit_code == 0
    mock_run.assert_called_once()
