"""Tests for ExecTool — observer wiring, SANDBOX_MODE resolution, tool behaviour.

Covers AC1–AC13 for Story 6.4:
- SANDBOX_ACTOR_CLASSES dict (AC1)
- ExecTool fields (AC2)
- observer() raises ValueError when orchestrator is None (AC3)
- observer() creates LocalSandboxActor with SANDBOX_MODE=local (AC4)
- observer() creates DockerSandboxActor with SANDBOX_MODE=docker (AC5)
- observer() reuses existing actor — no second createActor call (AC6)
- observer() raises KeyError on unknown SANDBOX_MODE (AC7)
- exec_command returns formatted stdout/stderr/exit_code (AC8)
- exec_command catches CommandNotAllowedError → error string (AC9)
- get_tools() returns [] when exec_command=False (AC10)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from akgentic.core.actor_address import ActorAddress

from akgentic.tool.sandbox.actor import (
    SANDBOX_ACTOR_NAME,
    CommandNotAllowedError,
    ExecResult,
    SandboxActor,
    SandboxConfig,
)
from akgentic.tool.sandbox.docker import DockerSandboxActor
from akgentic.tool.sandbox.local import LocalSandboxActor
from akgentic.tool.sandbox.tool import SANDBOX_ACTOR_CLASSES, ExecTool

# ---------------------------------------------------------------------------
# Mock observer infrastructure
# ---------------------------------------------------------------------------


class MockObserver:
    """Minimal ActorToolObserver stub for ExecTool unit tests."""

    def __init__(
        self,
        has_orchestrator: bool = True,
        existing_actor: ActorAddress | None = None,
    ) -> None:
        self._team_id = "team-test"
        self.myAddress = MagicMock(spec=ActorAddress)
        self.orchestrator = MagicMock(spec=ActorAddress) if has_orchestrator else None

        # Set up orchestrator proxy mock
        self._orch_proxy = MagicMock()
        self._orch_proxy.get_team_member.return_value = existing_actor  # None = not found
        new_addr = MagicMock(spec=ActorAddress)
        self._orch_proxy.createActor.return_value = new_addr
        self._new_actor_addr = new_addr

    def proxy_ask(
        self,
        actor: ActorAddress,
        actor_type: object = None,
        timeout: int | None = None,
    ) -> object:
        if actor is self.orchestrator:
            return self._orch_proxy
        return MagicMock()  # sandbox proxy

    def notify_event(self, event: object) -> None:
        pass


# ---------------------------------------------------------------------------
# AC1 — SANDBOX_ACTOR_CLASSES registry
# ---------------------------------------------------------------------------


def test_sandbox_actor_classes_has_local_key() -> None:
    """AC1: SANDBOX_ACTOR_CLASSES['local'] maps to LocalSandboxActor."""
    assert "local" in SANDBOX_ACTOR_CLASSES
    assert SANDBOX_ACTOR_CLASSES["local"] is LocalSandboxActor


def test_sandbox_actor_classes_has_docker_key() -> None:
    """AC1: SANDBOX_ACTOR_CLASSES['docker'] maps to DockerSandboxActor."""
    assert "docker" in SANDBOX_ACTOR_CLASSES
    assert SANDBOX_ACTOR_CLASSES["docker"] is DockerSandboxActor


def test_sandbox_actor_classes_is_mutable_dict() -> None:
    """AC1: SANDBOX_ACTOR_CLASSES is a regular dict (mutable — injection window)."""
    assert isinstance(SANDBOX_ACTOR_CLASSES, dict)


# ---------------------------------------------------------------------------
# AC2 — ExecTool field defaults
# ---------------------------------------------------------------------------


def test_exec_tool_name_default() -> None:
    """AC2: ExecTool.name defaults to 'Exec'."""
    tool = ExecTool()
    assert tool.name == "Exec"


def test_exec_tool_description_default() -> None:
    """AC2: ExecTool.description is the expected string."""
    tool = ExecTool()
    assert tool.description == "Execute sandboxed shell commands in the team workspace"


def test_exec_tool_exec_command_default_is_true() -> None:
    """AC2: ExecTool.exec_command defaults to True."""
    tool = ExecTool()
    assert tool.exec_command is True


# ---------------------------------------------------------------------------
# AC3 — observer() raises ValueError when orchestrator is None
# ---------------------------------------------------------------------------


def test_observer_raises_value_error_when_orchestrator_is_none() -> None:
    """AC3: observer() raises ValueError when observer.orchestrator is None."""
    tool = ExecTool()
    observer = MockObserver(has_orchestrator=False)

    with pytest.raises(ValueError, match="orchestrator"):
        tool.observer(observer)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AC4 — observer() creates LocalSandboxActor when SANDBOX_MODE=local
# ---------------------------------------------------------------------------


def test_observer_creates_local_sandbox_actor(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC4: observer() with SANDBOX_MODE=local creates LocalSandboxActor."""
    monkeypatch.setenv("SANDBOX_MODE", "local")
    observer = MockObserver(existing_actor=None)
    tool = ExecTool()

    tool.observer(observer)  # type: ignore[arg-type]

    observer._orch_proxy.createActor.assert_called_once()
    call_args = observer._orch_proxy.createActor.call_args
    assert call_args[0][0] is LocalSandboxActor


def test_observer_creates_actor_with_correct_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC4: SandboxConfig passed to createActor has name, role, and team_id."""
    monkeypatch.setenv("SANDBOX_MODE", "local")
    observer = MockObserver(existing_actor=None)
    tool = ExecTool()

    tool.observer(observer)  # type: ignore[arg-type]

    call_kwargs = observer._orch_proxy.createActor.call_args[1]
    config: SandboxConfig = call_kwargs["config"]
    assert config.name == SANDBOX_ACTOR_NAME
    assert config.role == "ToolActor"
    assert config.team_id == "team-test"


def test_observer_stores_sandbox_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC4: observer() stores a non-None _sandbox_proxy after wiring."""
    monkeypatch.setenv("SANDBOX_MODE", "local")
    observer = MockObserver(existing_actor=None)
    tool = ExecTool()

    tool.observer(observer)  # type: ignore[arg-type]

    assert tool._sandbox_proxy is not None


# ---------------------------------------------------------------------------
# AC5 — observer() creates DockerSandboxActor when SANDBOX_MODE=docker
# ---------------------------------------------------------------------------


def test_observer_creates_docker_sandbox_actor(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC5: observer() with SANDBOX_MODE=docker creates DockerSandboxActor."""
    monkeypatch.setenv("SANDBOX_MODE", "docker")
    observer = MockObserver(existing_actor=None)
    tool = ExecTool()

    tool.observer(observer)  # type: ignore[arg-type]

    observer._orch_proxy.createActor.assert_called_once()
    call_args = observer._orch_proxy.createActor.call_args
    assert call_args[0][0] is DockerSandboxActor


# ---------------------------------------------------------------------------
# AC6 — observer() reuses existing actor — does NOT call createActor again
# ---------------------------------------------------------------------------


def test_observer_reuses_existing_actor(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC6: when #SandboxActor already exists, createActor is NOT called."""
    monkeypatch.setenv("SANDBOX_MODE", "local")
    existing_addr = MagicMock(spec=ActorAddress)
    observer = MockObserver(existing_actor=existing_addr)
    tool = ExecTool()

    tool.observer(observer)  # type: ignore[arg-type]

    observer._orch_proxy.createActor.assert_not_called()


def test_observer_second_call_reuses_actor(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC6: calling observer() a second time (actor already exists) does not create a new one."""
    monkeypatch.setenv("SANDBOX_MODE", "local")

    # First call: no existing actor → creates one
    observer1 = MockObserver(existing_actor=None)
    tool = ExecTool()
    tool.observer(observer1)  # type: ignore[arg-type]
    assert observer1._orch_proxy.createActor.call_count == 1

    # Second call: actor now exists
    existing_addr = MagicMock(spec=ActorAddress)
    observer2 = MockObserver(existing_actor=existing_addr)
    tool.observer(observer2)  # type: ignore[arg-type]

    observer2._orch_proxy.createActor.assert_not_called()


# ---------------------------------------------------------------------------
# AC7 — observer() raises KeyError on unknown SANDBOX_MODE
# ---------------------------------------------------------------------------


def test_observer_raises_key_error_on_unknown_sandbox_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC7: SANDBOX_MODE=unknown → KeyError (fail-fast, no error handling added)."""
    monkeypatch.setenv("SANDBOX_MODE", "unknown-backend")
    observer = MockObserver(existing_actor=None)
    tool = ExecTool()

    with pytest.raises(KeyError):
        tool.observer(observer)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AC8 — exec_command returns formatted stdout/stderr/exit_code
# ---------------------------------------------------------------------------


def test_exec_command_returns_formatted_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC8: exec_command returns 'stdout:\\n...\\nstderr:\\n...\\nexit_code: 0'."""
    monkeypatch.setenv("SANDBOX_MODE", "local")
    observer = MockObserver(existing_actor=None)
    tool = ExecTool()
    tool.observer(observer)  # type: ignore[arg-type]

    # Replace proxy with a controlled mock
    mock_proxy = MagicMock(spec=SandboxActor)
    mock_proxy.exec.return_value = ExecResult(
        stdout="===== 5 passed =====", stderr="", exit_code=0
    )
    tool._sandbox_proxy = mock_proxy

    tools = tool.get_tools()
    assert len(tools) == 1
    result = tools[0](cmd="pytest tests/ -v")

    assert "exit_code: 0" in result
    assert "5 passed" in result
    assert "stdout:" in result
    assert "stderr:" in result


def test_exec_command_includes_stderr_in_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC8: exec_command includes stderr in the returned string."""
    monkeypatch.setenv("SANDBOX_MODE", "local")
    observer = MockObserver(existing_actor=None)
    tool = ExecTool()
    tool.observer(observer)  # type: ignore[arg-type]

    mock_proxy = MagicMock(spec=SandboxActor)
    mock_proxy.exec.return_value = ExecResult(stdout="", stderr="SyntaxError", exit_code=1)
    tool._sandbox_proxy = mock_proxy

    tools = tool.get_tools()
    result = tools[0](cmd="python bad.py")

    assert "SyntaxError" in result
    assert "exit_code: 1" in result


# ---------------------------------------------------------------------------
# AC9 — exec_command catches CommandNotAllowedError → error string
# ---------------------------------------------------------------------------


def test_exec_command_catches_command_not_allowed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC9: CommandNotAllowedError is caught and returned as an error string — not raised."""
    monkeypatch.setenv("SANDBOX_MODE", "local")
    observer = MockObserver(existing_actor=None)
    tool = ExecTool()
    tool.observer(observer)  # type: ignore[arg-type]

    mock_proxy = MagicMock(spec=SandboxActor)
    mock_proxy.exec.side_effect = CommandNotAllowedError("malware not allowed")
    tool._sandbox_proxy = mock_proxy

    tools = tool.get_tools()
    result = tools[0](cmd="malware --install")

    assert "CommandNotAllowedError" in result
    assert not result.startswith("Traceback")  # must not have raised


def test_exec_command_error_string_lists_allowed_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC9: error string contains the sorted list of ALLOWED_COMMANDS."""
    monkeypatch.setenv("SANDBOX_MODE", "local")
    observer = MockObserver(existing_actor=None)
    tool = ExecTool()
    tool.observer(observer)  # type: ignore[arg-type]

    mock_proxy = MagicMock(spec=SandboxActor)
    mock_proxy.exec.side_effect = CommandNotAllowedError("malware not in allowlist")
    tool._sandbox_proxy = mock_proxy

    tools = tool.get_tools()
    result = tools[0](cmd="malware --install")

    # Should list at least some allowed binaries
    for binary in ["pytest", "python"]:
        assert binary in result


# ---------------------------------------------------------------------------
# AC10 — get_tools() returns [] when exec_command=False
# ---------------------------------------------------------------------------


def test_get_tools_returns_empty_list_when_exec_command_disabled() -> None:
    """AC10: ExecTool(exec_command=False).get_tools() returns []."""
    tool = ExecTool(exec_command=False)
    assert tool.get_tools() == []


def test_get_tools_returns_one_callable_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_tools() returns exactly one callable when exec_command=True."""
    monkeypatch.setenv("SANDBOX_MODE", "local")
    observer = MockObserver(existing_actor=None)
    tool = ExecTool()
    tool.observer(observer)  # type: ignore[arg-type]

    mock_proxy = MagicMock(spec=SandboxActor)
    tool._sandbox_proxy = mock_proxy

    tools = tool.get_tools()
    assert len(tools) == 1
    assert callable(tools[0])


# ---------------------------------------------------------------------------
# observer() return value — method chaining
# ---------------------------------------------------------------------------


def test_observer_returns_self(monkeypatch: pytest.MonkeyPatch) -> None:
    """observer() returns the ExecTool instance for method chaining."""
    monkeypatch.setenv("SANDBOX_MODE", "local")
    observer = MockObserver(existing_actor=None)
    tool = ExecTool()

    result = tool.observer(observer)  # type: ignore[arg-type]

    assert result is tool
