"""Tests for ExecTool — observer wiring, mode field, tool behaviour.

Covers AC1–AC13 for Story 6.4 (updated for Story 6.5, Story 8.4, Story 29-5):
- SANDBOX_ACTOR_CLASSES dict (AC1)
- ExecTool fields including mode (AC2)
- observer() raises ValueError when orchestrator is None (AC3)
- observer() creates LocalSandboxActor with mode="local" (AC4)
- observer() creates DockerSandboxActor with mode="docker" (AC5)
- observer() reuses existing actor — no second createActor call (AC6)
- observer() raises KeyError on unknown mode (AC7)
- exec_command returns formatted stdout/stderr/exit_code (AC8)
- exec_command catches CommandNotAllowedError → error string (AC9)
- get_tools() returns [] when exec_command=False (AC10)
- Story 6.5: mode comes from ExecTool.mode field, not SANDBOX_MODE env var
- Story 8.4: bwrap/seatbelt keys in registry, auto-mode resolution, DeprecationWarning

**Story 29-5 changed the shape of this file, and for a structural reason rather
than a behavioural one.** ``ExecTool`` is now a shim over
``WorkspaceTool(workspace_exec=...)``: it wires ``#Workspace`` as well as
``#SandboxActor``, and ``exec_command`` routes through the former rather than
calling the latter directly. So the double below answers two
``getChildrenOrCreate`` calls instead of one — ``sandbox_config_of`` picks out
the sandbox one by actor class rather than by call order — and the
``exec_command`` tests stand in a workspace proxy where they used to stand in a
sandbox proxy. Every assertion about *behaviour* is unchanged.

The end-to-end equivalence of the two surfaces, against real actors and a real
journal, is asserted in ``tests/workspace/test_exec.py``.
"""

from __future__ import annotations

import warnings
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from akgentic.core.actor_address import ActorAddress

from akgentic.tool.sandbox.actor import (
    CommandNotAllowedError,
    SandboxActor,
    SandboxConfig,
    sandbox_actor_name,
)
from akgentic.tool.sandbox.bwrap import BwrapSandboxActor
from akgentic.tool.sandbox.docker import DockerSandboxActor
from akgentic.tool.sandbox.local import LocalSandboxActor
from akgentic.tool.sandbox.seatbelt import SeatbeltSandboxActor
from akgentic.tool.sandbox.tool import SANDBOX_ACTOR_CLASSES, ExecTool, _resolve_auto_mode
from akgentic.tool.workspace.execution import (
    ExecOutcome,
    ExecStart,
    ExecState,
    ExecStatus,
)

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
        self.team_id = "team-test"
        self.myAddress = MagicMock(spec=ActorAddress)
        self.orchestrator = MagicMock(spec=ActorAddress) if has_orchestrator else None

        # Set up orchestrator proxy mock
        self._orch_proxy = MagicMock()
        if existing_actor is not None:
            self._orch_proxy.getChildrenOrCreate.return_value = existing_actor
        else:
            new_addr = MagicMock(spec=ActorAddress)
            self._orch_proxy.getChildrenOrCreate.return_value = new_addr
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

    def proxy_tell(self, actor: ActorAddress, actor_type: object = None) -> object:
        return MagicMock()

    def notify_event(self, event: object) -> None:
        pass


def sandbox_call(observer: MockObserver) -> Any:
    """Return the one ``getChildrenOrCreate`` call that created a sandbox backend.

    The shim now makes two — ``#SandboxActor`` and ``#Workspace`` — so a test
    about the *backend* has to name which one it means. Selecting by actor class
    rather than by call index keeps that true if the wiring order ever changes.
    """
    calls = [
        call
        for call in observer._orch_proxy.getChildrenOrCreate.call_args_list
        if isinstance(call[0][0], type) and issubclass(call[0][0], SandboxActor)
    ]
    assert len(calls) == 1, f"expected exactly one sandbox creation, got {len(calls)}"
    return calls[0]


def sandbox_config_of(observer: MockObserver) -> SandboxConfig:
    """Return the ``SandboxConfig`` the backend was created with."""
    config = sandbox_call(observer)[1]["config"]
    assert isinstance(config, SandboxConfig)
    return config


def wire(tool: ExecTool, observer: MockObserver) -> None:
    """Wire *tool*, swallowing the card's own deprecation warning.

    Every wiring in this file now emits it — that is the point of the shim — and
    it is asserted on its own below rather than in every unrelated test.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        tool.observer(observer)  # type: ignore[arg-type]


class FakeWorkspaceProxy:
    """Stands in for ``#Workspace`` on the ask path, for the shim's closure.

    Answers ``request_exec`` with an issued run id and ``exec_status`` with
    whatever the test set up. It is deliberately not a ``MagicMock``: the closure
    branches on ``ExecStart.run_id`` and ``ExecStatus.settled``, and a mock would
    make both truthy and prove nothing.
    """

    def __init__(self, status: ExecStatus | None = None, refusal: str = "") -> None:
        self.status = status
        self.refusal = refusal
        self.requests: list[tuple[str, str, str]] = []

    def request_exec(self, agent_id: str, cmd: str, cwd: str = "") -> ExecStart:
        self.requests.append((agent_id, cmd, cwd))
        if self.refusal:
            return ExecStart(refusal=self.refusal)
        return ExecStart(run_id="abc12345")

    def exec_status(self, agent_id: str, run_id: str) -> ExecStatus:
        assert self.status is not None
        return self.status


class RaisingWorkspaceProxy:
    """A workspace proxy whose ``request_exec`` raises — the error-string paths."""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def request_exec(self, agent_id: str, cmd: str, cwd: str = "") -> ExecStart:
        raise self.error


def done(stdout: str = "", stderr: str = "", exit_code: int = 0) -> ExecStatus:
    """A settled, successful status carrying *stdout* / *stderr* / *exit_code*."""
    return ExecStatus(
        state=ExecState.DONE,
        run_id="abc12345",
        outcome=ExecOutcome(stdout=stdout, stderr=stderr, exit_code=exit_code),
    )


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
# AC2 — ExecTool field defaults (including mode)
# ---------------------------------------------------------------------------


def test_exec_tool_exec_command_default_is_true() -> None:
    """AC2: ExecTool.exec_command defaults to True."""
    tool = ExecTool()
    assert tool.exec_command is True


def test_exec_tool_mode_defaults_to_auto() -> None:
    """Story 6.5: ExecTool.mode defaults to 'auto'."""
    tool = ExecTool()
    assert tool.mode == "auto"


def test_exec_tool_mode_can_be_set_to_docker() -> None:
    """Story 6.5: ExecTool(mode='docker') stores mode='docker'."""
    tool = ExecTool(mode="docker")
    assert tool.mode == "docker"


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
# AC4 — observer() creates LocalSandboxActor when mode="local"
# ---------------------------------------------------------------------------


def test_observer_creates_local_sandbox_actor() -> None:
    """AC4: ExecTool(mode='local').observer() creates LocalSandboxActor."""
    observer = MockObserver(existing_actor=None)
    tool = ExecTool(mode="local")

    wire(tool, observer)

    assert sandbox_call(observer)[0][0] is LocalSandboxActor


def test_observer_creates_actor_with_correct_config() -> None:
    """AC4: SandboxConfig passed to createActor has name, role, team_id, and mode."""
    observer = MockObserver(existing_actor=None)
    tool = ExecTool(mode="local")

    wire(tool, observer)

    config = sandbox_config_of(observer)
    # No workspace_id on this card, so the workspace — and therefore the actor's
    # name — falls back to the team id, exactly as Filesystem resolution does.
    assert config.name == sandbox_actor_name("team-test")
    assert config.role == "ToolActor"
    assert config.team_id == "team-test"
    assert config.mode == "local"


def test_observer_stores_sandbox_proxy() -> None:
    """AC4: observer() stores a non-None _sandbox_proxy after wiring."""
    observer = MockObserver(existing_actor=None)
    tool = ExecTool(mode="local")

    wire(tool, observer)

    assert tool._sandbox_proxy is not None
    assert tool._workspace_proxy is not None


# ---------------------------------------------------------------------------
# AC5 — observer() creates DockerSandboxActor when mode="docker"
# ---------------------------------------------------------------------------


def test_observer_creates_docker_sandbox_actor() -> None:
    """AC5: ExecTool(mode='docker').observer() creates DockerSandboxActor."""
    observer = MockObserver(existing_actor=None)
    tool = ExecTool(mode="docker")

    wire(tool, observer)

    assert sandbox_call(observer)[0][0] is DockerSandboxActor


def test_observer_creates_docker_actor_config_has_mode_docker() -> None:
    """Story 6.5: SandboxConfig for docker mode has mode='docker'."""
    observer = MockObserver(existing_actor=None)
    tool = ExecTool(mode="docker")

    wire(tool, observer)

    assert sandbox_config_of(observer).mode == "docker"


# ---------------------------------------------------------------------------
# AC6 — observer() reuses existing actor — does NOT call createActor again
# ---------------------------------------------------------------------------


def test_observer_reuses_existing_actor() -> None:
    """AC6: getChildrenOrCreate is called and returns the existing actor."""
    existing_addr = MagicMock(spec=ActorAddress)
    observer = MockObserver(existing_actor=existing_addr)
    tool = ExecTool()

    wire(tool, observer)

    sandbox_call(observer)  # exactly one sandbox creation, no matter how many cards


def test_observer_second_call_reuses_actor() -> None:
    """AC6: calling observer() a second time still calls getChildrenOrCreate."""
    # First call: no existing actor → getChildrenOrCreate creates one
    observer1 = MockObserver(existing_actor=None)
    tool = ExecTool()
    wire(tool, observer1)
    sandbox_call(observer1)

    # Second call: actor now exists — getChildrenOrCreate returns existing
    existing_addr = MagicMock(spec=ActorAddress)
    observer2 = MockObserver(existing_actor=existing_addr)
    wire(tool, observer2)

    sandbox_call(observer2)


# ---------------------------------------------------------------------------
# AC7 — observer() raises KeyError on unknown mode value
# ---------------------------------------------------------------------------


def test_observer_raises_key_error_on_unknown_mode() -> None:
    """AC7: ExecTool(mode=...) with an unregistered mode → KeyError (fail-fast)."""
    observer = MockObserver(existing_actor=None)
    # Bypass Literal validation by using object.__setattr__
    tool = ExecTool()
    object.__setattr__(tool, "mode", "unknown-backend")

    with pytest.raises(KeyError):
        wire(tool, observer)


# ---------------------------------------------------------------------------
# AC8 — exec_command returns formatted stdout/stderr/exit_code
# ---------------------------------------------------------------------------


def test_exec_command_returns_formatted_output() -> None:
    """AC8: exec_command returns 'stdout:\\n...\\nstderr:\\n...\\nexit_code: 0'."""
    observer = MockObserver(existing_actor=None)
    tool = ExecTool(mode="local")
    wire(tool, observer)

    tool._workspace_proxy = FakeWorkspaceProxy(done(stdout="===== 5 passed ====="))  # type: ignore[assignment]

    tools = tool.get_tools()
    assert len(tools) == 1
    result = tools[0](cmd="pytest tests/ -v")

    assert "exit_code: 0 (OK)" in result
    assert "5 passed" in result
    assert "stdout:" in result
    assert "stderr" in result


def test_exec_command_includes_stderr_in_output() -> None:
    """AC8: exec_command includes stderr in the returned string."""
    observer = MockObserver(existing_actor=None)
    tool = ExecTool(mode="local")
    wire(tool, observer)

    tool._workspace_proxy = FakeWorkspaceProxy(done(stderr="SyntaxError", exit_code=1))  # type: ignore[assignment]

    tools = tool.get_tools()
    result = tools[0](cmd="python bad.py")

    assert "SyntaxError" in result
    assert "exit_code: 1" in result


# ---------------------------------------------------------------------------
# AC9 — exec_command catches CommandNotAllowedError → error string
# ---------------------------------------------------------------------------


def test_exec_command_catches_command_not_allowed_error() -> None:
    """AC9: CommandNotAllowedError is caught and returned as an error string — not raised.

    **This covers the defensive branch, not the production path.** From story 29-5
    the allowlist is checked in ``SandboxActor.exec``, which runs on the worker's
    thread, so the error reaches the agent as a *reported failure* rather than as
    an exception out of ``request_exec``. What an agent is really told for
    ``git status`` is asserted end to end in
    ``tests/workspace/test_exec.py::TestADisallowedCommand``. The handler stays
    because a synchronous raise is still representable and must not escape.
    """
    observer = MockObserver(existing_actor=None)
    tool = ExecTool(mode="local")
    wire(tool, observer)

    tool._workspace_proxy = RaisingWorkspaceProxy(  # type: ignore[assignment]
        CommandNotAllowedError("malware not allowed")
    )

    tools = tool.get_tools()
    result = tools[0](cmd="malware --install")

    assert "CommandNotAllowedError" in result
    assert not result.startswith("Traceback")  # must not have raised


def test_exec_command_catches_subprocess_error() -> None:
    """AC3 (Story 8.5): When sandbox proxy raises SubprocessError, exec_command
    returns an error string instead of crashing.
    """
    import subprocess

    observer = MockObserver(existing_actor=None)
    tool = ExecTool(mode="local")
    wire(tool, observer)

    tool._workspace_proxy = RaisingWorkspaceProxy(  # type: ignore[assignment]
        subprocess.SubprocessError("Exception occurred in preexec_fn.")
    )

    tools = tool.get_tools()
    result = tools[0](cmd="echo hello")

    assert "SandboxError" in result
    assert "SubprocessError" in result
    assert "preexec_fn" in result


def test_exec_command_catches_generic_exception() -> None:
    """AC3 (Story 8.5): When sandbox proxy raises any Exception, exec_command
    returns an error string instead of raising.
    """
    observer = MockObserver(existing_actor=None)
    tool = ExecTool(mode="local")
    wire(tool, observer)

    tool._workspace_proxy = RaisingWorkspaceProxy(RuntimeError("sandbox crashed"))  # type: ignore[assignment]

    tools = tool.get_tools()
    result = tools[0](cmd="echo hello")

    assert "SandboxError" in result
    assert "RuntimeError" in result
    assert "sandbox crashed" in result


def test_exec_command_error_string_lists_allowed_commands() -> None:
    """AC9: error string contains the sorted list of ALLOWED_COMMANDS.

    Defensive branch, as above — on the production path the same list reaches the
    agent inside the reported failure, because ``CommandNotAllowedError``'s own
    message carries it.
    """
    observer = MockObserver(existing_actor=None)
    tool = ExecTool(mode="local")
    wire(tool, observer)

    tool._workspace_proxy = RaisingWorkspaceProxy(  # type: ignore[assignment]
        CommandNotAllowedError("malware not in allowlist")
    )

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


def test_get_tools_returns_one_callable_when_enabled() -> None:
    """get_tools() returns exactly one callable when exec_command=True."""
    observer = MockObserver(existing_actor=None)
    tool = ExecTool(mode="local")
    wire(tool, observer)

    tools = tool.get_tools()
    assert len(tools) == 1
    assert callable(tools[0])
    assert tools[0].__name__ == "exec_command"


# ---------------------------------------------------------------------------
# observer() return value — method chaining
# ---------------------------------------------------------------------------


def test_observer_returns_self() -> None:
    """observer() returns the ExecTool instance for method chaining."""
    observer = MockObserver(existing_actor=None)
    tool = ExecTool()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = tool.observer(observer)  # type: ignore[arg-type]

    assert result is tool


# ---------------------------------------------------------------------------
# Story 6.5: no SANDBOX_MODE env var dependency
# ---------------------------------------------------------------------------


def test_exec_tool_mode_not_affected_by_sandbox_mode_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Story 6.5: SANDBOX_MODE env var has no effect — mode is read from ExecTool.mode."""
    # Even if SANDBOX_MODE is set, ExecTool must use self.mode exclusively
    monkeypatch.setenv("SANDBOX_MODE", "docker")
    observer = MockObserver(existing_actor=None)
    tool = ExecTool(mode="local")  # explicit local

    wire(tool, observer)

    # Despite env var, LocalSandboxActor must be chosen (mode="local")
    assert sandbox_call(observer)[0][0] is LocalSandboxActor


# ---------------------------------------------------------------------------
# Story 6.6: workspace_id field on ExecTool and pass-through to SandboxConfig
# ---------------------------------------------------------------------------


def test_exec_tool_workspace_id_defaults_to_none() -> None:
    """FR-SB-32: ExecTool.workspace_id defaults to None."""
    tool = ExecTool()
    assert tool.workspace_id is None


def test_exec_tool_workspace_id_can_be_set() -> None:
    """FR-SB-32: ExecTool(workspace_id='test') stores workspace_id='test'."""
    tool = ExecTool(workspace_id="test")
    assert tool.workspace_id == "test"


def test_observer_passes_workspace_id_to_sandbox_config() -> None:
    """FR-SB-32: ExecTool.observer() passes workspace_id through to SandboxConfig."""
    observer = MockObserver(existing_actor=None)
    tool = ExecTool(workspace_id="test")

    wire(tool, observer)

    assert sandbox_config_of(observer).workspace_id == "test"


def test_observer_passes_workspace_id_none_to_sandbox_config() -> None:
    """FR-SB-32: ExecTool() (no workspace_id) passes workspace_id=None to SandboxConfig."""
    observer = MockObserver(existing_actor=None)
    tool = ExecTool()

    wire(tool, observer)

    assert sandbox_config_of(observer).workspace_id is None


def test_observer_config_has_team_id_and_workspace_id_independently() -> None:
    """FR-SB-32: SandboxConfig gets both team_id and workspace_id, independently set."""
    observer = MockObserver(existing_actor=None)
    observer.team_id = "t1"
    tool = ExecTool(workspace_id="my-ws")

    wire(tool, observer)

    config = sandbox_config_of(observer)
    assert config.team_id == "t1"
    assert config.workspace_id == "my-ws"


# ---------------------------------------------------------------------------
# Story 8.4 — registry extension: bwrap and seatbelt keys
# ---------------------------------------------------------------------------


def test_sandbox_actor_classes_has_bwrap_key() -> None:
    """AC1 (8.4): SANDBOX_ACTOR_CLASSES['bwrap'] maps to BwrapSandboxActor."""
    assert "bwrap" in SANDBOX_ACTOR_CLASSES
    assert SANDBOX_ACTOR_CLASSES["bwrap"] is BwrapSandboxActor


def test_sandbox_actor_classes_has_seatbelt_key() -> None:
    """AC1 (8.4): SANDBOX_ACTOR_CLASSES['seatbelt'] maps to SeatbeltSandboxActor."""
    assert "seatbelt" in SANDBOX_ACTOR_CLASSES
    assert SANDBOX_ACTOR_CLASSES["seatbelt"] is SeatbeltSandboxActor


# ---------------------------------------------------------------------------
# Story 8.4 — ExecTool mode field accepts new values
# ---------------------------------------------------------------------------


def test_exec_tool_mode_can_be_set_to_bwrap() -> None:
    """AC2 (8.4): ExecTool(mode='bwrap').mode == 'bwrap'."""
    tool = ExecTool(mode="bwrap")
    assert tool.mode == "bwrap"


def test_exec_tool_mode_can_be_set_to_seatbelt() -> None:
    """AC2 (8.4): ExecTool(mode='seatbelt').mode == 'seatbelt'."""
    tool = ExecTool(mode="seatbelt")
    assert tool.mode == "seatbelt"


def test_exec_tool_mode_can_be_set_to_auto() -> None:
    """AC3 (8.4): ExecTool(mode='auto').mode == 'auto'."""
    tool = ExecTool(mode="auto")
    assert tool.mode == "auto"


# ---------------------------------------------------------------------------
# Story 8.4 — SandboxConfig accepts new mode values
# ---------------------------------------------------------------------------


def test_sandbox_config_mode_accepts_bwrap() -> None:
    """AC2 (8.4): SandboxConfig(team_id='t', mode='bwrap') validates without error."""
    config = SandboxConfig(team_id="t", mode="bwrap")
    assert config.mode == "bwrap"


def test_sandbox_config_mode_accepts_seatbelt() -> None:
    """AC2 (8.4): SandboxConfig(team_id='t', mode='seatbelt') validates without error."""
    config = SandboxConfig(team_id="t", mode="seatbelt")
    assert config.mode == "seatbelt"


def test_sandbox_config_mode_accepts_auto() -> None:
    """AC3 (8.4): SandboxConfig(team_id='t', mode='auto') validates without error."""
    config = SandboxConfig(team_id="t", mode="auto")
    assert config.mode == "auto"


# ---------------------------------------------------------------------------
# Story 8.4 — observer() creates correct actor for bwrap and seatbelt modes
# ---------------------------------------------------------------------------


def test_observer_creates_bwrap_sandbox_actor() -> None:
    """AC4 (8.4): ExecTool(mode='bwrap').observer() creates BwrapSandboxActor."""
    observer = MockObserver(existing_actor=None)
    tool = ExecTool(mode="bwrap")

    wire(tool, observer)

    assert sandbox_call(observer)[0][0] is BwrapSandboxActor


def test_observer_creates_seatbelt_sandbox_actor() -> None:
    """AC5 (8.4): ExecTool(mode='seatbelt').observer() creates SeatbeltSandboxActor."""
    observer = MockObserver(existing_actor=None)
    tool = ExecTool(mode="seatbelt")

    wire(tool, observer)

    assert sandbox_call(observer)[0][0] is SeatbeltSandboxActor


# ---------------------------------------------------------------------------
# Story 8.4 — _resolve_auto_mode() probe order
# ---------------------------------------------------------------------------


def test_resolve_auto_mode_returns_bwrap_when_bwrap_on_path() -> None:
    """AC6 (8.4): _resolve_auto_mode() returns 'bwrap' when bwrap is on PATH."""
    with patch("akgentic.tool.sandbox.tool.shutil.which", return_value="/usr/bin/bwrap"):
        result = _resolve_auto_mode()
    assert result == "bwrap"


def test_resolve_auto_mode_returns_seatbelt_on_darwin_without_bwrap() -> None:
    """AC7 (8.4): _resolve_auto_mode() returns 'seatbelt' on Darwin when sandbox-exec works."""

    def which_side_effect(cmd: str) -> str | None:
        return {
            "bwrap": None,
            "sandbox-exec": "/usr/bin/sandbox-exec",
            "docker": None,
        }.get(cmd)

    mock_probe = MagicMock(returncode=0)
    with (
        patch("akgentic.tool.sandbox.tool.shutil.which", side_effect=which_side_effect),
        patch("akgentic.tool.sandbox.tool.platform.system", return_value="Darwin"),
        patch("akgentic.tool.sandbox.tool.subprocess.run", return_value=mock_probe),
    ):
        result = _resolve_auto_mode()
    assert result == "seatbelt"


def test_resolve_auto_mode_skips_seatbelt_when_probe_fails() -> None:
    """_resolve_auto_mode() falls through to docker/local when sandbox-exec probe fails."""

    def which_side_effect(cmd: str) -> str | None:
        return {
            "bwrap": None,
            "sandbox-exec": "/usr/bin/sandbox-exec",
            "docker": "/usr/bin/docker",
        }.get(cmd)

    mock_probe = MagicMock(returncode=71)  # Operation not permitted
    with (
        patch("akgentic.tool.sandbox.tool.shutil.which", side_effect=which_side_effect),
        patch("akgentic.tool.sandbox.tool.platform.system", return_value="Darwin"),
        patch("akgentic.tool.sandbox.tool.subprocess.run", return_value=mock_probe),
    ):
        result = _resolve_auto_mode()
    assert result == "docker"


def test_resolve_auto_mode_returns_docker_when_docker_on_path() -> None:
    """AC (8.4): _resolve_auto_mode() returns 'docker' when docker on PATH, no bwrap/seatbelt."""

    def which_side_effect(cmd: str) -> str | None:
        return {
            "bwrap": None,
            "sandbox-exec": None,
            "docker": "/usr/bin/docker",
        }.get(cmd)

    with (
        patch("akgentic.tool.sandbox.tool.shutil.which", side_effect=which_side_effect),
        patch("akgentic.tool.sandbox.tool._seatbelt_available", return_value=False),
    ):
        result = _resolve_auto_mode()
    assert result == "docker"


def test_resolve_auto_mode_returns_local_when_nothing_found() -> None:
    """AC8 (8.4): _resolve_auto_mode() returns 'local' when no backends found."""
    with (
        patch("akgentic.tool.sandbox.tool.shutil.which", return_value=None),
        patch("akgentic.tool.sandbox.tool._seatbelt_available", return_value=False),
    ):
        result = _resolve_auto_mode()
    assert result == "local"


# ---------------------------------------------------------------------------
# Story 8.4 — observer() auto-mode creates correct actor
# ---------------------------------------------------------------------------


def test_observer_auto_mode_creates_bwrap_actor() -> None:
    """AC6 (8.4): mode='auto' → _resolve_auto_mode='bwrap' → BwrapSandboxActor created."""
    observer = MockObserver(existing_actor=None)
    tool = ExecTool(mode="auto")

    with patch("akgentic.tool.sandbox.tool._resolve_auto_mode", return_value="bwrap"):
        wire(tool, observer)

    assert sandbox_call(observer)[0][0] is BwrapSandboxActor


def test_observer_auto_mode_creates_seatbelt_actor() -> None:
    """AC7 (8.4): mode='auto' → _resolve_auto_mode='seatbelt' → SeatbeltSandboxActor created."""
    observer = MockObserver(existing_actor=None)
    tool = ExecTool(mode="auto")

    with patch("akgentic.tool.sandbox.tool._resolve_auto_mode", return_value="seatbelt"):
        wire(tool, observer)

    assert sandbox_call(observer)[0][0] is SeatbeltSandboxActor


def test_observer_auto_mode_fallback_to_local_emits_deprecation_warning() -> None:
    """AC8 (8.4): mode='auto' fallback to 'local' emits DeprecationWarning."""
    observer = MockObserver(existing_actor=None)
    tool = ExecTool(mode="auto")

    with (
        patch("akgentic.tool.sandbox.tool._resolve_auto_mode", return_value="local"),
        pytest.warns(DeprecationWarning, match="no isolation backend found"),
    ):
        tool.observer(observer)  # type: ignore[arg-type]

    assert sandbox_call(observer)[0][0] is LocalSandboxActor


def test_observer_auto_mode_config_uses_resolved_mode() -> None:
    """AC6 (8.4): SandboxConfig.mode stores resolved mode, not 'auto'."""
    observer = MockObserver(existing_actor=None)
    tool = ExecTool(mode="auto")

    with patch("akgentic.tool.sandbox.tool._resolve_auto_mode", return_value="bwrap"):
        wire(tool, observer)

    assert sandbox_config_of(observer).mode == "bwrap"


def test_observer_auto_mode_creates_docker_actor() -> None:
    """AC (8.4): mode='auto' → _resolve_auto_mode='docker' → DockerSandboxActor created."""
    observer = MockObserver(existing_actor=None)
    tool = ExecTool(mode="auto")

    with patch("akgentic.tool.sandbox.tool._resolve_auto_mode", return_value="docker"):
        wire(tool, observer)

    assert sandbox_call(observer)[0][0] is DockerSandboxActor


def test_observer_local_mode_explicit_does_not_emit_the_fallback_warning() -> None:
    """AC8/AC9 (8.4): the auto-fallback warning fires only on the auto path.

    ``ExecTool`` itself is now deprecated and warns on every wiring, so this can
    no longer be "no DeprecationWarning at all". The property it was written to
    guard is unchanged and is what is asserted: an explicit ``mode='local'`` is a
    choice, not a fallback, and must not be reported as one.
    """
    observer = MockObserver(existing_actor=None)
    tool = ExecTool(mode="local")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tool.observer(observer)  # type: ignore[arg-type]

    messages = [str(warning.message) for warning in caught]
    assert not any("no isolation backend found" in message for message in messages)
    assert sandbox_call(observer)[0][0] is LocalSandboxActor


# ---------------------------------------------------------------------------
# Story 29-5 — the card is a shim: it warns on use, and wires #Workspace too
# ---------------------------------------------------------------------------


def test_wiring_warns_and_names_its_replacement() -> None:
    """AC2: the deprecation fires on use and says what to use instead."""
    observer = MockObserver(existing_actor=None)
    tool = ExecTool(mode="local")

    with pytest.warns(DeprecationWarning, match=r"WorkspaceTool\(workspace_exec="):
        tool.observer(observer)  # type: ignore[arg-type]


def test_importing_the_module_does_not_warn() -> None:
    """AC2: the warning is on *use*, never on import.

    An import-time warning fires for anybody who merely has this module in a
    dependency's ``__init__`` — which is nobody's decision to change. The module
    object is cached, so what this asserts is that nothing at module scope emits
    one; the sibling test above proves the warning does exist.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        import importlib  # noqa: PLC0415

        importlib.import_module("akgentic.tool.sandbox.tool")


def test_wiring_creates_the_workspace_actor_as_well() -> None:
    """AC2: the shim owns no tree of its own — it goes through ``#Workspace``."""
    observer = MockObserver(existing_actor=None)
    tool = ExecTool(mode="local", workspace_id="shared")

    wire(tool, observer)

    names = [
        call[1]["config"].name
        for call in observer._orch_proxy.getChildrenOrCreate.call_args_list
    ]
    assert sandbox_actor_name("shared") in names
    assert "#Workspace-shared" in names


def test_the_sandbox_actor_is_named_per_workspace_like_the_workspace_actor() -> None:
    """Two workspaces in one team must not collapse onto one sandbox actor.

    ``getChildrenOrCreate`` resolves on ``config.name`` alone, so a constant name
    handed the second card the first card's actor — whose directory is the first
    card's tree. Asserted on the two names being distinct, because that is the
    whole of what the orchestrator keys on.
    """
    names: list[str] = []
    for workspace in ("alpha", "beta"):
        observer = MockObserver(existing_actor=None)
        wire(ExecTool(mode="local", workspace_id=workspace), observer)
        names.append(sandbox_config_of(observer).name)

    assert names == [sandbox_actor_name("alpha"), sandbox_actor_name("beta")]
    assert names[0] != names[1]


def test_exec_command_routes_through_the_workspace_actor() -> None:
    """AC2: the command reaches ``request_exec``, not the sandbox proxy."""
    observer = MockObserver(existing_actor=None)
    tool = ExecTool(mode="local")
    wire(tool, observer)
    fake = FakeWorkspaceProxy(done(stdout="hi"))
    tool._workspace_proxy = fake  # type: ignore[assignment]

    result = tool.get_tools()[0](cmd="echo hi", cwd="src")

    assert [(cmd, cwd) for _, cmd, cwd in fake.requests] == [("echo hi", "src")]
    assert "hi" in result


def test_exec_command_returns_a_busy_refusal_rather_than_raising() -> None:
    """A lease held elsewhere is a returned string, as every other failure here is."""
    observer = MockObserver(existing_actor=None)
    tool = ExecTool(mode="local")
    wire(tool, observer)
    busy = FakeWorkspaceProxy(refusal="workspace busy — exec run x")
    tool._workspace_proxy = busy  # type: ignore[assignment]

    result = tool.get_tools()[0](cmd="echo hi")

    assert "workspace busy" in result


def test_exec_command_hands_back_a_run_id_when_the_poll_runs_out() -> None:
    """A run still going past the poll budget degrades to the run id, echoed verbatim."""
    observer = MockObserver(existing_actor=None)
    tool = ExecTool(mode="local")
    wire(tool, observer)
    running = ExecStatus(state=ExecState.RUNNING, run_id="abc12345")
    tool._workspace_proxy = FakeWorkspaceProxy(running)  # type: ignore[assignment]

    with patch("akgentic.tool.core.deferred.time.sleep"):
        result = tool.get_tools()[0](cmd="pytest")

    assert "abc12345" in result
    assert "in progress" in result
