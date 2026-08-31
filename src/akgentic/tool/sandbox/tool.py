"""``ExecTool`` — the deprecated card that ``WorkspaceTool(workspace_exec=...)`` replaced.

The class still works and is still importable from here; what changed is what it
*is*. Sandboxed execution is now a capability of ``WorkspaceTool``, because exec
and the write gate share one resource — the tree — and two cards over one tree
means two mailboxes that interleave (ADR-036 §5, Alternative F). This card is a
shim over that path: same lease, same worker, same discovery, same commit.

**The warning is on use, not on import**, following the package's migration
policy. An import-time warning fires for anybody who merely has the module in a
dependency's ``__init__``, which is nobody's decision to change.

``SANDBOX_ACTOR_CLASSES``, ``_resolve_auto_mode`` and the four backend classes
stay here and are not deprecated at all — they are the exec *backend*, and
``workspace_exec`` resolves through them.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from pydantic import PrivateAttr

from akgentic.core.actor_address import ActorAddress
from akgentic.core.orchestrator import Orchestrator
from akgentic.tool.core import TOOL_CALL, BaseToolParam, Channels, ToolCard, _resolve
from akgentic.tool.core.deferred import DEFAULT_WORKER_TIMEOUT_S, poll_deferred
from akgentic.tool.core.observer import ActorToolObserver
from akgentic.tool.sandbox.actor import (
    ALLOWED_COMMANDS,
    CardMode,
    CommandNotAllowedError,
    SandboxActor,
)
from akgentic.tool.sandbox.bwrap import BwrapSandboxActor
from akgentic.tool.sandbox.docker import DockerSandboxActor
from akgentic.tool.sandbox.local import LocalSandboxActor
from akgentic.tool.sandbox.seatbelt import SeatbeltSandboxActor

if TYPE_CHECKING:  # pragma: no cover — types only, never imported at runtime
    from akgentic.tool.workspace.actor import WorkspaceActor
    from akgentic.tool.workspace.execution import ExecStatus

##
## Every reference to ``akgentic.tool.workspace`` in this module is deferred to
## call time, and that is structural rather than stylistic.  ``workspace``
## imports ``sandbox`` at module level — the worker needs the backend registry
## and ``SandboxConfig`` — so a module-level import back the other way makes the
## pair a cycle that fails on whichever package is imported first.  The edge runs
## ``workspace`` → ``sandbox`` and only that way.
##

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# SANDBOX_ACTOR_CLASSES — mutable injection window for runtime registration
# ---------------------------------------------------------------------------

SANDBOX_ACTOR_CLASSES: dict[str, type[SandboxActor]] = {
    "local": LocalSandboxActor,
    "bwrap": BwrapSandboxActor,
    "seatbelt": SeatbeltSandboxActor,
    "docker": DockerSandboxActor,
    # "e2b": E2BSandboxActor  ← injected by akgentic-infra at runtime
}


# ---------------------------------------------------------------------------
# Auto-mode resolution
# ---------------------------------------------------------------------------


def _seatbelt_available() -> bool:
    """Return True if sandbox-exec is on PATH and actually works at runtime.

    macOS 15+ may block ``sandbox_apply`` even when ``sandbox-exec`` is present.
    A quick probe with ``(allow default)`` detects this at negligible cost.
    """
    if shutil.which("sandbox-exec") is None or platform.system() != "Darwin":
        return False
    try:
        result = subprocess.run(
            ["sandbox-exec", "-p", "(version 1)(allow default)", "/usr/bin/true"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _resolve_auto_mode() -> Literal["local", "bwrap", "seatbelt", "docker"]:
    """Probe the host and return the best available sandbox backend.

    Probe order:
    1. ``bwrap`` on PATH → ``"bwrap"`` (Linux bubblewrap)
    2. ``sandbox-exec`` on PATH + Darwin → ``"seatbelt"`` (macOS)
    3. ``docker`` on PATH → ``"docker"``
    4. fallback → ``"local"`` (no filesystem isolation)

    Returns:
        String key matching an entry in SANDBOX_ACTOR_CLASSES.
    """
    if shutil.which("bwrap") is not None:
        logger.debug("_resolve_auto_mode: selected bwrap")
        return "bwrap"
    if _seatbelt_available():
        logger.debug("_resolve_auto_mode: selected seatbelt")
        return "seatbelt"
    if shutil.which("docker") is not None:
        logger.debug("_resolve_auto_mode: selected docker")
        return "docker"
    logger.debug("_resolve_auto_mode: fallback to local (no isolation backend found)")
    return "local"


# ---------------------------------------------------------------------------
# Capability parameter model
# ---------------------------------------------------------------------------


class ExecCommand(BaseToolParam):
    """Execute a sandboxed shell command in the team workspace."""

    expose: set[Channels] = {TOOL_CALL}


# ---------------------------------------------------------------------------
# ExecTool ToolCard
# ---------------------------------------------------------------------------


class ExecTool(ToolCard):
    """Deprecated card: use ``WorkspaceTool(workspace_exec=...)`` instead.

    Kept working, and working *identically* — the same lease, the same worker,
    the same discovery, the same commit — because existing configurations should
    not break on a version bump. What it no longer is, is a second owner of the
    tree: ``exec_command`` routes through ``#Workspace`` like every other
    mutation, which is what makes an exec run and a gated write share one
    serialization domain.

    Note what ``getChildrenOrCreate`` implies for an agent carrying this card
    *alone*: it creates the ``#Workspace`` actor for its workspace, and the first
    card to create that actor decides its configuration. So an ``ExecTool``-only
    agent gets the journal's defaults, exactly as a bare ``WorkspaceTool()``
    would.
    """

    exec_command: ExecCommand | bool = True
    mode: CardMode = "auto"
    workspace_id: str | None = None

    _sandbox_proxy: SandboxActor | None = PrivateAttr(default=None)
    _workspace_proxy: WorkspaceActor | None = PrivateAttr(default=None)
    _agent_id: str = PrivateAttr(default="")

    def observer(self, observer: ActorToolObserver) -> ExecTool:  # type: ignore[override]
        """Warn, then wire the same two actors ``WorkspaceTool(workspace_exec=...)`` wires.

        The mode is resolved at call time, not import time, so a backend injected
        by a deployment package before any card is constructed is still found.
        ``self.workspace_id`` is forwarded to both actors, so the sandbox's
        directory and the workspace the runs are journalled into are the same
        one.

        Args:
            observer: Actor-aware observer providing orchestrator access.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If observer.orchestrator is None.
            KeyError: If ``self.mode`` names an unregistered backend.
        """
        super().observer(observer)  # store the observer weakly via the base setter
        if observer.orchestrator is None:
            raise ValueError("ExecTool requires access to the orchestrator.")
        warnings.warn(
            "ExecTool is deprecated — use WorkspaceTool(workspace_exec=...) instead. It "
            "exposes the same execution through workspace_exec and workspace_exec_result, "
            "over the same sandbox backend.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._bind(observer, observer.orchestrator)
        return self

    def _bind(self, observer: ActorToolObserver, orchestrator: ActorAddress) -> None:
        """Bring up ``#Workspace`` and ``#SandboxActor``, and introduce them to each other."""
        from akgentic.tool.workspace.actor import (  # noqa: PLC0415 — see the module note
            WORKSPACE_ACTOR_ROLE,
            WorkspaceActor,
            workspace_actor_name,
        )
        from akgentic.tool.workspace.execution import (  # noqa: PLC0415 — see the module note
            DEFAULT_EXEC_TIMEOUT_S,
            ExecConfig,
            resolve_mode,
            sandbox_config,
        )
        from akgentic.tool.workspace.models import WorkspaceConfig  # noqa: PLC0415

        mode, actor_class = resolve_mode(self.mode)
        config = ExecConfig(
            mode=mode,
            team_id=str(observer.team_id),
            workspace_id=self.workspace_id,
            timeout_s=DEFAULT_EXEC_TIMEOUT_S,
        )
        workspace_name = self.workspace_id or str(observer.team_id)
        orchestrator_proxy = observer.proxy_ask(orchestrator, Orchestrator)
        sandbox_addr = orchestrator_proxy.getChildrenOrCreate(
            actor_class, config=sandbox_config(config)
        )
        workspace_addr = orchestrator_proxy.getChildrenOrCreate(
            WorkspaceActor,
            config=WorkspaceConfig(
                name=workspace_actor_name(workspace_name),
                role=WORKSPACE_ACTOR_ROLE,
                workspace_name=workspace_name,
            ),
        )
        self._sandbox_proxy = observer.proxy_ask(sandbox_addr, SandboxActor)
        self._workspace_proxy = observer.proxy_ask(workspace_addr, WorkspaceActor)
        self._agent_id = str(observer.myAddress.agent_id)
        tell = observer.proxy_tell(workspace_addr, WorkspaceActor)
        self._announce(tell, observer, config)

    def _announce(self, tell: Any, observer: ActorToolObserver, config: Any) -> None:
        """Register this agent's name and the exec backend — fire and forget, never fatal.

        A harness that hands back a stand-in proxy without these methods, or an
        actor that is already gone, must not stop a card binding. The degradation
        is a journal authored by agent id, or an exec request refused because no
        backend was announced — both visible, neither a crash at wiring time.
        """
        try:
            tell.register_agent(self._agent_id, str(observer.myAddress.name))
            tell.configure_exec(config)
        except Exception:
            logger.debug("Could not announce ExecTool's agent or backend", exc_info=True)

    def get_tools(self) -> list[Callable[..., Any]]:
        """Return the exec_command tool callable when enabled."""
        tools: list[Callable[..., Any]] = []
        ec = _resolve(self.exec_command, ExecCommand)
        if ec is not None and TOOL_CALL in ec.expose:
            tools.append(self._exec_command_factory(ec))
        return tools

    def _exec_command_factory(self, params: ExecCommand) -> Callable[..., Any]:
        """Build the exec_command callable, routed through ``#Workspace``.

        The name and the signature are unchanged, which is the whole point of a
        shim — renaming ``exec_command`` would defeat it.
        """
        from akgentic.tool.workspace.execution import (  # noqa: PLC0415 — see the module note
            DEFAULT_EXEC_POLL_ATTEMPTS,
            DEFAULT_EXEC_POLL_DELAY_S,
            DEFAULT_EXEC_TIMEOUT_S,
            format_status,
            poll_attempts_within,
            timed_out,
        )

        proxy = self._workspace_proxy
        agent_id = self._agent_id
        # The shim carries the capability's defaults and no way to change them,
        # so it resolves them the same way the capability does. Handing
        # ``poll_deferred`` the raw default would pass it the wait-out-the-run
        # sentinel, whose ``range(-1)`` is zero looks — the shim would take a run
        # id without ever having looked for a result.
        run_budget = min(DEFAULT_EXEC_TIMEOUT_S, DEFAULT_WORKER_TIMEOUT_S)
        attempts = poll_attempts_within(
            DEFAULT_EXEC_POLL_ATTEMPTS, DEFAULT_EXEC_POLL_DELAY_S, run_budget
        )

        def exec_command(cmd: str, cwd: str = "") -> str:
            """Execute a sandboxed shell command in the team workspace.

            Args:
                cmd: Full command string. The binary (first token) must be in the allow-list.
                cwd: Subdirectory relative to workspace root. Defaults to workspace root.

            Returns:
                Combined stdout, stderr, and exit code summary as a string.
                On disallowed command: error string listing allowed commands.
            """
            if proxy is None:
                return "SandboxError: RuntimeError: ExecTool was not wired to an orchestrator"
            try:
                start = proxy.request_exec(agent_id, cmd, cwd)
                if not start.run_id:
                    return start.refusal
                run_id = start.run_id
                settled = poll_deferred(
                    lambda: _settled(proxy.exec_status(agent_id, run_id)),
                    attempts=attempts,
                    delay=DEFAULT_EXEC_POLL_DELAY_S,
                )
                if settled is not None:
                    return format_status(settled)
                return timed_out(run_id, run_budget)
            except CommandNotAllowedError as e:
                return f"CommandNotAllowedError: {e}. Allowed commands: {sorted(ALLOWED_COMMANDS)}"
            except Exception as e:
                logger.warning("Sandbox exec failed: %s: %s", type(e).__name__, e)
                return f"SandboxError: {type(e).__name__}: {e}"

        allowed_str = ", ".join(sorted(ALLOWED_COMMANDS))
        base_doc = (exec_command.__doc__ or "") + f"\n\n            Allowed binaries: {allowed_str}"
        exec_command.__doc__ = params.format_docstring(base_doc)
        return exec_command


def _settled(status: ExecStatus) -> ExecStatus | None:
    """Answer a poll only once the run has something final to say."""
    return status if status.settled else None
