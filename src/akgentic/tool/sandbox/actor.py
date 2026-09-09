"""SandboxActor — abstract base class for sandbox execution backends.

Defines the actor's models, its module constants and the lifecycle/exec
contract. Concrete subclasses (LocalSandboxActor, DockerSandboxActor) implement
_start_sandbox, _stop_sandbox and _exec by delegating to the strategy in
:mod:`akgentic.tool.sandbox.backend` and its four backend modules — the actor
holds no execution path of its own.

The base has **two** entry points and a backend implements neither. ``exec()``
is the ask: a validated command in, an ``ExecResult`` out, raising on anything
else — what a harness or a direct caller uses. ``receiveMsg_ExecRequest`` is the
tell: what ``#Workspace`` uses, and the only one that reports. The tell handler
wraps the ask, so a backend that implements ``_exec`` has already implemented
both.
"""

from __future__ import annotations

import logging
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from akgentic.core.actor_address import ActorAddress
from akgentic.core.agent import Akgent
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.tool.sandbox.backend import (
    ALLOWED_COMMANDS,
    DEFAULT_BACKEND_TIMEOUT_S,
    CardMode,
    CommandNotAllowedError,
    CommandParseError,
    ExecReport,
    ExecResult,
    SandboxMode,
    validate_command,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ALLOWED_COMMANDS",
    "DEFAULT_BACKEND_TIMEOUT_S",
    "SANDBOX_ACTOR_NAME",
    "SANDBOX_ACTOR_ROLE",
    "CardMode",
    "CommandNotAllowedError",
    "CommandParseError",
    "ExecReport",
    "ExecRequest",
    "ExecResult",
    "SandboxActor",
    "SandboxConfig",
    "SandboxMode",
    "SandboxState",
    "sandbox_actor_name",
    "validate_command",
]
"""The allowlist, the two exceptions, ``ExecResult``, ``ExecReport`` and the two
mode aliases now live in :mod:`akgentic.tool.sandbox.backend` and are re-exported
here.

They moved with the backends they belong to; the re-export is what keeps
``from akgentic.tool.sandbox.actor import ALLOWED_COMMANDS`` — and everything
that reaches them through the package — resolving to the same objects.

``ExecReport`` was the last of them, and it moved for a different reason: this
module is what the sandbox actor's retirement deletes, and the report is still
told by whatever performed the run — which is now ``#Workspace``'s own worker
thread. ``ExecRequest`` stays, because only :meth:`SandboxActor.receiveMsg_ExecRequest`
uses it and it dies with that method.
"""

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SANDBOX_ACTOR_NAME: str = "#SandboxActor"
"""Base actor name. The live name appends the workspace — see :func:`sandbox_actor_name`.

The ``#`` prefix is the orchestrator's teardown invariant: it is what classifies
the actor as a tool actor during the two-phase stop.
"""

SANDBOX_ACTOR_ROLE: str = "ToolActor"


def sandbox_actor_name(workspace_name: str) -> str:
    """Return the sandbox actor name owning *workspace_name*'s tree.

    ``getChildrenOrCreate`` resolves purely on ``config.name``, so a fixed
    ``#SandboxActor`` collapses two exec-capable cards on different
    workspaces onto the **first** actor — whose directory is the
    *other* card's tree. The second agent's commands then run in tree ``a`` while
    ``#Workspace-b`` gates, discovers and commits tree ``b``: tree ``a`` is
    mutated entirely outside the gate, with nothing raised and nothing logged.

    Same rule as the workspace actor, in the one place it was not applied — the
    unicity domain of an actor must equal the resource it owns, and the resource
    is a tree.

    Args:
        workspace_name: The **resolved** two-segment workspace path, exactly as
            the card derived it and as ``Filesystem`` receives it. The slash it
            contains is carried verbatim: nothing parses an actor name, and the
            path is injective by construction, so a second encoding here would
            only add an injectivity proof nobody needs.

    Returns:
        ``#SandboxActor-<workspace_name>``.
    """
    return f"{SANDBOX_ACTOR_NAME}-{workspace_name}"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class SandboxConfig(BaseConfig):
    """Configuration for a SandboxActor.

    Attributes:
        team_id: Identifier of the team that owns this sandbox.  It names the
            docker **container** — containers are per-team execution resources —
            and nothing else: no directory is derived from it.
        workspace_path: The already-resolved two-segment path of the tree this
            sandbox mounts, relative to ``AKGENTIC_WORKSPACES_ROOT``.  It
            replaces the raw ``workspace_id`` override this config used to
            carry: a backend that joins a path it was handed cannot open a
            different directory from the one the card, the write gate and the
            journal are working on.
        mode: Execution backend — ``"local"`` (subprocess), ``"bwrap"``
            (Linux bubblewrap), ``"seatbelt"`` (macOS Apple Seatbelt),
            ``"docker"`` (persistent container), or ``"auto"`` (automatic
            selection of the best available backend).  Defaults to ``"local"``.
    """

    team_id: str
    workspace_path: str
    mode: CardMode = "local"


class SandboxState(BaseState):
    """Runtime state for a SandboxActor.

    Attributes:
        workspace_path: Path to the workspace directory on the host, or None if
            the sandbox has not been started yet.
        container_name: Name of the Docker container, or None if not applicable.
    """

    workspace_path: Path | None = None
    container_name: str | None = None


class ExecRequest(SerializableBaseModel):
    """One command handed to the sandbox, and where to report it.

    **Not a** ``Message``. ``Akgent.on_receive`` emits the ``ReceivedMessage`` /
    ``ProcessedMessage`` telemetry sandwich only for ``Message`` instances, and
    anything built on those two types derives "who is working" from them — so a
    ``Message`` here would surface the sandbox as a busy team member every time
    an agent ran a command. It is the same reason ``DeferredPayload`` is not one.

    A model rather than five arguments because it crosses an actor boundary
    (Golden Rule #1): a positional tell is where a cwd and a command get
    swapped.

    Attributes:
        run_id: The requester's id for this run, echoed back in the report. It
            is what lets a report be matched against the run that is actually
            holding the tree, so a late one cannot close out a newer run.
        cmd: The command string, exactly as the agent gave it.
        cwd: Working directory below the workspace root.
        timeout_s: Wall-clock budget for the command, already clamped by the
            requester. The sandbox does not clamp it again — one owner, one
            budget.
        reply_to: Where the :class:`ExecReport` goes. An address rather than a
            proxy so the sandbox imports nothing from ``workspace/``: the edge
            is one-directional and stays that way.
    """

    run_id: str
    cmd: str
    cwd: str = ""
    timeout_s: float
    reply_to: ActorAddress


# ---------------------------------------------------------------------------
# Abstract actor
# ---------------------------------------------------------------------------


class SandboxActor(Akgent[SandboxConfig, SandboxState], ABC):
    """Abstract sandbox actor. Concrete subclasses provide the execution backend.

    Responsibilities of this base class:
    - Initialize and manage SandboxState via on_start / on_stop lifecycle hooks.
    - Enforce the command allowlist before delegating to _exec.
    - Define the abstract interface (_start_sandbox, _stop_sandbox, _exec) that
      subclasses must implement.
    """

    def on_start(self) -> None:
        """Initialize SandboxState and start the sandbox backend.

        Registers the actor as a state observer (required for Pykka telemetry),
        then delegates to _start_sandbox() for backend-specific setup.
        """
        self.state = SandboxState()
        self.state.observer(self)
        self._start_sandbox()

    def on_stop(self) -> None:
        """Stop the sandbox backend, swallowing any exceptions.

        Calls _stop_sandbox() inside a try/except so that any backend error
        does not prevent super().on_stop() from running. Leaving Pykka actors
        in a broken state by raising in on_stop() is a critical failure mode
        that this pattern prevents.
        """
        try:
            self._stop_sandbox()
        except Exception:
            logger.warning(
                "SandboxActor._stop_sandbox() raised during on_stop — swallowing",
                exc_info=True,
            )
        super().on_stop()

    def receiveMsg_ExecRequest(self, request: ExecRequest) -> None:
        """TELL, from ``#Workspace``. Run the command and **always** report it.

        The base's second entry point, beside :meth:`exec`, and the only one
        that reports. ``#Workspace`` sends here and goes back to draining its
        mailbox; the subprocess blocks this actor's own thread, which is what
        this actor is for. No ask travels in either direction, so neither side
        can be parked waiting on the other.

        Dispatched by name: ``Akgent.on_receive`` routes a non-``Message``
        payload to ``receiveMsg_<Type>``, which is why the request is not a
        ``Message`` and why ``N802`` is suppressed for this convention.

        **Nothing escapes this method, and that is the whole contract.** An
        exception out of a tell handler stops the actor, and a stopped sandbox
        reports nothing — which is the exact failure the retired worker had,
        moved one actor over, except that there is no longer anything holding a
        budget that could time it out. So every exit builds a report, the
        ``finally`` covers the exits nothing else thought of, and the send
        itself is guarded: a dead ``#Workspace`` is nobody to report to, not a
        reason to take this actor down with it.

        The three outcomes are three different things to the agent waiting:
        a command that ran is a result whatever it exited with; a command the
        budget killed is an answer that says so; anything else — the backend
        raised, the allowlist refused the binary, the quotes would not balance —
        is a failure with the reason in it.

        Args:
            request: The command, its budget, and where to report.
        """
        report: ExecReport | None = None
        try:
            result = self.exec(request.cmd, request.cwd, timeout=request.timeout_s)
            report = ExecReport(run_id=request.run_id, result=result)
        except subprocess.TimeoutExpired:
            report = ExecReport(run_id=request.run_id, timed_out=True)
        except Exception as exc:  # noqa: BLE001 — every failure is an answer, never a crash
            # ``or repr(exc)`` is load-bearing, not defensive. ``str(exc)`` is
            # the empty string for any exception raised with no message —
            # ``raise RuntimeError()`` — and an empty ``error`` fails
            # ``ExecReport``'s exactly-one validator, so the report that was
            # meant to carry the failure raises *inside this except clause* and
            # propagates out of the handler, stopping the actor: the one thing
            # this method exists to prevent. ``repr`` always names the type.
            report = ExecReport(run_id=request.run_id, error=str(exc) or repr(exc))
        finally:
            if report is None:
                # Reachable only if building one of the reports above raises,
                # which the ``or repr(exc)`` overhead is there to stop — so this
                # is the branch for the exit nothing thought of. Kept rather
                # than argued: a run whose report is dropped holds the tree
                # until the gate's grace releases it.
                report = ExecReport(
                    run_id=request.run_id,
                    error="The sandbox produced no report for this run.",
                )
            try:
                request.reply_to.tell(report)
            except Exception:
                logger.warning(
                    "SandboxActor could not report run %s to %s — swallowing",
                    request.run_id,
                    request.reply_to.name,
                    exc_info=True,
                )

    def exec(self, cmd: str, cwd: str = "", timeout: float | None = None) -> ExecResult:
        """Execute a command inside the sandbox after allowlist validation.

        The string is tokenised the way a shell tokenises one — quotes group,
        backslashes escape — and only the first token (the binary name) is
        checked against ALLOWED_COMMANDS. Argument-level filtering is out of
        scope, and so is the allowlist as a security boundary — see the note
        above the set.

        **No shell interprets the string.** Every backend hands the same
        ``shlex.split(cmd)`` tokens straight to the binary, so ``&&``, ``|``,
        ``>``, ``$VAR`` and globs arrive as literal arguments. Wrap anything
        needing shell syntax in ``bash -c '…'``.

        **The filter exists in exactly one place**, :func:`validate_command` in
        ``backend.py``, and both this method and the backend call it. Calling it
        here is what keeps the raise ahead of the backend — a refused command
        must never reach a process — and calling it there is what keeps the
        validated binary and the executed one the same string, because the
        backend runs the very tokens the check returned.

        Args:
            cmd: Full command string to execute (e.g. "python main.py").
            cwd: Working directory inside the sandbox. Defaults to "".
            timeout: Wall-clock budget for the command, in seconds. ``None``
                keeps the backend's own default, so no existing caller changes
                behaviour. A caller that owns a budget must pass it: a budget
                that stops at the proxy is decoration, because a Python thread
                cannot be cancelled. ``#Workspace`` is that caller, and it
                passes the budget on :attr:`ExecRequest.timeout_s` — the
                subprocess is what actually stops, and this actor's thread holds
                the orchestrator's blocking stop open until it returns.

        Returns:
            ExecResult with stdout, stderr, and exit_code from the backend.

        Raises:
            CommandParseError: If the command string cannot be tokenised —
                an unbalanced quote. Raised before the backend is reached.
            CommandNotAllowedError: If the command binary is not in ALLOWED_COMMANDS.
        """
        validate_command(cmd)
        return self._exec(cmd, cwd, timeout)

    # ------------------------------------------------------------------
    # Abstract methods — must be implemented by concrete subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _start_sandbox(self) -> None:
        """Start the sandbox execution environment.

        Called from on_start() after SandboxState is initialized. Subclasses
        should provision any resources needed (e.g., create a temp directory,
        start a Docker container).
        """

    @abstractmethod
    def _stop_sandbox(self) -> None:
        """Stop and clean up the sandbox execution environment.

        Called from on_stop() inside a try/except. Subclasses should release
        resources (e.g., remove temp directory, stop a Docker container).
        May raise — the caller swallows all exceptions.
        """

    @abstractmethod
    def _exec(self, cmd: str, cwd: str, timeout: float | None = None) -> ExecResult:
        """Execute a pre-validated command inside the sandbox.

        Called by exec() after the allowlist check passes. Subclasses handle
        the actual process execution (subprocess, Docker exec API, etc.) and
        MUST hand *timeout* to it — a budget the backend drops is no budget.

        Args:
            cmd: Full command string (already validated by exec()).
            cwd: Working directory inside the sandbox.
            timeout: Wall-clock budget in seconds, or ``None`` for the
                backend's own default.

        Returns:
            ExecResult with captured stdout, stderr, and exit code.
        """
