"""SandboxActor — abstract base class for sandbox execution backends.

Defines models, the command allowlist, module constants, and the lifecycle/exec
contract. Concrete subclasses (LocalSandboxActor, DockerSandboxActor) provide
the execution backend by implementing _start_sandbox, _stop_sandbox, and _exec.

The base has **two** entry points and a backend implements neither. ``exec()``
is the ask: a validated command in, an ``ExecResult`` out, raising on anything
else — what a harness or a direct caller uses. ``receiveMsg_ExecRequest`` is the
tell: what ``#Workspace`` uses, and the only one that reports. The tell handler
wraps the ask, so a backend that implements ``_exec`` has already implemented
both.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

from akgentic.core.actor_address import ActorAddress
from akgentic.core.agent import Akgent
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.core.utils.serializer import SerializableBaseModel

logger = logging.getLogger(__name__)

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
    ``#SandboxActor`` collapses two exec-capable cards carrying different
    ``workspace_id`` values onto the **first** actor — whose directory is the
    *other* card's tree. The second agent's commands then run in tree ``a`` while
    ``#Workspace-b`` gates, discovers and commits tree ``b``: tree ``a`` is
    mutated entirely outside the gate, with nothing raised and nothing logged.

    Same rule as the workspace actor, in the one place it was not applied — the
    unicity domain of an actor must equal the resource it owns, and the resource
    is a tree.

    Args:
        workspace_name: The resolved workspace — a card's ``workspace_id``, or
            the team id when it has none. It must be derived exactly as
            ``Filesystem`` resolution derives it, or the actor's name and the
            directory it opens would disagree.

    Returns:
        ``#SandboxActor-<workspace_name>``.
    """
    return f"{SANDBOX_ACTOR_NAME}-{workspace_name}"

SandboxMode = Literal["local", "bwrap", "seatbelt", "docker"]
"""A backend that has been resolved. ``"auto"`` is not one of these."""

CardMode = Literal["local", "bwrap", "seatbelt", "docker", "auto"]
"""What a card may ask for, which includes ``"auto"``: probe the host and pick.

Both aliases live here, in the module with no dependencies of its own, because
both sides of the exec merge need them and a second definition in either would
be a second place to add a backend to.
"""

DEFAULT_BACKEND_TIMEOUT_S: float = 30.0
"""What a backend gives a command when the caller names no budget.

Strictly at the orchestrator's stop backstop rather than above it: a worker
cannot cancel a Python thread, so a subprocess still running past the backstop
holds its parent's ``stop_children(blocking=True)`` open for the difference.
Callers that own a tighter budget pass it to :meth:`SandboxActor.exec`.
"""

##
## Only the FIRST token of a command is checked against this set, and both
## ``bash`` and ``sh`` are in it — so ``bash -c "<anything>"`` walks straight
## past it.  Nothing may rely on this allowlist for safety: it is a usability
## filter that keeps an obvious mistake from running, and a way to tell an agent
## what the sandbox offers.
##
## ``git`` is on the list.  It was briefly removed, to stop a ``git reset
## --hard`` from destroying the workspace journal — but the bypass above meant
## that stopped nobody, while costing an agent the use of git in a directory
## that *is* a git repository.  The real guarantee is a filesystem fact: the
## journal lives at the sibling ``<root>.git``, outside the mount of every
## backend that constructs one, so it is not there to be reached.  Note that
## ``LocalSandboxActor`` constructs no mount, so that guarantee does not cover
## it — use an isolating backend where it matters.
##
ALLOWED_COMMANDS: frozenset[str] = frozenset(
    {
        ## Python
        "python",
        "python3",
        "pytest",
        "ruff",
        "mypy",
        "uv",
        "pip",
        ## Web
        "node",
        "npm",
        "npx",
        ## bash
        "sh",
        "bash",
        "cat",
        "echo",
        "ls",
        "cp",
        "mv",
        "rm",
        "mkdir",
        "find",
        "grep",
        "sed",
        "awk",
        "jq",
        "wc",
        "xargs",
        "touch",
        "make",
        "git",
        "kill",
        ## Network
        "curl",
        "wget",
    }
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class SandboxConfig(BaseConfig):
    """Configuration for a SandboxActor.

    Attributes:
        team_id: Identifier of the team that owns this sandbox.
        workspace_id: Optional workspace directory name override.  When ``None``
            (default), the workspace directory is named after ``team_id``.  When
            set, the named directory is used instead.  Docker container name
            always uses ``team_id`` — containers are per-team, not per-workspace.
        mode: Execution backend — ``"local"`` (subprocess), ``"bwrap"``
            (Linux bubblewrap), ``"seatbelt"`` (macOS Apple Seatbelt),
            ``"docker"`` (persistent container), or ``"auto"`` (automatic
            selection of the best available backend).  Defaults to ``"local"``.
    """

    team_id: str
    workspace_id: str | None = None
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


class ExecResult(BaseModel):
    """Result of a sandbox command execution.

    Attributes:
        stdout: Captured standard output from the command.
        stderr: Captured standard error from the command.
        exit_code: Process exit code (0 indicates success).
    """

    stdout: str
    stderr: str
    exit_code: int


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


class ExecReport(SerializableBaseModel):
    """What the sandbox tells back when a run is over — however it ended.

    Exactly one of the three outcomes is carried, and it is **enforced**: the
    receiving actor branches on them in order, so a report carrying none would
    close a run out with no answer at all and one carrying two would deliver an
    outcome for a run it had also failed.

    It lives here rather than beside ``ExecOutcome`` because ``sandbox/`` cannot
    import ``workspace/`` — the edge runs the other way — and the report has to
    be constructible on this side.

    Attributes:
        run_id: The run being reported, verbatim from the request.
        result: What the command produced, when it ran to completion — well or
            badly. A non-zero exit code is a **result**, not a failure.
        timed_out: True when the budget killed the command. Also an answer an
            agent can read, which is why it is not folded into *error*.
        error: Why there is no result — the backend raised, the allowlist
            refused the binary, the quoting would not parse.
    """

    run_id: str
    result: ExecResult | None = None
    timed_out: bool = False
    error: str = ""

    @model_validator(mode="after")
    def _exactly_one(self) -> ExecReport:
        """Reject a report that carries no outcome, or more than one."""
        carried = sum((self.result is not None, self.timed_out, bool(self.error)))
        if carried != 1:
            raise ValueError(
                "ExecReport carries exactly one of result, timed_out or error — "
                f"got {carried}."
            )
        return self


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CommandNotAllowedError(Exception):
    """Raised when exec() is called with a command binary not in ALLOWED_COMMANDS.

    Only the first token (binary name) of the command string is checked.
    Argument-level filtering is out of scope for the base class.
    """


class CommandParseError(Exception):
    """Raised when the command string cannot be tokenised at all.

    A quoting mistake — an unbalanced ``"`` or ``'`` — not a binary outside the
    allowlist. The distinction is deliberate and visible in the output: the tool
    surface appends ``Allowed commands: [...]`` to a
    :class:`CommandNotAllowedError`, which is the wrong answer to a quoting
    mistake and sends the caller hunting for a binary it already has. Kept a
    sibling of that class rather than a subclass, so an existing
    ``except CommandNotAllowedError`` cannot swallow one.
    """


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
            report = ExecReport(run_id=request.run_id, error=str(exc))
        finally:
            if report is None:
                # Unreachable through the branches above, and deliberately still
                # here: a run whose report is dropped holds the tree until the
                # gate's grace releases it, so "no exit path reports nothing" is
                # worth one branch rather than an argument.
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

        Tokenising here with the same function the backends use is what keeps
        the validated binary and the executed one the same string: a check that
        split on whitespace would validate ``"my`` and run something else.

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
        try:
            tokens = shlex.split(cmd)
        except ValueError as exc:
            raise CommandParseError(
                f"Command could not be parsed ({exc}): {cmd!r}. "
                "Balance the quotes, or wrap shell syntax in bash -c '...'."
            ) from exc
        if not tokens:
            raise CommandNotAllowedError(
                "Command string is empty — no binary to validate against the allowlist."
            )
        binary = tokens[0]
        if binary not in ALLOWED_COMMANDS:
            raise CommandNotAllowedError(
                f"Command '{binary}' is not in the allowed commands list. "
                f"Allowed: {sorted(ALLOWED_COMMANDS)}"
            )
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
