"""SandboxActor — abstract base class for sandbox execution backends.

Defines models, the command allowlist, module constants, and the lifecycle/exec
contract. Concrete subclasses (LocalSandboxActor, DockerSandboxActor) provide
the execution backend by implementing _start_sandbox, _stop_sandbox, and _exec.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from akgentic.core.agent import Akgent
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState

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
## ``bash`` and ``sh`` are in it — so ``bash -c "git reset --hard"`` walks
## straight past it.  Nothing may rely on this allowlist for safety.  The real
## guarantee that a sandboxed run cannot reach the workspace journal is a
## filesystem fact: the repository lives at the sibling ``<root>.git``, outside
## every backend's mount, so it is not there to be reached.  ``git`` is absent
## below as defence in depth, not as the boundary.
##
ALLOWED_COMMANDS: frozenset[str] = frozenset(
    {
        "python",
        "python3",
        "pytest",
        "ruff",
        "mypy",
        "uv",
        "pip",
        "cat",
        "ls",
        "find",
        "grep",
        "mkdir",
        "cp",
        "mv",
        "rm",
        "echo",
        "touch",
        "git",
        "curl",
        "wget",
        "make",
        "bash",
        "sh",
        "node",
        "npm",
        "npx",
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


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CommandNotAllowedError(Exception):
    """Raised when exec() is called with a command binary not in ALLOWED_COMMANDS.

    Only the first token (binary name) of the command string is checked.
    Argument-level filtering is out of scope for the base class.
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

    def ready(self) -> bool:
        """Answer as soon as this actor can serve messages at all.

        A **FIFO barrier, not a health check**. Its whole value is the mailbox's
        own ordering: a message cannot be delivered before ``on_start`` has
        returned, so an ask for this method is answered only once the backend has
        finished provisioning — with no flag, no polling and no state. It says
        the actor has finished starting and nothing whatever about whether docker
        still works.

        It exists so a caller can put a **timeout** on the wait for the backend
        separately from the timeout on the command. Conflating the two is what
        lets a slow cold start spend a run's whole budget before the command has
        started (see ``ExecWorker.produce``).

        Returns:
            ``True``, always. The value carries no information; the fact that it
            arrived does.
        """
        return True

    def exec(self, cmd: str, cwd: str = "", timeout: float | None = None) -> ExecResult:
        """Execute a command inside the sandbox after allowlist validation.

        Only the first whitespace-delimited token (the binary name) is checked
        against ALLOWED_COMMANDS. Argument-level filtering is out of scope, and
        so is the allowlist as a security boundary — see the note above the set.

        Args:
            cmd: Full command string to execute (e.g. "python main.py").
            cwd: Working directory inside the sandbox. Defaults to "".
            timeout: Wall-clock budget for the command, in seconds. ``None``
                keeps the backend's own default, so no existing caller changes
                behaviour. A caller that owns a budget — a ``DeferredWorker``
                above all — must pass it: a budget that stops at the proxy is
                decoration, because a Python thread cannot be cancelled and the
                worker holds its parent's teardown open until the subprocess
                returns.

        Returns:
            ExecResult with stdout, stderr, and exit_code from the backend.

        Raises:
            CommandNotAllowedError: If the command binary is not in ALLOWED_COMMANDS.
        """
        tokens = cmd.split()
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
