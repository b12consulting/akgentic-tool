"""The sandbox backend contract, and the process machinery every backend shares.

A backend is a plain strategy: no actor, no mailbox, no lifecycle beyond
``start`` / ``exec`` / ``kill`` / ``stop``. :class:`SandboxBackend` is the whole
of what a caller may rely on, and :class:`ProcessBackend` is the one place the
``Popen`` dance is written — four hand-rolled copies of it is precisely where
four backends would drift apart.

**``kill`` is a new obligation, not a rename.** Every backend used to run
``subprocess.run(timeout=)``, which owns the child for the duration of the call
and returns only when it is over: there was no handle, so there was nothing a
caller could end. A backend now keeps the handle of the run in flight, so a
caller that no longer wants a run can stop it.

This module imports nothing else from the package, which is what lets
``actor.py`` import it and re-export the names it moved from there.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import threading
from collections.abc import Callable
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

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
Callers that own a tighter budget pass it to :meth:`SandboxBackend.exec`.
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
## ``LocalBackend`` constructs no mount, so that guarantee does not cover
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


class ExecResult(BaseModel):
    """Result of a sandbox command execution.

    Attributes:
        stdout: Captured standard output from the command.
        stderr: Captured standard error from the command.
        exit_code: Process exit code (0 indicates success). A process ended by a
            signal carries the negative signal number, as ``Popen.returncode``
            reports it — ``-9`` for the ``SIGKILL`` that :meth:`ProcessBackend.kill`
            sends.
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
# The command filter
# ---------------------------------------------------------------------------


def validate_command(cmd: str) -> list[str]:
    """Tokenise *cmd* the way a shell would, and check its binary against the allowlist.

    Quotes group, backslashes escape, and only the first token — the binary name
    — is checked. Argument-level filtering is out of scope, and so is the
    allowlist as a security boundary; see the note above :data:`ALLOWED_COMMANDS`.

    The tokens are returned rather than discarded so that the validated binary
    and the executed one are the same string. A check that split on whitespace
    would validate ``"my`` and run something else.

    Args:
        cmd: Full command string, exactly as the caller gave it.

    Returns:
        The ``shlex`` tokens, ready to hand to a process.

    Raises:
        CommandParseError: The string cannot be tokenised — an unbalanced quote.
        CommandNotAllowedError: The string is empty, or its binary is not in
            :data:`ALLOWED_COMMANDS`.
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
    return tokens


# ---------------------------------------------------------------------------
# The Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SandboxBackend(Protocol):
    """What ``workspace_exec`` runs a command on, and the whole of it.

    ``@runtime_checkable`` buys an ``isinstance`` check on **method names only**
    — not a signature, not an argument count, not a return type. A backend whose
    ``exec()`` took the wrong arguments would pass it. The real conformance check
    is mypy over ``src/``; the runtime check is a smoke test, and any sentence
    written about it should say the four names are present rather than that the
    Protocol is satisfied.
    """

    def start(self, workspace_path: str) -> None:
        """Provision the backend for the already-resolved two-segment *workspace_path*."""
        ...

    def exec(self, cmd: str, cwd: str, timeout: float | None) -> ExecResult:
        """Run one validated command and return what it produced."""
        ...

    def kill(self) -> None:
        """End the run in flight, if there is one. Idempotent and best-effort."""
        ...

    def stop(self) -> None:
        """Kill the run in flight, then release the backend's own resources."""
        ...


# ---------------------------------------------------------------------------
# The shared process machinery
# ---------------------------------------------------------------------------


class ProcessBackend:
    """The ``Popen`` dance, written once for the four backends that share it.

    A subclass builds its own argv and hands it to :meth:`_run`; everything about
    keeping the handle, reproducing ``subprocess.run``'s timeout semantics and
    ending a run early lives here. It is deliberately not a
    :class:`SandboxBackend` itself — it implements neither ``start`` nor ``exec``,
    which is the whole of what distinguishes one backend from another.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running: subprocess.Popen[str] | None = None

    def _run(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        timeout: float,
        env: dict[str, str] | None = None,
        preexec_fn: Callable[[], None] | None = None,
    ) -> ExecResult:
        """Run *argv* to completion, holding its handle for the duration.

        Reproduces ``subprocess.run(timeout=)`` exactly, and that exactness is
        load-bearing: on expiry it kills the child, drains it, and re-raises
        ``TimeoutExpired``. ``Popen.communicate(timeout=)`` only raises — it does
        not kill — so skipping the kill-and-drain would leave a zombie and change
        the answer ``receiveMsg_ExecRequest`` gives an agent from "too slow" to
        "failed".

        The handle is cleared in a ``finally`` on every exit. That is not
        tidiness: a retained handle to an exited process makes a later
        :meth:`kill` act on a pid the OS may have recycled.

        Raises:
            subprocess.TimeoutExpired: The command outlived *timeout*.
        """
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            preexec_fn=preexec_fn,
        )
        with self._lock:
            self._running = proc
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            raise subprocess.TimeoutExpired(
                argv, timeout, output=stdout, stderr=stderr
            ) from None
        finally:
            with self._lock:
                self._running = None
        return ExecResult(stdout=stdout, stderr=stderr, exit_code=proc.returncode)

    def kill(self) -> None:
        """End the run in flight, if there is one.

        Idempotent by construction: with no handle there is nothing to signal and
        the call returns. A child that exited between reading the handle and
        signalling it is already what this method wanted, so the resulting
        ``ProcessLookupError`` is swallowed rather than raised at a caller who
        asked for exactly that outcome.

        **The direct child only.** ``Popen.kill`` sends ``SIGKILL`` to the process
        it started, not to its process group, even where ``os.setpgrp()`` ran in
        a ``preexec_fn``. Killing ``sh`` may therefore orphan what ``sh`` started.
        """
        with self._lock:
            proc = self._running
        if proc is None:
            return
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass

    def stop(self) -> None:
        """Kill the run in flight, then release the backend's own resources.

        The order is the point: releasing under a live child is how a container
        gets torn down around a command that is still writing to the tree.
        """
        self.kill()
        self._release()

    def _release(self) -> None:
        """Release whatever this backend provisioned in ``start()``.

        A no-op here because three of the four backends provision nothing that
        outlives a run. :class:`~akgentic.tool.sandbox.docker.DockerBackend`
        overrides it to stop its container.
        """
