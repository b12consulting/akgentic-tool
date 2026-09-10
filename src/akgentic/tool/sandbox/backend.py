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

**A kill reaches the child's whole process group where one was made for it.**
``local`` and ``bwrap`` start the child under a ``preexec_fn`` that calls
``os.setpgrp()``, and the group exists precisely so that a timeout or a kill can
end the subtree rather than the shell at its root — ``sh -c '…'`` on Linux forks
the command as a grandchild, which used to survive a direct-child kill and hold
the pipes open for the rest of its life. A backend says so when it spawns —
``_run(..., process_group=True)`` beside the ``preexec_fn`` that creates the
group — and :meth:`ProcessBackend._signal` ends the group where it was told one
exists and the direct child everywhere else. Docker (one host process, the
``docker exec`` client) and seatbelt (no ``preexec_fn``) take the second path.

This module imports nothing else from ``akgentic.tool``. (It does reach
``akgentic.core`` for the serializer base, which is a package below this one and
cannot import back.)

**``ExecReport`` lives here, beside the ``ExecResult`` it carries.** The report
is told by whatever performed the run, which is ``#Workspace``'s own worker
thread; the sandbox actor that used to carry it is gone.
"""

from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
import threading
from collections.abc import Callable
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, model_validator

from akgentic.core.utils.serializer import SerializableBaseModel

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


class ExecReport(SerializableBaseModel):
    """What the runner tells back when a run is over — however it ended.

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
                f"ExecReport carries exactly one of result, timed_out or error — got {carried}."
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
    Protocol is satisfied. Declaring ``__init__`` below does not strengthen it in
    the slightest — every object has one — and mypy ignores ``__init__``
    entirely when deciding whether a class *implements* a Protocol.

    **What ``__init__`` is declared for is the other direction**: the registry
    holds ``type[SandboxBackend]`` and ``resolve_mode`` constructs from it, so
    "constructible with no arguments" is part of what registering a backend
    commits to, and a Protocol that did not say so would leave the one call site
    that builds one unable to be type-checked at all.
    """

    def __init__(self) -> None:
        """Build an unstarted backend. Everything it needs arrives through :meth:`start`."""
        ...

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

    **Every backend is constructed with no arguments**, so the caller building
    one from the registry has one uniform constructor to call and never a type
    switch on the mode it just resolved. The tree a backend runs in arrives
    through ``start(workspace_path)``; nothing about the team that asked does,
    because a hosted tree is shared by several teams and no backend depends on
    which one asked.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running: subprocess.Popen[str] | None = None
        self._leads_a_group: bool = False
        """Whether ``_running`` was spawned as the leader of its own process group.

        Set beside the handle, under the same lock, from the ``process_group``
        argument the backend gave :meth:`_run` — so :meth:`kill` signals the
        group only where the backend said it made one.
        """

    def _run(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        timeout: float,
        env: dict[str, str] | None = None,
        preexec_fn: Callable[[], None] | None = None,
        process_group: bool = False,
    ) -> ExecResult:
        """Run *argv* to completion, holding its handle for the duration.

        Args:
            argv: The command line, already validated.
            cwd: Working directory for the child, or ``None`` for this process's.
            timeout: The run's budget in seconds.
            env: The child's environment, or ``None`` to inherit.
            preexec_fn: Run in the child before ``exec``; ``local`` and ``bwrap``
                pass ``_make_preexec()``, which sets resource limits and calls
                ``os.setpgrp()``.
            process_group: ``True`` when *preexec_fn* makes the child the leader
                of a new process group, so that a kill or a timeout can end the
                whole subtree. **Pass it only beside a ``preexec_fn`` that calls
                ``os.setpgrp()``**: it is what tells :meth:`_signal` there is a
                group to signal, and nothing here checks the claim.

        Reproduces the part of ``subprocess.run(timeout=)`` the contract rests
        on: on expiry it kills the child and re-raises ``TimeoutExpired``, which
        is what ``ExecRunner.perform`` turns into "too slow" rather than
        "failed". ``Popen.communicate(timeout=)`` only raises — it does not kill
        — so the kill cannot be skipped without leaving a zombie.

        **It is not identical to ``subprocess.run``, and the difference is what
        makes the group kill necessary.** CPython's POSIX path calls
        ``proc.wait()`` after the kill, because ``_communicate`` already
        collected the output; this drains with a second ``communicate()``
        instead, which reads both pipes to EOF. A grandchild that inherited the
        pipes and outlived the killed child would hold that drain open past the
        budget and past the stop backstop :data:`DEFAULT_BACKEND_TIMEOUT_S` is
        sized against — which is why the expiry signals the child's whole
        process group through :meth:`_signal`, where the backend made one, rather
        than the child alone. What remains out of reach is a process that left
        the group of its own accord (``setsid``, ``nohup``-style daemonising);
        that one holds the drain exactly as before and is the sandbox's
        ordinary limit rather than a defect here.

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
            self._leads_a_group = process_group
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._signal(proc, process_group)
            stdout, stderr = proc.communicate()
            raise subprocess.TimeoutExpired(
                argv, timeout, output=stdout, stderr=stderr
            ) from None
        finally:
            with self._lock:
                self._running = None
                self._leads_a_group = False
        return ExecResult(stdout=stdout, stderr=stderr, exit_code=proc.returncode)

    @staticmethod
    def _signal(proc: subprocess.Popen[str], process_group: bool) -> None:
        """``SIGKILL`` *proc* — its whole process group where *process_group* says it leads one.

        A child started under ``_make_preexec`` called ``os.setpgrp()`` and is
        the **leader** of a group whose id is its own pid; the group is the
        subtree, and ``os.killpg(proc.pid, …)`` is what ends it — the shell and
        whatever the shell forked, together. A child that leads no group
        (docker's ``docker exec`` client, seatbelt's ``sandbox-exec``) shares
        this process's group and gets ``Popen.kill`` alone: ``killpg`` on *that*
        group would take the caller down with it, which is why nothing here
        guesses and the backend has to say so.

        Best-effort on every path. A child or a group that was already gone by
        the time the signal was sent is the outcome this asked for, so the
        ``OSError`` is swallowed; a group that could not be signalled still gets
        the direct-child kill, so the worst case is today's behaviour rather
        than no kill at all.
        """
        if process_group:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
                return
            except OSError:
                pass
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass

    def kill(self) -> None:
        """End the run in flight, if there is one.

        Idempotent by construction: with no handle there is nothing to signal and
        the call returns. A child that exited between reading the handle and
        signalling it is already what this method wanted, so the resulting
        ``ProcessLookupError`` is swallowed rather than raised at a caller who
        asked for exactly that outcome.

        **The whole subtree, where the backend made one.** ``local`` and
        ``bwrap`` put the child in a new process group, and the group — the
        shell *and* what the shell forked — is what :meth:`_signal` ends. That
        is the promise ``_make_preexec``'s docstring has always made, and on
        Linux, where ``sh -c '…'`` forks its command rather than exec'ing it in
        place, it is the difference between a kill that lands and one that
        leaves a grandchild holding the pipes. Docker and seatbelt create no
        group and are signalled as the direct child they are.
        """
        with self._lock:
            proc = self._running
            process_group = self._leads_a_group
        if proc is None:
            return
        self._signal(proc, process_group)

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
