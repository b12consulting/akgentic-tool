"""The exclusive hold on a tree — a marker file, not a queue in one actor's memory.

An actor's hold is exclusive **inside one process**. pykka has no remote
addressing, a workspace singleton is one instance per process, and every
multi-worker deployment pins a team to one worker while mounting **one** volume
into all of them. Two teams on two workers attached to one tree therefore get two
unsynchronised actors over one directory — and an ``O_EXCL`` create on the shared
volume is exclusive exactly where the actor is not (ADR-051 Decision 5).

**The marker is a sibling of the tree, never a child of it.** It lives at
``<meta>/exec.lock``, where ``<meta>`` is
:func:`~akgentic.tool.workspace.workspace.meta_dir_for`'s directory — so no read
capability can name it, and an ``rm -rf`` from inside a sandboxed run cannot
delete the lock guarding that very run. **This module is the first writer to
``<meta>``**, and it creates the directory lazily, inside :meth:`acquire`: a
backend is built for every card that binds, and a workspace that never runs a
command must provision nothing.

**There is no FIFO queue.** A caller that finds the tree held is refused and
retries; among several waiters the first to retry wins, not the first to ask.
That supersedes ADR-047's queue, which lived in one process's memory and could
not have ordered waiters across workers without becoming a distributed
scheduler. It is a deliberate product behaviour, not a regression.

**Imports run one way only** — this module imports ``workspace.py`` for
:func:`meta_dir_for` and ``execution.py`` for the grace, the id and the refusal
text. Neither may import this one, or the pair becomes a cycle.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import ValidationError, model_validator

from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.tool.workspace.execution import LEASE_GRACE_S, exec_busy, new_run_id
from akgentic.tool.workspace.workspace import meta_dir_for

logger = logging.getLogger(__name__)

EXEC_LOCK_FILENAME = "exec.lock"
"""The marker's name under ``<meta>``, spelled once and only here.

One tree, one marker: this hold is over the **whole** tree, exactly as the
actor's lease was. The per-path write locks are a different mechanism in a
different file — an ``fcntl.flock`` under ``<meta>/locks/``, taken and released
inside one mutation (:mod:`akgentic.tool.workspace.card.gate`) — and the two must
not be confused: this one fences a shell command whose write set is unknowable,
those ones close a check-then-write window on a named path.
"""

_MARKER_MODE = 0o600
"""Mode the marker is created with — nothing outside the owner needs to read it."""

DEFAULT_LOCK_BACKEND = "file"
"""What ``AKGENTIC_LOCK_BACKEND`` resolves to when it is unset or empty."""

LOCK_BACKEND_ENV = "AKGENTIC_LOCK_BACKEND"
"""Environment variable naming the registry entry to build."""


class LockTicket(SerializableBaseModel):
    """What an acquirer offers about the run it wants to start.

    Attributes:
        agent_id: Who is asking, as a string. Recorded in the marker, so a
            human reading a wedged tree can see whose run took it.
        agent_name: That agent's configured, human-readable name, recorded in
            the marker beside the id. A **different process**'s mutation gate
            reads the marker to refuse a write while this run holds the tree,
            and it has no name map of its own to look the id up in — so the
            name has to travel with the hold or the refusal degrades to a UUID,
            which is something a model can read and nothing it can act on.
            Optional and defaulted: an acquirer that has no name loses the name
            and nothing else.
        cmd: The command, exactly as the agent gave it. Carried for a backend
            that wants to record or reject on it; the file backend records only
            the two ids, since the marker is read by a *different process* and a
            command line is the one field that can carry arbitrary bytes.
        budget_s: The **effective** run budget, already clamped. Half of the
            staleness window, and it comes from the acquirer because only the
            acquirer knows what budget the run will actually get.
    """

    agent_id: str
    cmd: str
    budget_s: float
    agent_name: str = ""


class LockGrant(SerializableBaseModel):
    """The answer to "may I take this tree, and under what id".

    Exactly one of the two fields is non-empty, and it is **enforced** rather
    than documented — the same rule, and the same reason,
    :class:`~akgentic.tool.workspace.execution.ExecStart` already enforces:
    every caller branches on ``if not grant.run_id``, so an instance carrying
    neither would surface to the agent as an empty refusal, and one carrying
    both would run a command the backend had already decided to refuse.

    Attributes:
        run_id: The issued id, when the tree was taken. It is the id the run
            is known by everywhere — so the id an agent holds and the id in the
            marker are the same value by construction, not by agreement.
        refusal: Why not, when it was not. It names **no** run and **no** agent
            (see :func:`~akgentic.tool.workspace.execution.exec_busy`).
    """

    run_id: str = ""
    refusal: str = ""

    @model_validator(mode="after")
    def _exactly_one(self) -> LockGrant:
        """Reject an answer that is both, or neither."""
        if bool(self.run_id) == bool(self.refusal):
            raise ValueError(
                "LockGrant carries exactly one of run_id or refusal — never both, never neither."
            )
        return self


class LockMarker(SerializableBaseModel):
    """What the marker file holds — the whole of the on-disk state.

    Read by a **different process** from the one that wrote it, which is why it
    is a model with a validator rather than a line of text: a marker that does
    not parse is not guesswork, it is a marker this code did not write.

    Attributes:
        run_id: The run holding the tree. A release must quote it, which is what
            stops a release from stealing a hold it does not own.
        agent_id: Who started that run.
        agent_name: That agent's display name, as its acquirer knew it.
            **Optional and defaulted**, so a marker written before this field
            existed still parses — the reader falls back to the id, which is the
            same degradation an unregistered agent already got. It is here
            because the reader of a marker is routinely a *different process*
            from its writer, and the mutation refusal it composes is read by a
            model deciding what to do next.
    """

    run_id: str
    agent_id: str
    agent_name: str = ""


@runtime_checkable
class LockBackend(Protocol):
    """The exclusive hold over one tree, and the whole of it.

    Three methods, and deliberately no **refresh**: there is no "am I still the
    holder" question, because nothing refreshes a hold mid-run (see
    :meth:`FileLockBackend.acquire`). :meth:`holder` is not that question — it
    is "who holds this tree *now*", asked by a reader that never took the hold
    and never will, so that a mutation gate in any process can refuse against
    the tree's own on-disk state rather than against one actor's memory
    (ADR-051 Decision 5).

    ``@runtime_checkable`` buys an ``isinstance`` check on **method names only**
    — not a signature, not an argument count, not a return type — exactly as
    :class:`~akgentic.tool.sandbox.backend.SandboxBackend`'s does. The real
    conformance check is mypy over ``src/``.

    Every registered backend is constructed with **no arguments**, so the
    registry can build any entry with no type switch; everything a backend needs
    arrives per call, in *tree_key*.
    """

    def acquire(self, tree_key: str, ticket: LockTicket) -> LockGrant:
        """Take the tree for a new run, or refuse."""
        ...

    def release(self, tree_key: str, run_id: str) -> None:
        """Give the tree back, if *run_id* is what holds it."""
        ...

    def holder(self, tree_key: str, budget_s: float) -> LockMarker | None:
        """Return the hold *tree_key* is genuinely under, or ``None``.

        Read-only: it takes nothing, releases nothing and writes nothing.

        *budget_s* is the run budget the staleness window is measured against,
        and it is a parameter rather than a constant for the reason
        :meth:`FileLockBackend.acquire` already takes it on its ticket — the
        window is ``budget_s + LEASE_GRACE_S``, and only the caller knows what
        budget runs on this tree get. A hold past it is not a hold: its run is
        not going to answer, and mutations proceed.
        """
        ...


class FileLockBackend:
    """An ``O_EXCL`` marker under ``<meta>``, which is what two workers share.

    **Stateless by construction.** It holds nothing per tree — everything comes
    from *tree_key* on each call — so one instance serves every tree, and two
    instances in one process are exactly as exclusive as two in two processes.
    That is what makes the concurrency guard meaningful: it acquires from two
    *objects*, so nothing in-process could be doing the excluding.

    **Staleness is measured on the marker's mtime**, against
    ``budget_s + LEASE_GRACE_S`` — the actor's own wedge predicate, one clock
    over. The two agree because **nothing refreshes the marker mid-run**: it is
    written once, at acquire, so its mtime *is* the run's start. Do not add a
    heartbeat. A refreshed mtime turns a bounded takeover into an unbounded one,
    and the case the grace exists for — a child that ignores the kill, with the
    thread waiting on it not free to say so — is precisely the case that would
    never refresh.

    ``time.time()`` rather than ``time.monotonic()`` is forced, not chosen: an
    mtime is wall clock and the two are not comparable. That inherits wall
    clock's hazards (an NTP step, skew between two workers on one share) and
    they are accepted — the window is seconds, and the failure is a takeover
    early or late by the skew, never a lost hold.
    """

    def acquire(self, tree_key: str, ticket: LockTicket) -> LockGrant:
        """Take the tree for a fresh run, or refuse because somebody holds it.

        The metadata directory is created **here**, lazily, rather than at
        construction: a backend is built for every card that binds, and a
        workspace that never runs a command must not create a directory for a
        tree nothing writes to. It is the same reasoning
        :meth:`~akgentic.tool.workspace.execution.ExecRunner._exec` gives for
        its lazy ``backend.start()``.

        On a marker that already exists the takeover is attempted **exactly
        once**: stale marker unlinked, ``O_EXCL`` create retried. A lost retry
        is a refusal — somebody else won the takeover, which is the correct
        answer and not a case to loop over.

        Args:
            tree_key: The two-segment ``<scope>/<leaf>`` path
                :func:`~akgentic.tool.workspace.workspace.get_workspace` takes —
                the same string ``ExecConfig.workspace_path`` holds.
            ticket: Who is asking, with what, and under what budget.

        Returns:
            A grant carrying a fresh run id, or a refusal naming nobody.

        Raises:
            OSError: Whatever creating the metadata directory or the marker
                raised — an unwritable parent, a full disk. The caller turns it
                into a refusal; it is never allowed to reach an agent as a crash.
        """
        marker = self._marker(tree_key)
        marker.parent.mkdir(parents=True, exist_ok=True)
        grant = self._claim(marker, ticket)
        if grant is not None:
            return grant
        if not self._is_stale(marker, ticket.budget_s):
            return LockGrant(refusal=exec_busy())
        logger.warning(
            "Workspace %s: taking over the exec lock at %s — it is past its budget and the "
            "grace with nothing released, so its run is not going to answer.",
            tree_key,
            marker,
        )
        marker.unlink(missing_ok=True)
        grant = self._claim(marker, ticket)
        return grant if grant is not None else LockGrant(refusal=exec_busy())

    def release(self, tree_key: str, run_id: str) -> None:
        """Give the tree back, but only if *run_id* is genuinely what holds it.

        Three ways this is a no-op, and each of them is a hold somebody else
        owns or none at all:

        - **no marker** — already released, or never taken. Nothing to do.
        - **a marker that does not parse** — left **in place** and reclaimed by
          staleness. Unlinking a marker we cannot read is precisely how a
          release steals a hold it does not own.
        - **a marker naming another run** — the tree was taken over while this
          run was wedged, and removing it would hand the tree to a third
          acquirer while the takeover's run is still writing.

        **The read and the unlink are not atomic**, and the window is named
        rather than closed: between the parse and the unlink another acquirer
        could take the marker over as stale and write its own, which this
        release then removes. It is microseconds against a staleness window of
        ``budget + LEASE_GRACE_S`` (≥ 20 s at the shipped defaults), and the
        failure is one spurious extra grant, never a lost hold. Closing it needs
        a rename dance or a directory lock — a design change, not a hardening.

        Args:
            tree_key: The tree whose hold is being given back.
            run_id: The run that believes it holds it.
        """
        marker = self._marker(tree_key)
        try:
            raw = marker.read_text()
        except FileNotFoundError:
            return
        try:
            held = LockMarker.model_validate_json(raw)
        except ValidationError:
            logger.warning(
                "Workspace %s: the exec lock at %s does not parse — leaving it for staleness "
                "to reclaim rather than unlinking a hold this run may not own.",
                tree_key,
                marker,
            )
            return
        if held.run_id != run_id:
            return
        marker.unlink(missing_ok=True)

    def holder(self, tree_key: str, budget_s: float) -> LockMarker | None:
        """Return the hold *tree_key* is under, or ``None`` — see :meth:`LockBackend.holder`.

        **Nothing is created, taken or released here**, which is what makes it
        safe to call on the mutation path: a workspace that has never run a
        command has no metadata directory, and asking who holds it must not make
        one.

        Three ways the answer is ``None``, and all three mean "the tree is
        free": there is no marker; the marker does not parse, so it is not one
        this code wrote and nothing can be said about whose hold it is; or it is
        past ``budget_s + LEASE_GRACE_S``, in which case its run is not going to
        answer and :meth:`acquire` would take it over.

        Args:
            tree_key: The two-segment ``<scope>/<leaf>`` path.
            budget_s: The effective run budget the staleness window is measured
                against.

        Returns:
            The parsed marker, or ``None``.

        Raises:
            OSError: Whatever reading the marker raised. The caller decides what
                a tree it cannot inspect means; it is never turned into a hold
                here, because "I could not read it" and "somebody holds it" are
                different answers.
        """
        marker = self._marker(tree_key)
        try:
            raw = marker.read_text()
        except FileNotFoundError:
            return None
        try:
            held = LockMarker.model_validate_json(raw)
        except ValidationError:
            return None
        return None if self._is_stale(marker, budget_s) else held

    def _marker(self, tree_key: str) -> Path:
        """The marker belonging to *tree_key* — a sibling of the tree, never inside it."""
        return meta_dir_for(tree_key) / EXEC_LOCK_FILENAME

    def _claim(self, marker: Path, ticket: LockTicket) -> LockGrant | None:
        """Create *marker* exclusively and fill it, or answer ``None`` if it exists.

        ``O_EXCL`` is what makes this exclusive across processes: the create
        either wins or raises, with no window between a check and a write for a
        second acquirer to fit into. A ``marker.exists()`` test followed by a
        write would pass every sequential test and serialise nothing at all.
        """
        try:
            fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _MARKER_MODE)
        except FileExistsError:
            return None
        run_id = new_run_id()
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(
                    LockMarker(
                        run_id=run_id,
                        agent_id=ticket.agent_id,
                        agent_name=ticket.agent_name,
                    ).model_dump_json()
                )
        except BaseException:
            # A half-written marker is one nothing can parse, so ``release``
            # would leave it for the staleness window rather than clear it —
            # holding a tree for a run that never started. We created it, so we
            # own it, and removing it is the one safe cleanup there is.
            with contextlib.suppress(OSError):
                marker.unlink(missing_ok=True)
            raise
        return LockGrant(run_id=run_id)

    def _is_stale(self, marker: Path, budget_s: float) -> bool:
        """Whether the existing marker is past the budget and the grace.

        A marker that vanished between the failed create and this ``stat`` is
        **not** stale — there is nothing to take over, and the honest answer to
        a hold that has just been released is a refusal the caller retries
        through, not a takeover of a file that no longer exists.

        The grace is :data:`~akgentic.tool.workspace.execution.LEASE_GRACE_S`,
        derived rather than respelled: two literals for one window is two places
        for a change to be applied once and missed once.

        **One predicate, two readers.** :meth:`acquire` asks it to decide a
        takeover and :meth:`holder` asks it to decide a refusal, and they must
        agree — a mutation refused against a hold that the very next
        ``request_exec`` would take over is a tree that says two things about
        itself at once. Hence the plain float rather than a ticket: the holder
        query has no ticket to offer and inventing one would have been a second
        spelling of the same window.
        """
        try:
            mtime = marker.stat().st_mtime
        except FileNotFoundError:
            return False
        return time.time() - mtime > budget_s + LEASE_GRACE_S


LOCK_BACKEND_CLASSES: dict[str, type[LockBackend]] = {"file": FileLockBackend}
"""The registry, with the one entry this story ships.

Shaped like ``SANDBOX_BACKEND_CLASSES``: a mutable dict a deployment assigns its
own backend into, resolved **at call time** so an entry registered after this
module was imported is still found. A Redis or Dapr hold is a second entry here
and nothing else.
"""


def resolve_lock_backend() -> LockBackend:
    """Build the backend ``AKGENTIC_LOCK_BACKEND`` names, defaulting to ``file``.

    Called at bind time, unconditionally, beside ``get_workspace`` — so a typo
    in the variable fails the bind in front of the admin who set it rather than
    at the first command, exactly as
    :func:`~akgentic.tool.workspace.execution.resolve_mode` already fails for an
    unknown sandbox mode.

    An **empty** value falls back rather than being honoured: a compose file
    interpolating an unset variable and a bare ``FOO=`` in an env file both
    arrive here as ``""``, and neither is somebody asking for a backend called
    the empty string.

    Returns:
        A fresh backend. It holds no per-tree state, so the caller may share one
        instance across every tree it binds.

    Raises:
        KeyError: If the variable names no registered backend — a configuration
            error, deliberately at start-up.
    """
    name = os.environ.get(LOCK_BACKEND_ENV) or DEFAULT_LOCK_BACKEND
    return LOCK_BACKEND_CLASSES[name]()
