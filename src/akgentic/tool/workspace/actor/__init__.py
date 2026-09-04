"""``#Workspace-<workspace_name>``: the team singleton that owns one workspace tree.

29-2 wired an actor that knew what everyone had read and decided nothing. This
module is where it decides: every mutation now runs **here**, on the actor's own
tree handle, and only after the live file still matches what the writing agent
observed (ADR-036 §3).

**The check and the write are one mailbox turn.** Returning a verdict and letting
the agent write would reopen the window the gate exists to close — between the
answer and the write, a third agent can land a mutation. That is why the
mutation helpers moved into :mod:`akgentic.tool.workspace.edit` and why this
actor performs real file I/O rather than adjudicating from a distance.

**The hash is read from disk on every check, never cached.** A
``{path -> current_sha}`` map would pass almost every test written against this
module and fail exactly one: the file written behind the actor's back. That case
is not exotic — it is the frontend upload, resource seeding, a sandbox run, and a
second team sharing a ``workspace_id``. Four writers that never call this actor,
all caught for free, because the check consults the *file* rather than a record
of who wrote it. Do not optimise this into a cache.

**Reads never come here for content.** They report what they saw through a
fire-and-forget ``tell`` and go straight to the agent's own ``Filesystem``. From
this story the ask path hashes files, so a reader that waited on it would queue
behind another agent's mutation hashing a large file — and an ask carries no
timeout.

**The name carries the workspace, and that is load-bearing.**
``getChildrenOrCreate`` keys on the actor *name*, so a fixed ``#Workspace`` would
collapse two cards carrying different ``workspace_id`` values onto one actor
owning one of the two trees — silently, in a team where
``WorkspaceTool(workspace_id="shared")`` is a documented configuration. The
unicity domain of an actor must equal the resource it owns, and the resource is a
tree. The ``#`` prefix stays: it is the orchestrator's two-phase-stop invariant.

**Every accepted mutation is one commit, and the journal sits at the one place
they converge.** :meth:`~akgentic.tool.workspace.actor.gate.GateMixin._journalled`
wraps all six ``apply_*`` bodies, so the out-of-band commit happens before any of
them touches disk and the agent's own commit happens after exactly one of them
succeeds. A seventh mutation added later cannot forget it, because there is
nowhere else to put one.

**Exec is fenced, not gated, and that is the whole difference.** Every other
writer here says what it is about to do, so the gate can check a precondition
against the file it names. A shell command cannot, so ``workspace_exec`` takes an
exclusive lease over the tree instead, and its write set is *discovered*
afterwards from ``git status --porcelain -uall`` (ADR-036 §5). A mutation
arriving under that lease is refused immediately, naming the holder; reads are
untouched and keep working throughout.

The deferred-result mechanism (ADR-033) is **engaged** from story 29-5, and its
seven rules apply in full: the blocking sandbox call happens in a ``#defer-``
worker, never on this thread. Everything the ask path still does is bounded — one
file read, one write, a few short-lived ``git`` forks under an explicit timeout —
and never external.

**The class is assembled from three per-concern mixins** (ADR-045 §1): the
observation and last-writer maps in :mod:`~akgentic.tool.workspace.actor.observation`,
the gate and the six mutations in :mod:`~akgentic.tool.workspace.actor.gate`, and
the lease and the deferred surface in :mod:`~akgentic.tool.workspace.actor.execution`.
Each body is the same code with the same ``self``; what stays here is the class
itself, ``on_start``, ``worker_class`` and the startup sweep ``on_start`` calls.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections import OrderedDict
from pathlib import Path

from akgentic.tool.core.deferred import DeferredResultActor, DeferredWorker
from akgentic.tool.workspace.actor.execution import EXEC_CAPABILITY, ExecMixin
from akgentic.tool.workspace.actor.gate import GateMixin
from akgentic.tool.workspace.actor.observation import ObservationMixin
from akgentic.tool.workspace.edit import EditMatcher
from akgentic.tool.workspace.execution import (
    ExecConfig,
    ExecLease,
    ExecOutcome,
    ExecWorker,
)
from akgentic.tool.workspace.journal import GitJournal
from akgentic.tool.workspace.models import (
    STAGING_SWEEP_GRACE_S,
    LastWrite,
    Observation,
    WorkspaceConfig,
    WorkspaceState,
)
from akgentic.tool.workspace.workspace import Filesystem, get_workspace, is_staging_name

__all__ = [
    "EXEC_CAPABILITY",
    "WORKSPACE_ACTOR_NAME",
    "WORKSPACE_ACTOR_ROLE",
    "WorkspaceActor",
    "workspace_actor_name",
]

logger = logging.getLogger(__name__)

WORKSPACE_ACTOR_NAME = "#Workspace"
"""Base actor name. The live name appends the workspace — see :func:`workspace_actor_name`.

The ``#`` prefix is the orchestrator's teardown invariant: it is what classifies
the actor as a tool actor during the two-phase stop.
"""

WORKSPACE_ACTOR_ROLE = "ToolActor"


def workspace_actor_name(workspace_name: str) -> str:
    """Return the singleton actor name owning *workspace_name*.

    Args:
        workspace_name: The resolved workspace — a card's ``workspace_id``, or
            the team id when it has none.

    Returns:
        ``#Workspace-<workspace_name>``.
    """
    return f"{WORKSPACE_ACTOR_NAME}-{workspace_name}"


def _is_sweepable_orphan(entry: Path, cutoff: float) -> bool:
    """Whether *entry* is a staging file old enough to have been abandoned.

    Args:
        entry: A path found under the workspace root.
        cutoff: The mtime below which a staging file counts as orphaned.

    Returns:
        True only for a regular file carrying the full staging shape and last
        modified before *cutoff*. A failed ``stat`` answers False: an entry this
        process cannot inspect is one it must not delete.
    """
    if not is_staging_name(entry.name):
        return False
    try:
        return entry.is_file() and entry.stat().st_mtime < cutoff
    except OSError:
        return False


class WorkspaceActor(
    ExecMixin,
    GateMixin,
    ObservationMixin,
    DeferredResultActor[WorkspaceConfig, WorkspaceState, str, ExecOutcome],
):
    """Team singleton owning one workspace tree, the observations, the gate, and the lease.

    Neither map is a state field: recording is not persisted state (see
    :class:`WorkspaceState`). The observation map is keyed
    ``agent_id -> path -> Observation`` and each inner map is an
    :class:`~collections.OrderedDict`, which is the whole of the LRU — recording
    moves an entry to the end, eviction pops from the front. The last-writer map
    is keyed by path across all agents and is capped independently.

    From story 29-5 it is also a :class:`~akgentic.tool.core.deferred.DeferredResultActor`
    keyed by run id. The base supplies ``_slots`` and ``_in_flight``; do not
    shadow them and do not add a second cache. Its LRU and its negative TTL are
    two of the seven deferred rules, and reimplementing either is how a partial
    adoption starts. **The mixins ahead of it in the MRO must not shadow it
    either** — ``cache_capacity`` is a class attribute with a default, so a mixin
    defining one would resize the LRU with no error and no log line.

    :meth:`~akgentic.tool.workspace.actor.execution.ExecMixin.exec_status`
    **does** read ``_in_flight``, and that is deliberate: a run is running iff it
    is in flight, and the earlier prohibition forced the question onto a map with
    a different capacity, which answered ``RUNNING`` for settled runs. Reading an
    own attribute on an own thread is not a partial adoption; duplicating the
    cache would be.
    """

    def on_start(self) -> None:
        """Initialise state, take the tree handle, sweep staging files, open the journal.

        **The order is load-bearing and there is only one correct one**, and two
        different orderings are being satisfied at once.

        ``self.state`` is assigned *before* ``super().on_start()`` because
        ``DeferredResultActor.on_start`` touches ``self.state`` on its first
        line. It also does not chain to ``Akgent.on_start`` — which is a no-op
        today, so nothing is lost, but that is a fact about the current core
        rather than a guarantee, and it is why this comment exists rather than
        silence.

        Everything after it keeps 29-4's order: sweeping *after* the initial
        commit would commit orphaned staging files and then delete them, and
        seeding ``.gitignore`` after that commit would leave the sidecars inside
        it.
        """
        self.state = WorkspaceState()
        super().on_start()
        self._observations: dict[str, OrderedDict[str, Observation]] = {}
        self._last_writers: OrderedDict[str, LastWrite] = OrderedDict()
        self._agent_names: OrderedDict[str, str] = OrderedDict()
        self._touched: list[str] = []
        self._matcher = EditMatcher()
        self._exec_config: ExecConfig | None = None
        self._lease: ExecLease | None = None
        self._reclaimed: OrderedDict[str, ExecLease] = OrderedDict()
        self._run_errors: OrderedDict[str, str] = OrderedDict()
        self._recent_runs: dict[str, OrderedDict[str, str]] = {}
        self._workspace: Filesystem = get_workspace(self.config.workspace_name)
        self._sweep_staging_files()
        self._journal = GitJournal(
            self._workspace._root,
            enabled=self.config.git_journal,
            timeout_s=self.config.git_timeout_s,
        )
        if self._journal.initialise():
            self._journal.seed_gitignore(self._workspace.write)
            self._journal.commit_out_of_band()

    def worker_class(self) -> type[DeferredWorker]:
        """Return :class:`~akgentic.tool.workspace.execution.ExecWorker`."""
        return ExecWorker

    ##
    ## Startup housekeeping
    ##
    def _sweep_staging_files(self) -> None:
        """Delete staging files an interrupted write left behind, anywhere in the tree.

        ``Filesystem.write`` publishes by rename from ``.<name>.<32 hex>.tmp`` in
        the target's own directory. A process killed between the two steps leaves
        one behind for good, and nothing else ever removes it.

        The sweep runs **at actor start only** — never on a timer, never per
        mutation — and matches the full staging shape, so a user's own
        ``.notes.tmp`` survives. Every failure is suppressed: a directory this
        process cannot clean must not stop the team's workspace from starting.

        **A staging file younger than the grace window is left alone**, because
        it is being written *now* and — with a ``workspace_id`` shared between
        two teams, which is a supported configuration — by somebody who may not
        be us. A tool actor's unicity domain is the team, so two teams over one
        tree means two actors each sweeping the whole tree at start; unlinking
        the other's staged file in the window between ``os.open`` and
        ``os.replace`` turns their healthy write into a refusal. An orphan is
        minutes or a restart old, so no realistic window confuses the two.
        """
        root = self._workspace._root
        cutoff = time.time() - STAGING_SWEEP_GRACE_S
        staged: list[Path] = []
        with contextlib.suppress(OSError):
            staged = [entry for entry in root.rglob("*") if _is_sweepable_orphan(entry, cutoff)]
        removed = 0
        for entry in staged:
            with contextlib.suppress(OSError):
                entry.unlink()
                removed += 1
        if removed:
            logger.info(
                "Workspace %s: swept %d orphaned staging file(s) at start",
                self.config.workspace_name,
                removed,
            )
