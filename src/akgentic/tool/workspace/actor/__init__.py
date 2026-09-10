"""``#Workspace-<workspace_name>``: the hosted singleton that owns one workspace tree.

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

**The actor is hosted, not a team member.** Every card binds it through the
process's ``WorkspaceHost``, forwarded by its own team's orchestrator: the host
starts it with no orchestrator and no parent, so it is nobody's child, emits no
``StartMessage``, and is shared by every team whose cards resolve its path. Agents
``attach`` to it; teams do not own it.

**The name carries the workspace, and that is load-bearing.** The host keys its
registry on the actor *name*, so a fixed ``#Workspace`` would collapse two cards
carrying different ``workspace_id`` values onto one actor owning one of the two
trees — silently. The unicity domain of an actor must equal the resource it owns,
and the resource is a tree.

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

**The class is assembled from four per-concern mixins** (ADR-045 §1): the
extraction cache in :mod:`~akgentic.tool.workspace.actor.documents`, the
observation and last-writer maps in :mod:`~akgentic.tool.workspace.actor.observation`,
the gate and the six mutations in :mod:`~akgentic.tool.workspace.actor.gate`, and
the lease and the deferred surface in :mod:`~akgentic.tool.workspace.actor.execution`.
Each body is the same code with the same ``self``; what stays here is the class
itself, ``on_start``, ``init_state``, ``worker_class`` and the startup sweep
``on_start`` calls.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from akgentic.tool.core.deferred import DeferredResultActor, DeferredWorker
from akgentic.tool.workspace.actor.documents import DocumentsMixin
from akgentic.tool.workspace.actor.execution import EXEC_CAPABILITY, ExecMixin
from akgentic.tool.workspace.actor.gate import GateMixin
from akgentic.tool.workspace.actor.observation import ObservationMixin
from akgentic.tool.workspace.edit import EditMatcher
from akgentic.tool.workspace.execution import (
    ExecConfig,
    ExecOutcome,
    ExecRunner,
    QueuedExec,
    RunningExec,
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

if TYPE_CHECKING:
    from akgentic.core.actor_address import ActorAddress

    # Runtime slots the retrieval pipeline fills. Under ``TYPE_CHECKING`` because
    # the vector store lives behind an optional extra and ``card.params`` closes
    # an import cycle through this very module — neither is needed to annotate a
    # ``None`` at start.
    from akgentic.tool.vector_store.protocol import (
        EmbeddingProvider,
        VectorStoreParam,
        VectorStoreService,
    )
    from akgentic.tool.workspace.card.params import WorkspaceRagIndex
    from akgentic.tool.workspace.readers import DocumentReader

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

The ``#`` prefix stays although no orchestrator's teardown sees a hosted actor any
more: the full name is the ``WorkspaceHost``'s registry key and the store's scope,
and a bare path as a scope would collide with nothing today and something tomorrow.
"""

WORKSPACE_ACTOR_ROLE = "ToolActor"


def workspace_actor_name(workspace_name: str) -> str:
    """Return the singleton actor name owning *workspace_name*.

    Args:
        workspace_name: The **resolved** two-segment workspace path, exactly as
            the card derived it and as ``Filesystem`` receives it. Its slash is
            carried verbatim — nothing parses an actor name, and the path is
            injective by construction.

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
    DocumentsMixin,
    ExecMixin,
    GateMixin,
    ObservationMixin,
    DeferredResultActor[WorkspaceConfig, WorkspaceState, str, ExecOutcome],
):
    """Hosted singleton owning one tree, the extraction cache, the observations, the gate.

    One per resolved path per process, created by the ``WorkspaceHost`` and held
    by the agents that ``attach`` to it, from any number of teams. **The first
    bind fixes its configuration for every team on the tree**: the host ignores
    ``config`` on a hit, so a later card that disagrees gets the tree as it was
    created, with nothing raised — two cards disagreeing about one tree is a
    catalog inconsistency, not something the host arbitrates.

    ``DocumentsMixin`` sits ahead of ``ExecMixin`` in the MRO, so anything it
    named ``deliver`` or ``fail`` would silently take over the deferred delivery
    path. It defines neither, and the public-API guard asserts that.

    Neither observation map is a state field: recording is not persisted state,
    while the extraction cache is (see :class:`WorkspaceState`). The observation
    map is keyed
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
        self._holders: dict[str, ActorAddress] = {}
        self._observations: dict[str, OrderedDict[str, Observation]] = {}
        self._last_writers: OrderedDict[str, LastWrite] = OrderedDict()
        self._agent_names: OrderedDict[str, str] = OrderedDict()
        self._touched: list[str] = []
        self._matcher = EditMatcher()
        self._exec_config: ExecConfig | None = None
        self._runner: ExecRunner | None = None
        # One worker, unconditionally, for every workspace whether or not exec is
        # enabled. ``ThreadPoolExecutor`` spawns no thread until the first
        # ``submit``, so a workspace that never runs a command pays for the object
        # and nothing else — which is what makes the branch-free version correct
        # rather than merely tidy. One worker per tree is also what preserves the
        # serialisation the lease already guarantees: the tree admits one run at a
        # time, so a second worker could only ever idle.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"exec-{self.config.workspace_path}"
        )
        self._pending: Future[None] | None = None
        self._running: RunningExec | None = None
        self._queue: deque[QueuedExec] = deque()
        self._run_errors: OrderedDict[str, str] = OrderedDict()
        self._recent_runs: dict[str, OrderedDict[str, str]] = {}
        self._rag_params: WorkspaceRagIndex | None = None
        self._rag_reader: DocumentReader | None = None
        self._rag_collection: VectorStoreParam | None = None
        self._vs_proxy: VectorStoreService | None = None
        self._embedder: EmbeddingProvider | None = None
        self._index_active: set[str] = set()
        self._workspace: Filesystem = get_workspace(self.config.workspace_path)
        self._sweep_staging_files()
        self._journal = GitJournal(
            self._workspace._root,
            enabled=self.config.git_journal,
            timeout_s=self.config.git_timeout_s,
        )
        if self._journal.initialise():
            self._journal.seed_gitignore(self._workspace.write)
            self._journal.commit_out_of_band()

    def init_state(self, state: WorkspaceState) -> None:
        """Take a restored snapshot, then free anything left mid-embed (ADR-045 §7).

        **This is the resume hook, and ``on_start`` is not.** Story 45-7's
        acceptance criterion says the ``EMBEDDING`` reaper runs "on ``on_start``
        (resume)", and against this core it would run on an empty index: the first
        line of ``on_start`` assigns a fresh :class:`WorkspaceState`, and a
        restored snapshot arrives *afterwards* through this method — the one
        ``akgentic-team``'s restorer calls. Reaping in ``on_start`` is therefore
        provably a no-op, and the criterion's intent lands here instead.

        What it frees is a file whose ``EMBEDDING`` signal is never coming. The
        ``#embed-`` workers that would have reported it are children of this actor
        and died with the process; nothing survives a restart that could tell the
        file its batches are gone. Reverting it to ``PENDING`` costs one re-index
        and never a wrong answer.

        **``EMBEDDED`` rows are deliberately not re-marked here**, although on an
        in-memory engine they must be: the store child of a restored actor has no
        checkpoint and starts empty. At this moment the backend is unknown —
        ``_rag_collection`` is ``None`` from ``on_start`` until a card's
        ``enable_rag`` tell arrives, which the mailbox orders after this call —
        and a cluster engine loses nothing and must not be re-marked. The moment
        the in-memory child is created is the moment its emptiness is a fact, and
        it is the one that knows the backend, so the re-mark lives there:
        :meth:`~akgentic.tool.workspace.actor.documents.DocumentsMixin._requeue_embedded_rows`.

        Args:
            state: The snapshot to adopt.
        """
        super().init_state(state)
        self.reap_stale_embedding()

    def worker_class(self) -> type[DeferredWorker]:
        """Never called: nothing here is spawned through ``request()``.

        The base declares this abstract because its worker half spawns one actor
        per key through :meth:`~akgentic.tool.core.deferred.DeferredResultActor.request`,
        and ``core/deferred.py`` is shared with ``TeamTool``, which uses that half
        in full. This actor uses only the cache half — the sandbox performs the run
        and tells the report back — so ``request()`` is never called and there is
        nothing for this to return.

        **It does spawn actors directly, and that is not a contradiction.**
        ``DocumentsMixin`` creates an ``EmbeddingWorker`` per batch with
        ``createActor`` and hands it its payload directly, because that worker
        reports through this actor's ``receiveMsg_EmbeddingResult`` /
        ``receiveMsg_EmbeddingError`` rather than through ``deliver`` / ``fail`` —
        which here are the **exec** result cache. Routing it through ``request()``
        would put a batch of vectors into that cache and evict a running agent's
        exec outcome. The second actor spawned outside ``request()`` is the
        in-memory ``VectorStoreActor`` child — a storage engine this actor asks,
        not a worker that reports — created in the same way and stopped with
        this actor through ``stop_children``.

        Raising rather than returning a never-spawned stub: a stub would be dead
        code carrying a ``produce`` nobody runs, and the next reader would have to
        work out which of the two paths was live.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "#Workspace routes nothing through request() — the sandbox runs the "
            "command and reports back, and an embedding worker is spawned directly "
            "because it reports outside the deferred cache."
        )

    def on_stop(self) -> None:
        """Take exec down in its stated order, then chain to the base.

        The four steps and the reasoning behind their order live in
        :meth:`~akgentic.tool.workspace.actor.execution.ExecMixin._teardown_exec`,
        beside the exec code they tear down: cancel the queued runs, kill the
        running subprocess, drain the worker under a bound, release the backend.

        **A run in flight is no longer left alone.** It used to be, because it
        was on a second actor and its own budget was the only thing that could
        end it. It is now on this actor's own worker, and the tree it is writing
        to is this actor's, so a team that stops must not leave a command running
        in it. What bounds teardown is therefore the kill plus
        :data:`~akgentic.tool.workspace.execution.EXEC_SHUTDOWN_GRACE_S`, rather
        than one run's full budget.

        Nothing there may raise past ``super()``: leaving a Pykka actor part-way
        stopped is worse than any error a step could report, which is why every
        step is wrapped individually.
        """
        self._teardown_exec()
        super().on_stop()

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
        it is being written *now* and possibly by somebody who is not this
        actor: an upload, resource seeding, or — on the multi-worker tiers,
        where two processes can each host an actor over one tree — another
        process's ``#Workspace``. Unlinking such a staged file in the window
        between ``os.open`` and ``os.replace`` turns a healthy write into a
        refusal. An orphan is minutes or a restart old, so no realistic window
        confuses the two.
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
                self.config.workspace_path,
                removed,
            )
