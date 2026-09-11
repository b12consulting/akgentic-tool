"""``#Workspace-<workspace_name>``: the team child that owns one workspace tree's dispatch.

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

**The actor is an ordinary team child, and it holds no shared state** (ADR-051).
Its card creates it through ``getChildrenOrCreate``, so it belongs to exactly one
team and that team's teardown is the only thing that stops it. Two teams over one
tree therefore get two actors, which is correct rather than a regression: the
exec hold is a lock file, the document cache and the retrieval index are files
under ``<meta>``, and the write gate is an ``fcntl.flock`` — every shared thing is
on the tree, where the filesystem serialises it across processes as well as
teams. What is left here is **dispatch** over per-process resources: the sandbox
backend and its worker thread, the document reader, and the RAG indexing pipeline
whose ``IndexWorker`` and ``EmbeddingWorker`` children need a mailbox to report
to. That is the finished shape, not a leftover.

**The name carries the workspace, and that is load-bearing.** Get-or-create keys
on the actor *name*, so a fixed ``#Workspace`` would collapse two cards of one
team carrying different ``workspace_id`` values onto one actor owning one of the
two trees — silently. The unicity domain of an actor must equal the resource it
owns, and the resource is a tree.

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
itself, ``on_start``, ``worker_class``, the startup staging sweep ``on_start``
calls, and the agent-name map that gives a commit and a busy refusal a name to
print.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from akgentic.core.agent import Akgent
from akgentic.core.agent_state import BaseState
from akgentic.tool.core.deferred import DeferredResultActor, DeferredWorker
from akgentic.tool.workspace.actor.documents import DocumentsMixin
from akgentic.tool.workspace.actor.execution import EXEC_CAPABILITY, ExecMixin
from akgentic.tool.workspace.documents.store import DocumentStore
from akgentic.tool.workspace.execution import (
    ExecConfig,
    ExecOutcome,
    ExecRunner,
    RunningExec,
)
from akgentic.tool.workspace.journal import GitJournal, Identity
from akgentic.tool.workspace.lock import LockBackend
from akgentic.tool.workspace.models import STAGING_SWEEP_GRACE_S, WorkspaceConfig
from akgentic.tool.workspace.workspace import (
    Filesystem,
    get_workspace,
    is_staging_name,
    meta_dir_for,
)

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

The ``#`` prefix marks a tool actor for its team's teardown, and the full name is
the get-or-create key: two cards of one team on one tree get one actor, and two
cards on different trees get two.
"""

WORKSPACE_ACTOR_ROLE = "ToolActor"


def workspace_actor_name(workspace_name: str) -> str:
    """Return the singleton actor name owning *workspace_name*.

    Args:
        workspace_name: The **resolved** three-segment workspace path, exactly as
            the card derived it and as ``Filesystem`` receives it. Its slashes are
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
    DeferredResultActor[WorkspaceConfig, BaseState, str, ExecOutcome],
    # Redundant for the MRO — ``DeferredResultActor`` already is an ``Akgent`` —
    # and load-bearing for the restore. A store rebuilds this actor's state
    # through core's ``resolve_state_type``, which reads a direct
    # ``Akgent[Config, State]`` binding only; the one inherited through
    # ``DeferredResultActor`` carries type variables, so without this line the
    # answer is ``None`` and every restore is empty.
    Akgent[WorkspaceConfig, BaseState],
):
    """Team child owning one tree's exec dispatch and its RAG indexing pipeline.

    One per resolved path per team, created by its card through
    ``getChildrenOrCreate``. **The first card of a team to bind fixes the
    configuration every later card of that team gets on the tree**: get-or-create
    ignores ``config`` on a hit, so a second card that disagrees gets the actor as
    it was created, with nothing raised — two cards of one team disagreeing about
    one tree is a catalog inconsistency, not something a lookup arbitrates.

    **Its team owns its lifetime, and nothing else does.** It is in exactly one
    team's roster, so that team's two-phase teardown stops it like any other tool
    actor — through ``Akgent.stop``, whose ``stop_children`` takes any live
    ``#index-`` / ``#embed-`` worker down with it. There is no timer, no liveness
    sweep and no self-stop: the arrangement those existed for was an actor that
    sat outside every team's teardown, and this one does not.

    **It carries no state two instances could disagree about**, which is why two
    teams over one tree may each have one. ``state`` is a bare
    :class:`~akgentic.core.agent_state.BaseState` with no fields: the extraction
    cache and the retrieval index are one file per source document under the
    tree's sibling metadata directory (ADR-051 Decision 6), so a second actor —
    or a second *process* over the same mount — reads the same records with no
    shared memory. What the actor does keep in memory is the agent-name map, the
    exec run bookkeeping and the deferred result cache, all of them about runs it
    is dispatching itself.

    ``DocumentsMixin`` sits ahead of ``ExecMixin`` in the MRO, so anything it
    named ``deliver`` or ``fail`` would silently take over the deferred delivery
    path. It defines neither, and the public-API guard asserts that.

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
        self.state = BaseState()
        super().on_start()
        self._agent_names: OrderedDict[str, str] = OrderedDict()
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
        # Announced by the card at bind time, exactly as ``_exec_config`` is.
        # Until then ``request_exec`` refuses as unconfigured: a run admitted
        # under no hold is a run two workers could both admit.
        self._lock: LockBackend | None = None
        self._run_errors: OrderedDict[str, str] = OrderedDict()
        self._recent_runs: dict[str, OrderedDict[str, str]] = {}
        self._rag_params: WorkspaceRagIndex | None = None
        self._rag_reader: DocumentReader | None = None
        self._rag_collection: VectorStoreParam | None = None
        # Announced by the card at bind time, exactly as ``_lock`` is, and always
        # before ``enable_rag``: this actor resolves no store of its own, so
        # until the announcement lands there is nothing to enable retrieval over.
        self._vector_store: VectorStoreService | None = None
        self._vs_proxy: VectorStoreService | None = None
        self._embedder: EmbeddingProvider | None = None
        self._index_active: set[str] = set()
        # Announced by the card at bind time, exactly as ``_lock`` is. Until
        # then the document cache misses and the index looks empty: an ordinary,
        # visible degradation, never a raise (ADR-051 Decision 6).
        self._document_store: DocumentStore | None = None
        self._workspace: Filesystem = get_workspace(self.config.workspace_path)
        self._sweep_staging_files()
        self._journal = GitJournal(
            self._workspace._root,
            enabled=self.config.git_journal,
            timeout_s=self.config.git_timeout_s,
            meta_dir=meta_dir_for(self.config.workspace_path),
        )
        if self._journal.initialise():
            self._journal.seed_gitignore(self._workspace.write)
            self._journal.commit_out_of_band()

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

        The four exec steps and the reasoning behind their order live in
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

        It runs on two paths — its team's teardown, which is the ordinary one,
        and ``ActorSystem.shutdown`` through ``ActorRegistry.stop_all()``.

        Nothing here may raise past ``super()``: leaving a Pykka actor part-way
        stopped is worse than any error a step could report, which is why every
        step is wrapped individually.
        """
        self._teardown_exec()
        super().on_stop()

    ##
    ## Agent names — reached through the card's **ask** proxy, once, at bind time
    ##
    def attach(self, agent: ActorAddress, agent_name: str) -> None:
        """Record the name to print for *agent*, keyed by its id.

        Sent once per card, at bind, right after the get-or-create returns this
        actor's address — O(1), never on the mutation path. The map is keyed by
        ``str(agent.agent_id)``, the same string the card sends with every exec
        request.

        **The name is what a commit author and a refusal print, and that is the
        whole of what this call is for now.** The card can capture ``agent_id``
        without an edge back to the agent (ADR-030), but an id is a UUID: a
        refusal reading *"agent '3f2a…'"* tells a model nothing it can act on,
        and a journal entry authored by one names nobody a reader recognises.
        :meth:`_name_of` falls back to the id, so losing a name degrades the two
        messages rather than breaking either.

        It **is** capped, at ``max_tracked_writers``, because it is the one map
        here that grows with every agent that ever bound rather than with the
        runs in flight. Recording refreshes recency; the least recently recorded
        name is dropped over the cap.

        It still takes the whole address rather than the id alone: the id is
        derived here so that one caller cannot key the map differently from
        another, which is exactly how the exec refusal once printed a name for
        an agent and an id for the same agent one call later.

        **An ask, not a tell**, so a failure is seen by the card and fails the
        bind rather than leaving an agent bound to an actor that never heard of
        it.

        Args:
            agent: The binding agent's address.
            agent_name: Its configured, human-readable name.
        """
        agent_id = str(agent.agent_id)
        self._agent_names[agent_id] = agent_name
        self._agent_names.move_to_end(agent_id)
        while len(self._agent_names) > self.config.max_tracked_writers:
            self._agent_names.popitem(last=False)

    def _name_of(self, agent_id: str) -> str:
        """Return *agent_id*'s registered name, falling back to the id itself."""
        return self._agent_names.get(agent_id) or agent_id

    def _identity(self, agent_id: str) -> Identity:
        """Compose the git identity for *agent_id*: name to read, id to distinguish."""
        return Identity(self._name_of(agent_id), agent_id)

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
