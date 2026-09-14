"""``#Workspace-<workspace_name>``: the team child that owns one workspace tree's dispatch.

**This actor is dispatch, and nothing else** (ADR-053 Decision 6). It is created
only when a capability that dispatches is enabled — ``workspace_exec``, or one of
the three retrieval fields. A read-only or read/write card, which is the
overwhelming majority of them, creates no actor at all and loses nothing: what a
mailbox is genuinely needed for is a sandbox run whose report has to land
somewhere, and the ``#index-`` / ``#embed-`` children a Pydantic card cannot
parent.

**Mutations do not run here.** 29-2 wired an actor that knew what everyone had
read, and 29-3 made it decide; since story 52-5 the gate, the six mutation bodies
and the observation map are all **card-side**, in
:mod:`akgentic.tool.workspace.write.gate`. The lock that makes a check and a
write one unit is an ``fcntl.flock`` on the path — a lock on the *tree*, which is
what two workers over one mounted volume share, where a mailbox is only what one
process has. Reads do not come here either: a read runs on the calling agent's
own thread against its own ``Filesystem``, and records what it saw in the card's
own map.

**The hash is read from disk on every check, never cached** — a rule that is
still load-bearing and is now the **gate's**, stated where the gate lives. A
``{path -> current_sha}`` map would pass almost every test written against it and
fail exactly one: the file written behind its back. That case is not exotic — it
is the frontend upload, resource seeding, a sandbox run, and a second team
sharing a ``workspace_id``.

**The actor is an ordinary team child, and it holds no shared state** (ADR-051).
Its card creates it through ``getChildrenOrCreate``, so it belongs to exactly one
team and that team's teardown is the only thing that stops it. Two teams over one
tree therefore get two actors, which is correct rather than a regression: the
exec hold is a lock file, the document records are files under ``<meta>``, and the
write gate is an ``fcntl.flock`` — every shared thing is on the tree, where the
filesystem serialises it across processes as well as teams. What is left here is
**dispatch** over per-process resources: the sandbox backend and its worker
thread, the document reader, and the RAG indexing pipeline whose ``IndexWorker``
and ``EmbeddingWorker`` children need a mailbox to report to. That is the
finished shape, not a leftover.

**The name carries the workspace, and that is load-bearing.** Get-or-create keys
on the actor *name*, so a fixed ``#Workspace`` would collapse two cards of one
team carrying different ``workspace_id`` values onto one actor owning one of the
two trees — silently. **The unicity domain is ``(team, tree)``, and that is
correct**: one actor per team over a shared tree carries no correctness meaning,
because nothing shared is held here. Do not read the name rule as a claim that
two actors over one tree is a bug. It is the design.

**Exec is fenced, not gated, and that is the whole difference.** Every other
writer says what it is about to do, so the gate can check a precondition against
the file it names. A shell command cannot, so ``workspace_exec`` takes an
exclusive lease over the tree instead, and its write set is *discovered*
afterwards from ``git status --porcelain -uall`` (ADR-036 §5). A mutation
arriving under that lease is refused immediately, naming the holder; reads are
untouched and keep working throughout.

The deferred-result mechanism (ADR-033) is **engaged** from story 29-5, and its
seven rules apply in full. The blocking sandbox call is off this thread — on
``ExecRunner``'s own single worker since story 47-2, not in a ``#defer-`` worker.
Everything the ask path still does is bounded — one file read, one write, a few
short-lived ``git`` forks under an explicit timeout — and never external. The last
exception was ``rag_search``, which embedded a query and then searched a store
that may be a cluster client. **There is no ``rag_search`` here at all now**: a
search reads the document records, which is not dispatch, so the whole of it —
both legs, the fusion and the render — is the card's, in ``rag/search.py``, and
the snapshot render is in ``rag/context.py`` beside the models it builds.
``tests/workspace/test_rag_search_off_the_mailbox.py`` is what holds the sentence
to its word rather than leaving it a claim.

**The class is assembled from two per-concern mixins, and each one lives under
its own capability** (ADR-053 Decision 1): the retrieval pipeline in
:mod:`~akgentic.tool.workspace.rag.actor`, the lease and the deferred surface in
:mod:`~akgentic.tool.workspace.execution.actor`. Each body is the same code with
the same ``self``; what stays here is the class itself, ``on_start``,
``worker_class``, and the agent-name map that gives a commit and a busy refusal a
name to print. **This package therefore holds exactly one module**, and that is
the finished shape of the epic rather than a way-station: importing a mixin from
the capability that owns it is the assembly point doing its job, which is why
either capability's directory can still be deleted without touching the other.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from akgentic.core.agent import Akgent
from akgentic.core.agent_state import BaseState
from akgentic.tool.core.deferred import DeferredResultActor, DeferredWorker
from akgentic.tool.workspace.documents.cache import DocumentCache
from akgentic.tool.workspace.execution import (
    ExecConfig,
    ExecOutcome,
    ExecRunner,
    RunningExec,
)
from akgentic.tool.workspace.execution.actor import EXEC_CAPABILITY, ExecMixin
from akgentic.tool.workspace.journal import GitJournal
from akgentic.tool.workspace.lock import LockBackend
from akgentic.tool.workspace.models import Identity, WorkspaceConfig
from akgentic.tool.workspace.rag.actor import DocumentsMixin
from akgentic.tool.workspace.workspace import Filesystem, get_workspace, meta_dir_for

if TYPE_CHECKING:
    from akgentic.core.actor_address import ActorAddress

    # Runtime slots the retrieval pipeline fills. Under ``TYPE_CHECKING`` because
    # the vector store lives behind an optional extra — not needed to annotate a
    # ``None`` at start.
    from akgentic.tool.vector_store.protocol import VectorStoreParam, VectorStoreService
    from akgentic.tool.workspace.rag.params import WorkspaceRagIndex
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


class WorkspaceActor(
    DocumentsMixin,
    ExecMixin,
    DeferredResultActor[WorkspaceConfig, BaseState, str, ExecOutcome],
    # Redundant for the MRO — ``DeferredResultActor`` already is an ``Akgent`` —
    # but not for what core reads off the class. ``AgentCard`` coerces a dict
    # ``config`` to the agent class's config type, found by walking the MRO's
    # ``__orig_bases__`` for a concrete ``Akgent[Config, State]`` binding; the one
    # inherited through ``DeferredResultActor`` carries type variables, so without
    # this line the answer is ``None`` and the config stays a plain ``BaseConfig``.
    Akgent[WorkspaceConfig, BaseState],
):
    """Team child owning one tree's exec dispatch and its RAG indexing pipeline.

    **A per-team dispatch context that exists only when something dispatches.** It
    owns a sandbox run whose report must land somewhere, and the index/embed
    children a Pydantic card cannot parent. It holds the agent-name map those two
    print, and the deferred result cache the exec half keys on. Nothing else. Its
    unicity domain is ``(team, tree)``, and two actors over one tree is the design.

    That sentence is the end of a five-story arc and the file tree now agrees with
    it: every capability's actor-side code sits under the capability, and this
    package holds the class and nothing more.

    **Created only when the card enables exec or retrieval** (ADR-053 Decision
    6), and not otherwise: a read-only or read/write card binds a tree, seeds it,
    sweeps it, opens its journal and gates every mutation without one of these
    existing at all. One per resolved path per team where it does exist, created
    by its card through ``getChildrenOrCreate``. **The first card of a team to bind fixes the
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

    :meth:`~akgentic.tool.workspace.execution.actor.ExecMixin.exec_status`
    **does** read ``_in_flight``, and that is deliberate: a run is running iff it
    is in flight, and the earlier prohibition forced the question onto a map with
    a different capacity, which answered ``RUNNING`` for settled runs. Reading an
    own attribute on an own thread is not a partial adoption; duplicating the
    cache would be.
    """

    def on_start(self) -> None:
        """Initialise state, take the tree handle, and open the journal if there is one.

        ``self.state`` is assigned *before* ``super().on_start()`` because
        ``DeferredResultActor.on_start`` touches ``self.state`` on its first
        line. It also does not chain to ``Akgent.on_start`` — which is a no-op
        today, so nothing is lost, but that is a fact about the current core
        rather than a guarantee, and it is why this comment exists rather than
        silence.

        **Two duties left this method**, because they are not dispatch and this
        actor is only created when something dispatches (ADR-053 Decision 6).
        The staging sweep is
        :func:`~akgentic.tool.workspace.workspace.sweep_staging_files`, called
        from the card's ``observer()``; seeding ``.gitignore`` and making the
        initial out-of-band commit are the card's ``_open_journal``. A card with
        neither exec nor retrieval enabled has no actor to run them, and every
        one of them still has to happen — an orphaned staging file that nothing
        removes survives for ever, and 29-4's ordering argument (sweep before any
        write, seed before the first commit) is now satisfied inside
        ``observer()`` rather than here.

        **What is left is still gated.** The journal is built only when
        ``git_journal`` is on, mirroring the card's own ``_open_journal``. It used
        to be constructed and ``initialise``d on every bind whatever the setting,
        which is where the one WARNING a read-only bind logged came from.
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
        # The seven retrieval slots below are annotated on ``DocumentsMixin``, in
        # ``rag/actor.py``, and assigned here — which is exactly what ``ExecMixin``
        # does with its own eight. **The assignments stay in this method
        # deliberately**: moving them would mean giving a mixin an ``on_start`` and
        # inserting it into the chain, against the ordering ``self.state`` before
        # ``super().on_start()`` above states explicitly, and for no measured gain.
        # The actor is the assembly point — the role ``card/__init__.py`` plays for
        # the factory mixins — and an assembly point naming what it assembles is
        # not a capability leak.
        self._rag_params: WorkspaceRagIndex | None = None
        self._rag_reader: DocumentReader | None = None
        self._rag_collection: VectorStoreParam | None = None
        # Announced by the card at bind time, exactly as ``_lock`` is, and always
        # before ``enable_rag``: this actor resolves no store of its own, so
        # until the announcement lands there is nothing to enable retrieval over.
        self._vector_store: VectorStoreService | None = None
        self._vs_proxy: VectorStoreService | None = None
        # How many ``#index-`` workers **this process** is running, and nothing
        # else. Not the set it replaced: "is this path being worked on?" and "is
        # this path exempt from the reaper?" are questions about the *tree*, which
        # another process can answer differently, and both are now answered from
        # the row under its record hold. This is a resource bound on this process,
        # never a de-duplicator.
        self._index_workers: int = 0
        # Announced by the card at bind time, exactly as ``_lock`` is. Until
        # then the document cache misses and the index looks empty: an ordinary,
        # visible degradation, never a raise (ADR-051 Decision 6).
        self._document_cache: DocumentCache | None = None
        self._workspace: Filesystem = get_workspace(self.config.workspace_path)
        self._journal: GitJournal | None = None
        if self.config.git_journal:
            self._journal = GitJournal(
                self._workspace.root,
                enabled=self.config.git_journal,
                timeout_s=self.config.git_timeout_s,
                meta_dir=meta_dir_for(self.config.workspace_path),
            )
            self._journal.initialise()

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
        exec outcome. **It is the only actor spawned outside ``request()``**: the
        team's ``#VectorStore`` is not this actor's child and never was — the card
        creates it so that a planning or knowledge-graph card of the same team
        shares it, and ``_resolve_store`` says outright that this actor receives a
        store and creates no child.

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
        :meth:`~akgentic.tool.workspace.execution.actor.ExecMixin._teardown_exec`,
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
