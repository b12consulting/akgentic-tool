"""Workspace ToolCards — configurable read-only or full read/write/delete/edit access.

:class:`WorkspaceTool` exposes workspace operations as LLM-callable tools.
Pass ``read_only=True`` to restrict to read-side callables only (``workspace_read``,
``workspace_list``, ``workspace_glob``, ``workspace_grep``, ``workspace_view``).
The default ``read_only=False`` also includes write-side callables (``workspace_write``,
``workspace_delete``, ``workspace_edit``, ``workspace_multi_edit``, ``workspace_patch``,
``workspace_mkdir``).

**Reads and mutations both run here now.** A read runs on the calling agent's own
thread against its own :class:`~akgentic.tool.workspace.workspace.Filesystem` and
records what it saw in the card's own observation map. A mutation checks the live
file against that observation and performs the write, inside an ``fcntl.flock``
held on the path (ADR-051 Decision 3) — a lock on the *tree*, which is what two
workers over one mounted volume share, rather than a mailbox, which is what one
process has.

Nothing about the gate is visible in an LLM-facing signature: the six mutation
callables take exactly what they always took, and the precondition is derived
from what this agent observed. There is no digest, no ``expected``, and no
``force``.

**A workspace binds, gates and mutates in a process that hosts nothing.** There
is no resource host in this package any more and nothing here forwards to one:
the card creates its actor as an ordinary team child, which is the seam
``akgentic-infra``'s community wiring needs. That actor owns what is genuinely
per-process — the sandbox, the document reader, the retrieval pipeline — and no
shared state at all.

The factory bodies live in the five sibling mixins — ``card/read.py``,
``card/write.py``, ``card/gate.py``, ``card/execution.py``, ``card/rag.py`` — and
the capability parameters in ``card/params.py``. What stays here is the card
itself: its fields, ``observer()`` with its private binding helpers, and the
``get_tools`` / ``get_commands`` / ``get_context_states`` registration
(ADR-045 §1).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

from pydantic import Field, PrivateAttr, model_validator

from akgentic.core.actor_address import ActorAddress
from akgentic.core.orchestrator import Orchestrator
from akgentic.tool.core import (
    COMMAND,
    LLM_CONTEXT,
    TOOL_CALL,
    BaseToolParam,
    ContextState,
    ToolCard,
    _resolve,
)
from akgentic.tool.core.observer import ActorToolObserver
from akgentic.tool.vector_store.actor import (
    VS_ACTOR_NAME,
    VS_ACTOR_ROLE,
    VectorStoreActor,
    ensure_store_actor,
)
from akgentic.tool.vector_store.protocol import (
    VectorStoreConfig,
    VectorStoreParam,
    VectorStoreService,
    needs_store_actor,
    require_backend_configured,
    require_dimension_matches,
)
from akgentic.tool.vector_store.registry import BackendContext, get_backend_spec
from akgentic.tool.workspace.actor import (
    WORKSPACE_ACTOR_ROLE,
    WorkspaceActor,
    workspace_actor_name,
)
from akgentic.tool.workspace.card.execution import ExecFactories
from akgentic.tool.workspace.card.gate import CardGate
from akgentic.tool.workspace.card.params import (
    ExpandMediaRefs,
    Resource,
    ResourceType,
    WorkspaceDelete,
    WorkspaceEdit,
    WorkspaceExec,
    WorkspaceGlob,
    WorkspaceGrep,
    WorkspaceList,
    WorkspaceMkdir,
    WorkspaceMultiEdit,
    WorkspacePatch,
    WorkspaceRagIndex,
    WorkspaceRagList,
    WorkspaceRagSearch,
    WorkspaceRead,
    WorkspaceView,
    WorkspaceWrite,
)
from akgentic.tool.workspace.card.rag import (
    RagFactories,
    require_workspace_backend,
    workspace_backend,
)
from akgentic.tool.workspace.card.read import ReadFactories
from akgentic.tool.workspace.card.write import WriteFactories
from akgentic.tool.workspace.documents.models import EXTRACTOR_VERSION, derived_document_caps
from akgentic.tool.workspace.documents.store import DocumentStore, resolve_document_store
from akgentic.tool.workspace.event import WorkspaceAttached
from akgentic.tool.workspace.execution import (
    DEFAULT_EXEC_TIMEOUT_S,
    ExecConfig,
    effective_budget,
    resolve_mode,
)
from akgentic.tool.workspace.journal import GitJournal
from akgentic.tool.workspace.lock import LockBackend, resolve_lock_backend
from akgentic.tool.workspace.models import (
    DEFAULT_GIT_TIMEOUT_S,
    DEFAULT_MAX_OBSERVATIONS_PER_AGENT,
    Observation,
    WorkspaceConfig,
    content_sha,
)
from akgentic.tool.workspace.workspace import (
    Filesystem,
    get_workspace,
    meta_dir_for,
    resolve_workspace_path,
)

logger = logging.getLogger(__name__)

# Binds a capability's configuration field to the factory that consumes it.
_ParamT = TypeVar("_ParamT", bound=BaseToolParam)

__all__ = [
    "ExpandMediaRefs",
    "Resource",
    "ResourceType",
    "WorkspaceDelete",
    "WorkspaceEdit",
    "WorkspaceExec",
    "WorkspaceGlob",
    "WorkspaceGrep",
    "WorkspaceList",
    "WorkspaceMkdir",
    "WorkspaceMultiEdit",
    "WorkspacePatch",
    "WorkspaceRagIndex",
    "WorkspaceRagList",
    "WorkspaceRagSearch",
    "WorkspaceRead",
    "WorkspaceTool",
    "WorkspaceView",
    "WorkspaceWrite",
]


class WorkspaceTool(ReadFactories, WriteFactories, CardGate, ExecFactories, RagFactories, ToolCard):
    """Workspace access with configurable read-only or full read/write/delete/edit mode.

    Pass ``read_only=True`` to restrict to read-side tools only.  The default
    ``read_only=False`` also exposes write-side tools (write, delete, edit,
    multi_edit, patch, mkdir).

    Binary-extraction config lives on the nested :class:`WorkspaceRead` capability
    (``workspace_read=WorkspaceRead(document_reader=...)``), co-located with the read
    capability that uses it. ``WorkspaceRead.document_reader`` controls extraction:
    - ``True`` (default): uses a default ``DocumentReader()`` (Pass 1 only, no LLM).
    - ``False``: binary reads raise ``ValueError`` with install hint.
    - ``DocumentReader(...)`` instance: custom extraction config (e.g. with LLM).

    The five mixins carry the factory bodies and the gate, and declare no
    Pydantic field, so every field of this card is declared right here.
    """

    # Read capability fields (formerly in WorkspaceReadTool)
    workspace_id: str | None = None
    workspace_metadata_keys: list[str] = []
    """Metadata fields whose values key a workspace **shared** across teams and users.

    The third of three layouts (ADR-048 Decision 2). Declaring
    ``["customer_id", "case_id"]`` on a team whose metadata carries ``ACME`` and
    ``42`` resolves to ``_meta/customer_id-ACME__case_id-42`` — under a reserved
    scope rather than under anybody's principal, because sharing is the point.

    **Keys are a sequence, not a set:** they are joined in declaration order, so
    the list reads as a refinement path from the coarsest scope down and
    ``ls _meta/`` groups a customer's workspaces together. Two cards naming the
    same keys in different orders therefore address different workspaces —
    honest under the sequence model, and visible in the directory name. Values
    are percent-encoded, which is what keeps the join unforgeable.

    The resolver reads this field and nothing else reads it: the actor's config
    carries the resolved path, and a client learns which agent bound which tree
    from the ``WorkspaceAttached`` event the bind emits, never from a key list.

    Mutually exclusive with :attr:`workspace_id` — see :meth:`_one_layout`.
    """
    workspace_read: WorkspaceRead | bool = True
    workspace_view: WorkspaceView | bool = True
    workspace_list: WorkspaceList | bool = True
    workspace_glob: WorkspaceGlob | bool = True
    workspace_grep: WorkspaceGrep | bool = True
    expand_media_refs: ExpandMediaRefs | bool = True

    # Read-only gate (NEW)
    read_only: bool = False

    # Write capability fields
    workspace_write: WorkspaceWrite | bool = True
    workspace_delete: WorkspaceDelete | bool = True
    workspace_edit: WorkspaceEdit | bool = True
    workspace_multi_edit: WorkspaceMultiEdit | bool = True
    workspace_patch: WorkspacePatch | bool = True
    workspace_mkdir: WorkspaceMkdir | bool = True

    git_journal: bool = False
    """Whether accepted mutations are recorded in a git journal.

    A plain field, not a capability param: it exposes no tool, appears in no
    signature, and nothing about it is expressible by a model.

    **Off by default, because nothing in the system reads what it records.** The
    gate re-hashes the live file at mutation time and never consults the journal,
    and an agent's exec result carries ``exit_code``/``stdout``/``stderr`` and not
    the discovered write set — so the record exists only for a human reading
    ``git log`` afterwards. That is worth opting into, not worth three ``git``
    forks on every mutation by default. Turning it on buys history, attribution
    and out-of-band *detection*; leaving it off loosens the gate by nothing at
    all, because the gate is pure Python and independent.

    Note what get-or-create implies: the **first** card of a team to bind a tree
    decides that team's actor's configuration, exactly as the document caps
    already do. A second card arriving with ``git_journal=False`` does not turn
    off a journal that is already running, and a card arriving with it on does
    not start one on an actor already built without it — get-or-create ignores
    ``config`` on a hit. Two teams over one tree get two actors and so may
    disagree, and the tree survives that: the journal is a git repository on
    disk, which the team that enabled it writes and the team that did not simply
    leaves alone.
    """

    workspace_exec: WorkspaceExec | bool = False
    """Sandboxed shell execution — **off unless asked for**, and that is a security
    decision rather than a style one.

    Every other capability on this card defaults to on because every other one is
    a file operation the card already implies. Exec is not: defaulting it to
    ``True`` would give every ``WorkspaceTool()`` in existence sandboxed shell
    execution through a dependency bump, probe the host for docker at wiring
    time, and hand ``#Workspace`` a backend in teams that never asked for one.
    Capability escalation must be opt-in.

    It is also the one field that registers **two** callables — ``workspace_exec``
    and ``workspace_exec_result`` — breaking the card's otherwise strict
    one-field-one-callable convention. Deliberate: the result collector is
    meaningless without the runner, and separate fields would let a team enable
    the half that cannot do anything.

    Both live on the write side of ``read_only``: exec mutates the tree, whatever
    the command happens to be, so ``WorkspaceTool(read_only=True,
    workspace_exec=True)`` registers neither.
    """

    resources: list[Resource] = []
    """Files seeded into the team workspace at observer() time, before the
    agent's first turn. Each resource is written only if its path does not
    already exist — restoring a team never clobbers edited files."""

    workspace_rag_index: WorkspaceRagIndex | bool = False
    workspace_rag_list: WorkspaceRagList | bool = False
    workspace_rag_search: WorkspaceRagSearch | bool = False
    """Retrieval over the workspace tree — **all three off unless asked for**.

    This is ``workspace_exec``'s rationale one notch weaker. Every file capability
    on this card defaults to on because the card already implies file access;
    these three do not, because they reach the vector store and can spend
    embedding credits — a whole tree for the indexer, one query embed per call for
    the search. A capability that costs money on somebody else's account is
    opt-in.

    They are on the **read** side of ``read_only``: indexing and retrieval both
    derive from the tree and write nothing into it (ADR-045 §5), so
    ``WorkspaceTool(read_only=True, workspace_rag_index=True)`` registers the
    indexer and ``read_only=True, workspace_rag_search=True`` registers the search.
    """

    vector_store: VectorStoreParam = Field(default_factory=VectorStoreParam)
    """Backend, dimension, tenant, embedding model and provider of the one
    ``workspace_chunks`` collection.

    **The house name is now ``vector_store``, and this card no longer departs
    from it.** The old name ``rag_collection`` existed because a bare
    ``collection`` reads as "the workspace's collection of files" on a card whose
    other twenty fields are file operations — a real objection, which the new
    name answers rather than ignores: ``vector_store`` is unambiguous here, and
    ``PlanningTool`` and ``KnowledgeGraphTool`` carry the same field under the
    same name.

    **What an author writes here survives the bind unchanged, and is not what the
    store is built from.** ``observer()`` derives a copy carrying the resolved
    backend and this tree's ``<meta>`` as its ``root``; that copy is what the
    engine is built over and what reaches ``create_collection``. Two consequences
    follow. A ``root`` declared in a catalog is **inert** — the card overrides it
    unconditionally, so nothing an author writes can point one tree's index at
    another tree's directory. And a card that names no backend indexes into
    ``local``, files under its own tree's metadata directory, rather than into an
    index that dies with the process.

    Three things read it besides the collection itself: it decides the
    backend-derived document caps below, it is what ``require_backend_configured``
    and ``require_workspace_backend`` check, and it is what the resolved copy is
    derived from. All three only when a retrieval capability is actually enabled.
    """

    max_observations_per_agent: int = DEFAULT_MAX_OBSERVATIONS_PER_AGENT
    """Cap on the paths this agent's observation map remembers.

    **A card field because the map is the card's** (ADR-051 Decision 2). It was a
    ``WorkspaceConfig`` field while the actor held one map per agent — and was
    unreachable from a catalog even then, because the card never set it and the
    first bind fixed the tree's configuration for everybody. Here it is an
    ordinary declared field: the agent whose reads fill the map is the agent
    whose card declares the cap.

    Only the **path** dimension is bounded, and there is no other dimension: one
    card is one agent. Over the cap the least recently observed path is evicted,
    which costs a *refused* write rather than a stale accepted one — a
    correctness-preserving degradation, which is what makes an LRU safe here at
    all.
    """

    max_documents: int | None = None
    max_document_chars: int | None = None
    """Explicit overrides of the extraction cache's two caps.

    ``None`` is **not** zero and not "unset-so-use-the-default": it means *derive
    it*, from the vector backend and whether retrieval is on
    (:func:`~akgentic.tool.workspace.documents.models.derived_document_caps`). An
    explicit catalog value always wins (ADR-045 §7), which is what these two
    fields exist for.

    Note what get-or-create implies, exactly as ``git_journal`` records: the
    **first** card of a team to bind a tree decides that team's actor's
    configuration, so a second card of the same team arriving with different
    caps changes nothing.
    """

    # Private runtime state — not part of the serialised config.
    # Default None sentinel lets the workspace property detect uninitialized state
    # reliably under both normal execution and coverage instrumentation.
    _workspace: Filesystem | None = PrivateAttr(default=None)

    # Two proxies over the one ``#Workspace-<workspace>`` singleton, and the owning
    # agent's identity as a plain string.  All three are PrivateAttr: a proxy in a
    # Pydantic field breaks the card's serialisation contract, and the id is
    # captured as a string so no closure below holds an edge back to the agent
    # (ADR-030).  The proxies point at a *different* actor, so holding them
    # strongly roots nothing.
    #
    # The split is not stylistic.  Mutations must ask — the closure needs the
    # verdict.  Observations must tell — the reader needs nothing back, and an
    # ask would let a slow actor stall a read instead of refusing a write.
    _workspace_proxy: WorkspaceActor | None = PrivateAttr(default=None)
    _workspace_tell: WorkspaceActor | None = PrivateAttr(default=None)
    _agent_id: str = PrivateAttr(default="")
    # The owning agent's display **name**, captured beside its id and for the
    # same reason the actor's name map existed: a journal authored by UUID
    # satisfies the letter of "the git log is the who-changed-what record" and
    # defeats its purpose, and a refusal reading "last written by agent '3f2a…'"
    # tells a model nothing it can act on. A plain string, so it is no more an
    # edge back to the agent than the id is.
    _agent_name: str = PrivateAttr(default="")
    # The resolved two-segment path, kept because three runtime consumers need
    # it after ``observer()`` has returned: the exec hold's tree key, the
    # metadata directory, and the journal.
    _workspace_path: str = PrivateAttr(default="")
    # Where this tree's locks live — a **sibling** of the tree, so no read
    # capability can name a lock file. ``None`` until the bind.
    _meta_dir: Path | None = PrivateAttr(default=None)
    # The journal is runtime state, never a serializable field: it holds a
    # resolved git path and spawns subprocesses, and a ``ToolCard`` field holding
    # one would not round-trip (Golden Rule 1b).
    _journal: GitJournal | None = PrivateAttr(default=None)
    # This agent's slice of what has been read, and nobody else's. An
    # ``OrderedDict`` because it *is* the LRU. Never serialised: recording is not
    # persisted state.
    _observations: OrderedDict[str, Observation] = PrivateAttr(default_factory=OrderedDict)
    # The paths the mutation in flight has changed, collected on the accept path
    # for the commit and the stale-mark that follow it.
    _touched: list[str] = PrivateAttr(default_factory=list)
    # The effective run budget the exec hold's staleness window is measured
    # against, derived once at bind from this card's own exec configuration —
    # the same derivation ``ExecMixin._run_budget`` makes, so a mutation is never
    # refused against a hold the next ``request_exec`` would take over.
    _exec_budget_s: float = PrivateAttr(default=0.0)
    # Runtime state, never a serializable field: a backend is an object with a
    # method, and a ``ToolCard`` field holding one would not round-trip.
    _lock_backend: LockBackend | None = PrivateAttr(default=None)
    # Same reasoning: where this tree's document records live is an object with
    # methods, and a serializable field holding one would not round-trip
    # (Golden Rule 1b).
    _document_store: DocumentStore | None = PrivateAttr(default=None)
    # The author's ``vector_store`` with the backend resolved and this tree's
    # ``<meta>`` stamped into ``root`` — derived at bind, never written back onto
    # the field, so a stored card still says what its author wrote.
    _resolved_store: VectorStoreParam | None = PrivateAttr(default=None)
    # The engine itself: a proxy over the team's ``#VectorStore`` for a backend
    # that needs an actor, a freshly built client for one that does not. Runtime
    # state, so a ``PrivateAttr`` for the reason above.
    _vector_store: VectorStoreService | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _one_layout(self) -> WorkspaceTool:
        """Refuse a card that names its workspace twice.

        A **validation error**, not a precedence rule. "Metadata wins" would be a
        silent answer to a question the author got wrong, and two ways to name
        one tree on one card is a mistake worth surfacing at declaration time —
        where the person who wrote it is looking (ADR-048 Decision 2).

        Raises:
            ValueError: If both ``workspace_id`` and ``workspace_metadata_keys``
                are set.
        """
        if self.workspace_id is not None and self.workspace_metadata_keys:
            raise ValueError(
                "workspace_id and workspace_metadata_keys are mutually exclusive: "
                f"got workspace_id={self.workspace_id!r} and "
                f"workspace_metadata_keys={self.workspace_metadata_keys!r}"
            )
        return self

    def observer(  # type: ignore[override]
        self, observer: ActorToolObserver
    ) -> WorkspaceTool:
        """Attach observer, initialise the backend, and bind the workspace singleton.

        The workspace path is derived **once**, here, by the one resolver, and
        handed to everything downstream as an already-resolved value: the
        ``Filesystem``, the ``#Workspace`` actor's name and config, and the
        ``ExecConfig`` the sandbox is built from. Nothing below re-derives it,
        which is what makes it impossible for a backend to open a different
        directory from the one the gate and the journal are guarding.

        Args:
            observer: Actor tool observer; must have a non-None orchestrator.

        Returns:
            Self, enabling method chaining.

        Raises:
            ValueError: If ``observer.orchestrator`` is None, or if the workspace
                path cannot be derived — an unusable ``user_id``, an unusable
                ``workspace_id``, or any of the metadata conditions. That raise
                fails card binding and therefore team creation, **deliberately**:
                it surfaces in front of the admin who caused it, rather than
                silently collapsing several principals into one tree. It must
                never be caught and turned into a fallback.
            RuntimeError: Whatever a failing ``attach`` raises, propagated
                unchanged — an agent must never hold a tree that does not know
                it is held. It must never be caught, for the same reason.
        """
        if observer.orchestrator is None:
            raise ValueError("WorkspaceTool requires access to the orchestrator.")
        if self._rag_enabled():
            # Only when retrieval is on. ``protocol.py`` raises when a card names
            # a store the environment has not provisioned, which is correct for a
            # card that asked for durable shared storage and wrong to impose on
            # the overwhelming majority of ``WorkspaceTool()`` instances that
            # never enable retrieval at all.
            #
            # The guard is backend-agnostic now, not Weaviate-shaped. The old
            # check only tested ``backend == "weaviate"``, so a card naming
            # ``qdrant`` with no URL passed it and degraded silently at
            # ``enable_rag``; it now fails the team's build here, which is what
            # ``PlanningTool`` and ``KnowledgeGraphTool`` already did.
            require_backend_configured(self.vector_store, "WorkspaceTool")
            require_dimension_matches(self.vector_store, "WorkspaceTool")
            # One line later, and refusing rather than substituting: an author who
            # wrote the in-actor backend asked for an index that dies with the
            # process, under rows that do not. See ``WORKSPACE_IN_MEMORY_REFUSED``.
            require_workspace_backend(self.vector_store, "WorkspaceTool")
        super().observer(observer)  # store the observer weakly via the base setter
        ws_path = str(self._resolve_path(observer, observer.orchestrator))
        self._workspace_path = ws_path
        self._meta_dir = meta_dir_for(ws_path)
        self._exec_budget_s = self._mutation_budget()
        self._workspace = get_workspace(ws_path)
        # Unconditional, beside the filesystem and for the same reason: a bad
        # ``AKGENTIC_LOCK_BACKEND`` must fail the bind in front of the admin who
        # set it, not at the first command. Constructing one creates nothing —
        # the backend is stateless and touches the disk only on ``acquire``.
        self._lock_backend = resolve_lock_backend()
        # Unconditional for the same reason as the lock backend, and one line
        # later so the two failures are indistinguishable to an admin: a bad
        # ``AKGENTIC_DOCUMENT_STORE`` must fail the bind in front of whoever set
        # it, not at the first document read. Constructing one creates nothing —
        # the store is stateless and touches the disk only on a put.
        self._document_store = resolve_document_store()
        # Only when retrieval is on, for ``_bind_vector_store``'s own reason: a
        # bare ``WorkspaceTool()`` must create no ``#VectorStore`` and open no
        # client. Derived before the bind so the actor's config can be built from
        # the backend the collection will really use.
        self._resolved_store = self._resolve_store_param(ws_path)
        self._seed_resources()
        self._bind_workspace_actor(observer, observer.orchestrator, ws_path)
        # After the bind, because the actor's own ``on_start`` creates the
        # repository and seeds ``.gitignore``; this journal opens the same
        # repository to *commit* into rather than to create it. Before the first
        # mutation, which is all the gate needs.
        self._open_journal(ws_path)
        # Between the bind and the retrieval announcement, deliberately: the
        # actor must never be able to enable retrieval under a store it has not
        # been given, exactly as it must never admit a run under a hold it has
        # not been given.
        self._announce_document_store()
        self._bind_sandbox(observer, ws_path)
        self._bind_vector_store(observer, observer.orchestrator)
        # Before ``_announce_rag``, and that ordering is the invariant: the actor
        # enables retrieval on the turn it receives ``enable_rag``, so a store
        # announced afterwards would arrive to an actor that has already decided
        # it has none.
        self._announce_vector_store()
        self._announce_rag()
        return self

    def _mutation_budget(self) -> float:
        """The run budget this card measures an exec hold's staleness against.

        The same derivation
        :meth:`~akgentic.tool.workspace.actor.execution.ExecMixin._run_budget`
        makes, for the reason two spellings of one window would be two places
        for a change to be applied once and missed once: a mutation refused
        against a hold that the very next ``request_exec`` would take over is a
        tree saying two things about itself at once.

        **A card with exec off still has one**, and must: the hold it reads may
        have been taken by a *different* card — or a different process — and a
        read-only card still refuses mutations it does not have. The default
        budget is what that card knows.
        """
        params = _resolve(self.workspace_exec, WorkspaceExec)
        return effective_budget(params.timeout_s if params is not None else DEFAULT_EXEC_TIMEOUT_S)

    def _open_journal(self, workspace_path: str) -> None:
        """Open this card's view of the tree's git journal, or leave it inert.

        **One journal object per card, over one repository per tree**, which is
        the arrangement ADR-051 Decision 4 chose over routing commits through an
        actor: the repository is on disk and shared by every process that mounts
        the tree, so a single in-process owner could never have serialised the
        writers that matter. What does serialise them is the journal's own
        ``flock`` (:meth:`~akgentic.tool.workspace.journal.GitJournal._holding`).

        ``initialise`` runs **only when the card asked for a journal**. With
        ``git_journal=False`` the object is left un-initialised, which is the
        same inert state ``initialise`` would put it in — every method a no-op —
        minus a second "disabled by configuration" warning for a tree whose actor
        has just logged one.

        Args:
            workspace_path: The resolved two-segment path, whose metadata
                directory holds the commit lock.
        """
        assert self._workspace is not None
        journal = GitJournal(
            self._workspace._root,
            enabled=self.git_journal,
            timeout_s=DEFAULT_GIT_TIMEOUT_S,
            meta_dir=meta_dir_for(workspace_path),
        )
        if self.git_journal:
            journal.initialise()
        self._journal = journal

    def _resolve_store_param(self, workspace_path: str) -> VectorStoreParam | None:
        """The author's ``vector_store`` with the backend resolved and the root stamped.

        ``model_copy(update=...)`` rather than a rebuild: naming the six fields
        that exist today would silently drop the seventh added tomorrow, on a
        param that is persisted in a catalog and re-read (Golden Rule 12).

        **``self.vector_store`` is not mutated.** A card is a stored record of
        what its author declared; the substitution and the root belong to this
        bind, on this host, over this tree — and a root written back onto the
        field would be re-persisted into a catalog and then followed by a
        deployment whose workspaces root is somewhere else.

        The root is derived from the already-resolved *workspace_path* through
        the one resolver, so the index cannot land beside a different tree from
        the one the gate and the journal are guarding.

        Args:
            workspace_path: The resolved two-segment path this card is anchored to.

        Returns:
            The resolved param, or ``None`` when this card enables no retrieval
            and therefore owns no collection.
        """
        if not self._rag_enabled():
            return None
        return self.vector_store.model_copy(
            update={
                "backend": workspace_backend(self.vector_store),
                "root": str(meta_dir_for(workspace_path)),
            }
        )

    def _bind_vector_store(self, observer: ActorToolObserver, orchestrator: ActorAddress) -> None:
        """Resolve the engine this tree's chunks are written into — the card's job.

        The third instance of one pattern, not a new one: the card builds or
        resolves the runtime object in ``observer()`` and announces it, exactly as
        it does for the exec hold and the document store (ADR-051 Decision 8). The
        actor receives; it resolves nothing and asks nobody.

        **Which of the two branches runs is the registry's answer, never a name
        test.** A backend that needs an actor gets the team's one — created
        through ``ensure_store_actor`` and looked up by name, so three cards in a
        team share it — and one that does not is built straight from its factory,
        because there would be nothing for an actor to hold but a socket.

        **The team's real id is passed, and that is not the defect story 51-4
        neutralised.** That fix passed ``None`` because the workspace actor of
        the day minted a fresh id per lifetime, so the row identity moved and a
        re-add doubled;
        the identity no longer carries the team on a shared collection at all
        (:func:`~akgentic.tool.vector_store.protocol.row_object_id`), so the team
        is free to be what it is — and it has to be, since it is what a sweep and
        a team-scoped filter read.

        Every failure here **degrades**: one WARNING, no store announced, and the
        actor answers its existing unavailable sentence. A retrieval capability is
        one of twenty on a card whose others are file operations, and this method
        runs after the tree is already bound.

        Args:
            observer: The owning agent, live at bind time.
            orchestrator: Address of the orchestrator.
        """
        resolved = self._resolved_store
        if resolved is None:
            return
        try:
            self._vector_store = self._build_store(observer, orchestrator, resolved)
        except Exception as exc:  # noqa: BLE001 — a store is not worth failing a bind for
            logger.warning(
                "Workspace card: could not resolve the '%s' vector store: %s — retrieval "
                "stays off for this card",
                resolved.backend,
                exc,
            )

    def _build_store(
        self, observer: ActorToolObserver, orchestrator: ActorAddress, resolved: VectorStoreParam
    ) -> VectorStoreService | None:
        """Return the engine *resolved* names, over an actor or over a client."""
        orchestrator_proxy = observer.proxy_ask(orchestrator, Orchestrator)
        if not needs_store_actor(resolved):
            return get_backend_spec(resolved.backend).factory(
                BackendContext(
                    config=VectorStoreConfig(name=VS_ACTOR_NAME, role=VS_ACTOR_ROLE),
                    team_id=str(observer.team_id),
                    root=resolved.root,
                )
            )
        ensure_store_actor(resolved, orchestrator_proxy)
        address = orchestrator_proxy.get_team_member(VS_ACTOR_NAME)
        if address is None:
            logger.warning(
                "Workspace card: '%s' was not found in the team after ensure_store_actor "
                "— retrieval stays off for this card",
                VS_ACTOR_NAME,
            )
            return None
        return observer.proxy_ask(address, VectorStoreActor)

    def _announce_vector_store(self) -> None:
        """Tell the actor which engine this tree's chunks go into — fire and forget.

        Guarded exactly as :meth:`_announce_document_store` is, and it degrades
        the same way: without a store the actor leaves ``_vs_proxy`` at ``None``,
        logs one WARNING naming the workspace and answers the unavailable sentence
        from ``workspace_rag_index`` — visible, and recoverable by rebinding.

        **Conditional, unlike the document store's announcement**: nothing is
        resolved at all for a card with retrieval off, so there is nothing here to
        announce and no ``#VectorStore`` to have created.
        """
        tell = self._workspace_tell
        store = self._vector_store
        if tell is None or store is None:
            return
        try:
            tell.configure_vector_store(store)
        except Exception:
            logger.debug("Could not announce the vector store to #Workspace", exc_info=True)

    def _resolve_path(
        self, observer: ActorToolObserver, orchestrator: ActorAddress
    ) -> PurePosixPath:
        """Derive this card's two-segment workspace path, through the one resolver.

        ``observer.user_id`` is read as a **typed attribute**. A defaulted
        ``getattr`` would turn an observer that never received the identity into
        a silent fall-back to the anonymous scope — every user's tree quietly
        merged into one, with nothing raised and nothing logged.

        The team's metadata is fetched **only** when this card declares keys, so
        a bare ``WorkspaceTool()`` gains no bind-time round trip. It is the same
        call ``MetadataTool`` already makes at bind time.

        Args:
            observer: The owning agent, live at bind time.
            orchestrator: Address of the orchestrator, taken as a parameter
                already narrowed by the caller rather than re-tested here. A
                second ``is not None`` test inside the fetch condition would
                collapse "declares no keys" and "has no orchestrator" into the
                same ``None``, so a wiring failure would surface as the
                unrelated "the team carries no metadata".
        """
        metadata = (
            observer.proxy_ask(orchestrator, Orchestrator).get_metadata()
            if self.workspace_metadata_keys
            else None
        )
        return resolve_workspace_path(
            workspace_id=self.workspace_id,
            workspace_metadata_keys=self.workspace_metadata_keys,
            team_id=str(observer.team_id),
            user_id=observer.user_id,
            metadata=metadata,
        )

    def _enabled_exec(self) -> WorkspaceExec | None:
        """Return the exec configuration only when it will register callables.

        One predicate, because the two halves of this capability have to agree on
        what "on" means. They did not: the wiring looked at the field and
        ``read_only``, while ``_exec_tools`` also required the ``TOOL_CALL``
        channel — so a card that put exec off the tool channel still resolved the
        backend, still emitted the ``auto`` fallback warning, and — while a
        sandbox actor still existed — still brought one up (a running container,
        on the docker backend) to serve two callables it then never registered.

        Returns:
            The parameters, or ``None`` when nothing exec-related should happen.
        """
        params = _resolve(self.workspace_exec, WorkspaceExec)
        if params is None or self.read_only or TOOL_CALL not in params.expose:
            return None
        return params

    def _bind_sandbox(self, observer: ActorToolObserver, workspace_path: str) -> None:
        """Resolve the mode and tell ``#Workspace`` which backend to run on.

        **Nothing happens here when the capability is off** — no host probe, no
        message. That is the whole of what ``workspace_exec=False`` buys, and it
        is why the check is at the top rather than inside.

        **No actor is created here.** ``#Workspace`` owns its own backend and its
        own worker thread; the sandbox actor that used to be created at this
        point is gone, and one would be an actor nothing uses — on the docker
        backend, a container nobody execs in, provisioned at wiring time for a
        team that may never run a command.

        The order matters: this runs *after* ``_bind_workspace_actor``, because
        ``configure_exec`` travels over the tell proxy that method binds, and
        after ``attach``, so the actor can already name this agent in a refusal
        the first run causes.

        **``resolve_mode``'s instance is still dropped here, and that is
        correct**: this card does not run commands. What it needs from that call
        is the resolved mode and the ``"auto"`` probe's ``DeprecationWarning``,
        which must fire at wiring time in front of the admin who configured the
        card. ``#Workspace.configure_exec`` makes the same call with the concrete
        mode — so the probe short-circuits, no second warning fires — and keeps
        the instance.

        Args:
            observer: The owning agent, live at bind time.
            workspace_path: The already-resolved two-segment path this card is
                anchored to. Passed down rather than re-derived, so the backend
                cannot open a directory other than the one being gated.

        Raises:
            KeyError: If the configured mode names no registered backend —
                fail-fast at wiring time rather than at the first command.
        """
        params = self._enabled_exec()
        if params is None:
            return
        mode, _backend = resolve_mode(params.mode)
        # Before the exec config, deliberately: the actor refuses every run
        # until it has both, so announcing the hold first means it can never
        # admit a run under a backend it has not been given.
        self._announce_lock()
        self._announce_exec(
            ExecConfig(mode=mode, workspace_path=workspace_path, timeout_s=params.timeout_s)
        )

    def _announce_lock(self) -> None:
        """Tell the actor what the tree's exclusive hold is taken on — fire and forget.

        Guarded exactly as :meth:`_announce_exec` is, and it degrades the same
        way: without a backend the actor refuses every run as unconfigured —
        visible, and recoverable by rebinding. A raise at wiring time is neither.

        The backend is built in :meth:`observer`, so a card whose exec capability
        is off still resolves one; it simply never announces it. What that costs
        is nothing — the object is stateless and touches no disk until an
        ``acquire`` that will never come.
        """
        tell = self._workspace_tell
        backend = self._lock_backend
        if tell is None or backend is None:
            return
        try:
            tell.configure_lock(backend)
        except Exception:
            logger.debug("Could not announce the exec lock backend to #Workspace", exc_info=True)

    def _announce_document_store(self) -> None:
        """Tell the actor where this tree's document records live — fire and forget.

        Guarded exactly as :meth:`_announce_lock` is, and it degrades the same
        way: without a store the actor's document cache misses and
        ``workspace_rag_index`` answers its existing unavailable sentence —
        visible, and recoverable by rebinding. A raise at wiring time is neither,
        and this card's other twenty capabilities are file operations that have
        nothing to do with retrieval.

        **Unconditional, unlike the lock's announcement.** The store is not a
        retrieval capability: ``document_extract`` serves every read that goes
        through the extractor, whether or not any card on this tree ever enables
        an index. A card that gated this on ``_rag_enabled()`` would leave the
        common read path with no cache at all.
        """
        tell = self._workspace_tell
        store = self._document_store
        if tell is None or store is None:
            return
        try:
            tell.configure_document_store(store)
        except Exception:
            logger.debug("Could not announce the document store to #Workspace", exc_info=True)

    def _announce_exec(self, config: ExecConfig) -> None:
        """Tell the actor which backend to run commands on — fire and forget.

        Guarded, because a lost announcement degrades: a stand-in proxy that does
        not carry the method, or an actor that died between the bind and this
        line, costs an exec request refused for want of a backend — visible, and
        recoverable by rebinding. A raise at wiring time is neither.

        ``attach`` is deliberately **not** guarded like this, and the difference
        is the point: a lost announcement of a *backend* is recoverable by
        rebinding, while a lost ``attach`` leaves the actor with no name for this
        agent — so every commit it authors and every refusal that names it fall
        back to a UUID, for the life of the tree, with nothing raised.
        """
        tell = self._workspace_tell
        if tell is None:
            return
        try:
            tell.configure_exec(config)
        except Exception:
            logger.debug("Could not announce the exec backend to #Workspace", exc_info=True)

    def _bind_workspace_actor(
        self, observer: ActorToolObserver, orchestrator: ActorAddress, workspace_path: str
    ) -> None:
        """Bind the ``#Workspace-<workspace_path>`` actor that owns this tree, then attach.

        **Get-or-create as a team child, which is where this bind started.** The
        actor stopped being hosted because nothing shared lives on it any more:
        the exec hold is a marker file, the document cache is YAML under
        ``<meta>``, the retrieval index is files under ``<meta>``, and the write
        gate is an ``fcntl.flock`` — every one of them on the tree, which is the
        real unicity domain (ADR-051). Two teams over one tree therefore get two
        actors again, and that is **correct rather than a regression**: the
        arrangement epic 51 built existed to stop two actors sharing state, and
        there is no shared state left for them to disagree about. What they still
        own — a sandbox backend, a document reader, an embedding worker — is
        per-process by nature and was never the thing being protected.

        The actor's name carries the resolved path, so two cards on different
        trees get two actors and two cards of one team on one tree get one.

        **The card emits the event itself, through ``notify_event``.** It used to
        travel as an argument of the orchestrator forward this bind replaced,
        which is gone. ``ToolObserver.notify_event`` is the base protocol's
        own method for *"tools that only need to emit events"*, and
        ``Akgent.notify_event`` wraps the payload in the same ``EventMessage`` on
        the same team stream — so the envelope, the fan-out and the wire name are
        unchanged and only the emitter moved. **Do not hand-build an
        ``EventMessage`` here**: that would reimplement the envelope this seam
        exists to avoid. One event per successful bind, naming this agent, which
        is how a client's Workspace tab learns which agent bound which tree — and
        if it silently stopped, nothing would raise and the tab would simply
        render empty.

        Two proxies are bound over the one address: an ask proxy for exec, which
        needs the verdict, and a tell proxy for the retrieval signals, which need
        nothing back.

        **Then ``attach``, over the ask proxy and unguarded.** Its holder half
        went with the liveness sweep in 52-6; what it still does is register
        ``agent_id -> name``, which is the **only** source for the actor's
        ``_name_of``. That map is what the git journal authors a commit with and
        what the exec busy refusal names the holding agent by — stop making this
        call and both degrade to a raw UUID that tells a model nothing it can act
        on, silently and for the life of the tree. An ask rather than a tell, so
        a dead actor fails the bind rather than leaving an agent bound to a tree
        that never heard of it.

        **This method creates at most one actor.** The team's ``#VectorStore`` a
        retrieval card may need is created by :meth:`_bind_vector_store`, further
        down, so that it is shared with whatever planning or knowledge-graph card
        the team also carries. Nothing about retrieval happens here.

        Args:
            observer: The owning agent, live at bind time.
            orchestrator: Address of the orchestrator.
            workspace_path: The resolved two-segment path this card is anchored
                to, carried into the actor's name verbatim — slash included.
                Nothing parses an actor name, and the path is injective by
                construction, so carrying it whole avoids a second encoding
                whose injectivity would have to be proved separately.
        """
        orchestrator_proxy = observer.proxy_ask(orchestrator, Orchestrator)
        derived_documents, derived_chars = derived_document_caps(
            self.vector_store.backend, self._rag_enabled()
        )
        workspace_addr = orchestrator_proxy.getChildrenOrCreate(
            WorkspaceActor,
            config=WorkspaceConfig(
                name=workspace_actor_name(workspace_path),
                role=WORKSPACE_ACTOR_ROLE,
                workspace_path=workspace_path,
                git_journal=self.git_journal,
                max_documents=(
                    self.max_documents if self.max_documents is not None else derived_documents
                ),
                max_document_chars=(
                    self.max_document_chars
                    if self.max_document_chars is not None
                    else derived_chars
                ),
            ),
        )
        workspace = observer.proxy_ask(workspace_addr, WorkspaceActor)
        self._workspace_proxy = workspace
        self._workspace_tell = observer.proxy_tell(workspace_addr, WorkspaceActor)
        self._agent_id = str(observer.myAddress.agent_id)
        self._agent_name = str(observer.myAddress.name)
        observer.notify_event(
            WorkspaceAttached(agent_id=observer.myAddress.agent_id, workspace_path=workspace_path)
        )
        workspace.attach(observer.myAddress, self._agent_name)

    def _observation_recorder(self) -> Callable[[str, bytes, bool], None]:
        """Build the closure a read closure uses to report what it saw.

        **The card is captured here**, at ``get_tools`` time, and the map it
        records into is the card's own. There is no message any more: the
        observation and the gate that reads it are the same object, so a read
        cannot queue behind another agent's mutation and a recording cannot be
        lost in flight. That removes a whole failure mode rather than making one
        faster — the old fire-and-forget ``tell`` was chosen precisely because
        the alternative, an ask, could stall a read on a busy mailbox.

        Capturing ``self`` is not the retention ADR-030 forbids: a ``ToolCard``
        holds its observer **weakly**, so a closure rooting the card roots no
        stopped agent.

        A **document** read is the one read that records nothing at all — it
        asks :meth:`_extract_lookup` for a cached extraction instead — so it
        never reaches this closure.

        Returns:
            A callable taking the path, the file's raw bytes and whether the read
            covered the whole file. It never raises: a lost observation is a lost
            precondition, which the gate turns into a *refused* write — it must
            never turn into a failed read.
        """
        card = self

        def record(path: str, data: bytes, full: bool) -> None:
            try:
                card.record_observation(path, Observation(sha=content_sha(data), full=full))
            except Exception:
                # Deliberately blind: a lost precondition, never a lost read. The
                # gate reads a missing observation as "you have not read this" and
                # refuses the overwrite, so every failure here degrades towards
                # refusing a write rather than accepting a stale one.
                logger.debug("Could not record an observation for %s", path, exc_info=True)

        return record

    def _extract_lookup(self) -> Callable[[str, str], str | None]:
        """Build the closure a document read uses to ask for a cached extraction.

        An **ask**, because the answer is the whole point: the caller extracts
        when there is nothing to serve. It is the one place a read waits on the
        actor, and what makes that acceptable is what is behind it — a dict
        lookup and an LRU reorder, no I/O, and no notify.

        :data:`EXTRACTOR_VERSION` is captured **here**, at ``get_tools`` time,
        exactly as the agent id is above. The extractor a miss would run is a
        property of the code, never of the call, so it is not a parameter of the
        tool callable and no agent can choose it. Bumping the constant therefore
        invalidates every stored entry on the next read, with no sweep.

        Returns:
            A callable taking the path and the digest of the source bytes the
            caller just read, and returning the cached Markdown or ``None``. It
            never raises: every failure degrades to a miss, which costs one
            extraction and can never be a wrong answer.
        """
        proxy = self._workspace_proxy
        version = EXTRACTOR_VERSION

        def lookup(path: str, source_sha: str) -> str | None:
            if proxy is None:
                return None  # harness shapes that wire a bare observer never bind one
            try:
                return proxy.document_extract(path, source_sha, version)
            except Exception:
                # Fail open, towards the pre-cache behaviour: a miss re-extracts
                # from the tree, which is where every byte came from anyway.
                logger.debug("Could not read the document cache for %s", path, exc_info=True)
                return None

        return lookup

    def _extract_recorder(self) -> Callable[[str, str, str], None]:
        """Build the closure a document read uses to fill the extraction cache.

        A **tell**, because nothing comes back and a slow actor must not hold a
        read that has already produced its answer. The split against
        :meth:`_extract_lookup` is a correctness requirement, not a style
        choice: collapsing both onto one proxy either makes a fill blocking or
        makes a lookup answerless.

        Returns:
            A callable taking the path, the digest of the source bytes and the
            extracted Markdown. It never raises: a lost fill is a cache that did
            not grow, never a failed read.
        """
        proxy = self._workspace_tell
        version = EXTRACTOR_VERSION

        def remember(path: str, source_sha: str, markdown: str) -> None:
            if proxy is None:
                return  # harness shapes that wire a bare observer never bind one
            try:
                proxy.cache_document(path, source_sha, version, markdown)
            except Exception:
                logger.debug("Could not fill the document cache for %s", path, exc_info=True)

        return remember

    def _seed_resources(self) -> None:
        """Write each configured resource that is not already present.

        Idempotent: an existing file is never overwritten, so a team restore
        cannot clobber edits made to a seeded file since team creation.
        """
        assert self._workspace is not None
        for resource in self.resources:
            if self._workspace.exists(resource.file_name):
                continue
            self._workspace.write(resource.file_name, resource.to_bytes())

    @property
    def workspace(self) -> Filesystem:
        """Return the workspace backend (set after :meth:`observer` is called).

        Raises:
            RuntimeError: If :meth:`observer` has not been called yet.
        """
        if not isinstance(self._workspace, Filesystem):
            raise RuntimeError("WorkspaceTool.workspace accessed before observer() was called.")
        return self._workspace

    def get_tools(self) -> list[Callable[..., Any]]:
        """Return enabled workspace tool callables.

        Read tools are always included (when their capability field is enabled).
        Write tools are only included when ``read_only=False`` (the default).

        Returns:
            List of callables for all enabled capabilities.
        """
        tools = self._read_tools()
        if not self.read_only:
            tools += self._write_tools()
        return tools

    def _read_tools(self) -> list[Callable[..., Any]]:
        """Return enabled read-side callables — included regardless of ``read_only``."""
        candidates = [
            self._tool_if_enabled(self.workspace_read, WorkspaceRead, self._read_factory),
            self._tool_if_enabled(self.workspace_list, WorkspaceList, self._list_factory),
            self._tool_if_enabled(self.workspace_glob, WorkspaceGlob, self._glob_factory),
            self._tool_if_enabled(self.workspace_grep, WorkspaceGrep, self._grep_factory),
            self._tool_if_enabled(self.workspace_view, WorkspaceView, self._view_factory),
            # Indexing derives from the tree and writes nothing into it, so all
            # of retrieval lives on the read side of ``read_only`` (ADR-045 §5).
            self._tool_if_enabled(
                self.workspace_rag_index, WorkspaceRagIndex, self._rag_index_factory
            ),
            self._tool_if_enabled(
                self.workspace_rag_search, WorkspaceRagSearch, self._rag_search_factory
            ),
        ]
        return [tool for tool in candidates if tool is not None]

    def _write_tools(self) -> list[Callable[..., Any]]:
        """Return enabled write-side callables — omitted entirely when ``read_only``.

        Exec lands here rather than beside the reads because a command mutates
        the tree whatever it happens to be, so it belongs on the write side of
        the ``read_only`` gate.
        """
        candidates = [
            self._tool_if_enabled(self.workspace_write, WorkspaceWrite, self._write_factory),
            self._tool_if_enabled(self.workspace_delete, WorkspaceDelete, self._delete_factory),
            self._tool_if_enabled(self.workspace_edit, WorkspaceEdit, self._edit_factory),
            self._tool_if_enabled(
                self.workspace_multi_edit, WorkspaceMultiEdit, self._multi_edit_factory
            ),
            self._tool_if_enabled(self.workspace_patch, WorkspacePatch, self._patch_factory),
            self._tool_if_enabled(self.workspace_mkdir, WorkspaceMkdir, self._mkdir_factory),
        ]
        tools = [tool for tool in candidates if tool is not None]
        return tools + self._exec_tools()

    def _exec_tools(self) -> list[Callable[..., Any]]:
        """Return both exec callables, or neither.

        The one place in this card where a single capability field yields two
        callables. ``_tool_if_enabled`` encodes the 1:1 shape and is deliberately
        not used here — ``workspace_exec_result`` can do nothing without
        ``workspace_exec``, so the pair is enabled or absent as a unit.

        Shares :meth:`_enabled_exec` with the wiring, so the callables and the
        actor they need can never disagree about whether the capability is on.
        """
        params = self._enabled_exec()
        if params is None:
            return []
        return [self._exec_factory(params), self._exec_result_factory(params)]

    @staticmethod
    def _tool_if_enabled(
        value: _ParamT | bool,
        param_cls: type[_ParamT],
        factory: Callable[[_ParamT], Callable[..., Any]],
    ) -> Callable[..., Any] | None:
        """Build a capability's callable, or ``None`` when it is off the TOOL_CALL channel.

        Pairs the configuration field with the factory that consumes it, so the type
        checker verifies each row of :meth:`_read_tools` / :meth:`_write_tools`.
        """
        params = _resolve(value, param_cls)
        if params is None or TOOL_CALL not in params.expose:
            return None
        return factory(params)

    def get_commands(self) -> dict[type[BaseToolParam], Callable[..., Any]]:
        """Return COMMAND-channel capabilities for this tool.

        Returns:
            Dict mapping each enabled COMMAND capability to its callable —
            ``ExpandMediaRefs``, and the two retrieval capabilities.
        """
        commands: dict[type[BaseToolParam], Callable[..., Any]] = {}
        pr = _resolve(self.expand_media_refs, ExpandMediaRefs)
        if pr is not None:
            commands[ExpandMediaRefs] = self._expand_media_refs
        index = _resolve(self.workspace_rag_index, WorkspaceRagIndex)
        if index is not None and COMMAND in index.expose:
            commands[WorkspaceRagIndex] = self._rag_index_factory(index)
        listing = _resolve(self.workspace_rag_list, WorkspaceRagList)
        if listing is not None and COMMAND in listing.expose:
            commands[WorkspaceRagList] = self._rag_list_factory(listing)
        return commands

    def get_context_states(self) -> list[Callable[[], ContextState | None]]:
        """Return the retrieval-index provider, or nothing.

        The index is pushed into the context tail as a per-turn delta rather than
        re-rendered into the system prompt, which would invalidate the cached
        prefix every time one file changed status (ADR-037 §3).

        Returns:
            A single-element list when ``workspace_rag_list`` is enabled and
            exposed on ``LLM_CONTEXT``, otherwise an empty one.
        """
        listing = _resolve(self.workspace_rag_list, WorkspaceRagList)
        if listing is not None and LLM_CONTEXT in listing.expose:
            return [self._rag_list_state_factory(listing)]
        return []
