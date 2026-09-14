"""The retrieval pipeline — the actor-side half of this capability.

**This is the retrieval capability's mixin, and it lives under the capability**
(ADR-053 Decision 6), exactly as :mod:`akgentic.tool.workspace.execution.actor`
holds exec's. :class:`~akgentic.tool.workspace.actor.WorkspaceActor` imports
:class:`DocumentsMixin` from here and composes it; that import is the assembly
point doing its job and not a capability leak, which is why ``rag/`` stays
deletable without touching another capability.

**What this module knows about a document lives in a file, not in actor state.**
Both halves of one document — the cached extraction and the retrieval row — are
one :class:`~akgentic.tool.workspace.documents.store.DocumentEntry` under
``<meta>/rag/``, reached through the
:class:`~akgentic.tool.workspace.documents.cache.DocumentCache` the card builds
and announces at bind time (ADR-051 Decision 6). A second process over the same
mount therefore reads the same cache with no shared memory, and nothing here
sends a delta to anybody: the two state fields, both dirty sets, ``_persist``,
``_send_delta`` and the restore hook they existed for are gone.

**The extraction cache is not here, and what is left is precisely what needs a
mailbox.** Filling and reading an extraction is not dispatch, so it has to keep
working for a card that enables neither exec nor retrieval and therefore creates
no actor at all (ADR-053 Decision 6): the lookup, the fill, the eviction pass and
the stale-marking a mutation causes are all
:class:`~akgentic.tool.workspace.documents.cache.DocumentCache`'s, called
directly by the card. What this module keeps is the ``#index-`` and ``#embed-``
children a Pydantic card cannot parent, the reports they send back, and the
search's keyword leg — and it reads and fills the same cache through the same
object.

**There is exactly one writer of a row here, and it writes to disk on the turn it
is called.** :meth:`DocumentsMixin._put_row` goes through the cache's own
``put_row``, which reads the entry, copies it with the one half that changes, and
puts it back — never a field-by-field rebuild (Golden Rule 12). There is **no
dirty set and no batching**: the ``batches_landed`` counter that deliberately
persisted nothing on its own turn now writes, which costs
``ceil(chunks / EMBED_BATCH_SIZE)`` rewrites of one file and is accepted
deliberately. The alternative is in-memory write-behind state, which is exactly
what moving to files removes, and the counter has to survive a crash or its file
parks at ``EMBEDDING`` until the reaper.

**The rule that survives, unchanged and load-bearing: a read writes nothing.**
Reads are the majority of workspace traffic, and a document-cache **hit** is a
read: it performs one ``get_document`` and no write at all. A write on a *read*
path — of any kind, in any of these methods — is a defect until a decision says
otherwise.

**An unannounced cache is an ordinary state and every path degrades through
it**, never an ``assert`` and never a raise: a lost announcement means the cache
misses and the index looks empty, which is visible and recoverable by rebinding.
:meth:`DocumentsMixin._cache` is where that is said once — it hands back a cache
over no store rather than a ``None`` every delegate would have to branch on.

Everything on the ask path here is O(1)/O(n) dict work on the actor thread, plus
bounded file reads while queueing and bounded proxy calls to the store child —
and **no external round trip at all**. :meth:`DocumentsMixin.rag_search` used to
embed the query on this thread and then call ``search`` on a store that may be a
cluster client with no actor behind it, which made two network calls on one turn
of the mailbox that owns the write gate; while either was in flight,
``request_exec``, ``exec_status`` and every worker report queued behind it. Both
now happen in :func:`~akgentic.tool.workspace.rag._vector_hits`, on the **calling
agent's own thread** — which is already blocked waiting for that answer and
blocks nobody else — and this method is handed the hits. Do not put a network
call back on this turn. The two slow halves of indexing happen elsewhere too:
extraction and splitting in a ``#index-`` worker, embedding in an ``#embed-``
worker, and each batch's ``add()`` lands on its own turn when its worker reports.
Nothing here raises: an exception in a document handler would kill the actor
that owns the write gate, so a document path degrades — a miss, a file left
``FAILED``, a cache that did not grow — and never propagates.
"""

from __future__ import annotations

import contextlib
import logging
from datetime import UTC, datetime, timedelta
from enum import Enum
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
from uuid import uuid4

from akgentic.core.agent_config import BaseConfig
from akgentic.tool.vector_store.protocol import (
    PATH_PREFIX_REJECTED,
    PATH_PREFIX_WILDCARDS,
)
from akgentic.tool.workspace.documents.cache import DocumentCache
from akgentic.tool.workspace.documents.models import (
    EMBEDDING_STALE_AFTER_S,
    EXTRACTOR_VERSION,
    RAG_COLLECTION,
    NewFileMessage,
    RagChunk,
    RagFile,
    RagStatus,
)
from akgentic.tool.workspace.documents.store import DocumentEntry
from akgentic.tool.workspace.models import WorkspaceConfig, content_sha
from akgentic.tool.workspace.rag.context import RagFileRow, RagIndexState
from akgentic.tool.workspace.readers import _MIME_MAP, TEXT_EXTENSIONS, DocumentReader
from akgentic.tool.workspace.workspace import Filesystem

if TYPE_CHECKING:
    from collections.abc import Iterator

    from akgentic.core.agent import Akgent
    from akgentic.core.agent_state import BaseState
    from akgentic.tool.vector_store.embedding_actor import EmbeddingError, EmbeddingResult
    from akgentic.tool.vector_store.protocol import (
        SearchHit,
        VectorStoreParam,
        VectorStoreService,
    )
    from akgentic.tool.vector_store.vector import VectorEntry
    from akgentic.tool.workspace.rag.params import WorkspaceRagIndex
    from akgentic.tool.workspace.rag.worker import IndexFailure, IndexResult

    # The mixin consumes the actor's own surface — ``createActor``, the two proxy
    # builders, ``myAddress``, ``config``, ``state`` and ``team_id``. Naming the
    # base under ``if TYPE_CHECKING:`` is how ``ExecMixin`` already reaches its
    # own; at runtime the mixin contributes only ``object``, so the MRO the actor
    # declares is unchanged and nothing here shadows a sibling.
    _DocumentsBase = Akgent[WorkspaceConfig, BaseState]
else:
    _DocumentsBase = object

logger = logging.getLogger(__name__)

_INDEXABLE_EXTENSIONS: frozenset[str] = TEXT_EXTENSIONS | (
    DocumentReader.extensions - frozenset(_MIME_MAP)
)
"""What ``workspace_rag_index`` will accept — the set the read path already draws.

Not invented here and deliberately not re-listed: it is exactly the extensions a
read can already turn into text, minus the image formats. An OCR'd photograph is
not what this index is for, and ``_MIME_MAP`` is already the set that names them.
``read/`` draws the same line for ``expand_media_refs``; a second copy of
either extension list would drift.
"""

_UNAVAILABLE = "Retrieval indexing is not available for this workspace."
"""What ``workspace_rag_index`` answers with no vector store wired.

A sentence rather than an exception: this actor owns the write gate, and a
retrieval capability that raises would be a way for a misconfigured deployment to
take the gate down with it.
"""

_CHUNK_REF_TYPE = "workspace_chunk"
"""``VectorEntry.ref_type`` for every chunk this package stores."""


class _Spawn(Enum):
    """What one claim attempt did, and why a boolean no longer says enough.

    ``_spawn`` used to answer ``True``/``False`` and ``_drain`` stopped on any
    ``False``. With two processes draining one tree, "somebody else claimed it"
    became ordinary and has to *continue*, while a spawn that **raised** must
    still stop the pass — a process that cannot start actors should not keep
    trying to. One boolean cannot carry both.
    """

    STARTED = "started"
    """A worker is running for the path, and this process is counting it."""

    CLAIMED_ELSEWHERE = "claimed-elsewhere"
    """The row moved between the pending read and the hold. Try the next path."""

    FAILED = "failed"
    """The spawn raised, or retrieval is not configured. Stop the pass."""


_NO_HITS = (
    "Nothing in the retrieval index matched that query. "
    "Use workspace_rag_list to see which files are indexed."
)
"""What a search answers when retrieval works and nothing matched.

Deliberately **not** :data:`_UNAVAILABLE`: "nothing matched" and "nothing is
indexed" are different problems with different next steps, and an agent handed
one sentence for both would retry the query when it should have indexed the tree.
"""

_REJECTED_PREFIX = PATH_PREFIX_REJECTED
"""The sentence a rejected prefix answers with, identical to what a backend raises.

The characters and the sentence both live in ``vector_store/protocol.py``, beside
the two backends that read a prefix differently, so this capability and the guard
underneath it cannot drift apart. This layer **returns** the sentence rather than
raising it: nothing in a document handler raises, and a wildcard is a mistake the
agent can correct from the answer alone. Rejecting here as well as at the backend
is deliberate defence in depth — the actor's answer is a sentence, the backend's
is a ``ValueError``, and a caller that bypasses one still meets the other.
"""


class _KeywordMatch(NamedTuple):
    """One chunk the keyword leg hit, with everything its render needs.

    Carried because a keyword-only hit has no ``SearchHit`` behind it and would
    otherwise have no text at all — which would make a keyword-only search, the
    degraded mode this whole design turns on, render nothing.
    """

    path: str
    chunk: RagChunk
    text: str


class DocumentsMixin(_DocumentsBase):
    """The retrieval pipeline, over the cache the card announced.

    Declares no Pydantic field and no state field: everything it uses is owned by
    the actor and initialised in its ``on_start``, and what it *persists* lives on
    disk. It defines no sibling's method either — in particular not ``deliver``,
    ``fail`` or ``cache_capacity``, any of which would silently take over the
    deferred delivery path or resize the exec LRU, because this mixin precedes
    ``ExecMixin`` and ``DeferredResultActor`` in the MRO.

    **The seven slots below are annotated here and assigned in
    ``WorkspaceActor.on_start``**, which is ``ExecMixin``'s arrangement for its
    own eight rather than a divergence: this mixin has no ``on_start`` and adds
    nothing to the chain. There is no ``_embedder`` among them and none is coming
    back — the actor's own query leg was the only reader it ever had, and that leg
    is the card's.
    """

    _workspace: Filesystem
    _rag_params: WorkspaceRagIndex | None
    _rag_reader: DocumentReader | None
    _rag_collection: VectorStoreParam | None
    _vs_proxy: VectorStoreService | None
    _vector_store: VectorStoreService | None
    _index_workers: int
    _document_cache: DocumentCache | None

    ##
    ## The store — six delegates onto the cache the card announced
    ##
    def configure_vector_store(self, store: VectorStoreService) -> None:
        """Take the engine the card resolved — **tell** path, last writer wins.

        The shape ``configure_lock`` and ``configure_document_cache`` already
        have. It lands **before** ``enable_rag``, which is what makes it
        impossible for this actor to turn retrieval on under a store it has not
        been given; the card orders the two announcements and this method holds
        nothing but the slot.

        Last writer wins, and a replacement is not disruptive: two retrieval
        cards on one tree resolve the same collection from the same team, and
        the first ``enable_rag`` has already fixed the parameters for the tree,
        so a second card's store is bound only if the first never was.

        Args:
            store: The engine this tree's chunks are written into and searched.
        """
        self._vector_store = store

    def configure_document_cache(self, cache: DocumentCache) -> None:
        """Take the cache the card built — **tell** path, last writer wins.

        The shape ``configure_exec`` and ``configure_lock`` already have: the card
        builds the object in ``observer()`` and announces it here, so this actor
        never inspects a card and never resolves configuration of its own. A
        second announcement from a second card simply replaces the first; two
        cards on one tree resolve the same store from the same environment, so
        there is nothing for a first-call-wins rule to protect.

        **It carries the two document caps as well as the store**, which is what
        retired ``WorkspaceConfig.max_documents`` and ``max_document_chars``.
        Those travelled through get-or-create, so the first card of a team to
        bind fixed them for every later card of that team, with nothing raised;
        an announcement is last-writer-wins like every other one here.

        Args:
            cache: This tree's records, its two caps, and the store behind them.
        """
        self._document_cache = cache

    def _cache(self) -> DocumentCache:
        """The announced cache, or an empty stand-in over no store.

        **A missing announcement is an ordinary state and must stay one.** Every
        document path already degrades to a miss when no card has announced a
        store, and the six delegates below would each need the same ``None``
        branch to say so; handing back a cache whose store is ``None`` says it
        once, in the one place that knows the tree key.
        """
        cache = self._document_cache
        if cache is not None:
            return cache
        return DocumentCache(None, self.config.workspace_path, 0, 0)

    def _entry(self, path: str) -> DocumentEntry:
        """Return *path*'s stored record, or a fresh empty one."""
        return self._cache().entry(path)

    def _entries(self) -> list[DocumentEntry]:
        """Every stored record for this tree, in no guaranteed order.

        **One listing serves all three of its callers here** — the drain, the
        render and the keyword leg — because one file carries both halves of a
        document. Callers that need an order sort for themselves: a directory
        glob's order is the file system's, and leaning on it is how a render
        stops being stable across runs.
        """
        return self._cache().entries()

    @contextlib.contextmanager
    def _hold_record(self, path: str) -> Iterator[None]:
        """Serialise *path*'s record against every other process, for the block.

        Args:
            path: Workspace-relative path of the source document.
        """
        with self._cache().hold(path):
            yield

    def _next_pending(self, exclude: frozenset[str]) -> DocumentEntry | None:
        """One record waiting to be claimed, or ``None`` — never a whole listing."""
        return self._cache().next_pending(exclude)

    def _release_worker_slot(self) -> None:
        """Give back the slot one ``#index-`` worker was occupying in this process.

        One spawn produces exactly one report — ``IndexResult`` or
        ``IndexFailure`` — from a worker this actor created, so the count is
        balanced by construction. The floor at zero is there because it is a
        *count* rather than the set it replaced: an unknown path used to be a
        harmless ``discard``, and nothing in a resource bound is worth taking the
        gate's actor below zero over.
        """
        self._index_workers = max(0, self._index_workers - 1)

    ##
    ## The one row writer
    ##
    def _put_row(self, path: str, row: RagFile) -> None:
        """Write *row* as *path*'s index half, on this turn.

        The only writer of a row on this side, and it writes through the one
        writer there is —
        :meth:`~akgentic.tool.workspace.documents.cache.DocumentCache.put_row`,
        where the extraction half is preserved by construction and the suite's
        ``ast`` canary looks.

        Args:
            path: Workspace-relative path the row describes.
            row: The row, already derived by ``model_copy(update=...)`` or built
                fresh — never rebuilt by naming fields.
        """
        self._cache().put_row(path, row)

    ##
    ## Retrieval — enabling it, and the collection that is created lazily
    ##
    def enable_rag(
        self,
        agent_id: str,
        params: WorkspaceRagIndex,
        reader: DocumentReader,
        collection: VectorStoreParam,
    ) -> None:
        """Turn retrieval on for this tree — **tell** path, once per card.

        The actor cannot take any of this from :class:`WorkspaceConfig`, because
        the first bind fixes that for every card on the tree and the card that
        binds a tree first is routinely one with no retrieval capability at all.
        So a retrieval-capable card announces itself here instead, at bind time,
        exactly as ``configure_exec`` does. **The actor never inspects a card**;
        it has no handle on one.

        **The arbitration is gone; the keep-first that remains is defensive.**
        Deciding which chunking a tree uses is the *tree's* now, published at
        ``<meta>/policy.yaml`` and enforced at bind: a card carrying different
        parameters is refused there, naming both values, so no second card
        carrying different parameters can reach this method through a bind at all
        (:class:`~akgentic.tool.workspace.rag.TreePolicy`).

        The branch is kept rather than deleted, and narrowly. This actor **must
        never raise on a tell path** — it owns the write gate — so it cannot
        enforce anything itself; and a route that bypassed the bind gate (a direct
        ``enable_rag`` in a spec, a worker resolving a different ``<meta>`` root)
        would otherwise silently re-chunk a tree mid-flight. So the assignment
        stays idempotent, and what would have been an arbitration is one DEBUG
        line saying where the decision actually lives.

        This is also where the collection is created — **lazily, and never in
        ``on_start``**: a workspace with retrieval off must never create one. It
        follows ``PlanActor._acquire_vs_proxy`` with one deliberate divergence:
        where that actor *raises* when the team's store is absent, this one logs
        and degrades when no card has announced it a store. **It spawns no store
        child and never did on this path** — :meth:`_resolve_store` says so in as
        many words, and the constant that used to name what such a spawn raises
        went with the last reader of it. A missing vector
        store is a configuration error for a planning tool, whose whole purpose
        it is; here it must never be fatal, because this actor also owns the
        write gate.

        Args:
            agent_id: The announcing agent, for the log line only.
            params: The chunking configuration the whole tree will use.
            reader: The extraction configuration, which lives on the card.
            collection: The vector collection's backend, dimension and tenant.
        """
        try:
            if self._rag_params is not None:
                if self._rag_params != params:
                    logger.debug(
                        "Workspace %s: the tree's published policy record is the authority "
                        "on chunking and this actor is already carrying it; agent %s "
                        "reached enable_rag with %s past the bind gate and keeps %s",
                        self.config.workspace_path,
                        agent_id,
                        params,
                        self._rag_params,
                    )
                return
            self._rag_params = params
            self._rag_reader = reader
            self._rag_collection = collection
            self._acquire_vs_proxy()
        except Exception:
            logger.warning(
                "Workspace %s: could not enable retrieval — it stays off",
                self.config.workspace_path,
                exc_info=True,
            )

    def _acquire_vs_proxy(self) -> None:
        """Resolve the storage engine, create the collection, and bind it.

        **The slot holds a ``VectorStoreService``, not necessarily a proxy.** A
        backend that needs an actor is reached through a proxy over the team's
        ``#VectorStore``; a cluster backend is a client with no actor behind it.
        Which one arrived is the card's decision and is invisible here: the four
        methods this actor calls are exactly that protocol, so nothing below this
        line can tell the two apart. (The attribute keeps the name ``_vs_proxy``:
        renaming it to ``_store`` is ~200 mechanical private sites, routed to its
        own follow-up.)

        **Nothing is re-marked here, and nothing is to be re-added.** An earlier
        shape put every ``EMBEDDED`` row back to ``PENDING`` the moment an
        in-memory store child was created, because that child had no checkpoint
        and therefore held nothing the rows claimed. Both the rows
        and the index are on disk now and outlive every engine, so that premise is
        gone and porting the rule would blank a good cache on every process start
        — one full re-extraction and re-embedding of every indexed file, charged
        to whoever pays for embeddings. A guard in the suite goes red for anyone
        who re-adds it.

        **Every call through it is an ask.** ``create_collection`` has to be known
        to have worked before anything is added; ``remove`` re-raises a missing
        collection as a ``RetriableError`` this actor must see to keep the
        superseded ids for a later retry; and ``add`` is one ask per batch **on
        its own mailbox turn**, when that batch's worker reports — not thirty in a
        row on one turn, which is what made it a tell before the embedding
        pipeline moved to this actor.

        **No embedder is built here any more.** One was, from the card's own
        ``VectorStoreParam``, because the query leg embedded on this thread; that
        leg is the card's now and builds its own from the same param, so a second
        one here would be an object nobody reads. Every worker this actor spawns
        is still handed that param's model and provider, which is the only thing
        the embedding budget ever travelled as.

        **Every outage drops to degraded mode.** A store that was never announced
        and a ``create_collection`` that fails each log one WARNING and leave
        ``_vs_proxy`` ``None``, so ``workspace_rag_index`` answers a sentence;
        anything unexpected propagates to :meth:`enable_rag`, which keeps
        retrieval off and logs it with its traceback. The order is
        ``create_collection`` then bind — so a store whose ``create_collection``
        fails is simply not bound, and the card that announced it is unaffected.
        This actor also owns the write gate, so a missing store must never be
        fatal here the way it is for planning. **The up-front gate is no longer
        what protects a file from parking at ``EMBEDDING``**: the write happens
        on this actor's own turn
        inside a ``try``, so an ask that raises settles the file ``FAILED`` with
        the reason there and then.
        """
        if self._rag_collection is None:
            logger.warning(
                "Workspace %s: no collection configured — retrieval stays in degraded mode",
                self.config.workspace_path,
            )
            return
        store = self._resolve_store()
        if store is None:
            return
        try:
            store.create_collection(RAG_COLLECTION, self._rag_collection)
        except Exception as exc:
            logger.warning(
                "Workspace %s: create_collection(%s) failed: %s — degraded mode",
                self.config.workspace_path,
                RAG_COLLECTION,
                exc,
            )
            return
        self._vs_proxy = store

    def _resolve_store(self) -> VectorStoreService | None:
        """Return the engine the card announced, or ``None`` to stay degraded.

        **This actor receives a store; it does not make one.** It creates no
        child, names no backend, calls no factory and asks no orchestrator — the
        card resolved all of that in ``observer()`` and handed the object over
        through :meth:`configure_vector_store`, which is the shape
        ``configure_lock`` and ``configure_document_cache`` already have
        (ADR-051 Decision 8). Whether the object is a proxy over the team's
        ``#VectorStore`` or a cluster client is invisible here and must stay so:
        the four methods this actor calls are the ``VectorStoreService`` protocol
        and nothing below this line can tell the two apart.

        A **lost announcement** is the degraded case, and it is the only one left:
        one WARNING naming the workspace, ``_vs_proxy`` at ``None``, and
        ``workspace_rag_index`` answering its unavailable sentence — visible, and
        recoverable by rebinding. Nothing here can raise, because nothing here
        does anything.

        Returns:
            The announced store, or ``None`` when none was announced.
        """
        store = self._vector_store
        if store is None:
            logger.warning(
                "Workspace %s: no vector store was announced — degraded mode",
                self.config.workspace_path,
            )
        return store

    ##
    ## workspace_rag_index — the spawn side
    ##
    def index_paths(self, path: str = "", force: bool = False) -> str:
        """Queue every candidate under *path* and return what was accepted.

        Returns **immediately**. Everything here is O(n) over the candidate list
        on the actor thread — validate, read-and-hash, set ``PENDING``, spawn up
        to the concurrency cap — and no extraction, split or embedding happens on
        this turn.

        The ``_vs_proxy`` gate here refuses the whole capability when no vector
        store is wired. It is **not** what protects a file from parking at
        ``EMBEDDING``: each batch is written on the turn its worker reports, and a
        write that raises settles that file ``FAILED`` there and then.

        Args:
            path: A file, a directory, or ``""`` for the whole tree.
            force: Re-index a file that is already current at these bytes.

        Returns:
            The counts, or the degraded-mode sentence.
        """
        if self._vs_proxy is None or self._rag_params is None:
            return _UNAVAILABLE
        self.reap_abandoned_rows()
        candidates, unsupported = self._candidates(path)
        queued = current = 0
        for candidate in candidates:
            sha = self._digest(candidate)
            if sha is None:
                unsupported += 1
                continue
            # The test and the write are one read-modify-write of one record, so
            # they run inside that record's hold: without it an enqueue can race a
            # claim in another process and reset a run that is already in flight.
            with self._hold_record(candidate):
                if self._is_accounted_for(candidate, sha, force):
                    current += 1
                    continue
                self._enqueue(candidate, sha)
            queued += 1
        self._drain()
        return f"{queued} file(s) queued, {current} already current, {unsupported} unsupported"

    def _is_accounted_for(self, path: str, sha: str, force: bool) -> bool:
        """Whether *path* at *sha* needs no new work.

        True for a file already ``EMBEDDED`` at these bytes, and for one whose run
        over these same bytes is still in flight — re-queueing the latter would
        reset a live run and spawn a second worker for it. ``force`` overrides
        both, which is the whole of what ``force`` means.
        """
        if force:
            return False
        row = self._entry(path).row
        if row is None or row.indexed_sha != sha:
            return False
        return row.status in _IN_FLIGHT or row.status is RagStatus.EMBEDDED

    def _enqueue(self, path: str, sha: str) -> None:
        """Put *path* at ``PENDING`` for *sha*, keeping the old ids to supersede.

        The previous chunk set's ids move to ``superseded_chunk_ids`` **before**
        ``chunks`` is cleared, because re-index is add-then-remove and the old ids
        have to survive somewhere for the duration. Ids left over from a removal
        that previously failed are kept, so a later re-index retries them.
        """
        now = datetime.now(UTC)
        row = self._entry(path).row
        if row is None:
            self._put_row(
                path, RagFile(path=path, status=RagStatus.PENDING, indexed_sha=sha, updated_at=now)
            )
            return
        superseded = list(row.superseded_chunk_ids)
        # The membership set is built once. Rebuilding it per chunk is O(n²) on
        # the actor's mailbox turn, and an 800-page document is ~1,900 chunks.
        seen = set(superseded)
        for chunk in row.chunks:
            if chunk.chunk_id not in seen:
                superseded.append(chunk.chunk_id)
                seen.add(chunk.chunk_id)
        self._put_row(
            path,
            row.model_copy(
                update={
                    "status": RagStatus.PENDING,
                    "indexed_sha": sha,
                    "chunks": [],
                    "chunk_count": 0,
                    "batches_expected": 0,
                    "batches_landed": 0,
                    "superseded_chunk_ids": superseded,
                    "reason": None,
                    "updated_at": now,
                }
            ),
        )

    def _drain(self) -> None:
        """Spawn workers for ``PENDING`` files up to **this process's** concurrency cap.

        It writes rows for paths its caller never named — ``EXTRACTION`` or
        ``SPLITTING`` on a spawn, ``FAILED`` on a spawn that raised — and each of
        those reaches the disk on this turn, so it answers nothing and no caller
        has to guess whether it moved anything.

        **One filtered read per spawn, not one full listing.** ``next_pending``
        answers with the first waiting record it finds and reads no further, so a
        pass that meets a waiting record early costs one record read on a tree of
        a thousand. It is a short-circuit and not a bound: a pass offered nothing
        still walks the directory. What it never does is build a thousand records
        to pick one. The read is re-taken on every pass rather than snapshotted,
        because every spawn writes a row and a snapshot would re-offer a path
        already moved out of ``PENDING``.

        ``tried`` is **local to this pass and never instance state**: it holds the
        paths whose claim this process lost to another, so the loop moves on
        instead of being offered the same still-``PENDING``-looking record for
        ever. Instance state would carry a lost claim into the next call, where
        the path may legitimately be pending again.

        **A lost claim continues; a spawn that raised stops the pass.** "Somebody
        else claimed it" is ordinary once two processes drain one tree. A spawn
        that *raised* means this process cannot start actors, and it should not
        keep trying to.
        """
        from akgentic.tool.workspace.rag.worker import (  # noqa: PLC0415 — see _batch_size
            MAX_CONCURRENT_INDEX_WORKERS,
        )

        tried: set[str] = set()
        while self._index_workers < MAX_CONCURRENT_INDEX_WORKERS:
            waiting = self._next_pending(frozenset(tried))
            if waiting is None:
                return
            outcome = self._spawn(waiting.path)
            if outcome is _Spawn.FAILED:
                return
            if outcome is _Spawn.CLAIMED_ELSEWHERE:
                tried.add(waiting.path)

    def _spawn(self, path: str) -> _Spawn:
        """Claim *path* and start one ``#index-`` worker for it, under its record hold.

        **The whole claim runs inside ``hold``, on a row re-read inside it.** That
        re-read is the load-bearing line: a hold taken around a decision made on a
        read from *before* it serialises nothing, and the shape without it passes
        every sequential test while excluding nobody. Two processes draining one
        tree would otherwise both see ``PENDING`` and both spawn a worker for one
        file — two paid embedding runs.

        **The spawn is inside the hold too**, and that is deliberate rather than
        careless: a claim written and then not spawned is a row nobody carries
        until the reaper finds it ten minutes later. ``createActor`` plus a tell
        is microseconds, and no extraction, split or embedding happens behind it.

        The worker is spawned with ``createActor`` and handed everything it needs
        in one payload, including the card's extraction configuration. It is
        **not** a ``DeferredWorker`` and is deliberately not routed through
        ``DeferredResultActor.request()``: that mechanism's reports land in this
        actor's exec result cache.

        The status it moves to says which half of the work the worker actually
        has to do — ``EXTRACTION`` when the body has to be produced,
        ``SPLITTING`` when the cache could supply one.

        Returns:
            Which of the three things happened — see :class:`_Spawn`.
        """
        params, reader = self._rag_params, self._rag_reader
        if params is None or reader is None:
            return _Spawn.FAILED
        with self._hold_record(path):
            row = self._entry(path).row
            # RE-READ INSIDE THE HOLD. ``next_pending`` offered this path while it
            # was ``PENDING``; between that read and this hold another process may
            # have claimed it, re-queued it at other bytes, or dropped it.
            if row is None or row.indexed_sha is None or row.status is not RagStatus.PENDING:
                return _Spawn.CLAIMED_ELSEWHERE
            markdown = self._cache().lookup(path, row.indexed_sha, EXTRACTOR_VERSION)
            if not self._start_index_worker(path, row.indexed_sha, markdown, params, reader):
                return _Spawn.FAILED
            self._index_workers += 1
            self._put_row(
                path,
                row.model_copy(
                    update={
                        "status": (
                            RagStatus.SPLITTING if markdown is not None else RagStatus.EXTRACTION
                        ),
                        "updated_at": datetime.now(UTC),
                    }
                ),
            )
        return _Spawn.STARTED

    def _start_index_worker(
        self,
        path: str,
        source_sha: str,
        markdown: str | None,
        params: WorkspaceRagIndex,
        reader: DocumentReader,
    ) -> bool:
        """Create *path*'s worker and hand it its request, or fail the file.

        Split out of :meth:`_spawn` so the claim there reads as the sequence it is
        — re-read, decide, start, write. A spawn that raised leaves the file
        ``FAILED`` on this turn rather than at ``PENDING`` for ever, and says so
        to the caller, which stops the pass.

        Returns:
            Whether a worker is now running for *path*.
        """
        from akgentic.tool.workspace.rag.worker import (  # noqa: PLC0415 — see _batch_size
            IndexRequest,
            IndexWorker,
            index_worker_name,
        )

        scope = self.config.workspace_path
        try:
            address = self.createActor(
                IndexWorker, config=BaseConfig(name=index_worker_name(scope, path))
            )
            self.proxy_tell(address, IndexWorker).receiveMsg_IndexRequest(
                IndexRequest(
                    path=path,
                    scope=scope,
                    source_sha=source_sha,
                    markdown=markdown,
                    params=params,
                    reader=reader,
                )
            )
        except Exception as exc:
            logger.warning("Workspace %s: could not spawn an index worker for %s", scope, path)
            self._fail(path, source_sha, f"{type(exc).__name__}: {exc}")
            return False
        return True

    ##
    ## Candidate discovery — every path through ``Filesystem``, never its root
    ##
    def _candidates(self, path: str) -> tuple[list[str], int]:
        """Return the indexable files under *path*, and how many were unsupported.

        Every path goes through :class:`~akgentic.tool.workspace.workspace.Filesystem`,
        whose every entry point validates internally. Joining onto its private
        root instead is the traversal bypass this package has already had to close
        once.

        A path that escapes, does not exist, or cannot be listed is **skipped with
        a log line**, never an error: ``workspace_rag_index`` is reachable from a
        model, and a raise here would land in the agent's next turn as a failure it
        cannot act on.
        """
        try:
            found = self._walk(path)
        except NotADirectoryError:
            found = [path]  # a single file, which is a legal argument
        except OSError as exc:
            logger.info(
                "Workspace %s: %r is not indexable: %s",
                self.config.workspace_path,
                path,
                exc,
            )
            return [], 0
        supported = [
            candidate
            for candidate in found
            if Path(candidate).suffix.lower() in _INDEXABLE_EXTENSIONS
        ]
        return supported, len(found) - len(supported)

    def _walk(self, root: str) -> list[str]:
        """List every file under *root*, depth-first, through the backend only.

        Dot-prefixed names are skipped whole. That covers the atomic-write staging
        files (``.<name>.<32 hex>.tmp``) and the vestigial extraction sidecars
        (``.<name>.md``) — indexing either would put a temporary file or a stale
        copy of a document into the corpus.
        """
        found: list[str] = []
        for entry in self._workspace.list(root):
            if entry.name.startswith("."):
                continue
            relative = f"{root}/{entry.name}" if root else entry.name
            if entry.is_dir:
                with contextlib.suppress(OSError):
                    found.extend(self._walk(relative))
            else:
                found.append(relative)
        return found

    def _digest(self, path: str) -> str | None:
        """Return the digest of *path*'s current bytes, or ``None`` if unreadable."""
        try:
            return content_sha(self._workspace.read(path))
        except OSError as exc:
            logger.info(
                "Workspace %s: skipping %r while indexing: %s",
                self.config.workspace_path,
                path,
                exc,
            )
            return None

    ##
    ## The settle side
    ##
    def receiveMsg_IndexResult(self, msg: IndexResult) -> None:  # noqa: N802
        """TELL, from a worker. Take its chunks and issue the embedding batches."""
        try:
            self._on_index_result(msg)
        except Exception:
            logger.warning(
                "Workspace %s: could not record the index result for %s",
                self.config.workspace_path,
                msg.path,
                exc_info=True,
            )

    def _on_index_result(self, msg: IndexResult) -> None:
        """Record *msg* and issue its ``add()`` batches.

        A report whose row has moved on writes nothing itself, but the
        ``_drain`` it frees a slot for spawns the next file, and that spawn's
        row reaches the disk on this turn like any other.
        """
        self._release_worker_slot()
        entry = self._live_entry(msg.path, msg.source_sha)
        if entry is None:
            self._drain()
            return
        if msg.extracted:
            # The worker did the extraction, so the cache learns from it — a fill
            # like any other, and the one write in this method that is not the
            # file's own transition.
            self._cache().fill(msg.path, msg.source_sha, EXTRACTOR_VERSION, msg.markdown)
        if len(msg.texts) != len(msg.chunks):
            self._fail(msg.path, msg.source_sha, "the worker returned mismatched chunks and texts")
            self._drain()
            return
        batches = ceil(len(msg.chunks) / _batch_size())
        self._put_row(
            msg.path,
            entry.model_copy(
                update={
                    "status": RagStatus.EMBEDDING if msg.chunks else RagStatus.EMBEDDED,
                    "chunks": msg.chunks,
                    "chunk_count": len(msg.chunks),
                    "batches_expected": batches,
                    "batches_landed": 0,
                    "reason": None,
                    "updated_at": datetime.now(UTC),
                }
            ),
        )
        if msg.chunks:
            self._issue_batches(msg)
        else:
            # An empty document is indexed the moment it is split: there is
            # nothing to embed and nothing to wait for.
            self._drop_superseded(msg.path)
        self._drain()

    def _issue_batches(self, msg: IndexResult) -> None:
        """Spawn one ``#embed-`` worker per ``EMBED_BATCH_SIZE`` chunks of *msg*.

        The entries are built whole here and handed to the worker whole, so the
        ``scope``, ``path`` and ``ordinal`` every scoped removal and every scoped
        search filters on never leave this actor's own objects.
        ``batches_expected`` is already written by the caller, before the first
        worker is spawned.

        A spawn that raises fails the file and stops issuing. Every later report
        for that path is then dropped by the status guard, exactly as the first
        error for a file wins today.
        """
        from akgentic.tool.vector_store.vector import VectorEntry  # noqa: PLC0415 — optional extra

        entries = [
            VectorEntry(
                ref_type=_CHUNK_REF_TYPE,
                ref_id=chunk.chunk_id,
                text=text,
                vector=[],
                scope=self.config.workspace_path,
                path=msg.path,
                ordinal=chunk.ordinal,
            )
            for chunk, text in zip(msg.chunks, msg.texts, strict=True)
        ]
        size = _batch_size()
        for start in range(0, len(entries), size):
            if not self._spawn_embedding(msg.path, msg.source_sha, entries[start : start + size]):
                return

    def _spawn_embedding(self, path: str, source_sha: str, batch: list[VectorEntry]) -> bool:
        """Start one ``#embed-`` worker for *batch*, or fail *path* and say so.

        The model and the provider come from ``self._rag_collection`` — the card's
        own param, which is authoritative from this story on. The worker is
        spawned with ``createActor`` and handed the batch in one payload, the shape
        :meth:`_spawn` uses for an index worker.

        **Every exit either spawns or fails the file.** ``batches_expected`` is
        already written when this is called, so a return that neither starts a
        worker nor settles the row leaves the file at ``EMBEDDING`` waiting for a
        report that nobody will send, until the reaper reverts it ten minutes
        later and the whole extraction repeats.

        Returns:
            Whether a worker is now embedding *batch*.
        """
        from akgentic.tool.core.deferred import WORKER_ROLE  # noqa: PLC0415 — cycle
        from akgentic.tool.vector_store.embedding_actor import (  # noqa: PLC0415 — optional extra
            EmbeddingRequest,
            EmbeddingWorker,
            embedding_worker_name,
        )

        collection = self._rag_collection
        if collection is None:
            self._fail(path, source_sha, "retrieval has no collection parameters")
            return False
        request_id = str(uuid4())
        try:
            address = self.createActor(
                EmbeddingWorker,
                config=BaseConfig(
                    name=embedding_worker_name(RAG_COLLECTION, request_id), role=WORKER_ROLE
                ),
            )
            self.proxy_tell(address, EmbeddingWorker).receiveMsg_DeferredPayload(
                EmbeddingRequest(
                    deferred_key=request_id,
                    collection=RAG_COLLECTION,
                    entries=batch,
                    request_ref=path,
                    embedding_model=collection.embedding_model,
                    embedding_provider=collection.embedding_provider,
                )
            )
        except Exception as exc:
            logger.warning(
                "Workspace %s: could not spawn an embedding worker for %s: %s",
                self.config.workspace_path,
                path,
                exc,
            )
            self._fail(path, source_sha, f"{type(exc).__name__}: {exc}")
            return False
        return True

    def receiveMsg_IndexFailure(self, msg: IndexFailure) -> None:  # noqa: N802
        """TELL, from a worker. Mark the file ``FAILED`` and free its slot."""
        try:
            self._release_worker_slot()
            if self._live_entry(msg.path, msg.source_sha) is not None:
                self._fail(msg.path, msg.source_sha, msg.reason)
            self._drain()
        except Exception:
            logger.warning(
                "Workspace %s: could not record the index failure for %s",
                self.config.workspace_path,
                msg.path,
                exc_info=True,
            )

    def receiveMsg_EmbeddingResult(self, msg: EmbeddingResult) -> None:  # noqa: N802
        """TELL, from an ``#embed-`` worker. Write the batch, count it, settle at the last.

        **The write happens here, and this is where the epic's gate now lives.**
        One ask per worker report, on its own mailbox turn — never thirty on one
        turn. An ask that raises settles the file ``FAILED`` with the reason on
        this turn, so a batch that could not land can no longer leave a file
        sitting at ``EMBEDDING`` for the reaper to find ten minutes later.

        Only the **final** transition persists on the success path. A batch that
        lands without settling its file writes ``batches_landed`` through
        ``_put_row`` and sends nothing on its own turn: the row is dirty and rides
        on the next delta — this file's final transition, or any other path's —
        so a 1,900-chunk document costs one delta rather than thirty, and the
        store is never wrong about a status, only one counter behind.

        The **first** report that fails the file marks it ``FAILED``, and every
        later report for the same path is dropped without a second transition —
        the status guard below is what does it.
        """
        try:
            self._on_embedding_result(msg)
        except Exception:
            logger.warning(
                "Workspace %s: could not record an embedded batch",
                self.config.workspace_path,
                exc_info=True,
            )

    def receiveMsg_EmbeddingError(self, msg: EmbeddingError) -> None:  # noqa: N802
        """TELL, from an ``#embed-`` worker. Fail the file with the worker's reason."""
        try:
            self._on_embedding_error(msg)
        except Exception:
            logger.warning(
                "Workspace %s: could not record an embedding failure",
                self.config.workspace_path,
                exc_info=True,
            )

    def _counting_row(self, collection: str, path: str | None) -> RagFile | None:
        """Return the row *path* is counting batches into, or ``None`` to drop the report.

        A report for another collection, for a path this actor never queued, or
        for a file that has already left ``EMBEDDING`` — settled, failed, or
        re-queued at other bytes — belongs to nothing and is dropped.
        """
        if collection != RAG_COLLECTION or path is None:
            return None
        row = self._entry(path).row
        if row is None or row.status is not RagStatus.EMBEDDING:
            return None
        return row

    def _on_embedding_error(self, msg: EmbeddingError) -> None:
        """Fail the file one worker could not embed for."""
        entry = self._counting_row(msg.collection, msg.request_ref)
        if entry is None or msg.request_ref is None:
            return
        self._fail(msg.request_ref, entry.indexed_sha, msg.error)

    def _on_embedding_result(self, msg: EmbeddingResult) -> None:
        """Write one embedded batch, then apply it to the row that is counting it."""
        entry = self._counting_row(msg.collection, msg.request_ref)
        if entry is None or msg.request_ref is None:
            return
        path = msg.request_ref
        proxy = self._vs_proxy
        try:
            if proxy is None:
                raise RuntimeError("no vector store is bound")
            proxy.add(RAG_COLLECTION, msg.entries)
        except Exception as exc:
            self._fail(path, entry.indexed_sha, f"{type(exc).__name__}: {exc}")
            return
        landed = entry.batches_landed + 1
        if landed < entry.batches_expected:
            # Written on this turn, where it used to ride on the next delta. The
            # cost is one rewrite of this file per landing batch; the counter has
            # to survive a crash or the file parks at ``EMBEDDING`` until the
            # reaper, and the only state-free alternative is not persisting it.
            self._put_row(path, entry.model_copy(update={"batches_landed": landed}))
            return
        self._put_row(
            path,
            entry.model_copy(
                update={
                    "status": RagStatus.EMBEDDED,
                    "batches_landed": landed,
                    "updated_at": datetime.now(UTC),
                }
            ),
        )
        self._drop_superseded(path)

    def _drop_superseded(self, path: str) -> None:
        """Remove the previous chunk set, now that the new one has landed.

        **Add-then-remove, and never the other way round.** ``chunk_id`` includes
        the source digest, so the new ids cannot collide with the old ones — which
        is what makes this ordering safe, and what makes the other ordering leave a
        file absent from search for minutes while the list still calls it stale.

        The call is wrapped, because the store's ``remove`` re-raises a
        missing collection as a ``RetriableError``. A failure leaves
        ``superseded_chunk_ids`` populated so a later re-index retries it, and
        never fails the file: the worst case is a few orphaned vectors, and the
        alternative is a file that is ``FAILED`` because of a cleanup.

        **The two record touches run under the hold; ``proxy.remove`` does not.**
        That is ``_gated``'s split, for its reason: a network call inside a
        cross-process lock is how a lock becomes a bottleneck, and a doubled
        ``remove`` is idempotent. With ``_enqueue``'s append and this clear both
        serialised, the list cannot be lost — which is the whole of the orphan.

        **The clear subtracts what was actually removed rather than blanking the
        field**, because the list is not this call's to own: a concurrent
        ``_enqueue`` on the next turn appends the ids *it* has just superseded,
        and blanking would drop them with nothing removed and nothing raised.
        Sequentially there is nothing to subtract and the field still ends empty.
        """
        with self._hold_record(path):
            row = self._entry(path).row
            if row is None or not row.superseded_chunk_ids:
                return
            owed = list(row.superseded_chunk_ids)
        proxy = self._vs_proxy
        if proxy is None:
            return
        try:
            proxy.remove(RAG_COLLECTION, owed, scope=self.config.workspace_path)
        except Exception as exc:
            logger.warning(
                "Workspace %s: could not remove %d superseded chunk(s) of %s: %s — "
                "they are kept for the next re-index to retry",
                self.config.workspace_path,
                len(owed),
                path,
                exc,
            )
            return
        removed = set(owed)
        with self._hold_record(path):
            current = self._entry(path).row
            if current is None:
                return
            remaining = [
                stale_id for stale_id in current.superseded_chunk_ids if stale_id not in removed
            ]
            self._put_row(path, current.model_copy(update={"superseded_chunk_ids": remaining}))

    def _live_entry(self, path: str, source_sha: str) -> RagFile | None:
        """Return *path*'s row when it is still the one *source_sha* was indexing.

        A report whose file has since been re-indexed at other bytes belongs to a
        run nobody is waiting for, and applying it would overwrite the live run's
        chunk set with a stale one.
        """
        row = self._entry(path).row
        if row is None or row.indexed_sha != source_sha:
            logger.debug(
                "Workspace %s: dropping an index report for %s — the row has moved on",
                self.config.workspace_path,
                path,
            )
            return None
        return row

    def _fail(self, path: str, source_sha: str | None, reason: str) -> None:
        """Mark *path* ``FAILED``, keeping whatever chunks it already had.

        The chunk set is deliberately not cleared: a previously indexed file stays
        searchable at its previous content, which is what makes a failure a
        degradation rather than a loss.
        """
        row = self._entry(path).row
        if row is None or (source_sha is not None and row.indexed_sha != source_sha):
            return
        self._put_row(
            path,
            row.model_copy(
                update={
                    "status": RagStatus.FAILED,
                    "reason": reason,
                    "updated_at": datetime.now(UTC),
                }
            ),
        )

    ##
    ## The ``EMBEDDING`` bound, and the gate's staleness signal
    ##
    def reap_abandoned_rows(self) -> bool:
        """Re-queue every in-flight row no live worker is carrying, and say if any moved.

        **Runs at the top of ``index_paths``, and never on a turn path** — not in
        the context-state provider, not in ``rag_snapshot``, not in the gate. It
        is a mutation, and one that fired on every turn of every agent carrying
        the card would be both wasteful and a write from a render.

        **It is the backstop for every row a worker was carrying and no longer
        is, whichever process that worker belonged to.** Every other way a batch
        can fail arrives as a message — a worker that could not embed tells
        ``EmbeddingError``, and a write that could not land raises on the turn it
        is attempted. A worker that died silently reports neither, and its file
        would otherwise stay ``EXTRACTION``, ``SPLITTING`` or ``EMBEDDING`` for
        ever: :meth:`_drain` spawns ``PENDING`` only and :meth:`_is_accounted_for`
        counts in-flight as current, so nothing else would ever move it.

        **One predicate, and it is the age bound alone.** With the index in one
        actor's memory there was a moment called "restore" at which every
        in-flight row was abandoned *by construction*, and a separate hook
        re-queued them with no age bound; folding that into this method left a
        second discriminator beside the bound, ``_index_active``, exempting the
        rows this process was carrying. That exemption is **deleted**, and the
        deletion is a stated behaviour change rather than a tidy-up.

        It was silently false for every worker in another process — the case the
        records moved to disk to support — so it never answered the question it
        looked like it answered. Honestly stated, the rule is: a row older than
        :data:`~akgentic.tool.workspace.documents.models.EMBEDDING_STALE_AFTER_S`
        is re-queued, in this process exactly as in another. What that costs is a
        local worker still running past ten minutes being joined by a second one,
        and the duplicate is harmless because
        :func:`~akgentic.tool.workspace.documents.models.chunk_id` is
        deterministic and every backend now **upserts** a row rather than
        appending it.

        ``chunks`` and ``superseded_chunk_ids`` are kept — the superseded ids are
        still owed a removal, and the chunk set keeps the row's heading paths
        renderable until a worker replaces it.

        **It keeps the full listing, unlike ``_drain``**, and legitimately: the
        question it asks is about every row on the tree at once, so there is no
        filtered read that would answer it. The render is the other such caller. A
        remote ``DocumentStore`` would want a filtered form of this too.

        Returns:
            Whether any row was re-queued.
        """
        cutoff = datetime.now(UTC) - timedelta(seconds=EMBEDDING_STALE_AFTER_S)
        now = datetime.now(UTC)
        requeued = 0
        for entry in self._entries():
            row = entry.row
            if row is None or row.status not in _CARRIED_BY_A_WORKER:
                continue
            if row.updated_at >= cutoff:
                continue
            self._put_row(entry.path, _requeued(row, now))
            requeued += 1
        if requeued:
            logger.info(
                "Workspace %s: %d file(s) left in flight past %.0fs with no live worker "
                "are queued again",
                self.config.workspace_path,
                requeued,
                EMBEDDING_STALE_AFTER_S,
            )
        return requeued > 0

    ##
    ## workspace_rag_list — a render, and therefore free
    ##
    def rag_snapshot(self, max_pending_shown: int) -> RagIndexState:
        """Return the index as rows, capped on ``PENDING`` only.

        **No file access inside the tree, and no tree sweep**: this is asked once
        per turn by every agent carrying the card, and a ``stat`` per candidate
        file would put a tree walk on the hot path for a display. Reading
        ``<meta>/rag/`` is not that — it is one directory scan plus one parse per
        record, bounded by ``max_documents`` (32, or 8 when the vector backend is
        in-memory). That is real where the metadata root is a network share, and
        it has no mitigation on offer: ADR-051 Decision 9 once suggested pointing
        ``AKGENTIC_WORKSPACE_META_ROOT`` at a tmpfs for exactly this, and ADR-053
        Decision 5 **withdrew** that suggestion — the directory holds every
        cross-process lock, so a per-machine filesystem silently deletes the
        exclusion two workers over one tree depend on. **No read-through cache is
        added here** either: it would be exactly the in-memory state this move
        removes.

        The rows are sorted by **path**, so the render is stable across runs. A
        directory glob's order is the file system's, and a display that reordered
        itself between two turns would look like the index had changed.

        Everything that is not ``PENDING`` is always shown — those rows each say
        something different. ``PENDING`` rows all say the same thing, so a
        10,000-file tree would otherwise flood the context window with them.

        Args:
            max_pending_shown: How many ``PENDING`` rows to render.

        Returns:
            The state, never ``None`` and never raising.
        """
        rows: list[RagFileRow] = []
        hidden = 0
        pending_shown = 0
        for entry in sorted(self._entries(), key=lambda stored: stored.path):
            row = entry.row
            if row is None:
                continue
            if row.status is RagStatus.PENDING:
                if pending_shown >= max_pending_shown:
                    hidden += 1
                    continue
                pending_shown += 1
            rows.append(
                RagFileRow(
                    path=entry.path,
                    status=row.status.value,
                    chunk_count=row.chunk_count,
                    reason=row.reason or "",
                )
            )
        return RagIndexState(rows=rows, pending_hidden=hidden)

    ##
    ## workspace_rag_search — two legs, fused, and every failure degrades
    ##
    def rag_search(
        self,
        query: str,
        hits: dict[str, SearchHit] | None = None,
        top_k: int = 5,
        path_prefix: str = "",
        alpha: float | None = None,
    ) -> str:
        """Fuse the vector *hits* it is handed with its own keyword leg, and render.

        Two legs, combined by the one fusion rule the package shares — but only
        one of them runs here. The **vector** leg is the caller's: the query embed
        and the scoped similarity search against ``workspace_chunks`` are two
        external round trips, and they happen on the calling agent's own thread
        inside :func:`~akgentic.tool.workspace.rag._vector_hits`, which is the
        card's. What runs here is the half whose inputs are this actor's state and
        nothing else: a case-insensitive term match over the extraction bodies it
        already holds, the fusion, and the render that resolves each hit's heading
        path through the rows on disk.

        **That split is why this method makes no external call at all**, which is
        what ``actor/__init__.py``'s "and never external" claims of every ask on
        this thread. An embed here — let alone a ``search`` against a cluster
        backend on the same turn — held every ``request_exec``, every
        ``exec_status`` and every worker report behind an HTTP request.

        The vector leg is **not** routed through
        :func:`~akgentic.tool.vector_store.hybrid.semantic_scores`, and that is a
        correctness requirement rather than a preference. That helper takes no
        ``scope`` and no ``path_prefix``, and one ``workspace_chunks`` class holds
        every workspace of every team — a search through it would return another
        workspace's chunks. It also reduces its result to ``{ref_id: score}``,
        discarding the ``SearchHit`` that carries the text a hit renders and the
        ``path`` / ``ordinal`` its heading path is looked up by. ``fuse`` and the
        two constants are what this module reuses.

        **The unavailable gate stays here and is unchanged.** A tree with no store
        announced, or with parameters never announced, answers the sentence
        whatever the caller handed it.

        **Every failure degrades and none of them raises.** A vector leg that
        failed hands an empty mapping — it has already logged its own warning —
        and the keyword leg answers alone. This actor owns the write gate, and a
        retrieval capability that raised would be a way for a misconfigured
        deployment to take it down.

        Args:
            query: What to look for, in natural language.
            hits: ``{ref_id: hit}`` from the caller's vector leg, already filtered
                by its score threshold. ``None`` means the caller ran no vector
                leg, or ran one that degraded — the keyword leg then answers alone.
            top_k: How many hits to render, applied **after** filtering.
            path_prefix: Restrict the search to paths starting with this. Must not
                contain ``*`` or ``?`` — see
                :data:`~akgentic.tool.vector_store.protocol.PATH_PREFIX_WILDCARDS`.
                Checked here as well as at the caller: this is an ask with a
                public-shaped signature, and the gate must not depend on which
                caller reached it.
            alpha: Weight of the vector leg. ``None`` takes the fusion module's
                own default, which is the value the Weaviate client sends.

        Returns:
            The rendered hits, or one of the three sentences: retrieval
            unavailable, the prefix refused, or nothing matched.
        """
        from akgentic.tool.vector_store.hybrid import DEFAULT_ALPHA, fuse

        if self._vs_proxy is None or self._rag_params is None:
            return _UNAVAILABLE
        if any(character in path_prefix for character in PATH_PREFIX_WILDCARDS):
            return _REJECTED_PREFIX
        budget = max(top_k, 1)
        hits = hits or {}
        matches = self._keyword_leg(query, path_prefix)
        fused = fuse(
            list(matches),
            {ref_id: hit.score for ref_id, hit in hits.items()},
            alpha=DEFAULT_ALPHA if alpha is None else alpha,
        )
        rendered: list[str] = []
        for ref_id, score in sorted(fused.items(), key=lambda item: item[1], reverse=True):
            line = self._render_hit(score, hits.get(ref_id), matches.get(ref_id))
            if line is not None:
                rendered.append(line)
            if len(rendered) >= budget:
                break
        return "\n\n".join(rendered) if rendered else _NO_HITS

    def _keyword_leg(self, query: str, path_prefix: str) -> dict[str, _KeywordMatch]:
        """Return the chunks whose own slice of their document carries a query term.

        Case-insensitive, over the bodies this actor already holds — no file is
        read and no chunk text is stored anywhere, because a chunk is a pair of
        offsets into an extraction and never a copy of one.

        **An evicted body contributes nothing and is never sliced** (ADR-045 §3,
        §4). The search degrades toward vector-only for that file and is never
        wrong; the file stays ``EMBEDDED`` and its vector hits still render from
        the store's own copy of the text. This is what makes ``max_documents`` a
        bound on the extraction cache rather than on the searchable corpus — and
        it bounds no state of this actor's at all any more: it is a field of the
        :class:`~akgentic.tool.workspace.documents.cache.DocumentCache` the card
        builds, applied over the records on disk.

        **A body that is not the one the offsets were cut from is skipped too.**
        The two halves have different lifetimes even inside one record: a file
        re-read after a change holds a new body while its row still describes the
        old chunk boundaries, and slicing one with the other yields text that
        belongs to neither. The offsets of such a row are provenance, exactly as
        an evicted file's are.

        The keys are ``chunk_id``s — the key space ``fuse`` combines on, and what
        ``SearchHit.ref_id`` carries. It is an **indicator** and not a score: a
        flat substring match is equally good everywhere, which is why ``fuse``
        does not normalise this leg.
        """
        terms = query.lower().split()
        matches: dict[str, _KeywordMatch] = {}
        if not terms:
            return matches
        for entry in self._entries():
            extract, row = entry.extract, entry.row
            if extract is None or extract.markdown is None:
                continue
            if path_prefix and not entry.path.startswith(path_prefix):
                continue
            if row is None or row.indexed_sha != extract.source_sha:
                continue
            body = extract.markdown
            lowered = body.lower()
            for chunk in row.chunks:
                if any(term in lowered[chunk.start : chunk.end] for term in terms):
                    matches[chunk.chunk_id] = _KeywordMatch(
                        path=entry.path, chunk=chunk, text=body[chunk.start : chunk.end]
                    )
        return matches

    def _render_hit(
        self, score: float, hit: SearchHit | None, match: _KeywordMatch | None
    ) -> str | None:
        """Render one fused hit — path, heading path, score label, and the text.

        **The text comes from** ``SearchHit.text`` **whenever there is a hit**,
        never from a slice of the cached body: that is what keeps a file whose
        body was evicted searchable and renderable. A keyword-only hit has no
        ``SearchHit`` behind it, and its text is its own slice — which is present
        by construction, since matching it is what put it here.

        Args:
            score: The fused score, unused in the label and kept for the caller's
                ordering. See :func:`_score_label` for what is actually shown.
            hit: The vector hit, or ``None`` for a keyword-only match.
            match: The keyword match, or ``None`` for a vector-only hit.

        Returns:
            The rendered block, or ``None`` when neither leg supplied anything —
            which the caller skips without spending a result slot.
        """
        chunk: RagChunk | None
        if match is not None:
            path, chunk = match.path, match.chunk
            text = hit.text if hit is not None else match.text
        elif hit is not None:
            path = hit.path or ""
            chunk = self._chunk_at(path, hit.ordinal)
            text = hit.text
        else:
            return None
        heading = " > ".join(chunk.heading_path) if chunk is not None else ""
        location = f"{path} > {heading}" if heading else (path or "(unknown file)")
        return f"{location} ({_score_label(hit, match)})\n{text.strip()}"

    def _chunk_at(self, path: str, ordinal: int | None) -> RagChunk | None:
        """Return *path*'s chunk at *ordinal* — one dict lookup, and no reverse map.

        Story 45-6 put ``path`` and ``ordinal`` on ``SearchHit`` precisely so that
        this is O(1). A hit whose ``path`` or ``ordinal`` is missing, or whose
        ordinal is out of range, resolves to ``None`` and renders with an empty
        heading path rather than being dropped — the chunk text is still the
        answer.
        """
        if not path or ordinal is None:
            return None
        row = self._entry(path).row
        if row is None or not 0 <= ordinal < len(row.chunks):
            return None
        chunk = row.chunks[ordinal]
        return chunk if chunk.ordinal == ordinal else None

    ##
    ## The upload handler — reachable from outside the framework
    ##
    def receiveMsg_NewFileMessage(self, msg: NewFileMessage) -> None:  # noqa: N802
        """TELL, from whatever accepted an upload. Index the paths it names.

        **It never raises, and that is the most load-bearing property here**
        (ADR-045 §5, §2). This handler is reachable from *outside* the framework,
        and an exception on this turn would kill the actor that owns the write
        gate for the whole team. Same contract and same shape as
        :meth:`receiveMsg_IndexResult`: the body is a private method, this wraps
        it, and a failure logs and leaves the actor alive.

        It returns ``None`` and the sender does not wait. The frontend's upload
        must not block on extraction, and a 500-page PDF must not hold an HTTP
        request open; progress is observed through ``workspace_rag_list``.

        **Its documented "records and does not spawn" path is unreachable today,
        and that is a consequence to know rather than a regression to repair.**
        A tree with no retrieval capability has no actor at all since story 55-8,
        so a notification naming it reaches nobody — and no sender exists outside
        this package to send one anyway (verified across ``akgentic-infra``, both
        deployment tiers, ``-agent``, ``-catalog`` and the frontend). The branch
        stays because the *first* card on a tree may enable indexing while a later
        one does not, which is the case it was written for; it is not a reason to
        keep an actor alive for a card that dispatches nothing.
        """
        try:
            self._on_new_files(msg)
        except Exception:
            logger.warning(
                "Workspace %s: could not accept a new-file notification",
                self.config.workspace_path,
                exc_info=True,
            )

    def _on_new_files(self, msg: NewFileMessage) -> None:
        """Validate, hash, queue and spawn — O(n) over the path list, and no more.

        **An upload indexes where a gate write only marks ``STALE``, and the
        asymmetry is the decision rather than an omission** (ADR-045 §4, §5). An
        upload is one deliberate human act; an agent write is a stream of them,
        and auto-indexing each would spend embedding credits on content that is
        about to change again.

        **With no retrieval capability enabled it records and does not spawn.**
        The handler has no capability flag of its own — it is a message handler,
        not a tool — so the actor's test is the state ``enable_rag`` left behind.
        Writing the rows ``PENDING`` means enabling retrieval later picks the
        files up; spawning would spend embedding credits in a team that never
        opted in. **That path is idempotent at the live digest too**, and for a
        stronger reason than tidiness: the index is on disk, so a fresh process
        over the same tree sees ``EMBEDDED`` rows before any retrieval card has
        enabled anything. Re-queueing such a row here would clear its chunk set
        into ``superseded_chunk_ids`` that no proxy will ever remove — losing the
        heading paths a search renders. The rows stay as they are, and a later
        ``workspace_rag_index`` is what moves them.
        """
        candidates = self._uploaded_candidates(msg.paths)
        if not candidates:
            return
        if self._rag_params is None or self._vs_proxy is None:
            recorded = 0
            for path, sha in candidates:
                if self._is_accounted_for(path, sha, msg.force):
                    continue
                self._enqueue(path, sha)
                recorded += 1
            logger.info(
                "Workspace %s: recorded %d of %d new file(s) from %s as pending — "
                "retrieval is not enabled on this tree",
                self.config.workspace_path,
                recorded,
                len(candidates),
                msg.source,
            )
            return
        queued = 0
        for path, sha in candidates:
            if self._is_accounted_for(path, sha, msg.force):
                continue
            self._enqueue(path, sha)
            queued += 1
        logger.info(
            "Workspace %s: %d of %d new file(s) from %s were queued for indexing",
            self.config.workspace_path,
            queued,
            len(candidates),
            msg.source,
        )
        self._drain()

    def _uploaded_candidates(self, paths: list[str]) -> list[tuple[str, str]]:
        """Return ``(path, digest)`` for every named path that can be indexed.

        **Every path goes through** :class:`~akgentic.tool.workspace.workspace.Filesystem`,
        which validates internally — never ``backend._root / path``. An upload
        handler taking caller-supplied paths is the most escape-prone surface this
        decision adds, and joining onto the private root is the traversal bypass
        this package has already had to close once.

        Every rejection is a **log line and a skip**, never an error, and the four
        that matter are all ordinary: a path that escapes the root, one that does
        not exist because the message raced the upload's own write, one whose type
        cannot be indexed, and one that is not a usable path at all.
        """
        found: list[tuple[str, str]] = []
        for path in paths:
            try:
                if Path(path).suffix.lower() not in _INDEXABLE_EXTENSIONS:
                    logger.info(
                        "Workspace %s: %r is not an indexable type — skipped",
                        self.config.workspace_path,
                        path,
                    )
                    continue
                # ``_digest`` reads through ``Filesystem``, whose ``resolve_path``
                # raises ``PathEscapeError`` — a ``PermissionError``, and therefore
                # an ``OSError`` the digest already absorbs alongside a missing file.
                sha = self._digest(path)
            except Exception:
                logger.info(
                    "Workspace %s: skipping an unusable entry in a new-file notification",
                    self.config.workspace_path,
                    exc_info=True,
                )
                continue
            if sha is not None:
                found.append((path, sha))
        return found


def _score_label(hit: SearchHit | None, match: _KeywordMatch | None) -> str:
    """Describe how one chunk was found, for its rendered line.

    The shape ``PlanningTool`` established and the house convention records: the
    number shown is the **raw** cosine score, which is the only absolute one — a
    fused score is normalised against the rest of one result set and means nothing
    outside it.
    """
    if hit is None:
        return "keyword match"
    return f"{'hybrid' if match is not None else 'semantic'}: {hit.score:.2f}"


_IN_FLIGHT = frozenset(
    {RagStatus.PENDING, RagStatus.EXTRACTION, RagStatus.SPLITTING, RagStatus.EMBEDDING}
)
"""Statuses meaning "a run over these bytes has not finished yet"."""

_CARRIED_BY_A_WORKER = _IN_FLIGHT - {RagStatus.PENDING}
"""Statuses a worker is carrying — the three a row can only leave by being reported.

**Not "orphaned on restore", which is what this was called and is no longer
true.** There is no restore any more, and therefore no moment at which every row
in one of these statuses is abandoned by construction: the records are on disk
and another process may be working on one of them right now.
:meth:`DocumentsMixin.reap_abandoned_rows` is what decides, and it needs exactly
one more fact than this set — the age bound. The per-process exemption that used
to sit beside it is gone: it could only ever speak for workers in *this* process,
which made it silently false for the case the records moved to disk to support.
"""


def _requeued(entry: RagFile, now: datetime) -> RagFile:
    """*entry* back at ``PENDING`` with both batch counters reset — its chunks kept.

    A copy with the four fields that change, never a rebuild (Golden Rule 12).
    One re-queue survives — :meth:`DocumentsMixin.reap_abandoned_rows` — where
    there were three, so this is now the shape that one uses rather than the
    agreement three had to keep.
    """
    return entry.model_copy(
        update={
            "status": RagStatus.PENDING,
            "batches_expected": 0,
            "batches_landed": 0,
            "updated_at": now,
        }
    )


def _batch_size() -> int:
    """Return ``EMBED_BATCH_SIZE``, read here rather than duplicated.

    **The cycle these three function-level imports existed for is gone, and they
    stay function-level anyway.** ``documents/worker.py`` imported ``card.params``
    at runtime and ``card`` imports this actor package, so the constant could not
    be reached from this module's import block at all. The worker is
    ``rag/worker.py`` now and takes its parameter from ``rag/params.py``, so
    hoisting the three would work — this module already imports ``rag.context``
    at the top, which executes the same package ``__init__``.

    Hoisting them is an optimisation with a real import-order risk and no
    behaviour to gain, so it is not taken here. The ``# noqa`` comments say what
    they are now rather than repeating a cycle that no longer exists.
    """
    from akgentic.tool.workspace.rag.worker import EMBED_BATCH_SIZE  # noqa: PLC0415 — see above

    return EMBED_BATCH_SIZE
