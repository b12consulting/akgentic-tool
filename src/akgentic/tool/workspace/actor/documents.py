"""The extraction cache, the retrieval index, and every notify this package makes.

**Before story 45-7 ``notify_state_change()`` was called in exactly one place in
the whole ``workspace/`` package, and it was :meth:`DocumentsMixin.cache_document`.**
Every other tool actor calls it freely; this one did not call it at all before
this module existed, which is how epic 29 kept the event-store write off the read
path (ADR-036 §NFR1). ADR-045 §4 is the decision that adds more: the retrieval
index is persisted state that has to survive a resume, so the transitions that
move it notify. The call sites are now, and only:

- :meth:`DocumentsMixin.cache_document` — a cache **fill**, amortised against the
  seconds of extraction that preceded it.
- :meth:`DocumentsMixin.index_paths` — a queueing pass that actually queued
  something, or a drain that actually spawned.
- :meth:`DocumentsMixin.receiveMsg_NewFileMessage` — the same, for an upload. It
  is the *queueing* that notifies, so a notification that named no usable path
  costs nothing.
- :meth:`DocumentsMixin.receiveMsg_IndexResult` / :meth:`DocumentsMixin.receiveMsg_IndexError`
  — one per file, at its transition.
- :meth:`DocumentsMixin.receiveMsg_EmbeddingCompleted` — **only** at the file's
  final transition. A batch that lands without settling the file mutates
  ``batches_landed`` in memory and notifies nothing, so a 1,900-chunk document
  costs one event rather than thirty.
- :meth:`DocumentsMixin.mark_paths_stale` — only when it actually changed a
  status, so a tree that has never been indexed pays nothing on the mutation path.
- :meth:`DocumentsMixin.reap_stale_embedding` — only when it actually reverted a
  row.

**The rule that survives, unchanged and load-bearing: no notify on a text read,
and none on a document-cache hit.** Reads are the majority of workspace traffic.
A new notify on a *read* path — of any kind, in any of these methods — is a
defect until a decision says otherwise.

Everything on the ask path here is O(1)/O(n) dict work on the actor thread, plus
bounded file reads while queueing and bounded proxy calls to ``#VectorStore`` —
including, on :meth:`DocumentsMixin.rag_search`, **one query embed per call**.
That is the one external round trip this package puts on the gate's own thread.
It is bounded (one call, not thirty — which is why the indexing path issues its
``add()`` batches as a *tell*) and every one of its failure modes degrades to the
keyword leg, which is what makes it acceptable rather than a defect. Do not add a
second network call to that turn. The
slow half — extraction and splitting — happens in a ``#index-`` worker and never
here. Nothing here raises: an exception in a document handler would kill the actor
that owns the write gate, so a document path degrades — a miss, a file left
``FAILED``, a cache that did not grow — and never propagates.
"""

from __future__ import annotations

import contextlib
import logging
from datetime import UTC, datetime, timedelta
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from akgentic.core.agent_config import BaseConfig
from akgentic.tool.workspace.documents.context import RagFileRow, RagIndexState
from akgentic.tool.workspace.documents.models import (
    EMBEDDING_STALE_AFTER_S,
    EXTRACTOR_VERSION,
    RAG_COLLECTION,
    DocumentExtract,
    NewFileMessage,
    RagChunk,
    RagFile,
    RagStatus,
    evict_document_bodies,
)
from akgentic.tool.workspace.models import WorkspaceConfig, WorkspaceState, content_sha
from akgentic.tool.workspace.readers import _MIME_MAP, TEXT_EXTENSIONS, DocumentReader
from akgentic.tool.workspace.workspace import Filesystem

if TYPE_CHECKING:
    from akgentic.core.agent import Akgent
    from akgentic.tool.vector_store.actor import VectorStoreActor
    from akgentic.tool.vector_store.embedding_actor import EmbeddingCompleted
    from akgentic.tool.vector_store.protocol import CollectionConfig, SearchHit
    from akgentic.tool.workspace.card.params import WorkspaceRagIndex
    from akgentic.tool.workspace.documents.worker import IndexError, IndexResult

    # The mixin consumes the actor's own surface — ``createActor``, the two proxy
    # builders, ``myAddress``, ``orchestrator``, ``config`` and ``state``. Naming
    # the base under ``if TYPE_CHECKING:`` is how ``ExecMixin`` already reaches
    # its own; at runtime the mixin contributes only ``object``, so the MRO the
    # actor declares is unchanged and nothing here shadows a sibling.
    _DocumentsBase = Akgent[WorkspaceConfig, WorkspaceState]
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
``card/read.py`` draws the same line for ``expand_media_refs``; a second copy of
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

_NO_HITS = (
    "Nothing in the retrieval index matched that query. "
    "Use workspace_rag_list to see which files are indexed."
)
"""What a search answers when retrieval works and nothing matched.

Deliberately **not** :data:`_UNAVAILABLE`: "nothing matched" and "nothing is
indexed" are different problems with different next steps, and an agent handed
one sentence for both would retry the query when it should have indexed the tree.
"""

_PREFIX_WILDCARDS = "*?"
"""Characters a ``path_prefix`` may not contain, on either backend.

Both are legal in a POSIX filename and both are wildcards in Weaviate's ``Like``
operator, which is what ``WeaviateBackend`` builds a prefix filter from; the
in-memory backend uses ``str.startswith`` and treats them literally. The v4
filter API offers no escape, so the same query would mean two different things
depending on where the collection happens to live. Rejecting them at the
parameter boundary is the only answer under which the two backends agree, and it
costs nothing real: a prefix is a filter rather than a filename, so a shorter
wildcard-free prefix still reaches the file.
"""

_REJECTED_PREFIX = (
    "A path_prefix cannot contain '*' or '?': they are wildcards on one vector "
    "backend and literal characters on the other, so the same filter would mean "
    "two different things. Use a shorter prefix without them."
)
"""The sentence a rejected prefix answers with, identical on both backends."""


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
    """The extraction cache and the retrieval index over ``WorkspaceState``.

    Declares no Pydantic field and no state field: every map it uses is owned by
    the actor and initialised in its ``on_start``. It defines no sibling's method
    either — in particular not ``deliver``, ``fail`` or ``cache_capacity``, any of
    which would silently take over the deferred delivery path or resize the exec
    LRU, because this mixin precedes ``ExecMixin`` and ``DeferredResultActor`` in
    the MRO.
    """

    _workspace: Filesystem
    _rag_params: WorkspaceRagIndex | None
    _rag_reader: DocumentReader | None
    _rag_collection: CollectionConfig | None
    _vs_proxy: VectorStoreActor | None
    _vs_tell: VectorStoreActor | None
    _index_active: set[str]

    ##
    ## The extraction cache — lookup (ask) and fill (tell)
    ##
    def document_extract(self, path: str, source_sha: str, extractor_version: int) -> str | None:
        """Return the cached Markdown for *path*, or ``None`` on any miss.

        A hit requires all four of: the entry exists, it was produced from these
        source bytes, it was produced by this extractor, and its body is still
        present. Four distinct reasons to miss, one answer — the caller
        re-extracts, which is correct in every one of them.

        On a hit the entry moves to the end of the LRU. **That reorder is an
        in-memory mutation with no notify**, deliberately: this is the read
        path. Persisted recency therefore lags live recency until the next fill,
        which costs at most one extra re-extraction after a resume and is not to
        be "fixed" into a notify.

        Args:
            path: Workspace-relative path of the source file.
            source_sha: Digest of the source bytes the caller just read.
            extractor_version: The extractor the caller would run on a miss.

        Returns:
            The cached Markdown, or ``None``.
        """
        entry = self.state.documents.get(path)
        if (
            entry is None
            or entry.source_sha != source_sha
            or entry.extractor_version != extractor_version
            or entry.markdown is None
        ):
            return None
        # LRU: most recently used moves to the end. In-memory only — no notify
        # on a hit, because a hit is what a read does.
        self.state.documents[path] = self.state.documents.pop(path)
        return entry.markdown

    def cache_document(
        self, path: str, source_sha: str, extractor_version: int, markdown: str
    ) -> None:
        """Cache *markdown* as the extraction of *path*, evict, and notify once.

        The notify follows the insert **and** the eviction, so one fill is one
        event however many entries it displaced — never twice, never once per
        evicted entry. It is amortised against the seconds of extraction that
        preceded it.

        ``char_count`` is computed here rather than taken as a parameter, so it
        cannot disagree with the body it describes. Re-filling a known path
        refreshes its recency instead of leaving it where it was.

        **An eviction here must never de-index a file.** Nothing anywhere may
        infer index membership from ``state.documents``; the two maps are
        independent and only one of them is capped.

        Args:
            path: Workspace-relative path of the source file.
            source_sha: Digest of the source bytes this body was extracted from.
            extractor_version: The extractor that produced this body.
            markdown: The extracted body.
        """
        self.state.documents.pop(path, None)  # a re-fill refreshes recency
        self.state.documents[path] = DocumentExtract(
            path=path,
            source_sha=source_sha,
            extractor_version=extractor_version,
            markdown=markdown,
            char_count=len(markdown),
            extracted_at=datetime.now(UTC),
        )
        evicted = evict_document_bodies(
            self.state.documents,
            max_documents=self.config.max_documents,
            max_document_chars=self.config.max_document_chars,
        )
        if evicted:
            # One line per fill, never one per path, and DEBUG rather than INFO:
            # on a workspace sitting at either cap this fires on every fill, and
            # the question it answers — "why does this document keep
            # re-extracting?" — is a debugging question. It is also the only
            # evidence an entry-cap eviction ever happened, since the row is
            # gone by the time anything else could look.
            #
            # "evicted" covers both remedies deliberately: the return is a flat
            # list of paths and cannot say whether a path lost its whole entry
            # or only its body (45-3's frozen shape).
            # The paths are passed through rather than joined here: at either cap
            # this line runs on every fill, and an eager join would build the
            # string even with DEBUG off. ``%s`` over the list defers all of it.
            logger.debug(
                "Filling the document cache for %s evicted (entry removed or body dropped): %s",
                path,
                evicted,
            )
        self.state.notify_state_change()

    ##
    ## Retrieval — enabling it, and the collection that is created lazily
    ##
    def enable_rag(
        self,
        agent_id: str,
        params: WorkspaceRagIndex,
        reader: DocumentReader,
        collection: CollectionConfig,
    ) -> None:
        """Turn retrieval on for this tree — **tell** path, once per card.

        The actor cannot take any of this from :class:`WorkspaceConfig`, because
        ``getChildrenOrCreate`` fixes that at creation and the card that creates
        the actor for a workspace is routinely one with no retrieval capability at
        all. So a retrieval-capable card announces itself here instead, at bind
        time, exactly as ``configure_exec`` and ``register_agent`` do. **The actor
        never inspects a card**; it has no handle on one.

        **First call wins.** Two agents on one team must not make one file chunk
        two ways, so a second call carrying different parameters emits one INFO
        line naming both and changes nothing. A second call carrying equal
        parameters is silent.

        This is also where the collection is created — **lazily, and never in
        ``on_start``**: a workspace with retrieval off must never create one. It
        follows ``PlanActor._acquire_vs_proxy`` with one deliberate divergence:
        where that actor *raises* when ``#VectorStore`` is absent, this one logs
        and degrades. A missing vector store is a configuration error for a
        planning tool, whose whole purpose it is; here it must never be fatal,
        because this actor also owns the write gate.

        Args:
            agent_id: The announcing agent, for the log line only.
            params: The chunking configuration the whole tree will use.
            reader: The extraction configuration, which lives on the card.
            collection: The vector collection's backend, dimension and tenant.
        """
        try:
            if self._rag_params is not None:
                if self._rag_params != params:
                    logger.info(
                        "Workspace %s: retrieval is already configured by an earlier card; "
                        "agent %s asked for %s and keeps %s",
                        self.config.workspace_name,
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
                self.config.workspace_name,
                exc_info=True,
            )

    def _acquire_vs_proxy(self) -> None:
        """Resolve ``#VectorStore``, bind both proxies, and create the collection.

        **Two proxies over one address, and the split is a correctness choice.**
        ``create_collection`` and ``remove`` are **asks**: the first has to be
        known to have worked before anything is added, and the second re-raises a
        missing collection as a ``RetriableError`` that this actor must see to
        keep the superseded ids for a later retry. ``add`` is a **tell**: a
        1,900-chunk document is thirty of them, and thirty asks would park the
        actor that owns the write gate on another actor's mailbox.

        Any failure drops to degraded mode — ``_vs_proxy`` stays ``None`` and
        ``workspace_rag_index`` answers a sentence. **Both proxies are bound
        together or neither is**: ``index_paths`` gates on ``_vs_proxy`` alone, so
        a half-bound actor would accept work whose ``add()`` calls go nowhere,
        leaving every file at ``EMBEDDING`` until the reaper queues it again — and
        again, every ten minutes, for ever.
        """
        from akgentic.core.orchestrator import Orchestrator  # noqa: PLC0415 — cycle
        from akgentic.tool.vector_store.actor import (  # noqa: PLC0415 — optional extra
            VS_ACTOR_NAME,
            VectorStoreActor,
        )

        if self.orchestrator is None or self._rag_collection is None:
            logger.warning(
                "Workspace %s: no orchestrator — retrieval stays in degraded mode",
                self.config.workspace_name,
            )
            return
        orch_proxy = self.proxy_ask(self.orchestrator, Orchestrator)
        vs_addr = orch_proxy.get_team_member(VS_ACTOR_NAME)
        if vs_addr is None:
            logger.warning(
                "Workspace %s: %s was not found — retrieval stays in degraded mode. "
                "Add VectorStoreTool to the team configuration.",
                self.config.workspace_name,
                VS_ACTOR_NAME,
            )
            return
        proxy = self.proxy_ask(vs_addr, VectorStoreActor)
        try:
            proxy.create_collection(RAG_COLLECTION, self._rag_collection)
        except Exception as exc:
            logger.warning(
                "Workspace %s: create_collection(%s) failed: %s — degraded mode",
                self.config.workspace_name,
                RAG_COLLECTION,
                exc,
            )
            return
        tell = self.proxy_tell(vs_addr, VectorStoreActor)
        self._vs_proxy = proxy
        self._vs_tell = tell

    ##
    ## workspace_rag_index — the spawn side
    ##
    def index_paths(self, path: str = "", force: bool = False) -> str:
        """Queue every candidate under *path* and return what was accepted.

        Returns **immediately**. Everything here is O(n) over the candidate list
        on the actor thread — validate, read-and-hash, set ``PENDING``, spawn up
        to the concurrency cap — and no extraction, split or embedding happens on
        this turn.

        Args:
            path: A file, a directory, or ``""`` for the whole tree.
            force: Re-index a file that is already current at these bytes.

        Returns:
            The counts, or the degraded-mode sentence.
        """
        if self._vs_proxy is None or self._rag_params is None:
            return _UNAVAILABLE
        changed = self.reap_stale_embedding()
        candidates, unsupported = self._candidates(path)
        queued = current = 0
        for candidate in candidates:
            sha = self._digest(candidate)
            if sha is None:
                unsupported += 1
                continue
            if self._is_accounted_for(candidate, sha, force):
                current += 1
                continue
            self._enqueue(candidate, sha)
            queued += 1
        changed = self._drain() or queued > 0 or changed
        if changed:
            self.state.notify_state_change()
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
        entry = self.state.rag_index.get(path)
        if entry is None or entry.indexed_sha != sha:
            return False
        return entry.status in _IN_FLIGHT or entry.status is RagStatus.EMBEDDED

    def _enqueue(self, path: str, sha: str) -> None:
        """Put *path* at ``PENDING`` for *sha*, keeping the old ids to supersede.

        The previous chunk set's ids move to ``superseded_chunk_ids`` **before**
        ``chunks`` is cleared, because re-index is add-then-remove and the old ids
        have to survive somewhere for the duration. Ids left over from a removal
        that previously failed are kept, so a later re-index retries them.
        """
        now = datetime.now(UTC)
        entry = self.state.rag_index.get(path)
        if entry is None:
            self.state.rag_index[path] = RagFile(
                path=path, status=RagStatus.PENDING, indexed_sha=sha, updated_at=now
            )
            return
        superseded = list(entry.superseded_chunk_ids)
        # The membership set is built once. Rebuilding it per chunk is O(n²) on
        # the actor's mailbox turn, and an 800-page document is ~1,900 chunks.
        seen = set(superseded)
        for chunk in entry.chunks:
            if chunk.chunk_id not in seen:
                superseded.append(chunk.chunk_id)
                seen.add(chunk.chunk_id)
        self.state.rag_index[path] = entry.model_copy(
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
        )

    def _drain(self) -> bool:
        """Spawn workers for ``PENDING`` files up to the concurrency cap.

        Returns:
            Whether anything moved, so the caller can make one notify.
        """
        from akgentic.tool.workspace.documents.worker import (  # noqa: PLC0415 — cycle
            MAX_CONCURRENT_INDEX_WORKERS,
        )

        changed = False
        while len(self._index_active) < MAX_CONCURRENT_INDEX_WORKERS:
            waiting = next(
                (
                    candidate
                    for candidate, entry in self.state.rag_index.items()
                    if entry.status is RagStatus.PENDING and candidate not in self._index_active
                ),
                None,
            )
            if waiting is None or not self._spawn(waiting):
                return changed
            changed = True
        return changed

    def _spawn(self, path: str) -> bool:
        """Start one ``#index-`` worker for *path*, or record why it could not start.

        The worker is spawned with ``createActor`` and handed everything it needs
        in one payload, including the card's extraction configuration. It is
        **not** a ``DeferredWorker`` and is deliberately not routed through
        ``DeferredResultActor.request()``: that mechanism's reports land in this
        actor's exec result cache.

        The status it moves to says which half of the work the worker actually
        has to do — ``EXTRACTION`` when the body has to be produced,
        ``SPLITTING`` when the cache could supply one.

        Returns:
            Whether a worker is now running for *path*.
        """
        from akgentic.tool.workspace.documents.worker import (  # noqa: PLC0415 — cycle
            IndexRequest,
            IndexWorker,
            index_worker_name,
        )

        entry = self.state.rag_index[path]
        params, reader = self._rag_params, self._rag_reader
        if params is None or reader is None or entry.indexed_sha is None:
            return False
        scope = self.config.workspace_name
        markdown = self.document_extract(path, entry.indexed_sha, EXTRACTOR_VERSION)
        try:
            address = self.createActor(
                IndexWorker, config=BaseConfig(name=index_worker_name(scope, path))
            )
            self.proxy_tell(address, IndexWorker).receiveMsg_IndexRequest(
                IndexRequest(
                    path=path,
                    scope=scope,
                    source_sha=entry.indexed_sha,
                    markdown=markdown,
                    params=params,
                    reader=reader,
                )
            )
        except Exception as exc:
            logger.warning("Workspace %s: could not spawn an index worker for %s", scope, path)
            self._fail(path, entry.indexed_sha, f"{type(exc).__name__}: {exc}")
            return False
        self._index_active.add(path)
        self.state.rag_index[path] = entry.model_copy(
            update={
                "status": RagStatus.SPLITTING if markdown is not None else RagStatus.EXTRACTION,
                "updated_at": datetime.now(UTC),
            }
        )
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
                self.config.workspace_name,
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
                self.config.workspace_name,
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
                self.config.workspace_name,
                msg.path,
                exc_info=True,
            )

    def _on_index_result(self, msg: IndexResult) -> None:
        """Record *msg*, issue its ``add()`` batches, and notify once."""
        self._index_active.discard(msg.path)
        entry = self._live_entry(msg.path, msg.source_sha)
        if entry is None:
            self._drain()
            return
        if msg.extracted:
            # The worker did the extraction, so the cache learns from it. This is
            # the one notify in this method that is not the file's own transition,
            # and it is a fill like any other.
            self.cache_document(msg.path, msg.source_sha, EXTRACTOR_VERSION, msg.markdown)
        if len(msg.texts) != len(msg.chunks):
            self._fail(msg.path, msg.source_sha, "the worker returned mismatched chunks and texts")
            self._drain()
            self.state.notify_state_change()
            return
        batches = ceil(len(msg.chunks) / _batch_size())
        self.state.rag_index[msg.path] = entry.model_copy(
            update={
                "status": RagStatus.EMBEDDING if msg.chunks else RagStatus.EMBEDDED,
                "chunks": msg.chunks,
                "chunk_count": len(msg.chunks),
                "batches_expected": batches,
                "batches_landed": 0,
                "reason": None,
                "updated_at": datetime.now(UTC),
            }
        )
        if msg.chunks:
            self._issue_batches(msg)
        else:
            # An empty document is indexed the moment it is split: there is
            # nothing to embed and nothing to wait for.
            self._drop_superseded(msg.path)
        self._drain()
        self.state.notify_state_change()

    def _issue_batches(self, msg: IndexResult) -> None:
        """Send one ``add()`` per ``EMBED_BATCH_SIZE`` chunks, correlated by path.

        ``requester`` is this actor and ``request_ref`` is the file's path, which
        is what lets :meth:`receiveMsg_EmbeddingCompleted` attribute a completion
        to the row that is counting it. ``batches_expected`` is already written by
        the caller, before the first call goes out.
        """
        from akgentic.tool.vector_store.vector import VectorEntry  # noqa: PLC0415 — optional extra

        tell = self._vs_tell
        if tell is None:
            return
        entries = [
            VectorEntry(
                ref_type=_CHUNK_REF_TYPE,
                ref_id=chunk.chunk_id,
                text=text,
                vector=[],
                scope=self.config.workspace_name,
                path=msg.path,
                ordinal=chunk.ordinal,
            )
            for chunk, text in zip(msg.chunks, msg.texts, strict=True)
        ]
        size = _batch_size()
        for start in range(0, len(entries), size):
            tell.add(
                RAG_COLLECTION,
                entries[start : start + size],
                requester=self.myAddress,
                request_ref=msg.path,
            )

    def receiveMsg_IndexError(self, msg: IndexError) -> None:  # noqa: N802
        """TELL, from a worker. Mark the file ``FAILED`` and free its slot."""
        try:
            self._index_active.discard(msg.path)
            if self._live_entry(msg.path, msg.source_sha) is not None:
                self._fail(msg.path, msg.source_sha, msg.reason)
            self._drain()
            self.state.notify_state_change()
        except Exception:
            logger.warning(
                "Workspace %s: could not record the index failure for %s",
                self.config.workspace_name,
                msg.path,
                exc_info=True,
            )

    def receiveMsg_EmbeddingCompleted(self, msg: EmbeddingCompleted) -> None:  # noqa: N802
        """TELL, from ``#VectorStore``. Count one batch, and settle the file at the last.

        Only the **final** transition notifies. A batch that lands without
        settling its file mutates ``batches_landed`` in memory and says nothing,
        so a 1,900-chunk document costs one event rather than thirty.

        The **first** completion carrying an error marks the file ``FAILED``, and
        every later completion for the same path is dropped without a second
        transition — the status guard below is what does it.
        """
        try:
            self._on_embedding_completed(msg)
        except Exception:
            logger.warning(
                "Workspace %s: could not record an embedding completion",
                self.config.workspace_name,
                exc_info=True,
            )

    def _on_embedding_completed(self, msg: EmbeddingCompleted) -> None:
        """Apply one settled batch to the row that is counting it."""
        path = msg.request_ref
        if msg.collection != RAG_COLLECTION or path is None:
            return
        entry = self.state.rag_index.get(path)
        if entry is None or entry.status is not RagStatus.EMBEDDING:
            return
        if msg.error is not None:
            self._fail(path, entry.indexed_sha, msg.error)
            self.state.notify_state_change()
            return
        landed = entry.batches_landed + 1
        if landed < entry.batches_expected:
            # In memory only: the file has not moved, so nothing is worth an event.
            self.state.rag_index[path] = entry.model_copy(update={"batches_landed": landed})
            return
        self.state.rag_index[path] = entry.model_copy(
            update={
                "status": RagStatus.EMBEDDED,
                "batches_landed": landed,
                "updated_at": datetime.now(UTC),
            }
        )
        self._drop_superseded(path)
        self.state.notify_state_change()

    def _drop_superseded(self, path: str) -> None:
        """Remove the previous chunk set, now that the new one has landed.

        **Add-then-remove, and never the other way round.** ``chunk_id`` includes
        the source digest, so the new ids cannot collide with the old ones — which
        is what makes this ordering safe, and what makes the other ordering leave a
        file absent from search for minutes while the list still calls it stale.

        The call is wrapped, because ``VectorStoreActor.remove`` re-raises a
        missing collection as a ``RetriableError``. A failure leaves
        ``superseded_chunk_ids`` populated so a later re-index retries it, and
        never fails the file: the worst case is a few orphaned vectors, and the
        alternative is a file that is ``FAILED`` because of a cleanup.
        """
        entry = self.state.rag_index.get(path)
        proxy = self._vs_proxy
        if entry is None or proxy is None or not entry.superseded_chunk_ids:
            return
        try:
            proxy.remove(
                RAG_COLLECTION,
                entry.superseded_chunk_ids,
                scope=self.config.workspace_name,
            )
        except Exception as exc:
            logger.warning(
                "Workspace %s: could not remove %d superseded chunk(s) of %s: %s — "
                "they are kept for the next re-index to retry",
                self.config.workspace_name,
                len(entry.superseded_chunk_ids),
                path,
                exc,
            )
            return
        current = self.state.rag_index[path]
        self.state.rag_index[path] = current.model_copy(update={"superseded_chunk_ids": []})

    def _live_entry(self, path: str, source_sha: str) -> RagFile | None:
        """Return *path*'s row when it is still the one *source_sha* was indexing.

        A report whose file has since been re-indexed at other bytes belongs to a
        run nobody is waiting for, and applying it would overwrite the live run's
        chunk set with a stale one.
        """
        entry = self.state.rag_index.get(path)
        if entry is None or entry.indexed_sha != source_sha:
            logger.debug(
                "Workspace %s: dropping an index report for %s — the row has moved on",
                self.config.workspace_name,
                path,
            )
            return None
        return entry

    def _fail(self, path: str, source_sha: str | None, reason: str) -> None:
        """Mark *path* ``FAILED``, keeping whatever chunks it already had.

        The chunk set is deliberately not cleared: a previously indexed file stays
        searchable at its previous content, which is what makes a failure a
        degradation rather than a loss.
        """
        entry = self.state.rag_index.get(path)
        if entry is None or (source_sha is not None and entry.indexed_sha != source_sha):
            return
        self.state.rag_index[path] = entry.model_copy(
            update={"status": RagStatus.FAILED, "reason": reason, "updated_at": datetime.now(UTC)}
        )

    ##
    ## The ``EMBEDDING`` bound, and the gate's staleness signal
    ##
    def reap_stale_embedding(self) -> bool:
        """Revert files stuck at ``EMBEDDING`` past the bound, and say if any moved.

        **Runs on resume and at the top of ``index_paths``, and never on a turn
        path** — not in the context-state provider, not in ``rag_snapshot``, not in
        the gate. It is a state mutation, and one that fired on every turn of every
        agent carrying the card would be both wasteful and a write from a render.

        The resume call site is ``WorkspaceActor.init_state``, not ``on_start``:
        ``on_start`` assigns a fresh :class:`WorkspaceState` on its first line and a
        restored snapshot arrives afterwards, so reaping there would run against an
        empty index. See that method for the whole of it.

        It is also the only thing that will ever free such a file. ``#VectorStore``
        keeps the map from an open request to its requester in a private attribute,
        so a resume drops it along with the pending requests themselves, and the
        store's own status then truthfully reads ``READY``. Nothing there knows a
        workspace file is still waiting.

        Returns:
            Whether any row was reverted, so the caller can make one notify.
        """
        cutoff = datetime.now(UTC) - timedelta(seconds=EMBEDDING_STALE_AFTER_S)
        reverted = 0
        for path, entry in list(self.state.rag_index.items()):
            if entry.status is not RagStatus.EMBEDDING or entry.updated_at >= cutoff:
                continue
            self.state.rag_index[path] = entry.model_copy(
                update={
                    "status": RagStatus.PENDING,
                    "batches_expected": 0,
                    "batches_landed": 0,
                    "updated_at": datetime.now(UTC),
                }
            )
            reverted += 1
        if reverted:
            logger.info(
                "Workspace %s: %d file(s) left embedding past %.0fs are queued again",
                self.config.workspace_name,
                reverted,
                EMBEDDING_STALE_AFTER_S,
            )
        return reverted > 0

    def mark_paths_stale(self, paths: list[str]) -> None:
        """Mark every indexed path in *paths* ``STALE`` — and re-index none of them.

        Called **directly on ``self``** from the one point the six mutations
        converge on, never as a cross-actor tell: there is no message to drop, no
        ordering question, and no chance of arriving after the target stopped.

        **It marks and returns.** An agent mid-task rewrites the same file
        repeatedly, and auto-indexing every accepted write would spend embedding
        credits on every save and queue workers behind a file that is about to
        change again. Gate writes mark stale; uploads index.

        It notifies **only when it actually changed a status**, so a tree that has
        never been indexed pays nothing on the mutation path — which is the common
        case, and must stay free.

        Args:
            paths: The mutation's own write set.
        """
        now = datetime.now(UTC)
        changed = False
        for path in paths:
            entry = self.state.rag_index.get(path)
            if entry is None or entry.status is RagStatus.STALE:
                continue
            self.state.rag_index[path] = entry.model_copy(
                update={"status": RagStatus.STALE, "updated_at": now}
            )
            changed = True
        if changed:
            self.state.notify_state_change()

    ##
    ## workspace_rag_list — a render, and therefore free
    ##
    def rag_snapshot(self, max_pending_shown: int) -> RagIndexState:
        """Return the index as rows, capped on ``PENDING`` only.

        **No file access of any kind**, and no tree sweep: this is asked once per
        turn by every agent carrying the card, and a ``stat`` per candidate would
        put a tree walk on the hot path for a display. It is O(n) dict work over
        rows that already exist.

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
        for path, entry in self.state.rag_index.items():
            if entry.status is RagStatus.PENDING:
                if pending_shown >= max_pending_shown:
                    hidden += 1
                    continue
                pending_shown += 1
            rows.append(
                RagFileRow(
                    path=path,
                    status=entry.status.value,
                    chunk_count=entry.chunk_count,
                    reason=entry.reason or "",
                )
            )
        return RagIndexState(rows=rows, pending_hidden=hidden)

    ##
    ## workspace_rag_search — two legs, fused, and every failure degrades
    ##
    def rag_search(
        self,
        query: str,
        top_k: int = 5,
        path_prefix: str = "",
        alpha: float | None = None,
        score_threshold: float = 0.0,
    ) -> str:
        """Search the indexed chunks and render the best *top_k*.

        Two legs, combined by the one fusion rule the package shares: a **scoped**
        similarity search against ``workspace_chunks``, and a case-insensitive term
        match over the extraction bodies this actor already holds.

        **The vector leg is run here rather than through**
        :func:`~akgentic.tool.vector_store.hybrid.semantic_scores`, and that is a
        correctness requirement rather than a preference. That helper takes no
        ``scope`` and no ``path_prefix``, and one ``workspace_chunks`` class holds
        every workspace of every team — a search through it would return another
        workspace's chunks. It also reduces its result to ``{ref_id: score}``,
        discarding the ``SearchHit`` that carries the text a hit renders and the
        ``path`` / ``ordinal`` its heading path is looked up by. ``fuse`` and the
        two constants are what this module reuses; only the leg that needs a
        filter is local.

        **Every failure degrades and none of them raises.** No proxy, an ``embed``
        that raises or returns nothing, a ``search`` that raises: each yields an
        empty vector mapping, one warning, and the keyword leg alone. This actor
        owns the write gate, and a retrieval capability that raised would be a way
        for a misconfigured deployment to take it down.

        Args:
            query: What to look for, in natural language.
            top_k: How many hits to render, applied **after** filtering.
            path_prefix: Restrict the search to paths starting with this. Must not
                contain ``*`` or ``?`` — see :data:`_PREFIX_WILDCARDS`.
            alpha: Weight of the vector leg. ``None`` takes the fusion module's
                own default, which is the value the Weaviate client sends.
            score_threshold: Minimum **raw** cosine score for a vector hit,
                applied before normalisation so it keeps its absolute meaning.

        Returns:
            The rendered hits, or one of the three sentences: retrieval
            unavailable, the prefix refused, or nothing matched.
        """
        from akgentic.tool.vector_store.hybrid import DEFAULT_ALPHA, fuse

        if self._vs_proxy is None or self._rag_params is None:
            return _UNAVAILABLE
        if any(character in path_prefix for character in _PREFIX_WILDCARDS):
            return _REJECTED_PREFIX
        budget = max(top_k, 1)
        hits = self._vector_leg(query, budget, path_prefix, score_threshold)
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

    def _vector_leg(
        self, query: str, top_k: int, path_prefix: str, score_threshold: float
    ) -> dict[str, SearchHit]:
        """Embed *query* and search the collection **within this workspace only**.

        The ``scope`` predicate is mandatory on every workspace query (ADR-045 §5,
        §7) and both predicates go to the backend, so the ``top_k`` budget is never
        spent on another scope's objects. The call over-fetches by ``OVERFETCH``
        because fusion reorders and the caller drops what it cannot resolve.

        Returns:
            ``{ref_id: hit}`` for the hits at or above *score_threshold*, or an
            empty mapping on any failure.
        """
        from akgentic.tool.vector_store.hybrid import OVERFETCH

        proxy = self._vs_proxy
        if proxy is None:
            logger.warning(
                "Workspace %s: no vector store — searching on the keyword leg alone",
                self.config.workspace_name,
            )
            return {}
        try:
            vectors = proxy.embed([query])
            if not vectors:
                logger.warning(
                    "Workspace %s: embedding a search query returned nothing — keyword only",
                    self.config.workspace_name,
                )
                return {}
            result = proxy.search(
                RAG_COLLECTION,
                vectors[0],
                top_k * OVERFETCH,
                scope=self.config.workspace_name,
                path_prefix=path_prefix or None,
            )
        except Exception:
            logger.warning(
                "Workspace %s: the vector leg of a search failed — keyword only",
                self.config.workspace_name,
                exc_info=True,
            )
            return {}
        return {hit.ref_id: hit for hit in result.hits if hit.score >= score_threshold}

    def _keyword_leg(self, query: str, path_prefix: str) -> dict[str, _KeywordMatch]:
        """Return the chunks whose own slice of their document carries a query term.

        Case-insensitive, over the bodies this actor already holds — no file is
        read and no chunk text is stored anywhere, because a chunk is a pair of
        offsets into an extraction and never a copy of one.

        **An evicted body contributes nothing and is never sliced** (ADR-045 §3,
        §4). The search degrades toward vector-only for that file and is never
        wrong; the file stays ``EMBEDDED`` and its vector hits still render from
        the store's own copy of the text. This is what makes ``max_documents`` a
        bound on the size of the actor's state rather than on the searchable
        corpus.

        **A body that is not the one the offsets were cut from is skipped too.**
        The two maps have different lifetimes: a file re-read after a change holds
        a new body while its row still describes the old chunk boundaries, and
        slicing one with the other yields text that belongs to neither. The
        offsets of such a row are provenance, exactly as an evicted file's are.

        The keys are ``chunk_id``s — the key space ``fuse`` combines on, and what
        ``SearchHit.ref_id`` carries. It is an **indicator** and not a score: a
        flat substring match is equally good everywhere, which is why ``fuse``
        does not normalise this leg.
        """
        terms = query.lower().split()
        matches: dict[str, _KeywordMatch] = {}
        if not terms:
            return matches
        for path, extract in self.state.documents.items():
            body = extract.markdown
            if body is None or (path_prefix and not path.startswith(path_prefix)):
                continue
            entry = self.state.rag_index.get(path)
            if entry is None or entry.indexed_sha != extract.source_sha:
                continue
            lowered = body.lower()
            for chunk in entry.chunks:
                if any(term in lowered[chunk.start : chunk.end] for term in terms):
                    matches[chunk.chunk_id] = _KeywordMatch(
                        path=path, chunk=chunk, text=body[chunk.start : chunk.end]
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
        entry = self.state.rag_index.get(path)
        if entry is None or not 0 <= ordinal < len(entry.chunks):
            return None
        chunk = entry.chunks[ordinal]
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
        """
        try:
            self._on_new_files(msg)
        except Exception:
            logger.warning(
                "Workspace %s: could not accept a new-file notification",
                self.config.workspace_name,
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
        opted in.
        """
        candidates = self._uploaded_candidates(msg.paths)
        if not candidates:
            return
        if self._rag_params is None or self._vs_proxy is None:
            for path, sha in candidates:
                self._enqueue(path, sha)
            logger.info(
                "Workspace %s: recorded %d new file(s) from %s as pending — "
                "retrieval is not enabled on this tree",
                self.config.workspace_name,
                len(candidates),
                msg.source,
            )
            self.state.notify_state_change()
            return
        queued = 0
        for path, sha in candidates:
            if self._is_accounted_for(path, sha, msg.force):
                continue
            self._enqueue(path, sha)
            queued += 1
        logger.info(
            "Workspace %s: %d of %d new file(s) from %s were queued for indexing",
            self.config.workspace_name,
            queued,
            len(candidates),
            msg.source,
        )
        if self._drain() or queued > 0:
            self.state.notify_state_change()

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
                        self.config.workspace_name,
                        path,
                    )
                    continue
                # ``_digest`` reads through ``Filesystem``, whose ``_validate_path``
                # raises ``PathEscapeError`` — a ``PermissionError``, and therefore
                # an ``OSError`` the digest already absorbs alongside a missing file.
                sha = self._digest(path)
            except Exception:
                logger.info(
                    "Workspace %s: skipping an unusable entry in a new-file notification",
                    self.config.workspace_name,
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


def _batch_size() -> int:
    """Return ``EMBED_BATCH_SIZE``, imported where the cycle cannot bite.

    ``documents/worker.py`` imports ``card.params`` at runtime, and ``card`` imports
    this actor package — so the constant cannot be reached from this module's
    import block. It is read here rather than duplicated, so there is one value.
    """
    from akgentic.tool.workspace.documents.worker import EMBED_BATCH_SIZE  # noqa: PLC0415 — cycle

    return EMBED_BATCH_SIZE
