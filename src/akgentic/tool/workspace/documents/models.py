"""The extraction cache, the RAG index, and the caps both of them answer to.

**This module is the final home of :class:`DocumentExtract`, :class:`RagFile`
and :class:`RagChunk`, chosen now on purpose.** ``serialize()`` stamps
``__model__ = "<module>.<name>"`` into every nested ``SerializableBaseModel``,
and the deserializer resolves that literal string with ``import_module`` plus
``getattr``. An extract — or an index row — persisted inside a
``WorkspaceState`` snapshot therefore pins this module path in deployments this
repository cannot see — the failure mode that forced ``workspace/tool.py`` to
stay on disk as a shim. Nothing here moves afterwards.

**This module adds no digest.** ``content_sha`` in
:mod:`akgentic.tool.workspace.models` is the one definition of the digest in the
package, and ``source_sha`` below is produced by the *caller* and only compared
here. A second digest expression is how a gate fails closed while looking
healthy; the same reasoning applies to a cache that would then never hit.

The import edge runs one way — :mod:`akgentic.tool.workspace.models` imports
from here — so nothing in this module may import from it.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from akgentic.core.utils.serializer import SerializableBaseModel

__all__ = [
    "CHUNK_ID_NAMESPACE",
    "DEFAULT_MAX_DOCUMENTS",
    "DEFAULT_MAX_DOCUMENT_CHARS",
    "EMBEDDING_STALE_AFTER_S",
    "EXTRACTOR_VERSION",
    "IN_MEMORY_MAX_DOCUMENTS",
    "IN_MEMORY_MAX_DOCUMENT_CHARS",
    "RAG_COLLECTION",
    "DocumentExtract",
    "RagChunk",
    "RagFile",
    "RagStatus",
    "chunk_id",
    "derived_document_caps",
    "evict_document_bodies",
]

EXTRACTOR_VERSION = 1
"""Stamp on every cached extract, and the whole of the invalidation protocol.

A cached extract is served only while its stamp equals this constant, so
**bumping it invalidates every cached extract in every workspace on the next
read** — no sweep, no migration, no code that has to find the stale rows. The
bodies simply stop being hits and are re-extracted once each, on demand.

That is what makes an extractor defect repairable after the fact. The standing
example is a ``U+0000`` a reader wrote verbatim into every sidecar it produced:
once the sanitiser lands, one increment here is the entire remediation.

Nothing in story 45-3 bumps it. A bump belongs to whoever changes what the
extractor *produces* from unchanged source bytes.
"""

DEFAULT_MAX_DOCUMENTS = 32
"""Bound on the number of rows in ``WorkspaceState.documents``.

Answers the **metadata** dimension: ~200 bytes per row, re-serialised in full by
``model_dump_json()`` on every notify, growing for the life of the team. An
uncapped map on a team singleton leaks exactly that way — the same reasoning
:data:`~akgentic.tool.workspace.models.DEFAULT_MAX_TRACKED_WRITERS` records.

This is the Weaviate / RAG-off default. A backend-derived override belongs at
the one ``WorkspaceConfig`` construction site and is 45-7's, not this story's.
"""

DEFAULT_MAX_DOCUMENT_CHARS = 2_000_000
"""Bound on the characters held across every entry that still has a body.

Answers the **bytes** dimension, which is a different pressure from the row
count and therefore has a different remedy — see :func:`evict_document_bodies`.

This is the Weaviate / RAG-off default, as above.
"""

IN_MEMORY_MAX_DOCUMENTS = 8
"""Row cap when the vector backend is in-memory **and** retrieval is on.

The vector volume derives from the document cap, and the in-memory backend keeps
every vector inside ``VectorStoreState`` — which is re-serialised in full on
every notify. The arithmetic these two constants encode, written down so nobody
"tidies" them later: at ``chunk_chars = 1200`` with 150 characters of overlap the
stride is ~1050 characters, so 2 MB of Markdown is ~1,900 chunks, and 1536
``text-embedding-3-small`` floats rendered as JSON is ~23 KB each. In-memory at
the Weaviate caps is therefore **~44 MB re-serialised on every notify**. That row
is what these two constants exist to make unreachable.
"""

IN_MEMORY_MAX_DOCUMENT_CHARS = 200_000
"""Character cap under the same condition — see :data:`IN_MEMORY_MAX_DOCUMENTS`."""

RAG_COLLECTION = "workspace_chunks"
"""The **one** collection every workspace's chunks live in (ADR-045 §7).

One class for every workspace, never one per tree: a class per workspace would
mean a Weaviate schema mutation per team. The workspace is carried as a
``scope`` **property** on each entry instead, and every read and every removal
passes that scope as a predicate.

It is also why :func:`chunk_id` puts the scope inside its digest — see there.
"""

CHUNK_ID_NAMESPACE = uuid.UUID("2f5c1a90-7b64-5c3e-9f21-0d8a4c6b1e73")
"""Namespace of every chunk id this package mints.

A **hard-coded literal**, and that is the whole of the mechanism: a namespace
minted with ``uuid.uuid4()`` at import would change per process, which makes
every id non-deterministic and destroys the idempotent-retry property re-indexing
depends on. Two processes must agree, and a spec pins the agreement against a
literal.
"""

EMBEDDING_STALE_AFTER_S = 600.0
"""How long a file may sit at :attr:`RagStatus.EMBEDDING` before it is reaped.

Not a card parameter. ``VectorStoreActor`` keeps the map from an open request to
its requester in a **private** attribute — correctly, because an ``ActorAddress``
inside a ``BaseState`` breaks ``notify_state_change()``
(``b12consulting/akgentic-core#131``) — so after a resume the store's own pending
requests are gone too and its derived status truthfully reads ``READY``. Nothing
in the store will ever tell a file left at ``EMBEDDING`` that its signal is not
coming. The reaper is the only thing that will, and reverting to
:attr:`RagStatus.PENDING` costs one re-index and never a wrong answer.
"""


def derived_document_caps(backend: str, rag_enabled: bool) -> tuple[int, int]:
    """Return ``(max_documents, max_document_chars)`` for a vector backend.

    8 / 200,000 on an in-memory vector backend with retrieval on; 32 / 2,000,000
    otherwise. **Retrieval off yields the larger pair whatever the backend is**
    (ADR-045 §7): no vectors exist, so nothing is derived from the document cap,
    and lowering it would shrink a cache for a cost that is not being paid.

    Args:
        backend: The collection's configured backend — ``"inmemory"`` or
            ``"weaviate"``.
        rag_enabled: Whether any retrieval capability is enabled on the card.

    Returns:
        The two caps, in the order :class:`~akgentic.tool.workspace.models.WorkspaceConfig`
        declares them.
    """
    if rag_enabled and backend == "inmemory":
        return IN_MEMORY_MAX_DOCUMENTS, IN_MEMORY_MAX_DOCUMENT_CHARS
    return DEFAULT_MAX_DOCUMENTS, DEFAULT_MAX_DOCUMENT_CHARS


def chunk_id(scope: str, path: str, source_sha: str, ordinal: int) -> str:
    """Mint the deterministic id of one chunk.

    Deterministic because re-indexing has to be idempotent: the same file at the
    same bytes produces the same ids in any process, on any host, so a retry
    overwrites rather than duplicates.

    **The scope is inside the digest, and leaving it out was a defect in this
    decision's first draft.** One ``workspace_chunks`` class holds every
    workspace, and ``InMemoryBackend._map_search_hits`` resolves ``{ref_id:
    entry}`` last-one-wins — so two trees holding the same file at the same path
    with identical bytes would mint the same ``ref_id`` and the collision would be
    silent and cross-workspace.

    ``source_sha`` is inside it for a different reason: re-index is
    add-then-remove, and that ordering is only safe while the new ids cannot
    collide with the old ones.

    Args:
        scope: The workspace the chunk belongs to.
        path: Workspace-relative path of the source file.
        source_sha: Digest of the source bytes the chunk was cut from.
        ordinal: Position of the chunk within its document.

    Returns:
        The chunk's id, as the string form of a UUID5.
    """
    return str(uuid.uuid5(CHUNK_ID_NAMESPACE, f"{scope}:{path}:{source_sha}:{ordinal}"))


class RagStatus(StrEnum):
    """Where one file stands in the indexing pipeline.

    The five transient values are a straight line —
    :attr:`PENDING` → :attr:`EXTRACTION` or :attr:`SPLITTING` →
    :attr:`EMBEDDING` → :attr:`EMBEDDED` — and the two terminal ones are reached
    from anywhere:

    - :attr:`FAILED` — an extraction, a split or an embedding batch failed. The
      file keeps whatever chunks it had, so a previously indexed file stays
      searchable at its previous content.
    - :attr:`STALE` — the tree changed underneath an indexed file. The gate marks
      it and does **not** re-index: an agent mid-task rewrites the same file
      repeatedly, and auto-indexing every accepted write would spend embedding
      credits on every save.
    """

    PENDING = "pending"
    EXTRACTION = "extraction"
    SPLITTING = "splitting"
    EMBEDDING = "embedding"
    EMBEDDED = "embedded"
    FAILED = "failed"
    STALE = "stale"


class RagChunk(SerializableBaseModel):
    """One embeddable region of one document — **offsets, never text**.

    A chunk is a pair of offsets into the file's extracted Markdown and nothing
    else. There is no ``text`` field, no ``prefix`` field and no body of any
    kind: storing the text would duplicate the document inside the actor's state,
    which is re-serialised on every notify, and would let the copy drift from the
    extraction it claims to describe. The composed text is built at embed time
    and handed straight to the vector store.

    Attributes:
        chunk_id: The chunk's identity, from :func:`chunk_id`. It is also the
            ``ref_id`` the vector store stores it under.
        ordinal: Position within the document, from zero.
        start: Character offset into the extracted Markdown.
        end: Exclusive end offset.
        heading_path: The enclosing heading texts, outermost first. Carried
            through from ``Span`` verbatim, and composed into a prefix at embed
            time when ``prepend_heading_path`` is on.
        header_start: Offset of a table's own header row, for a continuation
            piece of a table cut at the ceiling. ``None`` on every other chunk.
        header_end: Exclusive end of that header row. Set exactly when
            ``header_start`` is.
    """

    chunk_id: str
    ordinal: int
    start: int
    end: int
    heading_path: list[str] = []
    header_start: int | None = None
    header_end: int | None = None


class RagFile(SerializableBaseModel):
    """One file's row in the retrieval index.

    **Every transition on this model is a**
    ``model_copy(update=...)`` (Golden Rule #12). The row is persisted and
    re-saved on every status change — ``PENDING`` → ``EXTRACTION`` →
    ``SPLITTING`` → ``EMBEDDING`` → ``EMBEDDED``, plus ``STALE``, plus the
    reaper's revert, plus every landing batch — which is seven or more rewrites
    per indexed file. A field-by-field rebuild is correct on the day it is
    written and silently destroys the field added after it.

    Attributes:
        path: Workspace-relative path of the source file.
        status: See :class:`RagStatus`.
        indexed_sha: Digest of the source bytes **this row is about**. Set when
            indexing starts rather than when it finishes, because a landing
            embedding batch has to be attributable to the run that issued it: a
            batch whose file has since been re-indexed at other bytes is dropped
            on this comparison.
        chunks: The chunk set the row currently describes. During a re-index it
            holds the **new** set from the moment the worker reports, which is
            what lets a landing batch be attributed.
        chunk_count: ``len(chunks)``, carried so a render never has to walk them.
        batches_expected: How many ``add()`` calls this file's chunks were split
            into. Set before the first call is issued.
        batches_landed: How many of them have reported success. ``EMBEDDED`` is
            reached when and only when the two are equal.
        superseded_chunk_ids: The ids of the **previous** chunk set, waiting to be
            removed. Re-index is add-then-remove, so the old ids need somewhere to
            live for the duration — ``chunks`` cannot hold them, because it is
            already holding the new set. Cleared the moment the removal succeeds;
            left populated when it fails, so a later re-index retries it.
        reason: Why the file is ``FAILED``, or ``None``.
        updated_at: When the row last moved, in UTC.
    """

    path: str
    status: RagStatus
    indexed_sha: str | None = None
    chunks: list[RagChunk] = []
    chunk_count: int = 0
    batches_expected: int = 0
    batches_landed: int = 0
    superseded_chunk_ids: list[str] = []
    reason: str | None = None
    updated_at: datetime


class DocumentExtract(SerializableBaseModel):
    """One document's extracted Markdown, keyed in the cache by its workspace path.

    Every field is a primitive, so the model round-trips through Pydantic
    unaided: no ``arbitrary_types_allowed`` of its own, and no ``PrivateAttr`` —
    there is no runtime state here to keep out of a snapshot.

    Attributes:
        path: Workspace-relative path of the **source** file, not of a sidecar.
        source_sha: :func:`~akgentic.tool.workspace.models.content_sha` of the
            source bytes this extract was produced from. Supplied by the caller
            and only ever compared here, never recomputed.
        extractor_version: The value of :data:`EXTRACTOR_VERSION` in force when
            the body was produced. An entry stamped with anything else is a miss.
        markdown: The extracted body, or ``None`` when it was dropped under the
            character cap.

            **Nothing may slice this without first checking it is present.** The
            ``str | None`` type is the enforcement and mypy ``--strict`` is the
            gate: no ``# type: ignore`` and no ``assert x is not None`` on this
            field anywhere in ``src/``. A ``None`` body is an ordinary,
            *expected* state — the entry is metadata awaiting a re-extraction,
            not a broken row.

            It also says nothing about whether the file is indexed. See
            :func:`evict_document_bodies`.
        char_count: ``len(markdown)`` as produced, computed at the fill site so
            it cannot disagree with the body it describes. It **survives a body
            drop**, which is what lets a dropped body stop pressing on the
            character cap while the row still records how large it was.
        extracted_at: When the extraction ran, in UTC.
    """

    path: str
    source_sha: str
    extractor_version: int
    markdown: str | None
    char_count: int
    extracted_at: datetime


def _bodied_chars(documents: dict[str, DocumentExtract]) -> int:
    """Sum ``char_count`` over the entries that still carry a body."""
    return sum(entry.char_count for entry in documents.values() if entry.markdown is not None)


def evict_document_bodies(
    documents: dict[str, DocumentExtract], *, max_documents: int, max_document_chars: int
) -> list[str]:
    """Bring *documents* back under both caps, least-recently-used first.

    Pure: it takes a dict and two ints, and touches no actor, no ``self`` and no
    chunk structure. *documents* is mutated in place and iterated in insertion
    order, which is the whole of the LRU — the fill site re-inserts on every
    write and the lookup moves an entry to the end on every hit.

    **Two caps, two different remedies**, because they answer two different
    pressures:

    - Over *max_document_chars* — the least-recently-used entry that still has a
      body **keeps its metadata and drops its body**. The bytes are what press
      on the snapshot, and the row that remains is what lets a later read know
      the file was seen at all.
    - Over *max_documents* — the least-recently-used **entry is removed
      outright**. Dropping only bodies here would leave the row count unbounded,
      and an uncapped map on a team singleton leaks for the life of the team.

    Both are safe: every byte is regenerable from the tree, so an eviction costs
    one re-extraction and never a wrong answer. **A single document larger than
    *max_document_chars* drops its own body on insert** and becomes a permanent
    miss costing one re-extraction per read — correct, and deliberately not
    special-cased.

    **A dropped body must not de-index its file.** Nothing anywhere may infer
    that a file is indexed, searchable, or absent from the index from the
    presence, the absence, or the body-state of an entry in this map. Index
    membership lives in its own map and is keyed off that alone.

    Args:
        documents: The cache, mutated in place.
        max_documents: Cap on the number of entries.
        max_document_chars: Cap on the characters held across bodied entries.

    Returns:
        The paths whose entry was removed or whose body was dropped, in the
        order it happened, for the caller's log line.
    """
    evicted: list[str] = []
    while len(documents) > max_documents:
        oldest = next(iter(documents))
        del documents[oldest]
        evicted.append(oldest)
    while _bodied_chars(documents) > max_document_chars:
        victim = next(
            (path for path, entry in documents.items() if entry.markdown is not None), None
        )
        if victim is None:
            # Every body is already gone and the cap is still exceeded, which a
            # cap below one document's size makes reachable. Nothing is left to
            # drop, so the loop must exit rather than spin.
            break
        # Golden Rule #12: copy and override the one field that changes. A
        # field-by-field rebuild is correct on the day it is written and
        # silently destroys the seventh field the day one is added.
        documents[victim] = documents[victim].model_copy(update={"markdown": None})
        evicted.append(victim)
    return evicted
