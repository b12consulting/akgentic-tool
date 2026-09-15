"""A search, whole — both legs, the fusion and the result, over the records on disk.

**Nothing here holds a card, an actor or a proxy**, and nothing here imports
``card/``. The module is a pure function of *(cache, store, params, query)*: the
caller resolves those four and calls in, on whatever thread it is already on.
That is what makes "a search reaches no mailbox" a property of the code rather
than of a wiring convention — there is no mailbox in scope to reach.

**Both legs live here because a search has two, and splitting them is what went
wrong.** The *vector* leg — the query embed and the scoped similarity search —
moved onto the calling agent's thread in story 55-6 and sat in the capability's
wiring module; the *keyword* leg, the fusion and the render stayed on
``#Workspace`` behind a blocking ask, justified as *"the half whose inputs are
this actor's state and nothing else"*. That claim was false: the keyword leg
reads the document records off disk, through a
:class:`~akgentic.tool.workspace.documents.cache.DocumentCache` the card already
holds and built itself. So a search queued behind ``request_exec``,
``exec_status`` and every index worker's report, for work that is file reading —
and what needs a mailbox is a run whose report must land somewhere and the
children a Pydantic card cannot parent, which a search is neither (ADR-053
Decision 6). Leaving one leg in the wiring module while the other moved here
would have been the same split in a new place, so both are here.

**A search is still an ask.** Nothing here is ``async``, returns a future, spawns
a thread or acquires a lock: the calling agent is blocked for exactly as long as
before, and only the thread that reads the files has changed.

**A read writes nothing.** The keyword leg reads extraction bodies and chunk
offsets and writes neither; building a hit reads one row. A write on this path —
of any kind, in any function here — is a defect until a decision says otherwise.

**This module holds no gate.** The three degradation sentences belong to the
caller, which is the only layer that knows whether a card is bound at all; see
:meth:`~akgentic.tool.workspace.rag.RagFactories._rag_search_factory`, where the
ordering of them is load-bearing. What is here carries :data:`_NO_HITS` as the
result's ``note`` when retrieval worked and nothing matched, and never raises out
of a leg.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple

from pydantic import BaseModel

from akgentic.tool.workspace.documents.models import RAG_COLLECTION, RagChunk

if TYPE_CHECKING:
    from akgentic.tool.vector_store.protocol import (
        SearchHit,
        VectorStoreParam,
        VectorStoreService,
    )
    from akgentic.tool.workspace.documents.cache import DocumentCache

logger = logging.getLogger(__name__)

_NO_HITS = (
    "Nothing in the retrieval index matched that query. "
    "Use workspace_rag_list to see which files are indexed."
)
"""The ``note`` a search carries when retrieval works and nothing matched.

Deliberately **not** the unavailable sentence: "nothing matched" and "nothing is
indexed" are different problems with different next steps, and an agent handed
one sentence for both would retry the query when it should have indexed the tree.
"""


class MatchKind(StrEnum):
    """Which leg of the search answered for one hit.

    Total over the two legs: a hit is here because the vector leg supplied it,
    because the keyword leg did, or because both did. There is no fourth case —
    a fused key that resolves to neither leg becomes no hit at all.
    """

    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    KEYWORD = "keyword"


class RagSearchHit(BaseModel):
    """One matched chunk, and everything needed to go and read around it.

    This is what replaced a rendered line. The point of the model is the four
    coordinate fields: an agent that has found the right passage can call
    ``workspace_read(path, offset=start_line, limit=end_line - start_line + 1)``
    and widen from there, instead of pulling a whole document back into its
    context window or re-parsing a string to find out where the passage lives.

    **Plain** :class:`~pydantic.BaseModel` **and not**
    ``SerializableBaseModel``: this model is serialised into a prompt and
    consumed, never persisted and reconstructed, so the ``__model__`` module-path
    discriminator that serializer stamps on every dump would buy nothing and
    would put this module's own import path in front of the model once per hit.

    Attributes:
        path: Workspace-relative path of the file the chunk is in.
        ordinal: Position of the chunk within its document, from zero. ``None``
            only when the vector store reported none either.
        chunk_count: How many chunks that document holds, so a reader knows
            where in the file this one falls. ``None`` when the row could not be
            resolved.
        start_line: The 1-indexed line of the extracted Markdown the chunk starts
            on. ``None`` when the row could not be resolved, **and** on a file
            indexed before ranges were recorded — ``workspace_rag_index(path,
            force=True)`` fills those in. It is never *refuse*.
        end_line: The 1-indexed line the chunk's last character falls on —
            inclusive both ends. ``None`` under the same two conditions.

            **The text of the hit is not the slice this range names.**
            ``compose_chunk_text`` prepends the heading path and re-synthesises a
            cut table's header row, so what was embedded is longer than the range
            and holds lines present in no document. Both statements are true and
            they are not reconciled: the range locates the chunk in its document,
            and :attr:`text` is what was matched.
        heading_path: The enclosing heading texts, outermost first. ``[]`` when
            the row could not be resolved, or when the chunk has none.
        score: The **raw** cosine of the vector leg, which is the only absolute
            number here. ``None`` — never ``0.0`` — on a keyword-only hit: the
            keyword leg is an indicator that fusion does not normalise, so there
            is no number to report, and ``0.0`` would read as *matched, badly*.
            The **fused** score is in no field at all; it is spent on the order
            of :attr:`RagSearchResult.hits` and means nothing outside one result.
        match: Which leg answered — see :class:`MatchKind`.
        text: The chunk's text as the leg supplied it: the vector store's own
            copy whenever there is a vector hit, which is what keeps a file whose
            extraction body was evicted both searchable and readable, and the
            keyword leg's slice only when there is not.
    """

    path: str
    ordinal: int | None = None
    chunk_count: int | None = None
    start_line: int | None = None
    end_line: int | None = None
    heading_path: list[str] = []
    score: float | None = None
    match: MatchKind
    text: str


class RagSearchResult(BaseModel):
    """What a search answers — the hits, or the one sentence saying why not.

    Exactly one of the two carries the answer in practice: a search that matched
    has hits and no note, and a search that did not — nothing matched, retrieval
    is unavailable, the prefix was refused — has a note and no hits.

    Attributes:
        hits: The matched chunks, **best first** by the fused score. The budget
            is spent after filtering, so a fused key neither leg could resolve
            costs no slot.
        note: Why there is nothing to show, in the same words the callable used
            to answer bare. ``None`` when there are hits.
    """

    hits: list[RagSearchHit] = []
    note: str | None = None


class _KeywordMatch(NamedTuple):
    """One chunk the keyword leg hit, with everything its hit needs.

    Carried because a keyword-only hit has no ``SearchHit`` behind it and would
    otherwise have no text at all — which would make a keyword-only search, the
    degraded mode this whole design turns on, answer nothing.

    ``chunk_count`` rides along because this leg holds the row while it matches:
    taking it here costs nothing, and it keeps a keyword-only hit off
    :func:`_locate_chunk` — and therefore off the disk — entirely.
    """

    path: str
    chunk: RagChunk
    text: str
    chunk_count: int


def search_documents(
    cache: DocumentCache,
    store: VectorStoreService | None,
    resolved: VectorStoreParam | None,
    query: str,
    *,
    top_k: int,
    scope: str,
    path_prefix: str = "",
    alpha: float | None = None,
    score_threshold: float = 0.0,
) -> RagSearchResult:
    """Run both legs over *cache*, fuse them, and answer a structured result.

    The two legs are combined by the one fusion rule the package shares. The
    vector leg is **not** routed through
    :func:`~akgentic.tool.vector_store.hybrid.semantic_scores`, and that is a
    correctness requirement rather than a preference: that helper takes no
    ``scope`` and no ``path_prefix``, and one ``workspace_chunks`` class holds
    every workspace of every team — a search through it would return another
    workspace's chunks. It also reduces its result to ``{ref_id: score}``,
    discarding the ``SearchHit`` that carries the text a hit answers with and the
    ``path`` / ``ordinal`` its coordinates are looked up by. ``fuse`` and the two
    constants are what this module reuses.

    The ``top_k`` budget is spent **after** filtering, so a hit neither leg could
    supply costs no result slot.

    **Every failure degrades and none of them raises.** A vector leg that failed
    hands back an empty mapping — it has already logged its own warning — and the
    keyword leg answers alone.

    Args:
        cache: This tree's document records, as the caller already holds them.
        store: The engine the caller resolved, or ``None`` when it resolved none.
        resolved: The caller's resolved collection param, whose model and provider
            the query is embedded through, or ``None``.
        query: What to look for, in natural language.
        top_k: How many hits to answer with, clamped to at least one.
        scope: The resolved workspace path every query is scoped to.
        path_prefix: Restrict the search to paths starting with this. Already
            checked for metacharacters by the caller — see
            :data:`~akgentic.tool.vector_store.protocol.PATH_PREFIX_WILDCARDS`.
        alpha: Weight of the vector leg. ``None`` takes the fusion module's own
            default, which is the value the Weaviate client sends.
        score_threshold: Minimum **raw** cosine score, applied before fusion.

    Returns:
        A :class:`RagSearchResult` whose ``hits`` are best first, or one with no
        hits and :data:`_NO_HITS` as its ``note`` when nothing matched.
    """
    from akgentic.tool.vector_store.hybrid import DEFAULT_ALPHA, fuse  # noqa: PLC0415

    budget = max(top_k, 1)
    hits = _vector_hits(store, resolved, query, budget, scope, path_prefix, score_threshold)
    matches = _keyword_leg(cache, query, path_prefix)
    fused = fuse(
        list(matches),
        {ref_id: hit.score for ref_id, hit in hits.items()},
        alpha=DEFAULT_ALPHA if alpha is None else alpha,
    )
    found: list[RagSearchHit] = []
    for ref_id, _fused_score in sorted(fused.items(), key=lambda item: item[1], reverse=True):
        built = _hit(cache, hits.get(ref_id), matches.get(ref_id))
        if built is not None:
            found.append(built)
        if len(found) >= budget:
            break
    return RagSearchResult(hits=found) if found else RagSearchResult(note=_NO_HITS)


def _vector_hits(
    store: VectorStoreService | None,
    resolved: VectorStoreParam | None,
    query: str,
    top_k: int,
    scope: str,
    path_prefix: str,
    score_threshold: float,
) -> dict[str, SearchHit]:
    """Embed *query* and search the collection **within this workspace only**.

    **This is the whole external half of a search, and it runs on the calling
    agent's thread.** Both round trips are here — the embed, and the ``search``
    that on a cluster backend is a second HTTP call — because moving only the
    embed would leave a mailbox turn holding one of them, and a guard written
    against the embedder alone would go vacuous the moment it did.

    The ``scope`` predicate is mandatory on every workspace query (ADR-045 §5,
    §7) and both predicates go to the backend, so the ``top_k`` budget is never
    spent on another scope's objects. The call over-fetches by ``OVERFETCH``
    because fusion reorders and the caller drops what it cannot resolve.

    **A collection that was never created is the one failure with no gate ahead
    of it.** The actor used to search only once ``create_collection`` had
    succeeded; no card-side signal reproduces that, so a search against a
    collection that was never created reaches the backend and raises. The raise
    lands in the ``except`` below, yields an empty mapping and one warning — the
    existing *every failure degrades* contract rather than a new hole. It is also
    the one state that **still spends**: one embed and one failing ``search`` per
    query, as it did before the move. What changed is what becomes of the result —
    the keyword leg answers over the records on disk, where the actor's gate used
    to discard both round trips and answer its sentence instead.

    **The caller's availability gate ahead of this function saves no round
    trip**, and the docstring this one replaces claimed that it did. That gate is
    ``store is not None and resolved is not None`` — precisely the condition the
    first branch below already short-circuits on, for free. What the hoist saves
    is the keyword leg's directory scan and parse on a degraded tree, which is
    worth having and is not an embed.

    **It never raises**, including out of building the embedder, which imports an
    optional extra.

    Args:
        store: The engine the caller resolved, or ``None`` when it resolved none.
        resolved: The caller's resolved collection param, whose model and provider
            the query is embedded through, or ``None``.
        query: What to look for, in natural language.
        top_k: The result budget, already clamped by the caller.
        scope: The resolved workspace path every query is scoped to.
        path_prefix: The caller's prefix, already checked for metacharacters.
        score_threshold: Minimum **raw** cosine score, applied before fusion.

    Returns:
        ``{ref_id: hit}`` for the hits at or above *score_threshold*, or an empty
        mapping on any failure.
    """
    if store is None or resolved is None:
        logger.warning("Workspace %s: no vector store — searching on the keyword leg alone", scope)
        return {}
    try:
        from akgentic.tool.vector_store.embedding_actor import (  # noqa: PLC0415 — optional extra
            build_embedding_service,
        )
        from akgentic.tool.vector_store.hybrid import OVERFETCH  # noqa: PLC0415 — optional extra

        embedder = build_embedding_service(resolved.embedding_model, resolved.embedding_provider)
        vectors = embedder.embed([query])
        if not vectors:
            logger.warning(
                "Workspace %s: embedding a search query returned nothing — keyword only", scope
            )
            return {}
        result = store.search(
            RAG_COLLECTION,
            vectors[0],
            top_k * OVERFETCH,
            scope=scope,
            path_prefix=path_prefix or None,
        )
    except Exception:
        logger.warning(
            "Workspace %s: the vector leg of a search failed — keyword only",
            scope,
            exc_info=True,
        )
        return {}
    return {hit.ref_id: hit for hit in result.hits if hit.score >= score_threshold}


def _keyword_leg(cache: DocumentCache, query: str, path_prefix: str) -> dict[str, _KeywordMatch]:
    """Return the chunks whose own slice of their document carries a query term.

    Case-insensitive, over the extraction bodies *cache* holds — no file inside
    the tree is read and no chunk text is stored anywhere, because a chunk is a
    pair of offsets into an extraction and never a copy of one.

    **An evicted body contributes nothing and is never sliced** (ADR-045 §3, §4).
    The search degrades toward vector-only for that file and is never wrong; the
    file stays ``EMBEDDED`` and its vector hits still render from the store's own
    copy of the text. This is what makes ``max_documents`` a bound on the
    extraction cache rather than on the searchable corpus — and it bounds no
    actor state at all: it is a field of the
    :class:`~akgentic.tool.workspace.documents.cache.DocumentCache` the card
    builds, applied over the records on disk.

    **A body that is not the one the offsets were cut from is skipped too.** The
    two halves have different lifetimes even inside one record: a file re-read
    after a change holds a new body while its row still describes the old chunk
    boundaries, and slicing one with the other yields text that belongs to
    neither. The offsets of such a row are provenance, exactly as an evicted
    file's are.

    **An extraction is identified by two things, and the second is the
    extractor.** Bumping ``EXTRACTOR_VERSION`` leaves the source bytes untouched,
    so ``indexed_sha`` still matches while every cached body is a miss and is
    re-extracted — and this leg would then slice a *new* extraction with *old*
    offsets. The two conditions are checked as two statements rather than one
    conjunction because they are two different staleness stories and a reader has
    to be able to see which of them fired. A row whose
    ``indexed_extractor_version`` is ``None`` predates the field and matches any
    version; a mismatch costs the file its lexical leg and nothing else — the row
    stays ``EMBEDDED`` and its vector hits still render.

    The keys are ``chunk_id``s — the key space ``fuse`` combines on, and what
    ``SearchHit.ref_id`` carries. It is an **indicator** and not a score: a flat
    substring match is equally good everywhere, which is why ``fuse`` does not
    normalise this leg.

    Args:
        cache: This tree's document records.
        query: What to look for; split on whitespace into terms.
        path_prefix: Restrict to paths starting with this, or ``""`` for all.

    Returns:
        ``{chunk_id: match}`` for every chunk whose own slice carries a term.
    """
    terms = query.lower().split()
    matches: dict[str, _KeywordMatch] = {}
    if not terms:
        return matches
    for entry in cache.entries():
        extract, row = entry.extract, entry.row
        if extract is None or extract.markdown is None:
            continue
        if path_prefix and not entry.path.startswith(path_prefix):
            continue
        if row is None or row.indexed_sha != extract.source_sha:
            continue
        if row.indexed_extractor_version not in (None, extract.extractor_version):
            continue
        body = extract.markdown
        lowered = body.lower()
        for chunk in row.chunks:
            if any(term in lowered[chunk.start : chunk.end] for term in terms):
                matches[chunk.chunk_id] = _KeywordMatch(
                    path=entry.path,
                    chunk=chunk,
                    text=body[chunk.start : chunk.end],
                    chunk_count=row.chunk_count,
                )
    return matches


def _hit(
    cache: DocumentCache, hit: SearchHit | None, match: _KeywordMatch | None
) -> RagSearchHit | None:
    """Build one fused hit — its coordinates, its score, its leg and its text.

    **The text comes from** ``SearchHit.text`` **whenever there is a hit**, never
    from a slice of the cached body: that is what keeps a file whose body was
    evicted searchable and readable. A keyword-only hit has no ``SearchHit``
    behind it, and its text is its own slice — which is present by construction,
    since matching it is what put it here. It is carried as the leg supplied it,
    with no strip: there are no blocks to read cleanly any more.

    **A hit whose row will not resolve is still a hit.** Its coordinates go
    ``None`` and its ``heading_path`` empty, and ``ordinal`` falls back to what
    the vector store itself reported — the chunk text is still the answer, and
    discarding the store's own ordinal would cost the caller the one coordinate
    it has left.

    Args:
        cache: This tree's document records, for the coordinate lookup.
        hit: The vector hit, or ``None`` for a keyword-only match.
        match: The keyword match, or ``None`` for a vector-only hit.

    Returns:
        The hit, or ``None`` when neither leg supplied anything — which the
        caller skips without spending a result slot.
    """
    chunk: RagChunk | None
    chunk_count: int | None
    if match is not None:
        path, chunk, chunk_count = match.path, match.chunk, match.chunk_count
        text = hit.text if hit is not None else match.text
    elif hit is not None:
        path = hit.path or ""
        chunk, chunk_count = _locate_chunk(cache, path, hit.ordinal)
        text = hit.text
    else:
        return None
    if hit is None:
        kind = MatchKind.KEYWORD
    elif match is not None:
        kind = MatchKind.HYBRID
    else:
        kind = MatchKind.SEMANTIC
    return RagSearchHit(
        path=path,
        ordinal=chunk.ordinal if chunk is not None else (hit.ordinal if hit is not None else None),
        chunk_count=chunk_count,
        start_line=chunk.start_line if chunk is not None else None,
        end_line=chunk.end_line if chunk is not None else None,
        heading_path=list(chunk.heading_path) if chunk is not None else [],
        score=hit.score if hit is not None else None,
        match=kind,
        text=text,
    )


def _locate_chunk(
    cache: DocumentCache, path: str, ordinal: int | None
) -> tuple[RagChunk | None, int | None]:
    """Return *path*'s chunk at *ordinal*, and its row's ``chunk_count``, in one read.

    Story 45-6 put ``path`` and ``ordinal`` on ``SearchHit`` precisely so that
    the chunk lookup is O(1) once the row is in hand. A hit whose ``path`` or
    ``ordinal`` is missing, whose ordinal is out of range, or whose stored chunk
    contradicts it, resolves to ``(None, None)`` and is answered with empty
    coordinates rather than being dropped — the chunk text is still the answer.

    **The count comes back with the chunk because the row is the expensive
    part.** ``cache.entry(path)`` is a YAML file opened and parsed
    (``DocumentStore.get_document`` → ``_read``), once per vector hit; fetching
    ``chunk_count`` through a second ``entry()`` call would double a search's
    disk reads for one integer. The keyword leg never comes here at all — it
    holds the row already and carries the count on ``_KeywordMatch``.
    """
    if not path or ordinal is None:
        return None, None
    row = cache.entry(path).row
    if row is None or not 0 <= ordinal < len(row.chunks):
        return None, None
    chunk = row.chunks[ordinal]
    return (chunk, row.chunk_count) if chunk.ordinal == ordinal else (None, None)
