"""A search, whole — both legs, the fusion and the render, over the records on disk.

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
offsets and writes neither; the render reads one row. A write on this path — of
any kind, in any function here — is a defect until a decision says otherwise.

**This module holds no gate.** The three degradation sentences belong to the
caller, which is the only layer that knows whether a card is bound at all; see
:meth:`~akgentic.tool.workspace.rag.RagFactories._rag_search_factory`, where the
ordering of them is load-bearing. What is here answers :data:`_NO_HITS` when
retrieval worked and nothing matched, and never raises out of a leg.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

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
"""What a search answers when retrieval works and nothing matched.

Deliberately **not** the unavailable sentence: "nothing matched" and "nothing is
indexed" are different problems with different next steps, and an agent handed
one sentence for both would retry the query when it should have indexed the tree.
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
) -> str:
    """Run both legs over *cache*, fuse them, and render the result.

    The two legs are combined by the one fusion rule the package shares. The
    vector leg is **not** routed through
    :func:`~akgentic.tool.vector_store.hybrid.semantic_scores`, and that is a
    correctness requirement rather than a preference: that helper takes no
    ``scope`` and no ``path_prefix``, and one ``workspace_chunks`` class holds
    every workspace of every team — a search through it would return another
    workspace's chunks. It also reduces its result to ``{ref_id: score}``,
    discarding the ``SearchHit`` that carries the text a hit renders and the
    ``path`` / ``ordinal`` its heading path is looked up by. ``fuse`` and the two
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
        top_k: How many hits to render, clamped to at least one.
        scope: The resolved workspace path every query is scoped to.
        path_prefix: Restrict the search to paths starting with this. Already
            checked for metacharacters by the caller — see
            :data:`~akgentic.tool.vector_store.protocol.PATH_PREFIX_WILDCARDS`.
        alpha: Weight of the vector leg. ``None`` takes the fusion module's own
            default, which is the value the Weaviate client sends.
        score_threshold: Minimum **raw** cosine score, applied before fusion.

    Returns:
        The rendered hits, or :data:`_NO_HITS` when nothing matched.
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
    rendered: list[str] = []
    for ref_id, score in sorted(fused.items(), key=lambda item: item[1], reverse=True):
        line = _render_hit(cache, score, hits.get(ref_id), matches.get(ref_id))
        if line is not None:
            rendered.append(line)
        if len(rendered) >= budget:
            break
    return "\n\n".join(rendered) if rendered else _NO_HITS


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
    of it**, and it costs nothing now. The actor used to search only once
    ``create_collection`` had succeeded; no card-side signal reproduces that, so a
    search against a collection that was never created reaches the backend and
    raises. The raise lands in the ``except`` below, yields an empty mapping and
    one warning — the existing *every failure degrades* contract rather than a new
    hole. What is **not** paid for any more is the degraded tree: the caller's
    availability gate now runs ahead of this function rather than after it, so a
    tree with no store and no parameters spends no embed and no ``search`` at all.

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
        body = extract.markdown
        lowered = body.lower()
        for chunk in row.chunks:
            if any(term in lowered[chunk.start : chunk.end] for term in terms):
                matches[chunk.chunk_id] = _KeywordMatch(
                    path=entry.path, chunk=chunk, text=body[chunk.start : chunk.end]
                )
    return matches


def _render_hit(
    cache: DocumentCache, score: float, hit: SearchHit | None, match: _KeywordMatch | None
) -> str | None:
    """Render one fused hit — path, heading path, score label, and the text.

    **The text comes from** ``SearchHit.text`` **whenever there is a hit**, never
    from a slice of the cached body: that is what keeps a file whose body was
    evicted searchable and renderable. A keyword-only hit has no ``SearchHit``
    behind it, and its text is its own slice — which is present by construction,
    since matching it is what put it here.

    Args:
        cache: This tree's document records, for the heading-path lookup.
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
        chunk = _chunk_at(cache, path, hit.ordinal)
        text = hit.text
    else:
        return None
    heading = " > ".join(chunk.heading_path) if chunk is not None else ""
    location = f"{path} > {heading}" if heading else (path or "(unknown file)")
    return f"{location} ({_score_label(hit, match)})\n{text.strip()}"


def _chunk_at(cache: DocumentCache, path: str, ordinal: int | None) -> RagChunk | None:
    """Return *path*'s chunk at *ordinal* — one dict lookup, and no reverse map.

    Story 45-6 put ``path`` and ``ordinal`` on ``SearchHit`` precisely so that
    this is O(1). A hit whose ``path`` or ``ordinal`` is missing, or whose ordinal
    is out of range, resolves to ``None`` and renders with an empty heading path
    rather than being dropped — the chunk text is still the answer.
    """
    if not path or ordinal is None:
        return None
    row = cache.entry(path).row
    if row is None or not 0 <= ordinal < len(row.chunks):
        return None
    chunk = row.chunks[ordinal]
    return chunk if chunk.ordinal == ordinal else None


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
