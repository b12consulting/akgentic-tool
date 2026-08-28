"""Shared building blocks for hybrid (keyword + vector) search.

The vector-store backends answer pure similarity queries: ``search`` takes a
vector and nothing else. The lexical half of a hybrid search therefore runs in
the calling actor, over its own authoritative state, and the two halves are
combined here.

The pieces live here because both the knowledge-graph and the planning actor
need them and had drifted apart:

- :func:`semantic_scores` — embed a query, search a collection, and degrade to
  no hits instead of raising when the vector store is absent or failing.
- :func:`fuse` — the single rule that combines keyword and vector hits.
- :func:`hybrid_search` — the whole algorithm: semantic leg, fusion, ranking.

The fusion reproduces Weaviate's ``relativeScoreFusion``, the default behind
``collection.query.hybrid()``, at the client's default ``alpha``. That is a
deliberate choice: it keeps the in-memory ranking close to what the cluster
would return, so moving a collection to Weaviate — or one day pushing the whole
hybrid query down into the backend — does not reorder anybody's results.

Keeping the rule here rather than in the backends is what makes in-memory and
Weaviate agree today; the backends answer pure similarity queries only.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Final, NamedTuple

if TYPE_CHECKING:
    from akgentic.tool.vector_store.actor import VectorStoreActor

logger = logging.getLogger(__name__)

DEFAULT_ALPHA: Final[float] = 0.7
"""Weight of the vector leg in the fused score; the keyword leg gets ``1 - alpha``.

Mirrors the default the Weaviate client sends for ``hybrid(alpha=...)``, so the
ranking produced here stays close to what the cluster would produce natively.
``1.0`` is pure vector search, ``0.0`` pure keyword.
"""

OVERFETCH: Final[int] = 2
"""Multiplier applied to ``top_k`` when querying the vector store.

Fusion reorders candidates and callers drop those they cannot resolve or that
their own filters exclude, so asking the backend for exactly ``top_k`` would
leave the final result short.
"""


def semantic_scores(
    proxy: VectorStoreActor | None,
    collection: str,
    query_text: str,
    top_k: int,
) -> dict[str, float]:
    """Return ``{ref_id: cosine_score}`` for *query_text* against *collection*.

    Embeds the query through *proxy* and searches the collection. Every failure
    mode — no proxy wired, an embedding call that raises or returns nothing, a
    search that raises — yields an empty mapping and a warning rather than an
    exception, so a caller running a hybrid search degrades to keyword-only.

    Args:
        proxy: Vector store actor proxy, or ``None`` when none is wired.
        collection: Collection to search.
        query_text: Natural-language query to embed.
        top_k: Maximum number of hits to request.

    Returns:
        Mapping of reference ID to cosine similarity score. Empty when the
        semantic phase is unavailable.
    """
    if proxy is None:
        logger.warning("No vector store proxy — semantic search skipped")
        return {}
    try:
        vectors = proxy.embed([query_text])
        if not vectors:
            return {}
        result = proxy.search(collection, vectors[0], top_k)
    except Exception:  # noqa: BLE001
        logger.warning("Semantic search failed for collection '%s'", collection, exc_info=True)
        return {}
    return {hit.ref_id: hit.score for hit in result.hits}


def _normalise(scores: Mapping[str, float]) -> dict[str, float]:
    """Min-max scale *scores* onto ``[0, 1]``, as relativeScoreFusion does.

    A list whose scores are all equal — including a single hit — maps entirely
    to ``1.0``; there is no spread to rank on, so nothing is penalised.

    Args:
        scores: Raw scores from one search leg.

    Returns:
        The same keys with scores scaled onto ``[0, 1]``.
    """
    if not scores:
        return {}
    lowest, highest = min(scores.values()), max(scores.values())
    if highest == lowest:
        return dict.fromkeys(scores, 1.0)
    span = highest - lowest
    return {key: (score - lowest) / span for key, score in scores.items()}


def fuse(
    keyword_keys: Iterable[str],
    vector_scores: Mapping[str, float],
    *,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, float]:
    """Combine keyword and vector hits the way Weaviate's relativeScoreFusion does.

    Each leg is min-max normalised, then weighted::

        score = alpha * norm(vector) + (1 - alpha) * keyword

    The keyword leg is an indicator rather than a normalised score: our lexical
    match is a substring test, so every keyword hit is equally good and
    normalising a flat list yields ``1.0`` throughout. A key absent from a leg
    contributes nothing from it, which gives three outcomes — ``alpha * norm``
    for vector only, ``1 - alpha`` for keyword only, and their sum for a hit
    confirmed by both.

    Scores land in ``[0, 1]`` and are comparable only within one query, since
    normalisation is relative to the result set. Threshold on the **raw** score
    before calling this; the caller also owns sorting and any ``top_k`` slice.

    Args:
        keyword_keys: Keys hit by the keyword phase, in any order. Duplicates
            are collapsed, so a key is never counted twice.
        vector_scores: Raw cosine score per key from the vector phase.
        alpha: Weight of the vector leg. See :data:`DEFAULT_ALPHA`.

    Returns:
        Fused score per key, covering the union of both inputs.
    """
    fused = {key: alpha * score for key, score in _normalise(vector_scores).items()}
    for key in dict.fromkeys(keyword_keys):
        fused[key] = fused.get(key, 0.0) + (1.0 - alpha)
    return fused


class HybridResult(NamedTuple):
    """What a hybrid search produced, ranked and traceable.

    ``vector_scores`` carries the **raw** cosine scores the semantic leg
    returned, before normalisation and fusion. Callers that explain a hit to a
    user — "why did this match, and how strongly?" — need that absolute number;
    the fused score is only meaningful relative to the rest of this one result
    set.
    """

    ranked: list[tuple[str, float]]
    """``(key, fused score)`` pairs, best first."""

    vector_scores: dict[str, float]
    """Raw cosine score per key, for keys the semantic leg returned."""


def hybrid_search(
    keyword_keys: Iterable[str],
    proxy: VectorStoreActor | None,
    collection: str,
    query_text: str,
    *,
    top_k: int,
    score_threshold: float = 0.0,
    alpha: float = DEFAULT_ALPHA,
) -> HybridResult:
    """Run the semantic leg, fuse it with *keyword_keys*, and rank the result.

    This is the whole hybrid algorithm. Callers supply the keyword hits — only
    they can search their own state — and materialise the ranked keys back into
    domain objects afterwards.

    No special case is needed when the semantic leg comes back empty: every
    keyword key then ties at ``1 - alpha`` and the sort is stable, so the result
    is the keyword hits in the order they were given.

    Args:
        keyword_keys: Keys hit by the caller's keyword phase, best-first.
        proxy: Vector store actor proxy, or ``None`` for keyword-only.
        collection: Collection to search.
        query_text: Natural-language query.
        top_k: How many hits the caller intends to keep. The semantic leg is
            over-fetched past it, since fusion and the caller's own filtering
            both reorder and drop candidates.
        score_threshold: Minimum **raw** cosine score for a semantic hit.
            Applied before normalisation, so it keeps its absolute meaning.
        alpha: Weight of the vector leg. See :data:`DEFAULT_ALPHA`.

    Returns:
        A :class:`HybridResult`. Its ``ranked`` list is **not** cut to ``top_k``
        — slice after materialising, so that a key which no longer resolves does
        not consume a result slot.
    """
    vector_scores = {
        key: score
        for key, score in semantic_scores(
            proxy, collection, query_text, top_k * OVERFETCH
        ).items()
        if score >= score_threshold
    }
    fused = fuse(keyword_keys, vector_scores, alpha=alpha)
    return HybridResult(
        ranked=sorted(fused.items(), key=lambda item: item[1], reverse=True),
        vector_scores=vector_scores,
    )
