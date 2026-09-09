"""In-memory vector store backend with per-collection VectorIndex management.

Implements the ``VectorStoreService`` protocol using numpy-backed ``VectorIndex``
instances. Collections live in the actor's serialisable state, reached through
``get_state()`` / ``restore_state()``.
"""

from __future__ import annotations

import logging
from typing import Any

from akgentic.tool.vector_store.protocol import (
    CollectionStatus,
    SearchHit,
    SearchResult,
    VectorQuery,
    VectorStoreParam,
    check_path_prefix,
    check_shared_scope,
)
from akgentic.tool.vector_store.registry import BackendContext, BackendSpec, register_backend
from akgentic.tool.vector_store.vector import (
    VectorEntry,
    VectorIndex,
    _check_vector_search_dependencies,
)

logger = logging.getLogger(__name__)


def _entry_matches(
    entry: VectorEntry, scope: str | None, path_prefix: str | None
) -> bool:
    """Return whether *entry* satisfies both optional predicates.

    A predicate left at ``None`` matches everything. An entry that carries no
    ``path`` never matches a non-empty ``path_prefix``.

    Args:
        entry: The stored entry to test.
        scope: Required ``scope`` value, or ``None`` to ignore scope.
        path_prefix: Required ``path`` prefix, or ``None`` to ignore path.

    Returns:
        ``True`` when the entry satisfies every predicate given.
    """
    if scope is not None and entry.scope != scope:
        return False
    if path_prefix is not None and not (entry.path or "").startswith(path_prefix):
        return False
    return True


class InMemoryBackend:
    """In-memory vector store managing one ``VectorIndex`` per collection.

    This is a plain Python class (not a Pydantic model) because it holds
    non-serialisable runtime state (numpy arrays inside ``VectorIndex``).
    It satisfies the ``VectorStoreService`` protocol structurally.

    Args:
        None. Instantiation validates that ``[vector_search]`` extras are
        installed.
    """

    def __init__(self) -> None:
        _check_vector_search_dependencies()
        self._collections: dict[str, VectorIndex] = {}
        self._configs: dict[str, VectorStoreParam] = {}

    # ------------------------------------------------------------------
    # VectorStoreService protocol methods
    # ------------------------------------------------------------------

    def create_collection(self, name: str, config: VectorStoreParam) -> None:
        """Create a named collection. No-op if the collection already exists.

        Args:
            name: Unique collection identifier.
            config: Collection configuration.
        """
        if name in self._collections:
            return
        self._collections[name] = VectorIndex()
        self._configs[name] = config

    def add(self, collection: str, entries: list[VectorEntry]) -> None:
        """Ingest embedding entries into a collection.

        Args:
            collection: Target collection name.
            entries: List of vector entries to store.

        Raises:
            ValueError: If the collection does not exist.
        """
        index = self._get_index(collection)
        for entry in entries:
            index.add(entry)

    def remove(
        self,
        collection: str,
        ref_ids: list[str],
        scope: str | None = None,
        path_prefix: str | None = None,
    ) -> None:
        """Remove entries from a collection by reference ID.

        An entry is removed only when its ``ref_id`` is listed **and** it satisfies
        every predicate given, so ``remove(scope=...)`` is a scalpel: another scope's
        entries survive even when a ref-id collides across scopes. The predicate goes
        to ``VectorIndex.remove`` rather than being resolved to a set of ids here —
        an id-precise removal cannot tell two colliding entries apart, which is the
        one case the scalpel exists for.

        Args:
            collection: Target collection name.
            ref_ids: List of reference IDs to remove.
            scope: Restrict removal to entries carrying this ``scope``.
            path_prefix: Restrict removal to entries whose ``path`` starts with this.

        Raises:
            ValueError: If the collection does not exist, if ``path_prefix``
                contains ``*`` or ``?``, or if the collection is shared across
                teams and no ``scope`` was given.
        """
        check_path_prefix(path_prefix)
        check_shared_scope(collection, scope)
        index = self._get_index(collection)
        if scope is None and path_prefix is None:
            index.remove(set(ref_ids))
            return
        index.remove(
            set(ref_ids),
            matches=lambda entry: _entry_matches(entry, scope, path_prefix),
        )

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        scope: str | None = None,
        path_prefix: str | None = None,
        query: VectorQuery | None = None,
    ) -> SearchResult:
        """Search a collection by cosine similarity.

        The ``scope`` / ``path_prefix`` predicates are applied to the **scored**
        candidates before ``top_k`` is taken, so a scoped search returns a full
        ``top_k`` of its own entries rather than filtering an already-cut set down
        to a handful. Scoring every entry costs nothing extra: ``search_cosine``
        already sorts the whole index and slices.

        A ``query`` refines the same pass: its ``filters`` and ``score_threshold``
        are applied client-side alongside the predicates before the result is cut
        back to ``top_k``. ``query.params`` has no in-memory meaning and is ignored.

        Args:
            collection: Target collection name.
            query_vector: Query embedding vector.
            top_k: Maximum number of results to return.
            scope: Restrict the search to entries carrying this ``scope``.
            path_prefix: Restrict the search to entries whose ``path`` starts with this.
            query: Optional filters / score threshold.

        Returns:
            Search results with hits ranked by cosine similarity and collection
            status ``READY``.

        Raises:
            ValueError: If the collection does not exist, if ``path_prefix``
                contains ``*`` or ``?``, or if the collection is shared across
                teams and no ``scope`` was given. This backend has no team
                predicate to lose, but one team may hold two ``WorkspaceTool``
                cards on two trees, so an unscoped query crosses a boundary here
                too.
        """
        check_path_prefix(path_prefix)
        check_shared_scope(collection, scope)
        index = self._get_index(collection)
        scoped = scope is not None or path_prefix is not None
        refined = query is not None and (bool(query.filters) or query.score_threshold is not None)
        if not scoped and not refined:
            results = index.search_cosine(query_vector, top_k)
            hits = self._map_search_hits(index, results)
        else:
            hits = self._filtered_search(
                index, query_vector, top_k, query, scope=scope, path_prefix=path_prefix
            )
        return SearchResult(hits=hits, status=CollectionStatus.READY)

    # ------------------------------------------------------------------
    # actor_state snapshot
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Return a serialisable snapshot of all collections.

        The returned dict is suitable for inclusion in a Pydantic ``BaseState``
        model (Story 10.3). Each collection is stored as its config plus a list
        of ``VectorEntry`` dicts.

        Returns:
            Nested dict keyed by collection name.
        """
        return {
            "collections": {
                name: {
                    "config": self._configs[name].model_dump(),
                    "entries": [e.model_dump() for e in index._entries],
                }
                for name, index in self._collections.items()
            }
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Rebuild all collections from a previously-saved state snapshot.

        Args:
            state: Dict produced by ``get_state()``.
        """
        self._collections.clear()
        self._configs.clear()
        collections = state.get("collections", {})
        for name, col_data in collections.items():
            config = VectorStoreParam.model_validate(col_data["config"])
            self._configs[name] = config
            index = VectorIndex()
            for entry_data in col_data["entries"]:
                index.add(VectorEntry.model_validate(entry_data))
            self._collections[name] = index

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_index(self, collection: str) -> VectorIndex:
        """Return the ``VectorIndex`` for *collection* or raise.

        Args:
            collection: Collection name to look up.

        Returns:
            The ``VectorIndex`` instance.

        Raises:
            ValueError: If the collection does not exist.
        """
        try:
            return self._collections[collection]
        except KeyError:
            msg = f"Collection '{collection}' does not exist"
            raise ValueError(msg) from None

    def _filtered_search(
        self,
        index: VectorIndex,
        query_vector: list[float],
        top_k: int,
        query: VectorQuery | None = None,
        *,
        scope: str | None = None,
        path_prefix: str | None = None,
    ) -> list[SearchHit]:
        """Apply predicates / ``filters`` / ``score_threshold``, then cut to ``top_k``.

        Over-fetches the whole index (cheap at in-memory scale) so that
        constraints never starve the result the way filtering a pre-cut top-k
        would. The ``scope`` / ``path_prefix`` predicates and the ``query``
        refinements are ANDed together.

        Args:
            index: The VectorIndex to search.
            query_vector: Query embedding vector.
            top_k: Maximum number of results to return after filtering.
            query: Optional refinement carrying filters and/or a score threshold.
            scope: Restrict to entries carrying this ``scope``.
            path_prefix: Restrict to entries whose ``path`` starts with this.

        Returns:
            Up to ``top_k`` matching ``SearchHit`` models, best score first.
        """
        candidates = index.search_cosine(query_vector, len(index) or top_k)
        entries_by_id: dict[str, VectorEntry] = {e.ref_id: e for e in index._entries}
        threshold = query.score_threshold if query is not None else None
        filters = query.filters if query is not None else None
        kept: list[tuple[str, float]] = []
        for ref_id, score in candidates:
            if threshold is not None and score < threshold:
                continue
            entry = entries_by_id.get(ref_id)
            if entry is None:
                continue
            if not _entry_matches(entry, scope, path_prefix):
                continue
            if filters and not self._matches(entry, filters):
                continue
            kept.append((ref_id, score))
            if len(kept) >= top_k:
                break
        return self._map_search_hits(index, kept)

    @staticmethod
    def _matches(entry: VectorEntry, filters: dict[str, Any]) -> bool:
        """Return whether *entry* satisfies every exact-match *filter*.

        A filter value may be a scalar (equality) or a list/tuple/set
        (match-any). Keys that name no field on ``VectorEntry`` never match.

        Args:
            entry: The candidate entry.
            filters: Field-name to expected-value(s) mapping.

        Returns:
            ``True`` when every filter is satisfied.
        """
        for key, expected in filters.items():
            actual = getattr(entry, key, None)
            if isinstance(expected, (list, tuple, set)):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    @staticmethod
    def _map_search_hits(
        index: VectorIndex, results: list[tuple[str, float]]
    ) -> list[SearchHit]:
        """Convert raw ``(ref_id, score)`` tuples to ``SearchHit`` models.

        Builds a lookup from ``VectorIndex._entries`` for O(1) metadata
        resolution.

        Args:
            index: The VectorIndex that produced the results.
            results: Raw search output from ``search_cosine``.

        Returns:
            List of ``SearchHit`` models with full metadata.
        """
        entries_by_id: dict[str, VectorEntry] = {e.ref_id: e for e in index._entries}
        hits: list[SearchHit] = []
        for ref_id, score in results:
            entry = entries_by_id.get(ref_id)
            if entry is None:
                logger.warning("Search returned ref_id '%s' not found in entries", ref_id)
                continue
            hits.append(
                SearchHit(
                    ref_type=entry.ref_type,
                    ref_id=entry.ref_id,
                    text=entry.text,
                    score=score,
                    scope=entry.scope,
                    path=entry.path,
                    ordinal=entry.ordinal,
                )
            )
        return hits


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _make_inmemory_backend(context: BackendContext) -> InMemoryBackend:
    """Build an :class:`InMemoryBackend`. The context carries nothing it needs."""
    return InMemoryBackend()


register_backend(
    BackendSpec(
        name="inmemory",
        factory=_make_inmemory_backend,
        persists_in_actor_state=True,
        selectable_as_default=False,
        legacy_actor_accessor="_get_or_create_backend",
    ),
    replace=True,
)
