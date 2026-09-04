"""In-memory vector store backend with per-collection VectorIndex management.

Implements the ``VectorStoreService`` protocol using numpy-backed ``VectorIndex``
instances. Collections live in the actor's serialisable state, reached through
``get_state()`` / ``restore_state()``.
"""

from __future__ import annotations

import logging
from typing import Any

from akgentic.tool.vector_store.protocol import (
    CollectionConfig,
    CollectionStatus,
    SearchHit,
    SearchResult,
)
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
        self._configs: dict[str, CollectionConfig] = {}

    # ------------------------------------------------------------------
    # VectorStoreService protocol methods
    # ------------------------------------------------------------------

    def create_collection(self, name: str, config: CollectionConfig) -> None:
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
            ValueError: If the collection does not exist.
        """
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
    ) -> SearchResult:
        """Search a collection by cosine similarity.

        The predicates are applied to the **scored** candidates before ``top_k`` is
        taken, so a scoped search returns a full ``top_k`` of its own entries rather
        than filtering an already-cut set down to a handful. Scoring every entry costs
        nothing extra: ``search_cosine`` already sorts the whole index and slices.

        Args:
            collection: Target collection name.
            query_vector: Query embedding vector.
            top_k: Maximum number of results to return.
            scope: Restrict the search to entries carrying this ``scope``.
            path_prefix: Restrict the search to entries whose ``path`` starts with this.

        Returns:
            Search results with hits ranked by cosine similarity, collection
            status ``READY``, and ``indexing_pending=0``.

        Raises:
            ValueError: If the collection does not exist.
        """
        index = self._get_index(collection)
        if scope is None and path_prefix is None:
            results = index.search_cosine(query_vector, top_k)
        else:
            allowed = {
                e.ref_id for e in index._entries if _entry_matches(e, scope, path_prefix)
            }
            scored = index.search_cosine(query_vector, len(index))
            results = [(rid, s) for rid, s in scored if rid in allowed][:top_k]
        hits = self._map_search_hits(index, results)
        return SearchResult(
            hits=hits,
            status=CollectionStatus.READY,
            indexing_pending=0,
        )

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
            config = CollectionConfig.model_validate(col_data["config"])
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
