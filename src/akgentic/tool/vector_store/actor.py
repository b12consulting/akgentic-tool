"""VectorStoreActor singleton — centralised vector storage via Pykka proxy.

Exposes the ``VectorStoreService`` protocol methods as actor proxy calls,
routing all operations to ``InMemoryBackend``.  Follows the established
KnowledgeGraphActor / PlanActor singleton pattern with lazy backend
initialisation and catch/log/swallow error handling.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from pydantic import Field

from akgentic.core.agent import Akgent
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.tool.errors import RetriableError
from akgentic.tool.vector_store.protocol import (
    CollectionConfig,
    CollectionStatus,
    SearchResult,
    VectorStoreConfig,
)

if TYPE_CHECKING:
    from akgentic.tool.vector import EmbeddingService, VectorEntry
    from akgentic.tool.vector_store.embedding_actor import (
        EmbeddingError,
        EmbeddingResult,
    )
    from akgentic.tool.vector_store.inmemory import InMemoryBackend

logger = logging.getLogger(__name__)

VS_ACTOR_NAME: str = "#VectorStore"
"""Singleton actor name registered with the orchestrator."""

VS_ACTOR_ROLE: str = "ToolActor"
"""Actor role constant for ToolCard integration."""


# ---------------------------------------------------------------------------
# VectorStoreState
# ---------------------------------------------------------------------------


class VectorStoreState(BaseState):
    """Serialisable state for the vector store actor.

    Holds a snapshot of the ``InMemoryBackend`` state (via ``get_state()`` /
    ``restore_state()``) and per-collection lifecycle statuses.
    """

    backend_state: dict[str, Any] = Field(
        default_factory=dict,
        description="Serialisable snapshot from InMemoryBackend.get_state()",
    )
    collection_statuses: dict[str, CollectionStatus] = Field(
        default_factory=dict,
        description="Per-collection lifecycle status",
    )
    pending_entries: dict[str, list[dict[str, str]]] = Field(
        default_factory=dict,
        description="Raw entry metadata awaiting embedding, keyed by collection",
    )
    indexing_pending: dict[str, int] = Field(
        default_factory=dict,
        description="Count of entries pending embedding per collection",
    )


# ---------------------------------------------------------------------------
# VectorStoreActor
# ---------------------------------------------------------------------------


class VectorStoreActor(Akgent[VectorStoreConfig, VectorStoreState]):
    """Singleton actor exposing ``VectorStoreService`` via Pykka proxy.

    All vector operations are delegated to ``InMemoryBackend`` which is
    created lazily on first use. Mutations synchronise serialisable state
    and notify the orchestrator via ``state.notify_state_change()``.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_start(self) -> None:  # noqa: ANN201
        """Initialise state, attach observer, and prepare lazy runtime slots."""
        self.state = VectorStoreState()
        self.state.observer(self)
        self._backend: InMemoryBackend | None = None
        self._embedding_svc: EmbeddingService | None = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _get_or_create_backend(self) -> InMemoryBackend | None:
        """Return the ``InMemoryBackend``, creating it lazily on first call.

        If ``self.state.backend_state`` contains data the backend is restored
        from the persisted snapshot.  Returns ``None`` when ``[vector_search]``
        dependencies are missing.
        """
        if self._backend is not None:
            return self._backend
        try:
            from akgentic.tool.vector_store.inmemory import InMemoryBackend

            self._backend = InMemoryBackend()
            if self.state.backend_state:
                self._backend.restore_state(self.state.backend_state)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] Failed to initialize InMemoryBackend: %s",
                self.config.name,
                exc,
            )
            return None
        return self._backend

    def _get_or_create_embedding_svc(self) -> EmbeddingService | None:
        """Return the ``EmbeddingService``, creating it lazily on first call.

        Returns ``None`` when creation fails (e.g. missing deps or bad config).
        """
        if self._embedding_svc is not None:
            return self._embedding_svc
        try:
            from akgentic.tool.vector import EmbeddingService

            self._embedding_svc = EmbeddingService(
                model=self.config.embedding_model,
                provider=self.config.embedding_provider,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] Failed to initialize EmbeddingService: %s",
                self.config.name,
                exc,
            )
            return None
        return self._embedding_svc

    # ------------------------------------------------------------------
    # State synchronisation
    # ------------------------------------------------------------------

    def _sync_backend_state(self) -> None:
        """Copy the backend's serialisable snapshot into actor state."""
        if self._backend is not None:
            self.state.backend_state = self._backend.get_state()

    # ------------------------------------------------------------------
    # Proxy methods
    # ------------------------------------------------------------------

    def create_collection(self, name: str, config: CollectionConfig) -> None:
        """Create or reconfigure a named collection.

        Delegates to ``InMemoryBackend.create_collection()``, marks the
        collection as ``READY``, and notifies the orchestrator.

        Args:
            name: Unique collection identifier.
            config: Collection configuration.
        """
        backend = self._get_or_create_backend()
        if backend is None:
            logger.warning(
                "[%s] Backend unavailable, skipping create_collection",
                self.config.name,
            )
            return
        try:
            backend.create_collection(name, config)
            self.state.collection_statuses[name] = CollectionStatus.READY
            self._sync_backend_state()
            self.state.notify_state_change()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] create_collection failed: %s", self.config.name, exc)

    def add(self, collection: str, entries: list[VectorEntry]) -> None:
        """Ingest embedding entries into a collection.

        Entries with pre-populated vectors go through the synchronous path
        directly to the backend.  Entries without vectors are sent to a
        spawned ``EmbeddingActor`` for asynchronous embedding (AC2, AC9).

        If the collection is in ``ERROR`` status, a new ``add()`` resets
        it to ``INDEXING`` (AC8 -- retry after failure).

        Args:
            collection: Target collection name.
            entries: List of ``VectorEntry`` to store.
        """
        backend = self._get_or_create_backend()
        if backend is None:
            logger.warning("[%s] Backend unavailable, skipping add", self.config.name)
            return

        pre_embedded = [e for e in entries if len(e.vector) > 0]
        needs_embedding = [e for e in entries if len(e.vector) == 0]

        if pre_embedded:
            self._add_pre_embedded(collection, pre_embedded)

        if needs_embedding:
            self._add_needs_embedding(collection, needs_embedding)

    def _add_pre_embedded(self, collection: str, entries: list[VectorEntry]) -> None:
        """Add entries with pre-populated vectors directly to backend.

        Args:
            collection: Target collection name.
            entries: Entries with non-empty vector fields.
        """
        backend = self._get_or_create_backend()
        if backend is None:
            return
        try:
            backend.add(collection, entries)
            self._sync_backend_state()
            self.state.notify_state_change()
        except ValueError as exc:
            raise RetriableError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] _add_pre_embedded failed: %s", self.config.name, exc)

    def _add_needs_embedding(self, collection: str, entries: list[VectorEntry]) -> None:
        """Spawn an EmbeddingActor for entries that need embedding.

        Stores raw metadata in pending state, sets collection to INDEXING,
        and fires the request asynchronously (AC2, AC3, AC8).

        Args:
            collection: Target collection name.
            entries: Entries with empty vector fields.
        """
        from akgentic.tool.vector_store.embedding_actor import (
            EmbeddingActor,
            EmbeddingRequest,
        )

        request_id = str(uuid.uuid4())
        raw_entries = [
            {"ref_type": e.ref_type, "ref_id": e.ref_id, "text": e.text} for e in entries
        ]

        # Track pending entries in state
        pending = self.state.pending_entries.get(collection, [])
        pending.extend(raw_entries)
        self.state.pending_entries[collection] = pending

        count = self.state.indexing_pending.get(collection, 0)
        self.state.indexing_pending[collection] = count + len(entries)

        # Transition status (READY/ERROR -> INDEXING)
        self.state.collection_statuses[collection] = CollectionStatus.INDEXING

        # Spawn EmbeddingActor child
        embed_config = BaseConfig(name=f"embed-{collection}-{request_id}")
        embed_addr = self.createActor(EmbeddingActor, config=embed_config)

        request = EmbeddingRequest(
            collection=collection,
            entries=raw_entries,
            request_id=request_id,
            embedding_model=self.config.embedding_model,
            embedding_provider=self.config.embedding_provider,
        )
        embed_proxy = self.proxy_tell(embed_addr, EmbeddingActor)
        embed_proxy.receiveMsg_EmbeddingRequest(request)

        self.state.notify_state_change()

    def remove(self, collection: str, ref_ids: list[str]) -> None:
        """Remove entries from a collection by reference ID.

        Delegates to ``InMemoryBackend.remove()``. ``ValueError`` (non-existent
        collection) is re-raised as ``RetriableError``.

        Args:
            collection: Target collection name.
            ref_ids: List of reference IDs to remove.
        """
        backend = self._get_or_create_backend()
        if backend is None:
            logger.warning("[%s] Backend unavailable, skipping remove", self.config.name)
            return
        try:
            backend.remove(collection, ref_ids)
            self._sync_backend_state()
            self.state.notify_state_change()
        except ValueError as exc:
            raise RetriableError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] remove failed: %s", self.config.name, exc)

    def search(self, collection: str, query_vector: list[float], top_k: int) -> SearchResult:
        """Search a collection by cosine similarity.

        Read-only operation — does not call ``state.notify_state_change()``.
        ``ValueError`` (non-existent collection) is re-raised as ``RetriableError``.

        Args:
            collection: Target collection name.
            query_vector: Query embedding vector.
            top_k: Maximum number of results to return.

        Returns:
            Search results with hits and collection status.
        """
        backend = self._get_or_create_backend()
        if backend is None:
            logger.warning("[%s] Backend unavailable, returning empty search", self.config.name)
            return SearchResult(hits=[], status=CollectionStatus.READY, indexing_pending=0)
        try:
            result: SearchResult = backend.search(collection, query_vector, top_k)
            # Override status/pending from actor state (AC6)
            actor_status = self.state.collection_statuses.get(collection)
            if actor_status is not None:
                result = SearchResult(
                    hits=result.hits,
                    status=actor_status,
                    indexing_pending=self.state.indexing_pending.get(collection, 0),
                )
            return result
        except ValueError as exc:
            raise RetriableError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] search failed: %s", self.config.name, exc)
            return SearchResult(hits=[], status=CollectionStatus.READY, indexing_pending=0)

    # ------------------------------------------------------------------
    # Embedding result/error handlers
    # ------------------------------------------------------------------

    def receiveMsg_EmbeddingResult(self, msg: EmbeddingResult) -> None:  # noqa: N802
        """Handle successful embedding delivery from EmbeddingActor.

        Inserts fully-vectorised entries into the backend and updates
        collection status (AC5).

        Args:
            msg: Result containing entries with populated vectors.
        """
        backend = self._get_or_create_backend()
        if backend is None:
            logger.warning(
                "[%s] Backend unavailable, cannot insert embedding results",
                self.config.name,
            )
            return
        try:
            backend.add(msg.collection, msg.entries)
            self._remove_pending(msg.collection, len(msg.entries))
            self._sync_backend_state()
            self.state.notify_state_change()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] receiveMsg_EmbeddingResult failed: %s",
                self.config.name,
                exc,
            )

    def receiveMsg_EmbeddingError(self, msg: EmbeddingError) -> None:  # noqa: N802
        """Handle embedding failure from EmbeddingActor.

        Sets collection to ERROR and discards pending entries (AC7).

        Args:
            msg: Error details from the failed embedding batch.
        """
        logger.warning(
            "[%s] Embedding failed for collection '%s': %s",
            self.config.name,
            msg.collection,
            msg.error,
        )
        self.state.collection_statuses[msg.collection] = CollectionStatus.ERROR
        self.state.pending_entries.pop(msg.collection, None)
        self.state.indexing_pending[msg.collection] = 0
        self.state.notify_state_change()

    def _remove_pending(self, collection: str, count: int) -> None:
        """Decrement pending count and clean up on completion.

        Args:
            collection: Collection name.
            count: Number of entries that were successfully embedded.
        """
        pending = self.state.indexing_pending.get(collection, 0)
        pending = max(0, pending - count)
        self.state.indexing_pending[collection] = pending

        # Remove corresponding raw entries from pending list
        pending_list = self.state.pending_entries.get(collection, [])
        self.state.pending_entries[collection] = pending_list[count:]

        if pending <= 0:
            self.state.collection_statuses[collection] = CollectionStatus.READY
            self.state.pending_entries.pop(collection, None)
            self.state.indexing_pending.pop(collection, None)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via ``EmbeddingService``.

        Returns an empty list when the embedding service is unavailable or on
        any failure (catch/log/swallow pattern).

        Args:
            texts: List of strings to embed.

        Returns:
            List of float vectors, one per input text. Empty on failure.
        """
        svc = self._get_or_create_embedding_svc()
        if svc is None:
            return []
        try:
            result: list[list[float]] = svc.embed(texts)
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] embed failed: %s", self.config.name, exc)
            return []
