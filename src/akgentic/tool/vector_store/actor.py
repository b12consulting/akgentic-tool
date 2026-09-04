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
from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.tool.errors import RetriableError
from akgentic.tool.vector_store.protocol import (
    CollectionConfig,
    CollectionStatus,
    SearchResult,
    VectorStoreConfig,
)

if TYPE_CHECKING:
    from akgentic.core.actor_address import ActorAddress
    from akgentic.tool.vector_store.embedding_actor import (
        EmbeddingError,
        EmbeddingResult,
    )
    from akgentic.tool.vector_store.inmemory import InMemoryBackend
    from akgentic.tool.vector_store.vector import EmbeddingService, VectorEntry
    from akgentic.tool.vector_store.weaviate import WeaviateBackend

logger = logging.getLogger(__name__)

VS_ACTOR_NAME: str = "#VectorStore"
"""Singleton actor name registered with the orchestrator."""

VS_ACTOR_ROLE: str = "ToolActor"
"""Actor role constant for ToolCard integration."""


# ---------------------------------------------------------------------------
# PendingRequest
# ---------------------------------------------------------------------------


class PendingRequest(SerializableBaseModel):
    """One asynchronous embedding request the actor is still waiting on.

    The unit of accounting for the asynchronous path. Everything a settle needs is
    here, so a result can be attributed to the request that issued it rather than to
    whatever happens to sit at the front of a shared per-collection list — which is
    what made two concurrent ``add()`` calls into one collection settle each other's
    entries.

    The requester's ``ActorAddress`` is deliberately **not** a field: a ``BaseState``
    carrying one raises ``AttributeError`` on every ``notify_state_change()``
    (``b12consulting/akgentic-core#131``). It lives in a private attribute on the
    actor instead.
    """

    request_id: str = Field(description="Identifier minted when the request was issued")
    collection: str = Field(description="Collection the request writes into")
    request_ref: str | None = Field(
        default=None,
        description="The caller's own correlation key, echoed back on completion",
    )
    count: int = Field(description="Number of entries this request carries")
    entries: list[dict[str, str]] = Field(
        default_factory=list,
        description="Raw {ref_type, ref_id, text} of THIS request, awaiting embedding",
    )


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
        description="Per-collection lifecycle status, derived from pending_requests",
    )
    pending_requests: dict[str, PendingRequest] = Field(
        default_factory=dict,
        description="Open embedding requests, keyed by request_id",
    )
    indexing_pending: dict[str, int] = Field(
        default_factory=dict,
        description="Entries pending embedding per collection, derived from pending_requests",
    )
    collection_configs: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Serialised CollectionConfig per collection (for backend lookups)",
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
        """Initialise state, attach observer, and prepare lazy runtime slots.

        ``_request_requesters`` maps an open ``request_id`` to the address that asked
        for it. It is a private attribute rather than state for two reasons: an
        ``ActorAddress`` in a ``BaseState`` breaks ``notify_state_change()``, and a
        restored request could not be settled anyway — the ``EmbeddingActor`` children
        that would deliver its result are gone. Losing the map on resume therefore
        loses nothing that was still live.
        """
        self.state = VectorStoreState()
        self.state.observer(self)
        self._backend: InMemoryBackend | None = None
        self._weaviate_backend: WeaviateBackend | None = None
        self._embedding_svc: EmbeddingService | None = None
        self._request_requesters: dict[str, ActorAddress] = {}

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
            from akgentic.tool.vector_store.vector import EmbeddingService

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

    def _get_or_create_weaviate_backend(self) -> WeaviateBackend | None:
        """Return the ``WeaviateBackend``, creating it lazily on first call.

        Uses ``self.config.weaviate_url`` and ``self.config.weaviate_api_key``
        for connection. Returns ``None`` when ``weaviate-client`` is missing
        or connection fails.

        The owning team's id is taken from ``self.team_id`` — propagated by the
        actor system, never configured — and stamped onto every object the
        backend writes, so a deleted team's vectors stay findable.
        """
        if self._weaviate_backend is not None:
            return self._weaviate_backend
        try:
            from akgentic.tool.vector_store.weaviate import WeaviateBackend

            url = self.config.weaviate_url
            if not url:
                logger.warning(
                    "[%s] weaviate_url not configured, cannot create WeaviateBackend",
                    self.config.name,
                )
                return None
            self._weaviate_backend = WeaviateBackend(
                url=url,
                api_key=self.config.weaviate_api_key,
                team_id=str(self.team_id),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] Failed to initialize WeaviateBackend: %s",
                self.config.name,
                exc,
            )
            return None
        return self._weaviate_backend

    def _get_backend_for_collection(
        self, collection: str,
    ) -> InMemoryBackend | WeaviateBackend | None:
        """Return the correct backend for the given collection.

        Checks ``self.state.collection_configs`` to determine whether the
        collection uses the inmemory or weaviate backend.

        Args:
            collection: Collection name to look up.

        Returns:
            The appropriate backend, or ``None`` if unavailable.
        """
        cfg_data = self.state.collection_configs.get(collection, {})
        backend_type = cfg_data.get("backend", "inmemory")
        if backend_type == "weaviate":
            return self._get_or_create_weaviate_backend()
        return self._get_or_create_backend()

    # ------------------------------------------------------------------
    # State synchronisation
    # ------------------------------------------------------------------

    def _is_weaviate(self, collection: str) -> bool:
        """Return whether *collection* is configured to live in Weaviate.

        The single reading of the stored config, so the ``_sync_backend_state()``
        guards on the synchronous and asynchronous ingest paths cannot drift apart:
        the in-memory snapshot is meaningless for a Weaviate-backed collection and is
        taken for neither.

        Args:
            collection: Collection name to look up.

        Returns:
            ``True`` when the collection's configured backend is Weaviate.
        """
        cfg_data = self.state.collection_configs.get(collection, {})
        return bool(cfg_data.get("backend", "inmemory") == "weaviate")

    def _sync_backend_state(self) -> None:
        """Copy the backend's serialisable snapshot into actor state."""
        if self._backend is not None:
            self.state.backend_state = self._backend.get_state()

    # ------------------------------------------------------------------
    # Proxy methods
    # ------------------------------------------------------------------

    def create_collection(self, name: str, config: CollectionConfig) -> None:
        """Create or reconfigure a named collection.

        Routes to the appropriate backend based on ``config.backend``:
        - ``"inmemory"``: delegates to ``InMemoryBackend``
        - ``"weaviate"``: delegates to ``WeaviateBackend``

        Args:
            name: Unique collection identifier.
            config: Collection configuration.
        """
        try:
            if config.backend == "weaviate":
                wb = self._get_or_create_weaviate_backend()
                if wb is None:
                    logger.warning(
                        "[%s] Weaviate backend unavailable, skipping create_collection",
                        self.config.name,
                    )
                    return
                wb.create_collection(name, config)
            else:
                backend = self._get_or_create_backend()
                if backend is None:
                    logger.warning(
                        "[%s] Backend unavailable, skipping create_collection",
                        self.config.name,
                    )
                    return
                backend.create_collection(name, config)

            self.state.collection_configs[name] = {
                "dimension": config.dimension,
                "backend": config.backend,
                "tenant": config.tenant,
            }
            self.state.collection_statuses[name] = CollectionStatus.READY
            if config.backend != "weaviate":
                self._sync_backend_state()
            self.state.notify_state_change()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] create_collection failed: %s", self.config.name, exc)

    def add(
        self,
        collection: str,
        entries: list[VectorEntry],
        requester: ActorAddress | None = None,
        request_ref: str | None = None,
    ) -> None:
        """Ingest embedding entries into a collection.

        Entries with pre-populated vectors go through the synchronous path
        directly to the backend.  Entries without vectors are sent to a
        spawned ``EmbeddingActor`` for asynchronous embedding.

        **Concurrent adds into one collection now settle independently.** Each
        asynchronous call opens its own request, and a result or an error settles
        only that request's entries and count: a failure fails one batch rather than
        blanking the collection, and a result arriving out of order no longer discards
        another request's pending entries. The collection returns to ``READY`` only
        when the last open request against it settles.

        Both new arguments are optional and additive — the three existing call sites
        pass neither and are unaffected.

        Args:
            collection: Target collection name.
            entries: List of ``VectorEntry`` to store.
            requester: Address told ``EmbeddingCompleted`` when the asynchronous
                request settles, either way. ``None`` means no notification.
            request_ref: The caller's own correlation key, returned unchanged on
                ``EmbeddingCompleted``. ``add`` is reached by ``tell`` and returns
                nothing, so this is the only way a caller can recognise its request.
        """
        backend = self._get_backend_for_collection(collection)
        if backend is None:
            logger.warning("[%s] Backend unavailable, skipping add", self.config.name)
            return

        pre_embedded = [e for e in entries if len(e.vector) > 0]
        needs_embedding = [e for e in entries if len(e.vector) == 0]

        if pre_embedded:
            self._add_pre_embedded(collection, pre_embedded)

        if needs_embedding:
            self._add_needs_embedding(collection, needs_embedding, requester, request_ref)

    def _add_pre_embedded(self, collection: str, entries: list[VectorEntry]) -> None:
        """Add entries with pre-populated vectors directly to backend.

        Args:
            collection: Target collection name.
            entries: Entries with non-empty vector fields.
        """
        backend = self._get_backend_for_collection(collection)
        if backend is None:
            return
        try:
            backend.add(collection, entries)
            if not self._is_weaviate(collection):
                self._sync_backend_state()
            self.state.notify_state_change()
        except ValueError as exc:
            raise RetriableError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] _add_pre_embedded failed: %s", self.config.name, exc)

    def _add_needs_embedding(
        self,
        collection: str,
        entries: list[VectorEntry],
        requester: ActorAddress | None = None,
        request_ref: str | None = None,
    ) -> None:
        """Open one embedding request and spawn an EmbeddingActor to fulfil it.

        Records the request under its own ``request_id``, re-derives the collection's
        status and pending count from the open requests, and fires the batch
        asynchronously.

        Args:
            collection: Target collection name.
            entries: Entries with empty vector fields.
            requester: Address to tell when this request settles, or ``None``.
            request_ref: The caller's correlation key, echoed back on completion.
        """
        from akgentic.tool.vector_store.embedding_actor import (
            EmbeddingActor,
            EmbeddingRequest,
        )

        request_id = str(uuid.uuid4())
        raw_entries = [
            {"ref_type": e.ref_type, "ref_id": e.ref_id, "text": e.text} for e in entries
        ]

        self.state.pending_requests[request_id] = PendingRequest(
            request_id=request_id,
            collection=collection,
            request_ref=request_ref,
            count=len(entries),
            entries=raw_entries,
        )
        if requester is not None:
            self._request_requesters[request_id] = requester
        self._refresh_derived(collection)

        # Spawn EmbeddingActor child
        embed_config = BaseConfig(name=f"#embed-{collection}-{request_id}")
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

    def remove(
        self,
        collection: str,
        ref_ids: list[str],
        scope: str | None = None,
        path_prefix: str | None = None,
    ) -> None:
        """Remove entries from a collection by reference ID.

        Delegates to the appropriate backend. ``ValueError`` (non-existent
        collection) is re-raised as ``RetriableError``.

        Args:
            collection: Target collection name.
            ref_ids: List of reference IDs to remove.
            scope: Restrict removal to entries carrying this ``scope``.
            path_prefix: Restrict removal to entries whose ``path`` starts with this.
        """
        backend = self._get_backend_for_collection(collection)
        if backend is None:
            logger.warning("[%s] Backend unavailable, skipping remove", self.config.name)
            return
        try:
            backend.remove(collection, ref_ids, scope=scope, path_prefix=path_prefix)
            if not self._is_weaviate(collection):
                self._sync_backend_state()
            self.state.notify_state_change()
        except ValueError as exc:
            raise RetriableError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] remove failed: %s", self.config.name, exc)

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        scope: str | None = None,
        path_prefix: str | None = None,
    ) -> SearchResult:
        """Search a collection by cosine similarity.

        Read-only operation — does not call ``state.notify_state_change()``.
        ``ValueError`` (non-existent collection) is re-raised as ``RetriableError``.

        Args:
            collection: Target collection name.
            query_vector: Query embedding vector.
            top_k: Maximum number of results to return.
            scope: Restrict the search to entries carrying this ``scope``.
            path_prefix: Restrict the search to entries whose ``path`` starts with this.

        Returns:
            Search results with hits and collection status.
        """
        backend = self._get_backend_for_collection(collection)
        if backend is None:
            logger.warning("[%s] Backend unavailable, returning empty search", self.config.name)
            return SearchResult(hits=[], status=CollectionStatus.READY, indexing_pending=0)
        try:
            result: SearchResult = backend.search(
                collection, query_vector, top_k, scope=scope, path_prefix=path_prefix
            )
            # Status and pending count come from the open-request map, never from the
            # backend: the backend has no idea a batch is still being embedded.
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

        Resolves the backend **for the collection**, not the in-memory one: an
        asynchronously-embedded batch destined for Weaviate belongs in the cluster,
        and writing it into a process-local index instead loses it silently while the
        collection goes on reporting ``READY``.

        Settles only ``msg.request_id``. Any other request open against the same
        collection keeps its own count and entries, and the collection returns to
        ``READY`` only once the last of them settles.

        Args:
            msg: Result containing entries with populated vectors.
        """
        backend = self._get_backend_for_collection(msg.collection)
        if backend is None:
            logger.warning(
                "[%s] Backend unavailable, cannot insert embedding results",
                self.config.name,
            )
            self._settle_request(msg.request_id, "Backend unavailable")
            self.state.notify_state_change()
            return

        error: str | None = None
        try:
            backend.add(msg.collection, msg.entries)
            if not self._is_weaviate(msg.collection):
                self._sync_backend_state()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] receiveMsg_EmbeddingResult failed: %s",
                self.config.name,
                exc,
            )
            error = str(exc)

        self._settle_request(msg.request_id, error)
        self.state.notify_state_change()

    def receiveMsg_EmbeddingError(self, msg: EmbeddingError) -> None:  # noqa: N802
        """Handle embedding failure from EmbeddingActor.

        Fails **one request**, not the collection. The failed request's record is
        dropped and its requester told; every other request open against the same
        collection keeps its entries, its count and the collection's status. A
        collection reaches ``READY`` when its last open request settles, whichever way
        each of them settled.

        Args:
            msg: Error details from the failed embedding batch.
        """
        logger.warning(
            "[%s] Embedding failed for collection '%s': %s",
            self.config.name,
            msg.collection,
            msg.error,
        )
        self._settle_request(msg.request_id, msg.error)
        self.state.notify_state_change()

    def _settle_request(self, request_id: str, error: str | None = None) -> None:
        """Close one open request and tell its requester how it ended.

        A ``request_id`` with no record settles nothing — a duplicate or unknown
        delivery must not touch another request's bookkeeping.

        Args:
            request_id: The request being closed.
            error: ``None`` on success, the failure description otherwise.
        """
        record = self.state.pending_requests.pop(request_id, None)
        requester = self._request_requesters.pop(request_id, None)
        if record is None:
            return
        self._refresh_derived(record.collection)
        if requester is not None:
            self._tell_completed(requester, record, error)

    def _tell_completed(
        self, requester: ActorAddress, record: PendingRequest, error: str | None
    ) -> None:
        """Deliver ``EmbeddingCompleted`` for a settled request.

        Fire-and-forget: a requester that has since stopped must not take the settle
        down with it, so delivery failure is logged and swallowed.

        Args:
            requester: Address that asked for this request.
            record: The request that just settled.
            error: ``None`` on success, the failure description otherwise.
        """
        from akgentic.tool.vector_store.embedding_actor import EmbeddingCompleted

        completed = EmbeddingCompleted(
            request_id=record.request_id,
            request_ref=record.request_ref,
            collection=record.collection,
            count=record.count,
            error=error,
        )
        try:
            self.send(requester, completed)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] Failed to deliver EmbeddingCompleted for '%s': %s",
                self.config.name,
                record.request_id,
                exc,
            )

    def _refresh_derived(self, collection: str) -> None:
        """Recompute *collection*'s status and pending count from its open requests.

        The open-request map is the only authority: a collection is ``INDEXING`` iff at
        least one request is open against it, and ``indexing_pending`` is the sum of
        those requests' counts. Deriving both here is what keeps
        ``SearchResult.indexing_pending`` from ever disagreeing with the map.

        Args:
            collection: Collection whose derived values are stale.
        """
        open_counts = [
            r.count for r in self.state.pending_requests.values() if r.collection == collection
        ]
        if open_counts:
            self.state.indexing_pending[collection] = sum(open_counts)
            self.state.collection_statuses[collection] = CollectionStatus.INDEXING
        else:
            self.state.indexing_pending.pop(collection, None)
            self.state.collection_statuses[collection] = CollectionStatus.READY

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
