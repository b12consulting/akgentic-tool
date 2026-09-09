"""VectorStoreActor singleton — centralised vector storage via Pykka proxy.

Exposes the ``VectorStoreService`` protocol methods as actor proxy calls, routing
each one to the backend its collection was created with, built lazily on first
use. Follows the established KnowledgeGraphActor / PlanActor singleton pattern.

Reads and cleanups keep the catch/log/degrade convention — a failed ``search``
answers empty, a failed ``remove`` is logged — but :meth:`VectorStoreActor.add`
does **not**: it is the only signal a consumer gets that its batch landed, so a
write that could not land raises (ADR-049 Decision 1).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeGuard, cast

from pydantic import Field

from akgentic.core.agent import Akgent
from akgentic.core.agent_state import BaseState
from akgentic.core.orchestrator import Orchestrator
from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.tool.errors import RetriableError
from akgentic.tool.vector_store.protocol import (
    ActorStateBackend,
    CollectionStatus,
    SearchResult,
    VectorQuery,
    VectorStoreConfig,
    VectorStoreParam,
    VectorStoreService,
    needs_store_actor,
    require_dimension_matches,
)
from akgentic.tool.vector_store.registry import BackendContext, get_backend_spec

if TYPE_CHECKING:
    from akgentic.tool.vector_store.inmemory import InMemoryBackend
    from akgentic.tool.vector_store.vector import VectorEntry
    from akgentic.tool.vector_store.weaviate import WeaviateBackend

logger = logging.getLogger(__name__)

VS_ACTOR_NAME: str = "#VectorStore"
"""Singleton actor name registered with the orchestrator."""

VS_ACTOR_ROLE: str = "ToolActor"
"""Actor role constant for ToolCard integration."""


def _supports_actor_state(backend: object) -> TypeGuard[ActorStateBackend]:
    """Return whether *backend* provides the actor-state persistence contract."""
    return callable(getattr(backend, "get_state", None)) and callable(
        getattr(backend, "restore_state", None)
    )


def ensure_store_actor(param: VectorStoreParam, orchestrator_proxy: Orchestrator) -> None:
    """Create the store actor for *param*, if that backend needs one.

    Called by a consumer card at ``observer()`` time, **before** it creates its
    own consumer actor: that actor looks the store up during its ``on_start``,
    so the store has to exist first. This is where ``VectorStoreTool.observer``'s
    ``getChildrenOrCreate`` call went when the card was deleted — the three
    consumers each call it now, so a store's existence is decided by the
    consumer that needs it rather than by a singleton card someone had to
    remember to add.

    **A cluster backend returns without creating anything.** There is nothing
    for an actor to hold: the data is on the cluster and the consumer talks to
    the shared client directly (:func:`~akgentic.tool.vector_store.protocol.needs_store_actor`).

    **Idempotent, but not by anything here.** ``getChildrenOrCreate`` is
    idempotent per ADR-025, so three cards in one team each calling this resolve
    to the same actor and this helper keeps no bookkeeping of its own.

    The config it builds sets **neither connection field**. The deleted card was
    their only writer, and the actor this helper creates is the in-memory one by
    construction — it needs no URL and no key. A caller who reaches the actor
    with a cluster collection by hand still resolves, because
    ``_make_weaviate_backend`` falls back to the environment.

    Args:
        param: The consumer's vector store configuration.
        orchestrator_proxy: An ask proxy over the team's orchestrator.

    Raises:
        ValueError: When ``param.backend`` names no registered backend, out of
            :func:`~akgentic.tool.vector_store.protocol.needs_store_actor`.
    """
    if not needs_store_actor(param):
        return
    orchestrator_proxy.getChildrenOrCreate(
        VectorStoreActor,
        config=VectorStoreConfig(name=VS_ACTOR_NAME, role=VS_ACTOR_ROLE),
    )


# ---------------------------------------------------------------------------
# PendingRequest
# ---------------------------------------------------------------------------


class PendingRequest(SerializableBaseModel):
    """Tombstone. Nothing writes this model and nothing in ``src/`` reads it.

    It survives for exactly one reason: a checkpoint taken while a batch was in
    flight — before the embedding pipeline moved to the consumers — holds
    ``{"__model__": "akgentic.tool.vector_store.actor.PendingRequest", ...}`` under
    a ``pending_requests`` key that :class:`VectorStoreState` no longer declares.
    The ``SerializableBaseModel`` before-validator runs ``deserialize_object`` over
    every top-level value **before** Pydantic sees the keys, and that resolves every
    nested ``__model__`` tag with ``import_module`` + ``getattr``. With this class
    gone the lookup raises ``AttributeError`` and the whole state fails to load;
    with it here the record is rebuilt, then dropped with its unknown key.

    Its fields are frozen as the last version that was ever written. Removing the
    class is a checkpoint-format break: the day it goes, the spec that loads a
    tagged snapshot goes with it and ADR-049 records the break, exactly as the
    *Migration* section does for ``VectorStoreConfig``.
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
        description="Legacy serialisable snapshot from InMemoryBackend.get_state()",
    )
    backend_states: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Serialisable snapshots for registered actor-state-backed backends",
    )
    collection_statuses: dict[str, CollectionStatus] = Field(
        default_factory=dict,
        description="Per-collection lifecycle status, set to READY when a collection is created",
    )
    collection_configs: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Serialised VectorStoreParam per collection (for backend lookups)",
    )


# ---------------------------------------------------------------------------
# VectorStoreActor
# ---------------------------------------------------------------------------


class VectorStoreActor(Akgent[VectorStoreConfig, VectorStoreState]):
    """Singleton actor exposing ``VectorStoreService`` via Pykka proxy.

    **A storage engine and nothing else** (ADR-049 Decision 1). It creates
    collections, writes vectors it is handed, removes them and searches them. It
    does not embed: the code that owns a ``VectorStoreParam`` spawns its own
    ``EmbeddingWorker`` and hands this actor finished vectors, so a consumer's
    ``embedding_model`` means what it says and there is no second embedder to
    disagree with it.

    Operations are delegated to the backend the collection was created with, built
    lazily on first use. Mutations synchronise serialisable state and notify the
    orchestrator via ``state.notify_state_change()``.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_start(self) -> None:  # noqa: ANN201
        """Initialise state, attach observer, and prepare the lazy backend slots.

        There is no embedding service and no per-request bookkeeping here: a write
        arrives already embedded, is written on this turn, and is either done or
        raised. Nothing is left open for a later message to settle, which is what
        removed the requester and entry maps this method used to carry.
        """
        self.state = VectorStoreState()
        self.state.observer(self)
        self._backend: InMemoryBackend | None = None
        self._weaviate_backend: WeaviateBackend | None = None
        self._backends: dict[str, VectorStoreService] = {}

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

    def _get_or_create_weaviate_backend(self) -> WeaviateBackend | None:
        """Return the ``WeaviateBackend``, creating it lazily on first call.

        **A backstop, not a wired path.** After the store became a backend, no
        consumer routes a cluster collection through this actor: a card whose
        param names Weaviate creates no store actor at all and its consumer
        talks to the backend directly. What remains is the caller who reaches
        this actor with a cluster collection by hand, and for them this accessor
        is what makes that a misconfiguration rather than a crash.

        Obtains the process's shared client for ``self.config.weaviate_url`` and
        ``self.config.weaviate_api_key`` through the same cache the registered
        factory uses — one client per cluster per process, never one per actor —
        and wraps it in a backend scoped to this team. Returns ``None`` when
        ``weaviate-client`` is missing or the first connect to the cluster fails.

        Neither connection field has a writer in ``src/`` any more, so a config
        that carries one was built by hand; the factory's own environment
        fallback covers the rest.

        The owning team's id is taken from ``self.team_id`` — propagated by the
        actor system, never configured — and stamped onto every object the
        backend writes, so a deleted team's vectors stay findable.

        This actor has no ``on_stop`` on purpose: the client is shared across
        every team in the process, and one team stopping must not disconnect the
        others. ``close_all()`` at process exit is the only closer.
        """
        if self._weaviate_backend is not None:
            return self._weaviate_backend
        try:
            from akgentic.tool.vector_store.weaviate import WeaviateBackend, _weaviate_client

            url = self.config.weaviate_url
            if not url:
                logger.warning(
                    "[%s] weaviate_url not configured, cannot create WeaviateBackend",
                    self.config.name,
                )
                return None
            self._weaviate_backend = WeaviateBackend(
                client=_weaviate_client(url, self.config.weaviate_api_key),
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

    def _get_backend(
        self, name: str
    ) -> InMemoryBackend | WeaviateBackend | VectorStoreService | None:
        """Return the backend registered under *name*, building it lazily.

        Every backend resolves through its current :class:`BackendSpec`. Built-in
        specs retain their pre-registry private accessors through a compatibility
        hook; deliberately replacing either registration routes through the
        replacement factory like any other third-party backend.

        Args:
            name: Backend identifier (a ``VectorStoreParam.backend`` value).

        Returns:
            The backend instance, or ``None`` when it cannot be built (missing
            dependency, misconfiguration, unknown name) — logged and swallowed so
            a routed operation degrades rather than raising.
        """
        existing = self._backends.get(name)
        if existing is not None:
            return existing
        try:
            spec = get_backend_spec(name)
            if spec.legacy_actor_accessor:
                accessor = cast(
                    Callable[[], VectorStoreService | None],
                    getattr(self, spec.legacy_actor_accessor),
                )
                return accessor()
            backend = spec.factory(
                BackendContext(config=self.config, team_id=str(self.team_id))
            )
            if spec.persists_in_actor_state:
                self._restore_backend_state(name, backend)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] Failed to initialize '%s' backend: %s",
                self.config.name,
                name,
                exc,
            )
            return None
        self._backends[name] = backend
        return backend

    def _persists_in_actor_state(self, name: str) -> bool:
        """Return whether backend *name* stores its data inside the actor state.

        Drives the state-sync decision that used to be a ``!= "weaviate"`` name
        check. Reads the backend's registered capability; an unregistered name
        falls back to treating only ``inmemory`` as actor-state-backed.

        Args:
            name: Backend identifier.

        Returns:
            ``True`` for actor-state-backed backends (the in-memory index).
        """
        try:
            return get_backend_spec(name).persists_in_actor_state
        except ValueError:
            return name == "inmemory"

    def _get_backend_for_collection(
        self, collection: str,
    ) -> InMemoryBackend | WeaviateBackend | VectorStoreService | None:
        """Return the correct backend for the given collection.

        Checks ``self.state.collection_configs`` for the collection's backend
        name and resolves it through :meth:`_get_backend`.

        Args:
            collection: Collection name to look up.

        Returns:
            The appropriate backend, or ``None`` if unavailable.
        """
        cfg_data = self.state.collection_configs.get(collection, {})
        backend_type = cfg_data.get("backend", "inmemory")
        return self._get_backend(backend_type)

    # ------------------------------------------------------------------
    # State synchronisation
    # ------------------------------------------------------------------

    def _restore_backend_state(self, name: str, backend: VectorStoreService) -> None:
        """Restore a factory-built actor-state-backed backend's saved snapshot.

        Reads from the per-backend ``backend_states`` map. The built-in in-memory
        backend restores from the legacy ``backend_state`` slot inside its own
        accessor (:meth:`_get_or_create_backend`) and never reaches here.
        """
        if not _supports_actor_state(backend):
            msg = (
                f"Backend '{name}' declares persists_in_actor_state=True but does not "
                "implement get_state() and restore_state()."
            )
            raise TypeError(msg)
        if name in self.state.backend_states:
            backend.restore_state(self.state.backend_states[name])

    def _sync_backend_state(self, name: str, backend: VectorStoreService) -> None:
        """Copy an actor-state-backed backend's serialisable snapshot into state.

        The built-in in-memory backend (``self._backend``) writes the legacy
        ``backend_state`` slot for compatibility; every other actor-state-backed
        backend — including a registered replacement for ``inmemory`` — writes the
        per-backend ``backend_states`` map.
        """
        if not _supports_actor_state(backend):
            msg = (
                f"Backend '{name}' declares persists_in_actor_state=True but does not "
                "implement get_state() and restore_state()."
            )
            raise TypeError(msg)
        snapshot = backend.get_state()
        if name == "inmemory" and backend is self._backend:
            self.state.backend_state = snapshot
        else:
            self.state.backend_states[name] = snapshot

    # ------------------------------------------------------------------
    # Proxy methods
    # ------------------------------------------------------------------

    def create_collection(self, name: str, config: VectorStoreParam) -> None:
        """Create or reconfigure a named collection.

        Refuses a ``config`` whose ``dimension`` contradicts its ``embedding_model``
        **before** the backend is touched and outside the error handling below, so
        the ``ValueError`` reaches the caller instead of degrading into a WARNING
        and a collection that was never created. Records the whole param, never an
        enumeration of its fields.

        The param's embedding fields are recorded and not compared against
        anything: this actor embeds nothing, so there is no second model for them
        to disagree with.

        Routes to the appropriate backend based on ``config.backend``:
        - ``"inmemory"``: delegates to ``InMemoryBackend``
        - ``"weaviate"``: delegates to ``WeaviateBackend``

        **A backend that could not be built raises, rather than being logged
        and skipped.** Skipping it was the *cause* whose consequence ``add``
        already refuses to hide: the caller's ``create_collection`` returned
        normally, so a consumer kept its optimistic binding and every later
        write went nowhere until ``add`` raised with no explanation of why. The
        raise happens outside the error handling below, so it reaches the
        caller, where each consumer's existing ``try`` around this call turns it
        into the same one-WARNING degraded mode a cluster consumer enters when
        its factory raises. Both paths now answer an unbuildable backend the
        same way, and neither leaves a binding behind that cannot work.

        Args:
            name: Unique collection identifier.
            config: Vector store configuration for the collection.

        Raises:
            ValueError: When ``config.dimension`` is not the native width of a
                known ``config.embedding_model``.
            RetriableError: When no backend could be built for
                ``config.backend`` — a missing dependency, a misconfiguration,
                an unregistered name. Retriable for the same reason it is on
                ``add``: the fault is usually the environment, not the call.
        """
        require_dimension_matches(config, f"{self.config.name} collection '{name}'")
        backend = self._get_backend(config.backend)
        if backend is None:
            msg = (
                f"[{self.config.name}] no '{config.backend}' backend could be built; "
                f"collection '{name}' was not created."
            )
            logger.warning(msg)
            raise RetriableError(msg)
        try:
            backend.create_collection(name, config)

            record = config.model_dump()
            # The serializer's class tag would re-hydrate this record into a model on
            # the state's own round trip, and the field is a plain dict per collection.
            record.pop("__model__", None)
            self.state.collection_configs[name] = record
            self.state.collection_statuses[name] = CollectionStatus.READY
            if self._persists_in_actor_state(config.backend):
                self._sync_backend_state(config.backend, backend)
            self.state.notify_state_change()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] create_collection failed: %s", self.config.name, exc)

    def add(self, collection: str, entries: list[VectorEntry]) -> None:
        """Write pre-embedded entries into a collection.

        **A plain write, and a plain write does not hide its failure.** Every
        entry arrives with its vector already produced by the consumer's own
        ``EmbeddingWorker``, so there is nothing here to defer and nothing to
        settle later. That makes the outcome of this call the only signal the
        consumer will ever get about whether its batch landed, and an engine that
        swallows a write fault and returns as if it had succeeded is the one thing
        a storage engine may not do: the workspace indexer would mark a file
        ``EMBEDDED`` with no vectors behind it.

        Args:
            collection: Target collection name.
            entries: Entries to store, each carrying a non-empty ``vector``.

        Raises:
            ValueError: When any entry carries an empty vector. Not a
                ``RetriableError``: a retry cannot produce a vector the caller
                never supplied, so this is a programming error at the call site.
            RetriableError: When the write cannot land — a backend that could not
                be built, a missing collection, a dead cluster, a full disk. An
                unavailable backend raises here rather than degrading, unlike
                ``remove`` and ``search`` above it: a read that answers empty
                costs a miss, but a write that answers nothing while storing
                nothing is the swallow this method exists not to do.
        """
        empty = [entry.ref_id for entry in entries if len(entry.vector) == 0]
        if empty:
            msg = (
                f"[{self.config.name}] collection '{collection}': "
                f"ref_id(s) {empty} carry no vector. Entries must be embedded by "
                "their own consumer before they are added."
            )
            raise ValueError(msg)

        backend = self._get_backend_for_collection(collection)
        if backend is None:
            msg = (
                f"[{self.config.name}] no backend could be built for collection "
                f"'{collection}'; {len(entries)} entries were not written."
            )
            logger.warning(msg)
            raise RetriableError(msg)
        try:
            backend.add(collection, entries)
            backend_name = self.state.collection_configs.get(collection, {}).get(
                "backend", "inmemory"
            )
            if self._persists_in_actor_state(backend_name):
                self._sync_backend_state(backend_name, backend)
            self.state.notify_state_change()
        except ValueError as exc:
            raise RetriableError(str(exc)) from exc
        except Exception as exc:
            logger.warning("[%s] add failed: %s", self.config.name, exc)
            raise RetriableError(str(exc)) from exc

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
            backend_name = self.state.collection_configs.get(collection, {}).get(
                "backend", "inmemory"
            )
            if self._persists_in_actor_state(backend_name):
                self._sync_backend_state(backend_name, backend)
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
        query: VectorQuery | None = None,
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
            query: Optional per-call refinement (filters, score threshold,
                backend-native params). ``None`` preserves the historical call.

        Returns:
            Search results with hits and collection status.
        """
        backend = self._get_backend_for_collection(collection)
        if backend is None:
            logger.warning("[%s] Backend unavailable, returning empty search", self.config.name)
            return SearchResult(hits=[], status=CollectionStatus.READY, indexing_pending=0)
        try:
            if query is None:
                result: SearchResult = backend.search(
                    collection,
                    query_vector,
                    top_k,
                    scope=scope,
                    path_prefix=path_prefix,
                )
            else:
                result = backend.search(
                    collection,
                    query_vector,
                    top_k,
                    scope=scope,
                    path_prefix=path_prefix,
                    query=query,
                )
            # Returned exactly as the backend built it. There is no longer anything
            # this actor knows about a collection's progress that the backend does
            # not: a write either landed on this turn or raised.
            return result
        except ValueError as exc:
            raise RetriableError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] search failed: %s", self.config.name, exc)
            return SearchResult(hits=[], status=CollectionStatus.READY, indexing_pending=0)
