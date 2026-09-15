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
from typing import TYPE_CHECKING, Any, TypeGuard

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
    from akgentic.tool.vector_store.vector import VectorEntry

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
    ``getChildrenOrCreate`` call went when the card was deleted — the planning,
    knowledge-graph and workspace cards call it now, so a store's existence is
    decided by the consumer that needs it rather than by a singleton card someone
    had to remember to add. Three cards in one team therefore share one
    ``#VectorStore``, whichever backends they name.

    **A cluster backend returns without creating anything.** There is nothing
    for an actor to hold: the data is on the cluster and the consumer talks to
    the shared client directly (:func:`~akgentic.tool.vector_store.protocol.needs_store_actor`).

    **Idempotent, but not by anything here.** ``getChildrenOrCreate`` is
    idempotent per ADR-025, so three cards in one team each calling this resolve
    to the same actor and this helper keeps no bookkeeping of its own.

    The config it builds sets **no connection field**, because
    ``VectorStoreConfig`` has none: an actor holds no deployment settings at all.
    Every backend it routes to is built by that backend's registered factory,
    which reads what it needs from the environment and from the collection's own
    param — a cluster URL, or the filesystem root a locally persisting index
    hangs off.

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

    Holds one snapshot per actor-state-backed backend, keyed by its registered
    name (via ``get_state()`` / ``restore_state()``), and per-collection lifecycle
    statuses.

    **The legacy single-backend snapshot slot that preceded ``backend_states`` is
    gone, and is not migrated.** A stored state that still carries it loads —
    Pydantic's default ``extra="ignore"`` drops the key — but its contents are
    not restored: the in-memory index it held starts empty and is regenerated
    from its source documents. Every released snapshot's slot holds collection
    configs tagged with a class already deleted, so no released snapshot loses
    an index it could have loaded (ADR-049 *Migration*).
    """

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
        """Initialise state, attach observer, and prepare the lazy backend map.

        There is no embedding service and no per-request bookkeeping here: a write
        arrives already embedded, is written on this turn, and is either done or
        raised. Nothing is left open for a later message to settle, which is what
        removed the requester and entry maps this method used to carry.
        """
        self.state = VectorStoreState()
        self.state.observer(self)
        self._backends: dict[tuple[str, str | None], VectorStoreService] = {}

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _get_backend(self, name: str, root: str | None = None) -> VectorStoreService | None:
        """Return the backend registered under *name* over *root*, building it lazily.

        **Every backend is built here, one way: through its registered
        :class:`BackendSpec` factory**, with this actor's config, team id and the
        collection's root. A backend whose spec sets ``persists_in_actor_state``
        is then restored from its own ``backend_states`` snapshot. Nothing names a
        backend: a built-in and a third-party registration take the same path.

        **The cache is keyed on the pair, not on the name.** One team may hold two
        ``WorkspaceTool`` cards on two trees, and a locally persisting backend's
        identity *is* its root — so a name-keyed cache would hand the second tree
        the first tree's instance and write one tree's chunks into the other's
        directory, silently. A backend with no filesystem passes ``None`` and its
        entry is exactly the one it always had.

        The owning team's id is taken from ``self.team_id`` — propagated by the
        actor system, never configured — so a cluster backend stamps it onto every
        object it writes and a deleted team's vectors stay findable. This actor
        has no ``on_stop`` on purpose: a cluster client is shared across every
        team in the process, and one team stopping must not disconnect the others.

        Args:
            name: Backend identifier (a ``VectorStoreParam.backend`` value).
            root: The collection's filesystem root, or ``None`` for a backend
                that has none.

        Returns:
            The backend instance, or ``None`` when it cannot be built (missing
            dependency, misconfiguration, unknown name) — logged and swallowed so
            a routed operation degrades rather than raising.
        """
        key = (name, root)
        existing = self._backends.get(key)
        if existing is not None:
            return existing
        try:
            spec = get_backend_spec(name)
            backend = spec.factory(
                BackendContext(config=self.config, team_id=str(self.team_id), root=root)
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
        self._backends[key] = backend
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
        self,
        collection: str,
    ) -> VectorStoreService | None:
        """Return the correct backend for the given collection.

        Checks ``self.state.collection_configs`` for the collection's backend
        name **and its root** and resolves both through :meth:`_get_backend`.
        Reading the root here rather than anywhere else is what makes the whole
        param the per-collection channel: ``create_collection`` records the param
        whole, and this is the one place it is read back.

        Args:
            collection: Collection name to look up.

        Returns:
            The appropriate backend, or ``None`` if unavailable.
        """
        cfg_data = self.state.collection_configs.get(collection, {})
        backend_type = cfg_data.get("backend", "inmemory")
        return self._get_backend(backend_type, cfg_data.get("root"))

    # ------------------------------------------------------------------
    # State synchronisation
    # ------------------------------------------------------------------

    def _restore_backend_state(self, name: str, backend: VectorStoreService) -> None:
        """Restore a factory-built actor-state-backed backend's saved snapshot.

        Reads from the per-backend ``backend_states`` map.
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

        Every actor-state-backed backend, built-in or registered, writes the
        per-backend ``backend_states`` map under its registered name.
        """
        if not _supports_actor_state(backend):
            msg = (
                f"Backend '{name}' declares persists_in_actor_state=True but does not "
                "implement get_state() and restore_state()."
            )
            raise TypeError(msg)
        self.state.backend_states[name] = backend.get_state()

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

        Routes through the registry by ``config.backend``: the collection is
        created on whichever backend that name is registered to.

        **This method refuses; it does not degrade.** Both ways it used to
        swallow now raise ``RetriableError`` — a backend that could not be built,
        and anything the backend's own ``create_collection`` throws. Swallowing
        was the *cause* whose consequence ``add`` already refuses to hide: the
        caller returned normally with no collection created, kept its optimistic
        binding, and found out at the first write. It is on the write side of the
        read/write line this class draws, because everything downstream of a
        collection is a write.

        **Nothing is written before a raise** — no ``collection_configs`` record,
        no ``READY`` status, no state sync — so there is no partial state to
        unwind.

        **No consumer's policy changes.** Each of the three already wraps this
        call in a ``try`` and already has its own answer; refusing here makes
        those answers reachable rather than replacing them. Planning and the
        knowledge graph log one WARNING and stay keyword-only; the workspace
        leaves retrieval off.

        **Why this is not a build failure.** Failing the team's build would be
        consistent with ``require_backend_configured``, and it is refused for one
        concrete reason: the in-memory backend can fail to build *legitimately*.
        ``InMemoryBackend.__init__`` raises ``ImportError`` when the
        ``[vector_search]`` extra is absent, and degrading there is this
        package's documented optional-extra contract. This method cannot tell
        "the extra is not installed" from "the cluster is down", so refusing here
        and degrading at the consumer is the only split that serves both.

        ``remove`` and ``search`` keep degrading, and the asymmetry is the
        decision: a read that answers empty costs a miss.

        Args:
            name: Unique collection identifier.
            config: Vector store configuration for the collection.

        Raises:
            ValueError: When ``config.dimension`` is not the native width of a
                known ``config.embedding_model``. Raised from **outside** the
                error handling below, so a contradictory param reaches the caller
                as a ``ValueError`` rather than as a ``RetriableError``.
            RetriableError: When no backend could be built for ``config.backend``
                — a missing optional dependency, a misconfiguration, an
                unregistered name — or when the backend's own
                ``create_collection`` fails. Retriable for the same reason it is
                on ``add``: the fault is usually the environment, not the call.
        """
        require_dimension_matches(config, f"{self.config.name} collection '{name}'")
        backend = self._get_backend(config.backend, config.root)
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
        except Exception as exc:
            logger.warning("[%s] create_collection failed: %s", self.config.name, exc)
            raise RetriableError(str(exc)) from exc

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
            return SearchResult(hits=[], status=CollectionStatus.READY)
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
            return SearchResult(hits=[], status=CollectionStatus.READY)
