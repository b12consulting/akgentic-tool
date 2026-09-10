"""Qdrant vector store backend with team-scoped isolation.

Implements the :class:`VectorStoreService` protocol by delegating vector storage
to a Qdrant cluster via the ``qdrant-client`` API. Vectors are provided
externally (no Qdrant-side inference).

Team isolation mirrors the Weaviate backend: every point carries a ``team_id``
in its payload, written on ingest and used to scope every ``search`` / ``remove``
and to reap a deleted team's points. Qdrant has no native multi-tenant handle
like Weaviate's tenants, so tenancy is expressed as a payload field and folded
into the same filter.

The query-construction hooks (:meth:`QdrantBackend._build_filter`,
:meth:`QdrantBackend._search_kwargs`) are the subclassing seam: override them to
add operators or native search knobs, then register the subclass under its own
name via :func:`akgentic.tool.vector_store.registry.register_backend`.

Install with ``akgentic-tool[qdrant]``.
"""

from __future__ import annotations

import importlib.util
import logging
import uuid
from typing import TYPE_CHECKING, Any

from akgentic.tool.vector_store.client import ClusterKey, get_client
from akgentic.tool.vector_store.protocol import (
    CollectionStatus,
    SearchHit,
    SearchResult,
    VectorQuery,
    VectorStoreParam,
    check_path_prefix,
    check_shared_scope,
    collection_is_team_scoped,
    stable_object_id,
)
from akgentic.tool.vector_store.registry import BackendContext, BackendSpec, register_backend

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from qdrant_client import models as qmodels

    from akgentic.tool.vector_store.vector import VectorEntry

logger = logging.getLogger(__name__)

TEAM_ID_PAYLOAD: str = "team_id"
"""Payload key carrying the owning team's id on every stored point.

Written on ingest and used to scope search/remove and to reap a deleted team's
points — the Qdrant analogue of the Weaviate ``team_id`` schema property.
"""

TENANT_PAYLOAD: str = "tenant"
"""Payload key carrying an optional tenant id, folded into the team scope."""

REF_ID_PAYLOAD: str = "ref_id"
"""Payload key carrying the domain reference id (removal is filtered on this)."""

SCOPE_PAYLOAD: str = "scope"
"""Payload key carrying the optional ``scope`` predicate value."""

PATH_PAYLOAD: str = "path"
"""Payload key carrying the optional ``path`` used by ``path_prefix`` filtering."""

ORDINAL_PAYLOAD: str = "ordinal"
"""Payload key carrying the optional ``ordinal`` position within a document."""


def _optional_str(value: object) -> str | None:
    """Return *value* as a string, or ``None`` when the payload key is absent.

    A point stored before ``scope`` / ``path`` existed reads them back as
    ``None``, which must stay ``None`` on the hit rather than becoming ``"None"``.

    Args:
        value: Raw payload value read back from Qdrant.

    Returns:
        The value as a string, or ``None``.
    """
    return None if value is None else str(value)


# ---------------------------------------------------------------------------
# Environment, read like the Weaviate backend's
# ---------------------------------------------------------------------------

QDRANT_URL_ENV: str = "AKGENTIC_QDRANT_URL"
"""Environment variable naming the Qdrant cluster. Exporting it turns Qdrant on."""

QDRANT_API_KEY_ENV: str = "AKGENTIC_QDRANT_API_KEY"
"""Environment variable holding the Qdrant API key. Optional for an open cluster."""


def qdrant_url() -> str | None:
    """Return the configured Qdrant cluster URL, or ``None`` when unset.

    An exported but empty variable counts as unset.
    """
    import os

    return os.environ.get(QDRANT_URL_ENV) or None


def qdrant_api_key() -> str | None:
    """Return the configured Qdrant API key, or ``None`` when unset."""
    import os

    return os.environ.get(QDRANT_API_KEY_ENV) or None


def qdrant_is_configured() -> bool:
    """Return whether a Qdrant cluster URL is exported."""
    return qdrant_url() is not None


def require_qdrant_configured(card_name: str) -> None:
    """Raise when Qdrant is named but its URL or client library is unavailable.

    Args:
        card_name: Card class name, for the error message.

    Raises:
        ValueError: When ``AKGENTIC_QDRANT_URL`` is not set or the optional
            ``qdrant-client`` dependency is unavailable.
    """
    if not qdrant_url():
        raise ValueError(
            f"{card_name} configures backend='qdrant' but {QDRANT_URL_ENV} is not set. "
            f"Export {QDRANT_URL_ENV} (and {QDRANT_API_KEY_ENV} for an authenticated "
            f"cluster), or drop the backend setting to use the in-memory index."
        )
    if not qdrant_dependencies_available():
        raise ValueError(
            f"{card_name} configures backend='qdrant' but the 'qdrant-client' package "
            "is not installed. Install with: pip install akgentic-tool[qdrant]"
        )


# ---------------------------------------------------------------------------
# Dependency guard
# ---------------------------------------------------------------------------


def qdrant_dependencies_available() -> bool:
    """Return whether the optional Qdrant client can be imported, without importing it."""
    return importlib.util.find_spec("qdrant_client") is not None


def _check_qdrant_dependencies() -> None:
    """Validate that ``qdrant-client`` is installed.

    Raises:
        ImportError: With install instructions when ``qdrant-client`` is missing.
    """
    if not qdrant_dependencies_available():
        msg = (
            "Qdrant backend requires the 'qdrant-client' package. "
            "Install with: pip install akgentic-tool[qdrant]"
        )
        raise ImportError(msg)


# ---------------------------------------------------------------------------
# QdrantBackend
# ---------------------------------------------------------------------------


class QdrantBackend:
    """Qdrant-backed vector store implementing ``VectorStoreService``.

    A plain class (not a Pydantic model) because it holds non-serialisable
    runtime state: a handle to the process's **shared** Qdrant client, not a
    connection of its own. It satisfies the ``VectorStoreService`` protocol
    structurally.

    **It takes a client and never closes one**, exactly as ``WeaviateBackend``
    does. The client comes from
    :func:`akgentic.tool.vector_store.client.get_client` — one per cluster per
    process, keyed on the backend name as well as the connection — and is closed
    only by ``close_all()`` at process exit. What a backend owns is its scope
    (``tenant``, ``team_id``) and its created-collections bookkeeping.

    Args:
        client: The connected ``qdrant_client.QdrantClient`` for the cluster.
            The backend never connects and never closes it.
        tenant: Optional tenant id, stored on every point and folded into the
            team scope for search/remove.
        team_id: Owning team id, stamped onto every point so a later sweep can
            reap a deleted team's points.
    """

    def __init__(
        self,
        client: QdrantClient,
        tenant: str | None = None,
        team_id: str | None = None,
    ) -> None:
        _check_qdrant_dependencies()

        self._tenant = tenant
        self._team_id = team_id
        self._client: QdrantClient = client
        self._collections_created: set[str] = set()
        self._collection_tenants: dict[str, str] = {}

    # ------------------------------------------------------------------
    # VectorStoreService protocol methods
    # ------------------------------------------------------------------

    def create_collection(self, name: str, config: VectorStoreParam) -> None:
        """Create a named Qdrant collection. No-op if it already exists.

        The distance metric is cosine, matching the ``VectorStoreService``
        similarity-score contract. An explicit non-cosine distance is rejected.

        Args:
            name: Unique collection identifier.
            config: Collection configuration.
        """
        from qdrant_client import models

        tenant = config.tenant or self._tenant
        distance = self._resolve_distance(config)
        if self._client.collection_exists(name):
            self._require_existing_cosine(name)
            self._collections_created.add(name)
            if tenant:
                self._collection_tenants[name] = tenant
            return

        self._client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=config.dimension, distance=distance),
        )
        self._collections_created.add(name)
        if tenant:
            self._collection_tenants[name] = tenant

    def add(self, collection: str, entries: list[VectorEntry]) -> None:
        """Ingest embedding entries into a Qdrant collection.

        Every point is stamped with the backend's ``team_id`` (empty string when
        built without one) and optional ``tenant`` so ``delete_by_team`` and the
        team-scoped query can find it later.

        Args:
            collection: Target collection name.
            entries: List of vector entries to store.

        Raises:
            ValueError: If the collection has not been created.
        """
        from qdrant_client import models

        self._check_collection(collection)
        tenant = self._collection_tenants.get(collection) or self._tenant
        points = [
            models.PointStruct(
                id=self._point_id(entry.ref_id, tenant),
                vector=entry.vector,
                payload={
                    "ref_type": entry.ref_type,
                    REF_ID_PAYLOAD: entry.ref_id,
                    "text": entry.text,
                    TEAM_ID_PAYLOAD: self._team_id or "",
                    **({TENANT_PAYLOAD: tenant} if tenant else {}),
                    **({SCOPE_PAYLOAD: entry.scope} if entry.scope is not None else {}),
                    **({PATH_PAYLOAD: entry.path} if entry.path is not None else {}),
                    **({ORDINAL_PAYLOAD: entry.ordinal} if entry.ordinal is not None else {}),
                },
            )
            for entry in entries
        ]
        self._client.upsert(collection_name=collection, points=points)

    def remove(
        self,
        collection: str,
        ref_ids: list[str],
        scope: str | None = None,
        path_prefix: str | None = None,
    ) -> None:
        """Remove entries from a Qdrant collection by ref_id, scoped to this team.

        ``scope`` conjoins a native equality condition onto the team predicate.
        ``path_prefix`` is not expressible as a native Qdrant condition, so
        matching points are scrolled and filtered client-side, then deleted by id.

        Args:
            collection: Target collection name.
            ref_ids: List of reference IDs to remove.
            scope: Restrict removal to points carrying this ``scope``.
            path_prefix: Restrict removal to points whose ``path`` starts with this.

        Raises:
            ValueError: If the collection has not been created, ``path_prefix``
                contains ``*`` or ``?``, the collection is shared across teams
                and no ``scope`` was given, or — on a team-scoped collection
                only — the backend was built without a ``team_id``.
        """
        from qdrant_client import models

        check_path_prefix(path_prefix)
        check_shared_scope(collection, scope)
        self._check_collection(collection)
        selector = self._build_filter(collection, None, scope=scope)
        selector.must.append(  # type: ignore[union-attr]
            models.FieldCondition(key=REF_ID_PAYLOAD, match=models.MatchAny(any=ref_ids))
        )
        if path_prefix is None:
            self._client.delete(
                collection_name=collection,
                points_selector=models.FilterSelector(filter=selector),
            )
            return
        point_ids = self._scroll_ids_matching_prefix(collection, selector, path_prefix)
        if point_ids:
            self._client.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(points=point_ids),
            )

    def _scroll_ids_matching_prefix(
        self, collection: str, selector: qmodels.Filter, path_prefix: str
    ) -> list[int | str | uuid.UUID]:
        """Return ids of points matching *selector* whose ``path`` starts with *prefix*.

        Qdrant has no native prefix operator, so the candidates selected natively
        are paged in and filtered on their stored ``path`` payload here.
        """
        ids: list[int | str | uuid.UUID] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=collection,
                scroll_filter=selector,
                with_payload=True,
                limit=256,
                offset=offset,
            )
            for point in points:
                path = (point.payload or {}).get(PATH_PAYLOAD)
                if isinstance(path, str) and path.startswith(path_prefix):
                    ids.append(point.id)
            if offset is None:
                break
        return ids

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        scope: str | None = None,
        path_prefix: str | None = None,
        query: VectorQuery | None = None,
    ) -> SearchResult:
        """Search this team's points in a Qdrant collection by cosine similarity.

        The team predicate is passed to the cluster as ``query_filter=``, so it
        is applied before ``limit``. ``scope`` conjoins a native equality
        condition. When ``query`` is supplied its ``filters`` are AND-combined
        onto the team predicate (see :meth:`_build_filter`), its
        ``score_threshold`` becomes Qdrant's native ``score_threshold``, and its
        ``params`` are forwarded to ``query_points`` (see :meth:`_search_kwargs`).

        ``path_prefix`` has no native Qdrant operator, so the query over-fetches
        and the prefix is applied to the stored ``path`` payload client-side
        before the result is cut back to ``top_k``.

        Args:
            collection: Target collection name.
            query_vector: Query embedding vector.
            top_k: Maximum number of results to return.
            scope: Restrict the search to points carrying this ``scope``.
            path_prefix: Restrict the search to points whose ``path`` starts with this.
            query: Optional filters / score threshold / native params.

        Returns:
            Search results with hits ranked by cosine similarity.

        Raises:
            ValueError: If the collection has not been created, ``path_prefix``
                contains ``*`` or ``?``, the collection is shared across teams
                and no ``scope`` was given, or — on a team-scoped collection
                only — the backend was built without a ``team_id``.
        """
        check_path_prefix(path_prefix)
        check_shared_scope(collection, scope)
        self._check_collection(collection)
        limit = top_k if path_prefix is None else max(top_k * 4, top_k)
        response = self._client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=limit,
            query_filter=self._build_filter(collection, query, scope=scope),
            score_threshold=query.score_threshold if query else None,
            with_payload=True,
            **self._search_kwargs(query),
        )

        hits: list[SearchHit] = []
        for point in response.points:
            payload = point.payload or {}
            if path_prefix is not None:
                path = payload.get(PATH_PAYLOAD)
                if not (isinstance(path, str) and path.startswith(path_prefix)):
                    continue
            ordinal = payload.get(ORDINAL_PAYLOAD)
            hits.append(
                SearchHit(
                    ref_type=str(payload.get("ref_type", "")),
                    ref_id=str(payload.get(REF_ID_PAYLOAD, "")),
                    text=str(payload.get("text", "")),
                    score=max(0.0, float(point.score)),
                    scope=_optional_str(payload.get(SCOPE_PAYLOAD)),
                    path=_optional_str(payload.get(PATH_PAYLOAD)),
                    ordinal=int(ordinal) if isinstance(ordinal, (int, float)) else None,
                )
            )
            if len(hits) >= top_k:
                break

        return SearchResult(hits=hits, status=CollectionStatus.READY)

    # ------------------------------------------------------------------
    # Query construction hooks (override in a subclass for bespoke behaviour)
    # ------------------------------------------------------------------

    def _build_filter(
        self,
        collection: str,
        query: VectorQuery | None,
        *,
        scope: str | None = None,
    ) -> qmodels.Filter:
        """Combine the collection's own predicate with ``scope`` and ``query.filters``.

        ``scope`` and each entry in ``query.filters`` become a ``MatchValue``
        (scalar) or ``MatchAny`` (list) condition AND-ed onto that predicate.
        Override to support ranges, geo, or nested payload operators.

        On a team-scoped collection the predicate starts from
        :meth:`_team_scope`, which refuses a backend that does not know its team.
        On a collection listed in
        :data:`~akgentic.tool.vector_store.protocol.SHARED_COLLECTIONS` the team leg
        is not built at all — its rows belong to a filesystem tree rather than to a
        team — and only the tenant leg survives from the base. The result is never
        an empty conjunction, because
        :func:`~akgentic.tool.vector_store.protocol.check_shared_scope` has already
        made ``scope`` mandatory for such a collection.

        Args:
            collection: The collection being queried, which decides whether the
                team leg is part of the predicate. Its effective tenant is
                included either way.
            query: The active query, or ``None``.
            scope: Optional ``scope`` equality predicate.

        Returns:
            A Qdrant ``Filter``, scoped to this team unless *collection* is shared
            across teams.
        """
        from qdrant_client import models

        selector = (
            self._team_scope(collection)
            if collection_is_team_scoped(collection)
            else models.Filter(must=self._tenant_conditions(collection))
        )
        if scope is not None:
            selector.must.append(  # type: ignore[union-attr]
                models.FieldCondition(key=SCOPE_PAYLOAD, match=models.MatchValue(value=scope))
            )
        if query and query.filters:
            for key, value in query.filters.items():
                match = (
                    models.MatchAny(any=list(value))
                    if isinstance(value, (list, tuple, set))
                    else models.MatchValue(value=value)
                )
                selector.must.append(  # type: ignore[union-attr]
                    models.FieldCondition(key=key, match=match)
                )
        return selector

    def _search_kwargs(self, query: VectorQuery | None) -> dict[str, Any]:
        """Return extra keyword arguments forwarded to ``query_points``.

        Passes ``query.params`` through untouched, so a caller can set native
        knobs such as ``search_params`` (HNSW ``ef``, exact toggle). Override to
        whitelist or remap parameters.

        Args:
            query: The active query, or ``None``.

        Returns:
            Keyword arguments for ``client.query_points``.
        """
        if query is None or not query.params:
            return {}
        return dict(query.params)

    # ------------------------------------------------------------------
    # Team-scoped cleanup (not part of VectorStoreService)
    # ------------------------------------------------------------------

    def list_collections(self) -> list[str]:
        """Return the names of every collection present in the cluster."""
        return [c.name for c in self._client.get_collections().collections]

    def delete_by_team(self, collection: str, team_id: str) -> None:
        """Delete every point in *collection* stamped with *team_id*.

        Existence is checked against the cluster, not local bookkeeping: the
        caller is typically a sweeper reaping a team that no longer exists.

        **A collection shared across teams is refused, before any cluster call.**
        On such a collection ``team_id`` records *who wrote the point* and is read
        by nothing once the query predicate stops using it, so a sweeper pointed at
        it would delete points another live team is still reading, from a tree that
        still exists. The check consults a module-level fact rather than this
        backend's own bookkeeping, so it still bites on an administrative backend
        that has created no collection — the only kind a sweeper has.

        Args:
            collection: Target collection name.
            team_id: The team whose points are to be removed.

        Raises:
            ValueError: If the collection is shared across teams, or does not
                exist in the cluster.
        """
        from qdrant_client import models

        if not collection_is_team_scoped(collection):
            msg = (
                f"Collection '{collection}' is shared across teams, so deleting one "
                "team's points would remove rows another live team is still reading. "
                "Remove by ref_id with a scope instead."
            )
            raise ValueError(msg)

        if not self._client.collection_exists(collection):
            msg = f"Collection '{collection}' does not exist"
            raise ValueError(msg)

        self._client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=TEAM_ID_PAYLOAD, match=models.MatchValue(value=team_id)
                        )
                    ]
                )
            ),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _point_id(self, ref_id: str, tenant: str | None = None) -> str:
        """Derive the point id for ``ref_id`` under this backend's team and *tenant*.

        Qdrant point ids must be an unsigned int or a UUID, so the id is
        :func:`~akgentic.tool.vector_store.protocol.stable_object_id` — the
        derivation both cluster backends share, and the one every existing point
        was written under. Stable across re-ingest, and a team-scoped
        collection keeps the team inside it, so one team never overwrites
        another's point. ``ref_id`` itself stays in the payload for filtering.
        """
        return stable_object_id(self._team_id, tenant, ref_id)

    @staticmethod
    def _resolve_distance(config: VectorStoreParam) -> qmodels.Distance:
        """Return cosine distance or reject an incompatible metric."""
        from qdrant_client import models

        override = config.params.get("distance")
        if override is None:
            return models.Distance.COSINE
        distance = (
            override if isinstance(override, models.Distance) else models.Distance(str(override))
        )
        if distance != models.Distance.COSINE:
            msg = (
                "QdrantBackend supports only cosine distance because "
                "VectorStoreService scores are higher-is-better similarities."
            )
            raise ValueError(msg)
        return distance

    def _require_existing_cosine(self, collection: str) -> None:
        """Reject an existing collection whose vectors do not use cosine distance."""
        from qdrant_client import models

        info = self._client.get_collection(collection_name=collection)
        vectors = info.config.params.vectors
        if vectors is None or isinstance(vectors, dict):
            msg = (
                f"Qdrant collection '{collection}' does not expose one unnamed "
                "vector configuration, which QdrantBackend requires."
            )
            raise ValueError(msg)
        if vectors.distance != models.Distance.COSINE:
            msg = (
                f"Qdrant collection '{collection}' does not use cosine distance, "
                "which is required by the VectorStoreService score contract."
            )
            raise ValueError(msg)

    def _team_scope(self, collection: str) -> qmodels.Filter:
        """Return the predicate restricting a query to this backend's team/tenant.

        A backend with no ``team_id`` cannot query — filtering on ``""`` would
        conflate a real unattributed writer with "any team". Refusing is the only
        reading that does not invent an identity for the caller (mirrors the
        Weaviate backend's ``_team_filter``).

        Returns:
            A Qdrant ``Filter`` with the team (and optional tenant) predicate.

        Raises:
            ValueError: When the backend was built without a ``team_id``.
        """
        from qdrant_client import models

        if not self._team_id:
            msg = (
                "This backend was built without a team_id, so it cannot search or "
                "remove — those are scoped to the owning team. Pass team_id to the "
                "constructor. Cluster administration (list_collections, "
                "delete_by_team) needs no team and is unaffected."
            )
            raise ValueError(msg)

        must: list[qmodels.Condition] = [
            models.FieldCondition(key=TEAM_ID_PAYLOAD, match=models.MatchValue(value=self._team_id))
        ]
        must.extend(self._tenant_conditions(collection))
        return models.Filter(must=must)

    def _tenant_conditions(self, collection: str) -> list[qmodels.Condition]:
        """Return the tenant leg for *collection*, or no leg when none is configured.

        Split out of :meth:`_team_scope` because a shared collection drops the team
        leg and keeps this one: tenancy is a deployment partition, orthogonal to
        which team wrote a row, so it applies whether or not the collection is
        team-scoped. Duplicating it in two filter builders is how the two would
        drift.

        Args:
            collection: The collection whose effective tenant is wanted.

        Returns:
            A one-element list holding the tenant equality condition, or an empty
            list when neither the collection nor the backend names a tenant.
        """
        from qdrant_client import models

        tenant = self._collection_tenants.get(collection) or self._tenant
        if not tenant:
            return []
        return [models.FieldCondition(key=TENANT_PAYLOAD, match=models.MatchValue(value=tenant))]

    def _check_collection(self, collection: str) -> None:
        """Raise ``ValueError`` if *collection* was never created via this backend.

        Args:
            collection: Collection name to validate.

        Raises:
            ValueError: If the collection has not been created.
        """
        if collection not in self._collections_created:
            msg = f"Collection '{collection}' does not exist"
            raise ValueError(msg)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _connect_qdrant(key: ClusterKey) -> QdrantClient:
    """Open the cluster connection *key* names — the callable ``get_client`` calls.

    The one place in this package that constructs a ``QdrantClient``, which is
    what keeps ``client.py`` free of any vendor name. It builds the **remote**
    client only: the embedded implementation is reached exclusively through
    ``location=":memory:"`` or ``path=``, neither of which anything here passes,
    so a shared client is always the thread-safe remote one.

    Args:
        key: The cluster to connect to.

    Returns:
        A connected client.

    Raises:
        ImportError: When ``qdrant-client`` is not installed.
    """
    _check_qdrant_dependencies()
    from qdrant_client import QdrantClient as _QdrantClient

    scheme = "https" if key.secure else "http"
    return _QdrantClient(url=f"{scheme}://{key.host}:{key.port}", api_key=key.api_key)


def _make_qdrant_backend(context: BackendContext) -> QdrantBackend:
    """Build a :class:`QdrantBackend` from the environment.

    Like Weaviate, connection settings are read straight from the environment
    (``VectorStoreConfig`` carries no connection field), so a Qdrant deployment
    needs only the ``AKGENTIC_QDRANT_*`` variables and no card changes. The
    resolved pair goes through the shared cache, so every backend built for one
    Qdrant cluster in this process holds one client — and none of them closes it.

    ``default_port=6333`` is a keying concern: it collapses ``http://host`` and
    ``http://host:6333`` onto one key rather than opening two clients against
    one server.
    """
    url = qdrant_url()
    if not url:
        msg = "qdrant_url is not configured; cannot build QdrantBackend."
        raise ValueError(msg)
    key = ClusterKey.from_url("qdrant", url, qdrant_api_key(), default_port=6333)
    # Tenancy is per-collection (VectorStoreParam.tenant), not per-actor, so the
    # actor-level factory leaves it unset; a hand-built backend or subclass may
    # still pass tenant= directly.
    return QdrantBackend(
        client=get_client(key, _connect_qdrant),
        tenant=None,
        team_id=context.team_id,
    )


register_backend(
    BackendSpec(
        name="qdrant",
        factory=_make_qdrant_backend,
        persists_in_actor_state=False,
        selectable_as_default=True,
        is_configured=qdrant_is_configured,
        require_configured=require_qdrant_configured,
    ),
    replace=True,
)
