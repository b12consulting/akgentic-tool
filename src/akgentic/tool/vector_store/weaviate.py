"""Weaviate vector store backend with multi-tenancy support.

Implements the ``VectorStoreService`` protocol by delegating all vector
storage operations to a Weaviate cluster via the ``weaviate-client`` v4 API.
Vectors are provided externally (no Weaviate-side vectoriser).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from akgentic.tool.vector_store.protocol import (
    CollectionConfig,
    CollectionStatus,
    SearchHit,
    SearchResult,
)

if TYPE_CHECKING:
    import weaviate
    import weaviate.collections
    from weaviate.collections.classes.filters import FilterReturn

    from akgentic.tool.vector_store.vector import VectorEntry

logger = logging.getLogger(__name__)

TEAM_ID_PROPERTY: str = "team_id"
"""Schema property carrying the owning team's id on every stored object.

Stamped on ingest and read back as a predicate on every query this backend
issues: ``search`` and ``remove`` are restricted to the objects of their own
team, and ``delete_by_team`` reaps another team's on request. It is never
surfaced on a ``SearchHit``.

Collection names are module constants, so every team on a cluster shares the
same collections — this property is the only thing on a Weaviate object that
says who produced it, and therefore the only thing a query can be scoped by.
"""

SCOPE_PROPERTY: str = "scope"
"""Schema property partitioning a collection *within* one team.

The second dimension beside :data:`TEAM_ID_PROPERTY`, and the same mechanism: one
class for the whole deployment, narrowed by a property predicate rather than by a
class per producer. A team's workspace vectors and its planning vectors can therefore
share a collection without ever meeting in a result set.

Stamped only when the entry sets it, so an entry from planning or the knowledge graph
writes exactly the properties it wrote before this existed — and provokes no
auto-schema change on the ``Planning`` and ``Knowledge_graph`` classes, which were
created before the property and are never backfilled.
"""

PATH_PROPERTY: str = "path"
"""Schema property carrying an entry's source path, filtered by prefix.

Like :data:`SCOPE_PROPERTY`, stamped only when set."""

ORDINAL_PROPERTY: str = "ordinal"
"""Schema property carrying a chunk's position within its source.

Returned on a hit for ordering reassembly; never filtered on. Stamped only when set."""


# ---------------------------------------------------------------------------
# Dependency guard
# ---------------------------------------------------------------------------

try:
    import weaviate as _weaviate  # noqa: F811, F401
except ImportError:
    _WEAVIATE_AVAILABLE = False
else:
    _WEAVIATE_AVAILABLE = True


def _optional_str(value: object) -> str | None:
    """Return *value* as a string, or ``None`` when the property is absent.

    A class created before ``scope`` and ``path`` existed returns them as ``None``,
    which must stay ``None`` on the hit rather than becoming ``"None"``.

    Args:
        value: Raw property value read back from Weaviate.

    Returns:
        The value as a string, or ``None``.
    """
    return None if value is None else str(value)


def _check_weaviate_dependencies() -> None:
    """Validate that ``weaviate-client`` is installed.

    Raises:
        ImportError: With install instructions when ``weaviate-client`` is missing.
    """
    if not _WEAVIATE_AVAILABLE:
        msg = (
            "Weaviate backend requires the 'weaviate-client' package. "
            "Install with: pip install akgentic-tool[weaviate]"
        )
        raise ImportError(msg)


# ---------------------------------------------------------------------------
# WeaviateBackend
# ---------------------------------------------------------------------------


class WeaviateBackend:
    """Weaviate-backed vector store implementing ``VectorStoreService``.

    This is a plain Python class (not a Pydantic model) because it holds
    non-serialisable runtime state (the Weaviate client connection).
    It satisfies the ``VectorStoreService`` protocol structurally.

    **The backend is team-scoped by construction.** Every query it issues
    carries a predicate on ``team_id``: ``search`` sees only its own team's
    objects, and ``remove`` deletes only its own team's. The boundary lives
    here rather than on ``VectorStoreService`` precisely so that no caller has
    to pass a team and no caller can forget one (ADR-046 §D1). The two cleanup
    primitives — ``delete_by_team`` and ``list_collections`` — cross the
    boundary deliberately and say so in their signatures.

    Args:
        url: Weaviate cluster URL (e.g. ``http://localhost:8080``).
        api_key: Optional API key for authentication.
        tenant: Optional default tenant ID for multi-tenancy.
        team_id: Owning team id. Stamped onto every object written through this
            backend and used as the filter on every object it reads or removes.
            **Required to query:** ``search`` and ``remove`` raise ``ValueError``
            without one, rather than inventing an identity for the caller. It may
            be omitted only to build an administrative backend for
            ``list_collections`` and ``delete_by_team``, neither of which needs a
            team.
    """

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        tenant: str | None = None,
        team_id: str | None = None,
    ) -> None:
        _check_weaviate_dependencies()

        import weaviate as _wv
        from weaviate.auth import AuthApiKey

        self._tenant = tenant
        self._team_id = team_id
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 8080)
        use_https = parsed.scheme == "https"

        # gRPC defaults: same host, port 50051
        grpc_port = 50051

        auth = AuthApiKey(api_key) if api_key else None
        self._client: weaviate.WeaviateClient = _wv.connect_to_custom(
            http_host=host,
            http_port=port,
            http_secure=use_https,
            grpc_host=host,
            grpc_port=grpc_port,
            grpc_secure=use_https,
            auth_credentials=auth,
        )
        self._collections_created: set[str] = set()
        self._collection_tenants: dict[str, str] = {}

    # ------------------------------------------------------------------
    # VectorStoreService protocol methods
    # ------------------------------------------------------------------

    def create_collection(self, name: str, config: CollectionConfig) -> None:
        """Create a named Weaviate collection. No-op if it already exists.

        When multi-tenancy is enabled (``self._tenant`` or ``config.tenant``
        is set), the collection is created with multi-tenancy and the tenant
        is provisioned.

        Args:
            name: Unique collection identifier.
            config: Collection configuration.
        """
        from weaviate.classes.config import Configure, DataType, Property
        from weaviate.classes.tenants import Tenant

        tenant = getattr(config, "tenant", None) or self._tenant

        if self._client.collections.exists(name):
            self._collections_created.add(name)
            if tenant:
                self._collection_tenants[name] = tenant
                try:
                    collection = self._client.collections.get(name)
                    collection.tenants.create([Tenant(name=tenant)])
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "Tenant '%s' may already exist on collection '%s'",
                        tenant,
                        name,
                    )
            return

        properties = [
            Property(name="ref_type", data_type=DataType.TEXT),
            Property(name="ref_id", data_type=DataType.TEXT),
            Property(name="text", data_type=DataType.TEXT),
            Property(name=TEAM_ID_PROPERTY, data_type=DataType.TEXT),
            Property(name=SCOPE_PROPERTY, data_type=DataType.TEXT),
            Property(name=PATH_PROPERTY, data_type=DataType.TEXT),
            Property(name=ORDINAL_PROPERTY, data_type=DataType.INT),
        ]
        mt_config = Configure.multi_tenancy(enabled=True) if tenant else None

        self._client.collections.create(
            name=name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=properties,
            multi_tenancy_config=mt_config,
        )
        self._collections_created.add(name)

        # Create the tenant after multi-tenant collection is created
        if tenant:
            self._collection_tenants[name] = tenant
            collection = self._client.collections.get(name)
            collection.tenants.create([Tenant(name=tenant)])

    def add(self, collection: str, entries: list[VectorEntry]) -> None:
        """Ingest embedding entries into a Weaviate collection.

        Uses batch insertion with pre-populated vectors. Every object is stamped
        with the backend's ``team_id`` — the handle by which ``search``, ``remove``
        and ``delete_by_team`` later find it.

        A backend with no ``team_id`` stamps ``""`` rather than omitting the
        property, so that the schema stays uniform and a sweep never meets an object
        where it is absent. Writing is therefore still possible without a team where
        querying is not: the asymmetry is deliberate, since ``""`` is a value a
        sweeper can find and act on, whereas a *query* filtering on ``""`` would be
        answering as an identity the caller never claimed.

        ``scope``, ``path`` and ``ordinal`` follow the opposite rule and are written
        **only when the entry sets them**. A Weaviate class created before a property
        existed never gains it, so the live ``Planning`` and ``Knowledge_graph``
        classes have no such properties; stamping a default would ask the cluster to
        auto-extend a schema those producers never asked for. Omitting them keeps an
        entry from planning or the knowledge graph byte-identical to what it was
        before this dimension existed.

        Args:
            collection: Target collection name.
            entries: List of vector entries to store.

        Raises:
            ValueError: If the collection has not been created.
        """
        self._check_collection(collection)
        col = self._get_collection(collection)

        with col.batch.dynamic() as batch:
            for entry in entries:
                batch.add_object(
                    properties=self._object_properties(entry),
                    vector=entry.vector,
                )

    def _object_properties(self, entry: VectorEntry) -> dict[str, str | int]:
        """Build the Weaviate property payload for one entry.

        Args:
            entry: The entry being written.

        Returns:
            The four always-present properties, plus each of ``scope`` / ``path`` /
            ``ordinal`` that the entry actually sets.
        """
        props: dict[str, str | int] = {
            "ref_type": entry.ref_type,
            "ref_id": entry.ref_id,
            "text": entry.text,
            TEAM_ID_PROPERTY: self._team_id or "",
        }
        if entry.scope is not None:
            props[SCOPE_PROPERTY] = entry.scope
        if entry.path is not None:
            props[PATH_PROPERTY] = entry.path
        if entry.ordinal is not None:
            props[ORDINAL_PROPERTY] = entry.ordinal
        return props

    def remove(
        self,
        collection: str,
        ref_ids: list[str],
        scope: str | None = None,
        path_prefix: str | None = None,
    ) -> None:
        """Remove this team's entries from a Weaviate collection by ref_id.

        Uses ``delete_many`` with the **conjunction** of a membership filter on
        ``ref_id`` and this backend's query predicate. Both legs are required:
        ``ref_id`` alone deletes the matching object of every team on the
        cluster — and reference ids collide across teams, since planning ids
        are small integers — while the team leg alone deletes the collection.

        ``scope`` and ``path_prefix`` conjoin further legs onto the same ``where=``,
        so a scoped removal is one round trip and cannot touch another scope's
        objects even where a ref-id collides.

        Args:
            collection: Target collection name.
            ref_ids: List of reference IDs to remove.
            scope: Restrict removal to objects carrying this ``scope``.
            path_prefix: Restrict removal to objects whose ``path`` starts with this.

        Raises:
            ValueError: If the collection has not been created, or if this backend
                was built without a ``team_id``.
        """
        from weaviate.classes.query import Filter

        self._check_collection(collection)
        col = self._get_collection(collection)
        col.data.delete_many(
            where=Filter.by_property("ref_id").contains_any(ref_ids)
            & self._query_filter(scope, path_prefix),
        )

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        scope: str | None = None,
        path_prefix: str | None = None,
    ) -> SearchResult:
        """Search this team's objects in a Weaviate collection by cosine similarity.

        Every predicate is passed to the cluster as ``filters=``, so all of them are
        applied **before** ``limit``. Filtering the returned objects here instead
        would leave the caller with a short result set reporting itself complete, its
        budget already spent on other teams' or other scopes' objects.

        Args:
            collection: Target collection name.
            query_vector: Query embedding vector.
            top_k: Maximum number of results to return.
            scope: Restrict the search to objects carrying this ``scope``.
            path_prefix: Restrict the search to objects whose ``path`` starts with this.

        Returns:
            Search results with hits ranked by distance (converted to score).

        Raises:
            ValueError: If the collection has not been created, or if this backend
                was built without a ``team_id``.
        """
        from weaviate.classes.query import MetadataQuery

        self._check_collection(collection)
        col = self._get_collection(collection)

        result = col.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            filters=self._query_filter(scope, path_prefix),
            return_metadata=MetadataQuery(distance=True),
        )

        hits: list[SearchHit] = []
        for obj in result.objects:
            props = obj.properties
            distance = obj.metadata.distance if obj.metadata and obj.metadata.distance else 0.0
            score = max(0.0, 1.0 - distance)
            ordinal = props.get(ORDINAL_PROPERTY)
            hits.append(
                SearchHit(
                    ref_type=str(props.get("ref_type", "")),
                    ref_id=str(props.get("ref_id", "")),
                    text=str(props.get("text", "")),
                    score=score,
                    scope=_optional_str(props.get(SCOPE_PROPERTY)),
                    path=_optional_str(props.get(PATH_PROPERTY)),
                    ordinal=int(ordinal) if isinstance(ordinal, (int, float)) else None,
                )
            )

        return SearchResult(
            hits=hits,
            status=CollectionStatus.READY,
            indexing_pending=0,
        )

    # ------------------------------------------------------------------
    # Team-scoped cleanup (not part of VectorStoreService)
    # ------------------------------------------------------------------

    def list_collections(self) -> list[str]:
        """Return the names of every collection present in the cluster.

        Unlike the protocol methods this reads the cluster rather than the
        backend's own bookkeeping, so a cleanup process that never created a
        collection can still enumerate what is there.

        Returns:
            Collection names, in whatever order the cluster reports them.
        """
        return list(self._client.collections.list_all().keys())

    def delete_by_team(self, collection: str, team_id: str) -> int:
        """Delete every object in *collection* stamped with *team_id*.

        Existence is checked against the cluster, not ``_collections_created``:
        the caller is typically a sweeper reaping a team that no longer exists,
        so it never created the collection through this backend.

        The filter is on the **argument** alone. Unlike ``search`` and
        ``remove`` this method does not add the backend's own team predicate —
        anding it on would leave a sweeper able to reap only itself, which is
        the one team that is never being reaped.

        Args:
            collection: Target collection name.
            team_id: The team whose objects are to be removed.

        Returns:
            Number of objects deleted, or ``0`` when the cluster reports none.

        Raises:
            ValueError: If the collection does not exist in the cluster.
        """
        from weaviate.classes.query import Filter

        if not self._client.collections.exists(collection):
            msg = f"Collection '{collection}' does not exist"
            raise ValueError(msg)

        col = self._get_collection(collection)
        result = col.data.delete_many(
            where=Filter.by_property(TEAM_ID_PROPERTY).equal(team_id),
        )
        return int(getattr(result, "successful", 0) or 0)

    def close(self) -> None:
        """Disconnect the Weaviate client."""
        self._client.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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

    def _team_filter(self) -> FilterReturn:
        """Return the predicate restricting a query to this backend's own team.

        The single place the team predicate is built, so ``search`` and ``remove``
        cannot drift apart.

        A backend with no ``team_id`` **cannot query at all**. Filtering on ``""``
        instead would conflate two different things: ``""`` is a real value in the
        data — ``add`` stamps it for a writer that has no team — so a backend that
        does not know who it is would silently issue a query that looks valid and
        answers as the unattributed team. Refusing is the only reading that does not
        invent an identity for the caller (ADR-046 §D2).

        Unreachable in a running deployment: ``VectorStoreActor`` always passes
        ``str(self.team_id)``, and an actor's ``team_id`` is a UUID defaulted at
        construction. This guards hand-built backends — scripts and tests.

        Returns:
            An equality predicate on the ``team_id`` property.

        Raises:
            ValueError: When the backend was built without a ``team_id``.
        """
        from weaviate.classes.query import Filter

        if not self._team_id:
            msg = (
                "This backend was built without a team_id, so it cannot search or "
                "remove — those are scoped to the owning team. Pass team_id to the "
                "constructor. Cluster administration (list_collections, "
                "delete_by_team) needs no team and is unaffected."
            )
            raise ValueError(msg)
        return Filter.by_property(TEAM_ID_PROPERTY).equal(self._team_id)

    def _query_filter(
        self, scope: str | None, path_prefix: str | None
    ) -> FilterReturn:
        """Return the full predicate for a query: the team leg plus what was asked.

        Built **around** :meth:`_team_filter`, never instead of it — a scoped query is
        still a team's query, and no argument can widen it past its own team. A
        predicate left at ``None`` contributes no leg, so the default is exactly the
        team filter this backend has always applied.

        Args:
            scope: Restrict to objects carrying this ``scope``, or ``None``.
            path_prefix: Restrict to objects whose ``path`` starts with this, or ``None``.

        Returns:
            The conjunction of every applicable predicate.

        Raises:
            ValueError: When the backend was built without a ``team_id``.
        """
        from weaviate.classes.query import Filter

        predicate = self._team_filter()
        if scope is not None:
            predicate = predicate & Filter.by_property(SCOPE_PROPERTY).equal(scope)
        if path_prefix is not None:
            predicate = predicate & Filter.by_property(PATH_PROPERTY).like(f"{path_prefix}*")
        return predicate

    def _get_collection(self, name: str) -> weaviate.collections.Collection:
        """Return the Weaviate collection handle, with tenant if applicable.

        Resolves the effective tenant from the per-collection mapping first,
        falling back to the backend-level default tenant.

        Args:
            name: Collection name.

        Returns:
            Weaviate collection object (optionally scoped to tenant).
        """
        col = self._client.collections.get(name)
        tenant = self._collection_tenants.get(name) or self._tenant
        if tenant:
            col = col.with_tenant(tenant)
        return col
