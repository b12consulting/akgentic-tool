"""Weaviate vector store backend with multi-tenancy support.

Implements the ``VectorStoreService`` protocol by delegating all vector
storage operations to a Weaviate cluster via the ``weaviate-client`` v4 API.
Vectors are provided externally (no Weaviate-side vectoriser).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Final

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
)
from akgentic.tool.vector_store.registry import BackendContext, BackendSpec, register_backend

if TYPE_CHECKING:
    import weaviate
    import weaviate.collections
    from weaviate.collections.classes.filters import FilterReturn

    from akgentic.tool.vector_store.vector import VectorEntry

logger = logging.getLogger(__name__)

GRPC_PORT: Final[int] = 50051
"""gRPC port used for every cluster. Fixed, and therefore not part of ``ClusterKey``.

It lives here rather than in ``client.py`` for the same reason the dependency
check does: a gRPC port is a Weaviate connection detail, and the cache is shared
with backends that have none.
"""

WEAVIATE_MISSING_MESSAGE: Final[str] = (
    "Weaviate backend requires the 'weaviate-client' package. "
    "Install with: pip install akgentic-tool[weaviate]"
)
"""The ``ImportError`` text raised when ``weaviate-client`` is not installed."""


def _check_weaviate_dependencies() -> None:
    """Validate that ``weaviate-client`` is importable, at call time.

    Lives beside ``qdrant.py``'s ``_check_qdrant_dependencies``, which is what
    "generalised, not twinned" means for the client cache: each backend owns its
    own dependency guard, and ``client.py`` owns neither.

    Raises:
        ImportError: With install instructions when ``weaviate-client`` is missing.
    """
    try:
        import weaviate  # noqa: F401
    except ImportError as exc:
        raise ImportError(WEAVIATE_MISSING_MESSAGE) from exc


def _connect_weaviate(key: ClusterKey) -> weaviate.WeaviateClient:
    """Open the cluster connection *key* names — the callable ``get_client`` calls.

    The one place in this package that names ``weaviate.connect_to_custom``, and
    the reason ``client.py`` names no vendor at all. The keyword set is fixed:
    HTTP and gRPC share the host and the ``secure`` flag, the gRPC port is
    :data:`GRPC_PORT` for every cluster, and the credential is an ``AuthApiKey``
    or nothing — never an ``AuthCredentials`` object, which is the module
    docstring's invariant 2 in ``client.py``.

    Args:
        key: The cluster to connect to.

    Returns:
        A connected client.

    Raises:
        ImportError: When ``weaviate-client`` is not installed.
    """
    _check_weaviate_dependencies()
    import weaviate as _weaviate
    from weaviate.auth import AuthApiKey

    return _weaviate.connect_to_custom(
        http_host=key.host,
        http_port=key.port,
        http_secure=key.secure,
        grpc_host=key.host,
        grpc_port=GRPC_PORT,
        grpc_secure=key.secure,
        auth_credentials=AuthApiKey(key.api_key) if key.api_key else None,
    )


def _weaviate_client(url: str, api_key: str | None = None) -> weaviate.WeaviateClient:
    """Return the process's shared client for the cluster *url* names.

    The one entry point its callers use — the registered factory and a sweeper
    built by hand — so the key's shape and the connect keyword set are written
    once.

    Args:
        url: Cluster URL, e.g. ``http://localhost:8080``.
        api_key: API key, or ``None`` for an unauthenticated cluster.

    Returns:
        The shared, connected client.
    """
    return get_client(ClusterKey.from_url("weaviate", url, api_key), _connect_weaviate)


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


# ---------------------------------------------------------------------------
# WeaviateBackend
# ---------------------------------------------------------------------------


class WeaviateBackend:
    """Weaviate-backed vector store implementing ``VectorStoreService``.

    This is a plain Python class (not a Pydantic model) because it holds
    non-serialisable runtime state: a handle to the process's **shared**
    Weaviate client, not a connection of its own. The client is obtained from
    :func:`akgentic.tool.vector_store.client.get_client` — one per cluster per
    process — and closed by ``close_all()`` at process exit, never by a backend.
    What a backend owns is its scope (``tenant``, ``team_id``) and its own
    created-collections bookkeeping. It satisfies the ``VectorStoreService``
    protocol structurally.

    **The backend is team-scoped by construction.** Every query it issues
    carries a predicate on ``team_id``: ``search`` sees only its own team's
    objects, and ``remove`` deletes only its own team's. The boundary lives
    here rather than on ``VectorStoreService`` precisely so that no caller has
    to pass a team and no caller can forget one (ADR-046 §D1). The two cleanup
    primitives — ``delete_by_team`` and ``list_collections`` — cross the
    boundary deliberately and say so in their signatures.

    Args:
        client: The connected ``weaviate.WeaviateClient`` for the cluster, from
            ``get_client(url, api_key)``. The backend never connects and never
            closes it.
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
        client: weaviate.WeaviateClient,
        tenant: str | None = None,
        team_id: str | None = None,
    ) -> None:
        _check_weaviate_dependencies()

        self._client: weaviate.WeaviateClient = client
        self._tenant = tenant
        self._team_id = team_id
        self._collections_created: set[str] = set()
        self._collection_tenants: dict[str, str] = {}
        self._collection_handles: dict[tuple[str, str | None], weaviate.collections.Collection] = {}

    # ------------------------------------------------------------------
    # VectorStoreService protocol methods
    # ------------------------------------------------------------------

    def create_collection(self, name: str, config: VectorStoreParam) -> None:
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

        The batch context is opened here, on a handle fetched in this call, and
        left here — never stored on the instance, never shared between calls —
        because a shared batch object is the one thing the vendor says is not
        thread-safe, and the client underneath is shared by every consumer in the
        process.

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
            ValueError: If the collection has not been created, if
                ``path_prefix`` contains ``*`` or ``?``, if the collection is
                shared across teams and no ``scope`` was given, or — on a
                team-scoped collection only — if this backend was built without a
                ``team_id``.
        """
        from weaviate.classes.query import Filter

        check_path_prefix(path_prefix)
        check_shared_scope(collection, scope)
        self._check_collection(collection)
        col = self._get_collection(collection)
        col.data.delete_many(
            where=Filter.by_property("ref_id").contains_any(ref_ids)
            & self._query_filter(collection, scope, path_prefix),
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
        """Search this team's objects in a Weaviate collection by cosine similarity.

        Every predicate is passed to the cluster as ``filters=``, so all of them are
        applied **before** ``limit``. Filtering the returned objects here instead
        would leave the caller with a short result set reporting itself complete, its
        budget already spent on other teams' or other scopes' objects.

        When ``query`` is supplied, its ``filters`` are AND-combined with the
        team predicate (see :meth:`_build_filter`), its ``params`` are forwarded
        to ``near_vector`` (see :meth:`_near_vector_kwargs`), and its
        ``score_threshold`` drops low-scoring hits after distance conversion.

        Args:
            collection: Target collection name.
            query_vector: Query embedding vector.
            top_k: Maximum number of results to return.
            scope: Restrict the search to objects carrying this ``scope``.
            path_prefix: Restrict the search to objects whose ``path`` starts with this.
            query: Optional filters / score threshold / native ``near_vector`` params.

        Returns:
            Search results with hits ranked by distance (converted to score).

        Raises:
            ValueError: If the collection has not been created, if
                ``path_prefix`` contains ``*`` or ``?``, if the collection is
                shared across teams and no ``scope`` was given, or — on a
                team-scoped collection only — if this backend was built without a
                ``team_id``.
        """
        from weaviate.classes.query import MetadataQuery

        check_path_prefix(path_prefix)
        check_shared_scope(collection, scope)
        self._check_collection(collection)
        col = self._get_collection(collection)

        result = col.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            filters=self._build_filter(collection, query, scope=scope, path_prefix=path_prefix),
            return_metadata=MetadataQuery(distance=True),
            **self._near_vector_kwargs(query),
        )

        threshold = query.score_threshold if query else None
        hits: list[SearchHit] = []
        for obj in result.objects:
            props = obj.properties
            distance = obj.metadata.distance if obj.metadata and obj.metadata.distance else 0.0
            score = max(0.0, 1.0 - distance)
            if threshold is not None and score < threshold:
                continue
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
        path_prefix: str | None = None,
    ) -> FilterReturn:
        """Combine the collection's own predicate with ``query.filters``.

        The predicate :meth:`_query_filter` builds for *collection* — the team leg
        where the collection is team-scoped, narrowed by ``scope`` / ``path_prefix``
        — is always present; each entry in ``query.filters`` is AND-ed onto it as an
        equality (scalar) or ``contains_any`` (list) condition. Override to support
        richer operators.

        Args:
            collection: The collection being queried, which decides whether the
                team leg is part of the predicate at all.
            query: The active query, or ``None``.
            scope: Restrict to objects carrying this ``scope``.
            path_prefix: Restrict to objects whose ``path`` starts with this.

        Returns:
            A combined Weaviate ``Filter``, scoped to this team unless *collection*
            is shared across teams.
        """
        base = self._query_filter(collection, scope, path_prefix)
        if query is None or not query.filters:
            return base
        from weaviate.classes.query import Filter

        combined = base
        for key, value in query.filters.items():
            prop = Filter.by_property(key)
            condition = (
                prop.contains_any(list(value))
                if isinstance(value, (list, tuple, set))
                else prop.equal(value)
            )
            combined = combined & condition
        return combined

    def _near_vector_kwargs(self, query: VectorQuery | None) -> dict[str, Any]:
        """Return extra keyword arguments forwarded to ``near_vector``.

        Passes ``query.params`` through untouched, so a caller can set native
        knobs such as ``certainty`` or ``distance``. Override to whitelist or
        remap parameters.

        Args:
            query: The active query, or ``None``.

        Returns:
            Keyword arguments for ``collection.query.near_vector``.
        """
        if query is None or not query.params:
            return {}
        return dict(query.params)

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

        **A collection shared across teams is refused, before any cluster call.**
        On such a collection ``team_id`` records *who wrote the row* and is read by
        nothing once the query predicate stops using it, so a sweeper pointed at it
        would delete rows another live team is still reading, from a tree that still
        exists. This method has no caller anywhere in ``src/``, which is exactly why
        it needs a guard rather than a docstring: the first caller will be written
        by someone reading the signature. The check consults a module-level fact
        rather than this backend's own bookkeeping, so it still bites on an
        administrative backend that has created no collection — the only kind a
        sweeper has.

        Args:
            collection: Target collection name.
            team_id: The team whose objects are to be removed.

        Returns:
            Number of objects deleted, or ``0`` when the cluster reports none.

        Raises:
            ValueError: If the collection is shared across teams, or does not
                exist in the cluster.
        """
        from weaviate.classes.query import Filter

        if not collection_is_team_scoped(collection):
            msg = (
                f"Collection '{collection}' is shared across teams, so deleting one "
                "team's objects would remove rows another live team is still reading. "
                "Remove by ref_id with a scope instead."
            )
            raise ValueError(msg)

        if not self._client.collections.exists(collection):
            msg = f"Collection '{collection}' does not exist"
            raise ValueError(msg)

        col = self._get_collection(collection)
        result = col.data.delete_many(
            where=Filter.by_property(TEAM_ID_PROPERTY).equal(team_id),
        )
        return int(getattr(result, "successful", 0) or 0)

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
        self, collection: str, scope: str | None, path_prefix: str | None
    ) -> FilterReturn:
        """Return the full predicate for a query on *collection*.

        On a team-scoped collection this is built **around** :meth:`_team_filter`,
        never instead of it — a scoped query is still a team's query, and no
        argument can widen it past its own team. A predicate left at ``None``
        contributes no leg, so the default there is exactly the team filter this
        backend has always applied.

        On a collection listed in
        :data:`~akgentic.tool.vector_store.protocol.SHARED_COLLECTIONS` the team leg
        is not built at all: its rows belong to a filesystem tree rather than to a
        team, and two teams over one tree must read the same rows. The conjunction
        is then made from the scope and path legs alone — and it is never empty,
        because :func:`~akgentic.tool.vector_store.protocol.check_shared_scope` has
        already made ``scope`` mandatory for such a collection at the top of
        ``search`` and ``remove``. That guard is the guarantee, so there is no
        defensive branch here for a predicate with no legs.

        Args:
            collection: The collection being queried, which decides whether the
                team leg is part of the predicate.
            scope: Restrict to objects carrying this ``scope``, or ``None``.
            path_prefix: Restrict to objects whose ``path`` starts with this, or ``None``.

        Returns:
            The conjunction of every applicable predicate.

        Raises:
            ValueError: When the collection is team-scoped and the backend was
                built without a ``team_id``. A team-less backend can therefore
                query a shared collection and still cannot query ``planning`` —
                which is the correct reading of ADR-046 §D2: on a shared
                collection there is no identity to invent, because the boundary
                is the scope.
        """
        from weaviate.classes.query import Filter

        legs: list[FilterReturn] = []
        if collection_is_team_scoped(collection):
            legs.append(self._team_filter())
        if scope is not None:
            legs.append(Filter.by_property(SCOPE_PROPERTY).equal(scope))
        if path_prefix is not None:
            # ``check_path_prefix`` has already refused ``*`` and ``?`` for this reason:
            # both are wildcards in Weaviate's ``Like`` operator, and the v4 filter API
            # offers no escape, so a literal one in a prefix would widen the match here
            # while the in-memory backend's ``str.startswith`` reads it literally.
            legs.append(Filter.by_property(PATH_PROPERTY).like(f"{path_prefix}*"))
        predicate = legs[0]
        for leg in legs[1:]:
            predicate = predicate & leg
        return predicate

    def _get_collection(self, name: str) -> weaviate.collections.Collection:
        """Return the Weaviate collection handle, with tenant if applicable.

        Resolves the effective tenant from the per-collection mapping first,
        falling back to the backend-level default tenant.

        **The handle is cached per ``(name, tenant)``, and it has to be.**
        ``client.collections.get`` builds a *new* ``Collection`` on every call
        (``weaviate/collections/collections/executor.py:85-91``), and each one
        constructs a ``_BatchCollectionWrapper`` whose ``__init__`` creates a
        ``ThreadPoolExecutor`` (``weaviate/collections/batch/collection.py:127-128``)
        that nothing ever shuts down. Uncached, every ``add()`` leaked one idle
        pool for the lifetime of the process. ``with_tenant()`` returns a
        different ``Collection``, so the tenant is part of the key.

        **The invariant that makes a cached handle safe: a backend instance
        belongs to exactly one actor.** The vendor's "not thread-safe" caveat is
        about the batching algorithm, and the batch wrapper is per-``Collection``
        object: ``dynamic()`` assigns ``self._batch_mode`` and replaces
        ``self._batch_data`` on that shared wrapper before handing back the
        context (``weaviate/collections/batch/collection.py:151-170``), so one
        ``Collection`` reached from two threads is precisely the unsafe case.
        It is not reached from two threads here, because each consumer builds its
        own backend through ``BackendSpec.factory`` at bind time and every
        ``add`` / ``remove`` / ``search`` runs on that consumer's own mailbox
        turn. What is shared across threads is the **client**, which is verified
        safe; what is cached here is a **handle**, which is not.

        **If that invariant ever stops holding — one backend instance handed to
        a second actor — this cache is the first thing to undo.**

        Args:
            name: Collection name.

        Returns:
            Weaviate collection object (optionally scoped to tenant).
        """
        tenant = self._collection_tenants.get(name) or self._tenant
        cached = self._collection_handles.get((name, tenant))
        if cached is not None:
            return cached
        col = self._client.collections.get(name)
        if tenant:
            col = col.with_tenant(tenant)
        self._collection_handles[(name, tenant)] = col
        return col


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


WEAVIATE_URL_ENV: Final[str] = "AKGENTIC_WEAVIATE_URL"
"""Environment variable naming the Weaviate cluster.

Connection settings are infrastructure, never card fields: a card persisted in a
catalog would otherwise carry a cluster URL and an API key as plain configuration.
**Exporting this is what turns Weaviate on.**
"""

WEAVIATE_API_KEY_ENV: Final[str] = "AKGENTIC_WEAVIATE_API_KEY"
"""Environment variable holding the Weaviate API key. Optional — an unauthenticated
cluster needs only the URL."""


def weaviate_url() -> str | None:
    """Return the configured Weaviate cluster URL, or ``None`` when unset.

    An exported but *empty* variable counts as unset, so a deployment template that
    always exports the name does not read as a cluster at ``""``.
    """
    import os

    return os.environ.get(WEAVIATE_URL_ENV) or None


def weaviate_api_key() -> str | None:
    """Return the configured Weaviate API key, or ``None`` when unset."""
    import os

    return os.environ.get(WEAVIATE_API_KEY_ENV) or None


def _weaviate_is_configured() -> bool:
    """Return whether a Weaviate cluster URL is exported."""
    return weaviate_url() is not None


def _require_weaviate(card_name: str) -> None:
    """Raise when Weaviate is named but no cluster URL is exported."""
    if weaviate_url():
        return
    raise ValueError(
        f"{card_name} configures backend='weaviate' but {WEAVIATE_URL_ENV} is not set. "
        f"Export {WEAVIATE_URL_ENV} (and {WEAVIATE_API_KEY_ENV} for an authenticated "
        f"cluster), or drop the backend setting to use the in-memory index."
    )


def _make_weaviate_backend(context: BackendContext) -> WeaviateBackend:
    """Build a :class:`WeaviateBackend` from the environment.

    Connection settings are read from the environment only, exactly as the Qdrant
    factory reads its own: ``VectorStoreConfig`` carries no connection field. The
    resolved pair goes to :func:`get_client`, so every backend built for one
    cluster in this process shares its client.

    Raises:
        ValueError: When :data:`WEAVIATE_URL_ENV` is not set, before the client
            cache is touched.
    """
    url = weaviate_url()
    if not url:
        raise ValueError(f"{WEAVIATE_URL_ENV} is not set; cannot build WeaviateBackend.")
    client = _weaviate_client(url, weaviate_api_key())
    return WeaviateBackend(client=client, team_id=context.team_id)


register_backend(
    BackendSpec(
        name="weaviate",
        factory=_make_weaviate_backend,
        persists_in_actor_state=False,
        selectable_as_default=True,
        is_configured=_weaviate_is_configured,
        require_configured=_require_weaviate,
    ),
    replace=True,
)
