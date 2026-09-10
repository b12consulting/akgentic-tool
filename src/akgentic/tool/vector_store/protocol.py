"""Vector store protocol definitions, data models, and configuration.

Defines the structural contracts (``VectorStoreService``, ``EmbeddingProvider``)
and Pydantic models (``VectorStoreParam``, ``SearchHit``, ``SearchResult``,
``VectorStoreConfig``) for the centralised vector storage service.
"""

from __future__ import annotations

import json
import uuid
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, runtime_checkable

from pydantic import Field

from akgentic.core.agent_config import BaseConfig
from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.tool.vector_store import registry

if TYPE_CHECKING:
    from akgentic.tool.vector_store.vector import VectorEntry


def default_backend() -> str:
    """Return the backend a collection uses when its card names none.

    Delegates to :func:`akgentic.tool.vector_store.registry.resolve_default_backend`,
    which consults every registered backend's ``is_configured()`` probe. The answer
    is the first provisioned, default-selectable backend in registration order,
    else the in-actor fallback — the backend that keeps its data in actor state and
    needs no provisioning.

    Resolved per instantiation rather than at import, so a process that exports
    a backend's variables after the module loads — a test, a late-configured
    worker — still sees it.
    """
    return registry.resolve_default_backend()


# ---------------------------------------------------------------------------
# CollectionStatus
# ---------------------------------------------------------------------------


class CollectionStatus(StrEnum):
    """Lifecycle state of a vector collection.

    **``INDEXING`` is gone and ``ERROR`` stays, and the asymmetry has a reason.**
    ``INDEXING`` used to be derived from the store's open embedding requests. Those
    no longer exist — after the embedding pipeline left the store, a write either
    lands on the turn it arrives or raises — so there is no interval left for a
    collection to be "indexing" in, and the member could not regain a meaning.
    ``ERROR`` is unassigned today but retains one: a backend-level fault that really
    does invalidate a whole collection is what it is for.

    Deleting an enum *member* is a **third** persisted-record case, distinct from the
    two this package has catalogued (a deleted field is dropped by ``extra="ignore"``;
    a deleted class breaks the ``__model__`` tag resolution outright).
    ``VectorStoreState.collection_statuses`` is persisted actor state, so a
    checkpoint written before the pipeline moved that caught a collection mid-index
    carries ``{"planning": "indexing"}`` and is now rejected on the **value** —
    the key is still declared, so ``extra="ignore"`` cannot help. That break is
    accepted in ADR-049 *Migration* on the ground that the same release already
    breaks those records harder, and the exemption does not extend to the next
    removal.
    """

    READY = "ready"
    ERROR = "error"


# ---------------------------------------------------------------------------
# VectorStoreParam
# ---------------------------------------------------------------------------

EMBEDDING_DIMENSIONS: Final[dict[str, int]] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
"""Native vector width of every embedding model this package knows.

The table is **exact** because ``EmbeddingService.embed`` never passes a
``dimensions=`` argument, so each model returns its full native width and nothing
else. Whoever adds ``dimensions=`` to ``embed()`` (Matryoshka truncation on the
``text-embedding-3-*`` models) owns widening :func:`require_dimension_matches`
from an equality into a range check at the same time.

A model absent from the table is accepted at whatever dimension the author
declares: an Azure ``embedding_model`` is a deployment name, which is routinely
not a model name, and a test names a synthetic one.
"""


class VectorStoreParam(SerializableBaseModel):
    """How a tool reaches its vector store. One object, embedded by every consumer.

    ``PlanningTool``, ``KnowledgeGraphTool`` and ``WorkspaceTool`` each carry one
    of these as a single field, so a store's six settings — backend, root,
    dimension, tenant, embedding model and embedding provider — live in one place
    and a reader learns one shape.

    **``root`` is the one field a consumer derives rather than an author writing
    it**, and it is declared here rather than on a card because it has to travel:
    ``create_collection`` records the whole param, and the store actor reads the
    backend and the root back out of that record when it builds the engine. A
    consumer with no filesystem leaves it ``None``.

    **Not a ``BaseToolParam``.** That base carries ``instructions`` and ``expose``
    because it describes a capability the model can be shown; a vector store is
    backing configuration that is exposed through no channel and has no docstring
    to append to, so a catalog rendering ``expose`` on it would be a lie.

    A payload persisted before the workspace-persistence mode was deleted may still
    carry ``persistence`` and ``workspace_path``. Neither is a field any more; this
    model declares no ``extra="forbid"``, so Pydantic's default ``extra="ignore"``
    drops them on validation. No migration is needed.
    """

    backend: str = Field(
        default_factory=default_backend,
        description=(
            "Storage backend for this collection, matched against a registered "
            "BackendSpec.name; akgentic.tool.vector_store.registry.available_backends() "
            "lists them. Defaults to whichever backend the environment has provisioned, "
            "else the fallback that keeps its data in actor state. Custom backends can "
            "be added via akgentic.tool.vector_store.registry.register_backend."
        ),
    )
    root: str | None = Field(
        default=None,
        description=(
            "Filesystem directory a file-persisting backend hangs its index off, "
            "derived at bind time by the consumer that has a filesystem — the workspace "
            "stamps its own <meta> sibling here — and ignored by every backend that has "
            "none. A value written in a catalog is inert: the consumer overrides it "
            "unconditionally, so nothing an author writes can point one tree's index at "
            "another tree's directory."
        ),
    )
    dimension: int = Field(default=1536, ge=1, description="Embedding vector dimensionality")
    tenant: str | None = Field(
        default=None,
        description=(
            "Deployment partition for a backend that supports multi-tenancy; each "
            "backend maps it onto its own partitioning, and one without any ignores it."
        ),
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Backend-native collection/connection settings passed through untouched "
            "to the backend (e.g. HNSW tuning, distance metric overrides). Ignored by "
            "backends that do not recognise a given key. Schemaless by nature: each "
            "backend defines its own keys."
        ),
    )
    embedding_model: str = Field(
        default="text-embedding-3-small", description="Embedding model identifier"
    )
    embedding_provider: Literal["openai", "azure"] = Field(
        default="openai", description="Embedding API provider"
    )


def require_dimension_matches(param: VectorStoreParam, owner: str) -> None:
    """Raise when *param* declares a dimension its embedding model cannot produce.

    A consumer card calls this at ``observer()`` time, beside
    :func:`require_backend_configured`, so a self-contradictory configuration
    fails the team's build in front of its author rather than surfacing as a
    width error on the first insert. ``VectorStoreActor.create_collection`` calls
    it again, before its own error handling, for the caller that bypasses every
    card.

    Deliberately **not** a ``model_validator``: that would fire on every
    ``model_validate`` — a restored backend snapshot, a persisted config replay —
    and turn a mismatch into an unloadable state.

    Args:
        param: The vector store configuration carried by the owner.
        owner: Who declares it, for the error message — a card class name, or an
            actor and collection.

    Raises:
        ValueError: When ``param.embedding_model`` is in
            :data:`EMBEDDING_DIMENSIONS` and ``param.dimension`` differs from its
            native width. An unknown model is accepted at any dimension.
    """
    expected = EMBEDDING_DIMENSIONS.get(param.embedding_model)
    if expected is None or expected == param.dimension:
        return
    raise ValueError(
        f"{owner} declares dimension={param.dimension} for embedding_model="
        f"'{param.embedding_model}', which produces {expected}-dimensional vectors. "
        f"Set dimension={expected}, or name the model that produces {param.dimension}."
    )


def require_backend_configured(config: VectorStoreParam, card_name: str) -> None:
    """Raise when *config* names a backend the environment has not provisioned.

    It looks up the registered backend named by ``config.backend`` and asks that
    backend's own ``require_configured`` probe. A consumer card calls this at
    ``observer()`` time so a team that names a durable store — a cluster or a
    custom backend — fails to build rather than starting up silently pointed at an
    index confined to one process. A card that asks for a cluster has asked for durable,
    shared, tenant-isolated storage; an in-memory index instead is not a
    degradation, it is the wrong answer to a question the deployment already
    settled.

    The in-actor fallback's probe is a no-op, so a card that names no backend
    (already resolved to the in-actor fallback) never raises.

    Args:
        config: The collection configuration carried by the card.
        card_name: Card class name, for the error message.

    Raises:
        ValueError: When the named backend is unknown, or is named but the
            environment has not provisioned it.
    """
    spec = registry.get_backend_spec(config.backend)
    spec.require_configured(card_name)


def needs_store_actor(param: VectorStoreParam) -> bool:
    """Return whether this param's backend needs the team's store actor in front of it.

    ``persists_in_actor_state`` answered a larger question than its name once
    suggested: not "should a mutation be snapshotted into actor state" but
    **does this backend need an actor at all** (ADR-049 Decision 1). In memory
    the actor's state *is* the database, so a store actor is what holds the
    data and one must exist. On a cluster the data lives elsewhere and an actor
    would hold nothing but a socket — there is nothing to name, look up, host,
    checkpoint or reap — so the consumer calls the backend's client directly.

    **The two questions have since come apart, and this is their disjunction.**
    A file-persisting backend keeps its index in files, so snapshotting it
    into actor state would serialise a whole numpy matrix on every mutation — yet
    one actor still has to own the in-process matrix, so an actor is required.
    ``BackendSpec.needs_actor`` is that second answer, and a backend that
    declares neither flag still gets exactly the behaviour it had.

    Both halves of the wiring ask this and must get the same answer: a consumer
    card at ``observer()`` time, deciding whether to create the store actor
    (:func:`~akgentic.tool.vector_store.actor.ensure_store_actor`), and that
    consumer's actor at ``on_start`` time, deciding whether to resolve a proxy
    or build a backend.

    Nothing in the registry changes shape for this: a third-party backend gets
    the actor-or-no-actor behaviour for free from the flag it already declares.

    Args:
        param: The vector store configuration carried by a consumer.

    Returns:
        ``True`` when the named backend stores its data in actor state, or
        declares that it needs an actor for another reason.

    Raises:
        ValueError: When ``param.backend`` names no registered backend. A card
            naming a backend nobody registered must fail the build rather than
            silently taking one of the two branches.
    """
    spec = registry.get_backend_spec(param.backend)
    return spec.persists_in_actor_state or spec.needs_actor


def resolve_store_param(value: VectorStoreParam | bool) -> VectorStoreParam | None:
    """The declared store, or ``None`` when this consumer declines one.

    A consumer card's ``vector_store`` is the author's surface and carries three
    shapes — ``True`` for "enabled, defaults everywhere", ``False`` for "no store
    at all", and an explicit param for "enabled, configured like this". Below the
    card there are only two, and this is the one place the third collapses into
    them, for the same reason :func:`needs_store_actor` is one place: two halves
    of the wiring must get the same answer (ADR-049 Decision 1).

    **``False`` becomes ``None``, and the two obvious alternatives were refused.**
    Carrying a literal ``False`` downstream would give every consumer three shapes
    to test for — a param, a ``False``, and the ``None`` its store slot already
    means — which is the state this normaliser exists to prevent. Carrying a
    ``VectorStoreParam`` with a disabled flag would be a live object meaning "not
    live": it still names a ``backend``, so ``require_backend_configured``,
    :func:`needs_store_actor`, ``ensure_store_actor``, ``create_collection`` and
    ``BackendSpec.factory`` would each have to learn to ignore an object they were
    handed, and the flag would be a further catalog-settable field an author could
    flip on a param a second card shares by ``__ref__``. ``None`` is what
    ``_vs_proxy`` and ``_resolve_store`` already mean by "no store".

    **This is deliberately not a Pydantic validator on the card.** Coercing
    ``True`` into ``VectorStoreParam()`` at validation time would resolve
    :func:`default_backend` — which ``VectorStoreParam.backend`` resolves *per
    instantiation*, from the environment — and write the build environment's
    backend into a stored record whose author wrote ``true``. Catalogs are
    routinely promoted between tiers, so that record would then name a backend the
    next tier has not provisioned. The card keeps the author's declaration
    verbatim and the collapse happens here, at the point of use.

    Args:
        value: What the card's ``vector_store`` field holds.

    Returns:
        The declared param, a fresh default one for ``True``, or ``None`` for
        ``False``.
    """
    if value is True:
        return VectorStoreParam()
    if value is False:
        return None
    return value


# ---------------------------------------------------------------------------
# path_prefix validation — one constant, one sentence, both backends
# ---------------------------------------------------------------------------

PATH_PREFIX_WILDCARDS: Final[str] = "*?"
"""Characters a ``path_prefix`` may not contain, on any backend.

Both are legal in a POSIX filename. One registered backend compiles a prefix into
a pattern operator in which ``*`` and ``?`` are wildcards and no escape exists;
another compares strings and reads them literally. The same query would then mean
two different things depending on where the collection happens to live — and on
``remove()`` that is sharp rather than academic: a ``*`` widens a deletion on the
pattern-matching backend and narrows it to nothing on the literal one. The rule
protects every backend; the concrete evidence sits beside the code that builds
the pattern.

They live here, next to the protocol every backend implements, so the backends
cannot drift apart (ADR-045 §5).
"""

PATH_PREFIX_REJECTED: Final[str] = (
    "A path_prefix cannot contain '*' or '?': they are wildcards on one vector "
    "backend and literal characters on the other, so the same filter would mean "
    "two different things. Use a shorter prefix without them."
)
"""The one sentence a rejected prefix is refused with, wherever it is refused."""

SEMANTIC_DISABLED: Final[str] = (
    "Semantic search is switched off for this tool (vector_store=False). "
    "Keyword and hybrid search still work."
)
"""What a semantic-only search answers on a tool that declined a store.

The precedent is :data:`PATH_PREFIX_REJECTED` directly above: a capability that
cannot serve a request answers the model in a sentence rather than raising or
returning an empty result. ``mode="vector"`` on a store-less tool has nothing to
score, and an empty result would be read as "the graph holds nothing about this"
— an assertion about the data rather than about the configuration.

One constant, two readers, so the two tools cannot drift: ``search_planning`` on
``PlanActor`` returns it as its only line, and ``KnowledgeGraphTool``'s search
closure returns it in place of asking the actor at all.

It is only ever the answer for a store that was **declined**. A store that was
configured and then failed to build keeps returning an empty result, so a real
misconfiguration is not swallowed under a reassuring sentence (ADR-049 Decision 5).
"""


def check_path_prefix(path_prefix: str | None) -> None:
    """Raise when *path_prefix* carries a character the two backends read differently.

    Called at the top of ``search()`` and ``remove()`` on **both** backends, which
    is where the divergence physically lives. Callers that answer an agent rather
    than a program — ``workspace_rag_search`` — check the same constant and return
    :data:`PATH_PREFIX_REJECTED` as a sentence instead of raising; the message is
    the same either way.

    Args:
        path_prefix: The prefix to validate. ``None`` and ``""`` filter nothing
            and are always accepted.

    Raises:
        ValueError: When the prefix contains ``*`` or ``?``.
    """
    if path_prefix and any(character in path_prefix for character in PATH_PREFIX_WILDCARDS):
        raise ValueError(PATH_PREFIX_REJECTED)


# ---------------------------------------------------------------------------
# Per-collection scoping — one declaration, both cluster backends
# ---------------------------------------------------------------------------

SHARED_COLLECTIONS: Final[frozenset[str]] = frozenset({"workspace_chunks"})
"""Collections whose rows are read by every team that indexes the same tree.

A workspace collection holds the chunks of a *filesystem path*, not of a team, so
two teams pointed at one tree must see the same rows. Every other collection in
this package — ``planning``, ``knowledge_graph`` — holds rows that belong to one
team and keeps its team predicate.

**The declaration is by name, and it lives here rather than on a model, on a
constructor or in per-instance state.** Three alternatives were weighed:

- **A field on ``VectorStoreParam``** is catalog-settable, so an author could write
  ``team_scoped: false`` on ``id_planning`` and silently reproduce the
  cross-tenant destruction that ADR-046 exists to prevent, with no error and no
  log line.
- **A keyword on ``create_collection``, kept per instance**, is vacuous exactly
  where it is needed: a sweeper builds an administrative backend that has created
  no collection, so its per-instance map is empty when
  :meth:`delete_by_team`'s guard has to bite.
- **A flag on the backend constructor or ``BackendContext``** puts the declaration
  on the *wiring* rather than on the collection, and one backend instance serves
  several collections through the store actor's per-backend cache.

The one cost is a duplicated string: ``RAG_COLLECTION`` is declared in the
workspace package, which ``vector_store`` may not import. A spec in
``tests/vector_store/test_protocol.py`` imports that constant and asserts
membership, so renaming it reddens a test rather than silently un-sharing the
collection.

Deliberately a ``frozenset`` and deliberately not extensible at runtime: a
mutable registry anyone may add to is the catalog hazard in another shape, plus
an import-order trap for a bare sweeper script that imports no consumer. A
third-party shared collection, if ever needed, wants a ``BackendSpec``-style
registration with the same fail-loud discipline (ADR-049 open item).
"""

SHARED_SCOPE_REQUIRED: Final[str] = (
    "Collection '{collection}' is shared across teams, so a scope is the only boundary it "
    "has. Pass scope= to search or remove."
)
"""The one sentence an unscoped query against a shared collection is refused with."""


def collection_is_team_scoped(collection: str) -> bool:
    """Whether reads and removals on *collection* filter on the caller's own team.

    Consulted by both cluster backends when they build a query predicate and when
    they refuse a team-wide deletion. An unregistered name is team-scoped: the
    safe answer for a collection nobody declared is the isolating one.

    Args:
        collection: The collection name a query or a removal names.

    Returns:
        ``True`` unless *collection* is listed in :data:`SHARED_COLLECTIONS`.
    """
    return collection not in SHARED_COLLECTIONS


def check_shared_scope(collection: str, scope: str | None) -> None:
    """Raise when a shared collection is queried or modified without a scope.

    **The boundary moves; it does not disappear.** ADR-046 §D1 rejected a
    ``team_id`` argument on the protocol on the ground that "a boundary a caller
    can omit is a boundary that will be omitted". Dropping the team leg from a
    shared collection leaves ``scope`` as its only boundary, and ``scope`` is an
    optional argument — so an omitted one would read or delete every workspace's
    chunks on the cluster, a strictly worse version of the bug ADR-046 fixed.
    Making it mandatory here gives that job an owner.

    Called at the top of ``search()`` and ``remove()`` on **all three** backends,
    on the line after :func:`check_path_prefix`. In memory there is no team
    predicate to lose, but one team may hold two ``WorkspaceTool`` cards on two
    trees, so an unscoped query crosses a boundary there as well — and putting it
    on all three is what lets the suite exercise the rule without a cluster.

    Args:
        collection: The collection being queried or modified.
        scope: The partition the caller named, or ``None``.

    Raises:
        ValueError: When *collection* is shared and *scope* is ``None``.
    """
    if scope is None and not collection_is_team_scoped(collection):
        raise ValueError(SHARED_SCOPE_REQUIRED.format(collection=collection))


# ---------------------------------------------------------------------------
# Row identity — one derivation, both cluster backends
# ---------------------------------------------------------------------------

STABLE_OBJECT_ID_NAMESPACE: Final[uuid.UUID] = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")
"""The ``uuid5`` namespace every cluster row id is derived under.

Its value predates this module's ownership of it and **must never change**: every
point already stored on a cluster was written under it, so a new namespace would
orphan all of them and double every later re-add.
"""


def stable_object_id(team_id: str | None, tenant: str | None, ref_id: str) -> str:
    """The id a row for *ref_id* is stored under — the same for the same triple, always.

    A deterministic ``uuid5`` over ``[team_id or "", tenant or "", ref_id]``,
    serialised as compact ASCII JSON so no separator inside a value can make two
    triples collide. Both cluster backends store under it, and both clients
    **replace** a row whose id already exists — so a re-add overwrites instead of
    doubling, within one actor's lifetime and across two.

    **The team is inside the id, which is what keeps two teams' rows apart on a
    team-scoped collection**: equal ``ref_id``s from two teams are two rows. On a
    shared collection the writer carries no team, the first element is ``""``, and
    every lifetime of the tree writes the one row. The tenant is inside it too,
    for the same reason one level down.

    It lives here, beside :func:`check_path_prefix` and :data:`SHARED_COLLECTIONS`,
    for their reason: so the two cluster backends cannot drift apart on what makes
    a row the same row.

    Args:
        team_id: The writing backend's team, or ``None`` for a writer with none.
        tenant: The tenant the row is written under, or ``None``.
        ref_id: The caller's own identifier for the row.

    Returns:
        The id, as a UUID string.
    """
    identity = json.dumps(
        [team_id or "", tenant or "", ref_id], ensure_ascii=True, separators=(",", ":")
    )
    return str(uuid.uuid5(STABLE_OBJECT_ID_NAMESPACE, identity))


def row_object_id(collection: str, team_id: str | None, tenant: str | None, ref_id: str) -> str:
    """The id a row is stored under, with the team dropped on a shared collection.

    **The rule is per collection, not per caller**, and that is the whole reason
    this function exists rather than each backend deciding. A shared collection
    holds the chunks of a *filesystem path*: two teams indexing one tree must
    write **one** row per chunk, so the writer's team may not enter the identity
    — otherwise the same chunk is two points and every search returns it twice.
    A team-scoped collection is the opposite case and keeps the team inside the
    id, which is what stops one team's ``ref_id`` from overwriting another's.

    The workspace used to reach this the long way round: its backend was built
    with ``team_id=None``, so the first element was always ``""`` whoever wrote
    it. That was a property of one wiring rather than of the collection, and it
    ended the moment the card started passing its real team — which it must, so
    the rows can still be *filtered* by team. Dropping the team here instead
    moves the rule to where it can be stated once and read by both cluster
    backends, which is the same reason :data:`SHARED_COLLECTIONS` and
    :func:`check_path_prefix` live here.

    The team is still **stamped** on the object either way: filtering and
    sweeping need it, and only the identity derivation is affected.

    Args:
        collection: The collection the row is written into.
        team_id: The writing backend's team, or ``None`` for a writer with none.
        tenant: The tenant the row is written under, or ``None``.
        ref_id: The caller's own identifier for the row.

    Returns:
        The id, as a UUID string.
    """
    scoped_team = team_id if collection_is_team_scoped(collection) else None
    return stable_object_id(scoped_team, tenant, ref_id)


# ---------------------------------------------------------------------------
# SearchHit
# ---------------------------------------------------------------------------


class VectorQuery(SerializableBaseModel):
    """Optional per-call knobs that refine a similarity search.

    Threaded through :meth:`VectorStoreService.search` so a caller can tune a
    query without changing the backend or the collection. Every field is
    optional; a backend applies what it understands and ignores the rest, so the
    same query stays portable across backends of differing capability.

    Attributes:
        filters: Exact-match metadata constraints, e.g. ``{"ref_type": "entity"}``.
            Keys name stored properties (``ref_type`` / ``ref_id`` / ``text`` on
            the built-in schema); a value may be a scalar or a list (match-any).
        score_threshold: Drop hits whose raw cosine score is below this value.
        params: Backend-native query parameters passed through untouched — HNSW
            ``ef``, an exact-search toggle, a certainty floor — recognised only
            by the backend that defines them.
    """

    filters: dict[str, Any] | None = Field(
        default=None,
        description="Exact-match metadata filters, e.g. {'ref_type': 'entity'}.",
    )
    score_threshold: float | None = Field(
        default=None, description="Drop hits scoring below this cosine value."
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Backend-native query parameters passed through untouched.",
    )


class SearchHit(SerializableBaseModel):
    """A single result from a vector similarity search.

    References the source object via ``ref_type`` and ``ref_id`` with the
    original text and cosine similarity ``score``.
    """

    ref_type: str = Field(description="Domain-specific type label for the referenced object")
    ref_id: str = Field(description="Identifier of the referenced object")
    text: str = Field(description="The text that was embedded")
    score: float = Field(description="Cosine similarity score")
    scope: str | None = Field(
        default=None,
        description=(
            "Partition the entry belongs to within the collection — for the workspace, "
            "the workspace id. None for a producer that does not partition."
        ),
    )
    path: str | None = Field(
        default=None,
        description="Source path within the scope, filterable by prefix. None when there is none.",
    )
    ordinal: int | None = Field(
        default=None,
        description="Position of this chunk within its source, for ordering reassembly.",
    )


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


class SearchResult(SerializableBaseModel):
    """Aggregated search response from the vector store.

    Contains the ranked list of ``SearchHit`` items together with collection
    status metadata.
    """

    hits: list[SearchHit] = Field(description="Ranked search results")
    status: CollectionStatus = Field(description="Current collection lifecycle state")


# ---------------------------------------------------------------------------
# EmbeddingProvider (Protocol)
# ---------------------------------------------------------------------------


class EmbeddingProvider(Protocol):
    """Structural contract for embedding text into vectors.

    Any class that implements an ``embed`` method with the correct signature
    satisfies this protocol via structural subtyping.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts and return one vector per input.

        Args:
            texts: List of strings to embed.

        Returns:
            List of float vectors, one per input text.
        """
        ...


# ---------------------------------------------------------------------------
# VectorStoreService (Protocol)
# ---------------------------------------------------------------------------


class VectorStoreService(Protocol):
    """Structural contract for a centralised vector storage backend.

    Implementations manage named collections, handle ingestion, removal,
    and similarity search without exposing backend details.
    """

    def create_collection(self, name: str, config: VectorStoreParam) -> None:
        """Create or reconfigure a named collection.

        Args:
            name: Unique collection identifier.
            config: Vector store configuration for the collection.
        """
        ...

    def add(self, collection: str, entries: list[VectorEntry]) -> None:
        """Ingest embedding entries into a collection.

        Args:
            collection: Target collection name.
            entries: List of vector entries to store.
        """
        ...

    def remove(
        self,
        collection: str,
        ref_ids: list[str],
        scope: str | None = None,
        path_prefix: str | None = None,
    ) -> None:
        """Remove entries from a collection by reference ID.

        ``scope`` and ``path_prefix`` narrow the removal further: an entry is removed
        only when it matches the ref-id list **and** every predicate given. Both
        default to ``None``, which filters nothing.

        Args:
            collection: Target collection name.
            ref_ids: List of reference IDs to remove.
            scope: Restrict removal to entries carrying this ``scope``.
            path_prefix: Restrict removal to entries whose ``path`` starts with this.

        Raises:
            ValueError: When ``path_prefix`` contains ``*`` or ``?`` — see
                :func:`check_path_prefix`.
        """
        ...

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

        Both predicates are applied **before** ``top_k`` is taken, so a scoped search
        returns a full ``top_k`` of its own entries rather than a short set whose
        budget was spent on entries belonging to another scope.

        Args:
            collection: Target collection name.
            query_vector: Query embedding vector.
            top_k: Maximum number of results to return.
            scope: Restrict the search to entries carrying this ``scope``.
            path_prefix: Restrict the search to entries whose ``path`` starts with this.
            query: Optional refinement (filters, score threshold, backend-native
                params). ``None`` runs a plain top-k similarity search.

        Returns:
            Search results with hits and collection status.

        Raises:
            ValueError: When ``path_prefix`` contains ``*`` or ``?`` — see
                :func:`check_path_prefix`.
        """
        ...


# ---------------------------------------------------------------------------
# ActorStateBackend (Protocol)
# ---------------------------------------------------------------------------


@runtime_checkable
class ActorStateBackend(Protocol):
    """Persistence contract for backends stored in ``VectorStoreState``.

    A backend whose :class:`~akgentic.tool.vector_store.registry.BackendSpec`
    sets ``persists_in_actor_state=True`` must implement this protocol in
    addition to :class:`VectorStoreService`.
    """

    def get_state(self) -> dict[str, Any]:
        """Return a serialisable snapshot of the backend."""
        ...

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore a snapshot previously returned by :meth:`get_state`."""
        ...


# ---------------------------------------------------------------------------
# VectorStoreConfig
# ---------------------------------------------------------------------------


class VectorStoreConfig(BaseConfig):
    """Configuration for the vector store actor: its name and role, nothing more.

    It declares no field beyond ``BaseConfig``'s. The two embedding fields and the
    two cluster connection fields are gone: the embedding fields had been inert
    since the embedding pipeline moved to the consumers, and the connection fields
    had no writer once the card that set them was deleted. A backend reads its
    deployment from the environment, through its own factory.

    A persisted config that still carries any of the four keys loads unchanged:
    Pydantic's default ``extra="ignore"`` drops an undeclared key, so a removed
    *field* costs nothing (unlike a removed *class*, which the
    ``SerializableBaseModel`` before-validator cannot resolve — see
    :class:`~akgentic.tool.vector_store.actor.PendingRequest`). The value a stored
    record carried is not read.

    The class itself stays here, under this name: stored ``StartMessage`` records
    tag ``akgentic.tool.vector_store.protocol.VectorStoreConfig``.
    """
