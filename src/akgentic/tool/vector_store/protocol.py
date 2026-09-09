"""Vector store protocol definitions, data models, and configuration.

Defines the structural contracts (``VectorStoreService``, ``EmbeddingProvider``)
and Pydantic models (``VectorStoreParam``, ``SearchHit``, ``SearchResult``,
``VectorStoreConfig``) for the centralised vector storage service.
"""

from __future__ import annotations

import os
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, runtime_checkable

from pydantic import Field

from akgentic.core.agent_config import BaseConfig
from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.tool.vector_store import registry

if TYPE_CHECKING:
    from akgentic.tool.vector_store.vector import VectorEntry


# ---------------------------------------------------------------------------
# Weaviate deployment, read from the environment
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
    return os.environ.get(WEAVIATE_URL_ENV) or None


def weaviate_api_key() -> str | None:
    """Return the configured Weaviate API key, or ``None`` when unset."""
    return os.environ.get(WEAVIATE_API_KEY_ENV) or None


def default_backend() -> str:
    """Return the backend a collection uses when its card names none.

    Delegates to :func:`akgentic.tool.vector_store.registry.resolve_default_backend`,
    which consults every registered backend's ``is_configured()`` probe. With
    only the built-in backends this preserves the historical rule — ``weaviate``
    when a cluster URL is set, ``inmemory`` otherwise — while letting a
    registered external backend (e.g. Qdrant) also claim the default when it,
    and no earlier-registered backend, is the one the environment provisioned.

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

    ``INDEXING`` is derived, not assigned: a collection is ``INDEXING`` exactly while
    at least one embedding request is open against it, and returns to ``READY`` when
    the last one settles — whether that request succeeded or failed.

    **``ERROR`` is no longer reachable from a single failed batch.** It used to be:
    one failing ``EmbeddingActor`` marked the whole collection ``ERROR`` and discarded
    every other request's pending entries with it. A failure is now reported to the
    caller that asked for the batch, through ``EmbeddingCompleted.error``, and leaves
    every concurrent request untouched. Nothing in this package assigns ``ERROR``
    today; the member remains for a backend-level fault that really does invalidate a
    whole collection.
    """

    READY = "ready"
    INDEXING = "indexing"
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
    of these as a single field, so a store's five settings — backend, dimension,
    tenant, embedding model and embedding provider — live in one place and a
    reader learns one shape.

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
            "BackendSpec.name. Built-ins: 'inmemory' and 'weaviate' (plus 'qdrant' "
            "when akgentic-tool[qdrant] is installed). Defaults to whichever backend "
            "the environment has provisioned, else 'inmemory'. Custom backends can be "
            "added via akgentic.tool.vector_store.registry.register_backend."
        ),
    )
    dimension: int = Field(default=1536, ge=1, description="Embedding vector dimensionality")
    tenant: str | None = Field(
        default=None,
        description="Weaviate tenant ID for multi-tenancy (maps to workspace/team ID)",
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


def require_weaviate_configured(config: VectorStoreParam, card_name: str) -> None:
    """Raise when *config* asks for Weaviate and the environment has no cluster.

    Called by a consumer card at ``observer()`` time, so the team fails to build
    rather than starting up silently pointed at a process-local index. A card that
    asks for Weaviate has asked for durable, shared, tenant-isolated storage; giving
    it an in-memory index instead is not a degradation, it is the wrong answer to a
    question the deployment already settled.

    A card that names no backend never reaches here: :func:`default_backend` has
    already resolved it to ``inmemory`` in that environment.

    Args:
        config: The collection configuration carried by the card.
        card_name: Card class name, for the error message.

    Raises:
        ValueError: When ``config.backend == "weaviate"`` and no cluster URL is set.
    """
    if config.backend != "weaviate" or weaviate_url():
        return
    raise ValueError(
        f"{card_name} configures backend='weaviate' but {WEAVIATE_URL_ENV} is not set. "
        f"Export {WEAVIATE_URL_ENV} (and {WEAVIATE_API_KEY_ENV} for an authenticated "
        f"cluster), or drop the backend setting to use the in-memory index."
    )


def require_backend_configured(config: VectorStoreParam, card_name: str) -> None:
    """Raise when *config* names a backend the environment has not provisioned.

    The backend-agnostic generalisation of :func:`require_weaviate_configured`:
    it looks up the registered backend named by ``config.backend`` and delegates
    to that backend's ``require_configured`` probe. A consumer card calls this at
    ``observer()`` time so a team that names a durable store — Weaviate, Qdrant,
    a custom backend — fails to build rather than starting up silently pointed at
    a process-local index.

    The in-memory backend's probe is a no-op, so a card that names no backend
    (already resolved to ``inmemory``) never raises.

    Args:
        config: The collection configuration carried by the card.
        card_name: Card class name, for the error message.

    Raises:
        ValueError: When the named backend is unknown, or is named but the
            environment has not provisioned it.
    """
    spec = registry.get_backend_spec(config.backend)
    spec.require_configured(card_name)


# ---------------------------------------------------------------------------
# path_prefix validation — one constant, one sentence, both backends
# ---------------------------------------------------------------------------

PATH_PREFIX_WILDCARDS: Final[str] = "*?"
"""Characters a ``path_prefix`` may not contain, on either backend.

Both are legal in a POSIX filename and both are wildcards in Weaviate's ``Like``
operator, which is what ``WeaviateBackend`` builds a prefix filter from; the
in-memory backend uses ``str.startswith`` and treats them literally. The v4
filter API offers no escape, so the same query would mean two different things
depending on where the collection happens to live — and on ``remove()`` that is
sharp rather than academic: a ``*`` widens a deletion on Weaviate and narrows it
to nothing in memory.

They live here, next to the protocol both backends implement, so the two cannot
drift apart (ADR-045 §5).
"""

PATH_PREFIX_REJECTED: Final[str] = (
    "A path_prefix cannot contain '*' or '?': they are wildcards on one vector "
    "backend and literal characters on the other, so the same filter would mean "
    "two different things. Use a shorter prefix without them."
)
"""The one sentence a rejected prefix is refused with, wherever it is refused."""


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
    indexing_pending: int = Field(
        default=0, ge=0, description="Number of entries still being indexed"
    )


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
    """Configuration for the vector store actor.

    Specifies the embedding model, provider, and optional Weaviate connection
    details.
    """

    embedding_model: str = Field(
        default="text-embedding-3-small", description="Embedding model identifier"
    )
    embedding_provider: Literal["openai", "azure"] = Field(
        default="openai", description="Embedding API provider"
    )
    weaviate_url: str | None = Field(
        default=None, description="Weaviate cluster URL"
    )
    weaviate_api_key: str | None = Field(
        default=None, description="Weaviate API key"
    )
