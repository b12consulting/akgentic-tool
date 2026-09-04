"""Vector store protocol definitions, data models, and configuration.

Defines the structural contracts (``VectorStoreService``, ``EmbeddingProvider``)
and Pydantic models (``CollectionConfig``, ``SearchHit``, ``SearchResult``,
``VectorStoreConfig``) for the centralised vector storage service.
"""

from __future__ import annotations

import os
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Literal, Protocol

from pydantic import Field

from akgentic.core.agent_config import BaseConfig
from akgentic.core.utils.serializer import SerializableBaseModel

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


def default_backend() -> Literal["inmemory", "weaviate"]:
    """Return the backend a collection uses when its card names none.

    The environment decides: a cluster URL means Weaviate is deployed, and a
    collection that expresses no preference should land there rather than in a
    process-local index that disappears with the actor.

    Resolved per instantiation rather than at import, so a process that exports the
    variable after the module loads — a test, a late-configured worker — still sees it.
    """
    return "weaviate" if weaviate_url() else "inmemory"


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
# CollectionConfig
# ---------------------------------------------------------------------------


class CollectionConfig(SerializableBaseModel):
    """Configuration for a single vector collection.

    Controls the embedding dimensionality and storage backend for the collection.

    A payload persisted before the workspace-persistence mode was deleted may still
    carry ``persistence`` and ``workspace_path``. Neither is a field any more; this
    model declares no ``extra="forbid"``, so Pydantic's default ``extra="ignore"``
    drops them on validation. No migration is needed.
    """

    dimension: int = Field(default=1536, ge=1, description="Embedding vector dimensionality")
    backend: Literal["inmemory", "weaviate"] = Field(
        default_factory=default_backend,
        description=(
            "Storage backend for this collection. Defaults to 'weaviate' when "
            f"{WEAVIATE_URL_ENV} names a cluster, otherwise 'inmemory'."
        ),
    )
    tenant: str | None = Field(
        default=None,
        description="Weaviate tenant ID for multi-tenancy (maps to workspace/team ID)",
    )


def require_weaviate_configured(config: CollectionConfig, card_name: str) -> None:
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


# ---------------------------------------------------------------------------
# SearchHit
# ---------------------------------------------------------------------------


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

    def create_collection(self, name: str, config: CollectionConfig) -> None:
        """Create or reconfigure a named collection.

        Args:
            name: Unique collection identifier.
            config: Collection configuration.
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
        """
        ...

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        scope: str | None = None,
        path_prefix: str | None = None,
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

        Returns:
            Search results with hits and collection status.
        """
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
