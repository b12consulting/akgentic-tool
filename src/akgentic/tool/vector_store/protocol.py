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
    """Lifecycle state of a vector collection."""

    READY = "ready"
    INDEXING = "indexing"
    ERROR = "error"


# ---------------------------------------------------------------------------
# CollectionConfig
# ---------------------------------------------------------------------------


class CollectionConfig(SerializableBaseModel):
    """Configuration for a single vector collection.

    Controls the embedding dimensionality, storage backend, and persistence
    strategy for the collection.
    """

    dimension: int = Field(default=1536, ge=1, description="Embedding vector dimensionality")
    backend: Literal["inmemory", "weaviate"] = Field(
        default_factory=default_backend,
        description=(
            "Storage backend for this collection. Defaults to 'weaviate' when "
            f"{WEAVIATE_URL_ENV} names a cluster, otherwise 'inmemory'."
        ),
    )
    persistence: Literal["actor_state", "workspace"] = Field(
        default="actor_state", description="Persistence mode (inmemory backend only)"
    )
    workspace_path: str | None = Field(
        default=None, description="Filesystem path when persistence is workspace"
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

    def remove(self, collection: str, ref_ids: list[str]) -> None:
        """Remove entries from a collection by reference ID.

        Args:
            collection: Target collection name.
            ref_ids: List of reference IDs to remove.
        """
        ...

    def search(
        self, collection: str, query_vector: list[float], top_k: int
    ) -> SearchResult:
        """Search a collection by cosine similarity.

        Args:
            collection: Target collection name.
            query_vector: Query embedding vector.
            top_k: Maximum number of results to return.

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
