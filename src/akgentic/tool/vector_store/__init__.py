"""Centralised vector storage service — protocols, models, and configuration.

Re-exports all public types from ``protocol.py`` so consumers can import
directly from ``akgentic.tool.vector_store``.
"""

from __future__ import annotations

from akgentic.tool.vector_store.actor import (
    VS_ACTOR_NAME,
    VS_ACTOR_ROLE,
    VectorStoreActor,
    VectorStoreState,
)
from akgentic.tool.vector_store.embedding_actor import (
    EmbeddingActor,
    EmbeddingError,
    EmbeddingRequest,
    EmbeddingResult,
)
from akgentic.tool.vector_store.inmemory import InMemoryBackend

try:
    from akgentic.tool.vector_store.weaviate import WeaviateBackend
except ImportError:
    WeaviateBackend = None  # type: ignore[assignment,misc]
from akgentic.tool.vector_store.protocol import (
    WEAVIATE_API_KEY_ENV,
    WEAVIATE_URL_ENV,
    CollectionConfig,
    CollectionStatus,
    EmbeddingProvider,
    SearchHit,
    SearchResult,
    VectorStoreConfig,
    VectorStoreService,
    default_backend,
    require_weaviate_configured,
    weaviate_api_key,
    weaviate_url,
)
from akgentic.tool.vector_store.tool import VectorStoreTool
from akgentic.tool.vector_store.vector import EmbeddingService, VectorEntry, VectorIndex

# ``_check_vector_search_dependencies`` is private but imported by name from six modules
# and the test suite. The redundant ``as`` alias marks it a deliberate re-export (mypy
# strict turns off implicit re-export) without promoting it into ``__all__`` — the form
# ``core/__init__.py`` uses for ``_resolve`` and ``_topological_sort``.
from akgentic.tool.vector_store.vector import (  # noqa: F401
    _check_vector_search_dependencies as _check_vector_search_dependencies,
)

__all__ = [
    "WEAVIATE_API_KEY_ENV",
    "WEAVIATE_URL_ENV",
    "CollectionConfig",
    "CollectionStatus",
    "EmbeddingActor",
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingRequest",
    "EmbeddingResult",
    "EmbeddingService",
    "InMemoryBackend",
    "WeaviateBackend",
    "SearchHit",
    "SearchResult",
    "VS_ACTOR_NAME",
    "VS_ACTOR_ROLE",
    "VectorEntry",
    "VectorIndex",
    "VectorStoreActor",
    "VectorStoreConfig",
    "VectorStoreService",
    "VectorStoreState",
    "VectorStoreTool",
    "default_backend",
    "require_weaviate_configured",
    "weaviate_api_key",
    "weaviate_url",
]
