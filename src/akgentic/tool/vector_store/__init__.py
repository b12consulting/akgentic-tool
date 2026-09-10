"""Centralised vector storage service — protocols, models, and configuration.

Re-exports all public types from ``protocol.py`` so consumers can import
directly from ``akgentic.tool.vector_store``. The backend classes are re-exported
from the ``backends`` package; this module is their one public surface.
"""

from __future__ import annotations

from akgentic.tool.vector_store.actor import (
    VS_ACTOR_NAME,
    VS_ACTOR_ROLE,
    VectorStoreActor,
    VectorStoreState,
    ensure_store_actor,
)
from akgentic.tool.vector_store.backends.inmemory import InMemoryBackend
from akgentic.tool.vector_store.backends.qdrant import QdrantBackend
from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend
from akgentic.tool.vector_store.client import ClusterKey, close_all, get_client
from akgentic.tool.vector_store.embedding_actor import (
    EmbeddingError,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingWorker,
    build_embedding_service,
    embedding_worker_name,
)
from akgentic.tool.vector_store.protocol import (
    EMBEDDING_DIMENSIONS,
    ActorStateBackend,
    CollectionStatus,
    EmbeddingProvider,
    SearchHit,
    SearchResult,
    VectorQuery,
    VectorStoreConfig,
    VectorStoreParam,
    VectorStoreService,
    default_backend,
    needs_store_actor,
    require_backend_configured,
    require_dimension_matches,
)
from akgentic.tool.vector_store.registry import (
    BackendContext,
    BackendSpec,
    available_backends,
    get_backend_spec,
    is_registered,
    register_backend,
    resolve_default_backend,
    unregister_backend,
)
from akgentic.tool.vector_store.vector import EmbeddingService, VectorEntry, VectorIndex

# ``_check_vector_search_dependencies`` is private but imported by name from six modules
# and the test suite. The redundant ``as`` alias marks it a deliberate re-export (mypy
# strict turns off implicit re-export) without promoting it into ``__all__`` — the form
# ``core/__init__.py`` uses for ``_resolve`` and ``_topological_sort``.
from akgentic.tool.vector_store.vector import (  # noqa: F401
    _check_vector_search_dependencies as _check_vector_search_dependencies,
)

__all__ = [
    "EMBEDDING_DIMENSIONS",
    "ActorStateBackend",
    "BackendContext",
    "BackendSpec",
    "ClusterKey",
    "CollectionStatus",
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingRequest",
    "EmbeddingResult",
    "EmbeddingService",
    "EmbeddingWorker",
    "InMemoryBackend",
    "QdrantBackend",
    "WeaviateBackend",
    "SearchHit",
    "SearchResult",
    "VS_ACTOR_NAME",
    "VS_ACTOR_ROLE",
    "VectorEntry",
    "VectorIndex",
    "VectorQuery",
    "VectorStoreActor",
    "VectorStoreConfig",
    "VectorStoreParam",
    "VectorStoreService",
    "VectorStoreState",
    "available_backends",
    "build_embedding_service",
    "close_all",
    "default_backend",
    "embedding_worker_name",
    "ensure_store_actor",
    "get_backend_spec",
    "get_client",
    "is_registered",
    "needs_store_actor",
    "register_backend",
    "require_backend_configured",
    "require_dimension_matches",
    "resolve_default_backend",
    "unregister_backend",
]
