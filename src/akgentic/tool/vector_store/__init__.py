"""Centralised vector storage service — protocols, models, and configuration.

Re-exports all public types from ``protocol.py`` so consumers can import
directly from ``akgentic.tool.vector_store``.
"""

from __future__ import annotations

from akgentic.tool.vector_store.inmemory import InMemoryBackend
from akgentic.tool.vector_store.protocol import (
    CollectionConfig,
    CollectionStatus,
    EmbeddingProvider,
    SearchHit,
    SearchResult,
    VectorStoreConfig,
    VectorStoreService,
)

__all__ = [
    "CollectionConfig",
    "CollectionStatus",
    "EmbeddingProvider",
    "InMemoryBackend",
    "SearchHit",
    "SearchResult",
    "VectorStoreConfig",
    "VectorStoreService",
]
