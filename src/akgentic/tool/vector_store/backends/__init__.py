"""The vector store's built-in backends: ``inmemory``, ``weaviate`` and ``qdrant``.

Each module implements ``VectorStoreService`` for one engine and registers itself with
``registry.register_backend`` when it is imported. ``registry._ensure_builtins`` imports
the three modules; nothing here does, so a broken optional backend cannot break the
import of another.

This package is not a public surface. Import the backend classes from
``akgentic.tool.vector_store``, which re-exports them.
"""
