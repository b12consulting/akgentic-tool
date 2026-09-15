"""The vector store's built-in backends: ``inmemory``, ``local``, ``weaviate`` and ``qdrant``.

Each module implements ``VectorStoreService`` for one engine and registers itself with
``registry.register_backend`` when it is imported. This package imports none of them: the
package root imports them all to re-export their classes, and ``registry._ensure_builtins``
imports them again, idempotently. The root's imports are unguarded, so a backend module that
fails to import fails the whole package. A missing vendor client does not: each module imports
its client lazily, so every class exists without ``weaviate-client`` or ``qdrant-client``.

This package is not a public surface. Import the backend classes from
``akgentic.tool.vector_store``, which re-exports them.
"""
