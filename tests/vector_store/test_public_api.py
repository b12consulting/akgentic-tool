"""Test that vector_store.__all__ matches actual exports (AC10)."""

from __future__ import annotations

import importlib

import pytest

import akgentic.tool.vector_store as vs


class TestPublicApi:
    """Validate vector_store public API re-exports."""

    def test_all_matches_actual_exports(self) -> None:
        """Every name in __all__ must be importable from the package."""
        for name in vs.__all__:
            assert hasattr(vs, name), f"{name} listed in __all__ but not importable"

    def test_expected_names_in_all(self) -> None:
        """All expected public types are re-exported."""
        expected = {
            "ActorStateBackend",
            "BackendContext",
            "BackendSpec",
            "ClusterKey",
            "CollectionStatus",
            "EMBEDDING_DIMENSIONS",
            "EmbeddingError",
            "EmbeddingProvider",
            "EmbeddingRequest",
            "EmbeddingResult",
            "EmbeddingService",
            "EmbeddingWorker",
            "InMemoryBackend",
            "QdrantBackend",
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
            "WeaviateBackend",
            "available_backends",
            "build_embedding_service",
            "close_all",
            "ensure_store_actor",
            "default_backend",
            "embedding_worker_name",
            "get_backend_spec",
            "get_client",
            "is_registered",
            "needs_store_actor",
            "register_backend",
            "require_backend_configured",
            "require_dimension_matches",
            "resolve_default_backend",
            "unregister_backend",
        }
        assert set(vs.__all__) == expected

    @pytest.mark.parametrize(
        "name",
        [
            "WEAVIATE_URL_ENV",
            "WEAVIATE_API_KEY_ENV",
            "weaviate_url",
            "weaviate_api_key",
            "require_weaviate_configured",
        ],
    )
    def test_the_root_no_longer_carries_a_backend_s_environment(self, name: str) -> None:
        """The four Weaviate helpers moved to their backend module; the old guard is gone.

        Absent from the namespace, not only from ``__all__``: a leftover import would keep
        the name reachable from the root and pass an ``__all__``-only check.
        """
        assert name not in vs.__all__
        assert not hasattr(vs, name)

    @pytest.mark.parametrize(
        ("name", "module"),
        [
            ("InMemoryBackend", "inmemory"),
            ("WeaviateBackend", "weaviate"),
            ("QdrantBackend", "qdrant"),
        ],
    )
    def test_the_root_exports_each_backend_class_by_identity(self, name: str, module: str) -> None:
        """The root's class IS the backend module's class — identity, not ``hasattr``.

        ``hasattr`` is what let a ``try/except ImportError: X = None`` fallback hide a
        wrong import path: the name existed, and was ``None``.
        """
        backend_module = importlib.import_module(f"akgentic.tool.vector_store.backends.{module}")
        exported = getattr(vs, name)
        assert exported is not None
        assert isinstance(exported, type)
        assert exported is getattr(backend_module, name)
        assert exported.__module__ == f"akgentic.tool.vector_store.backends.{module}"

    def test_private_dependency_check_is_re_exported_but_unlisted(self) -> None:
        """Six modules import it by name, so it must resolve — without being public."""
        from akgentic.tool.vector_store.vector import _check_vector_search_dependencies

        assert vs._check_vector_search_dependencies is _check_vector_search_dependencies
        assert "_check_vector_search_dependencies" not in vs.__all__
