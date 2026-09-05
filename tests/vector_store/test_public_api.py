"""Test that vector_store.__all__ matches actual exports (AC10)."""

from __future__ import annotations

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
            "CollectionConfig",
            "CollectionStatus",
            "EmbeddingActor",
            "EmbeddingCompleted",
            "EmbeddingError",
            "EmbeddingProvider",
            "EmbeddingRequest",
            "EmbeddingResult",
            "EmbeddingService",
            "InMemoryBackend",
            "PendingRequest",
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
            "WeaviateBackend",
            "WEAVIATE_API_KEY_ENV",
            "WEAVIATE_URL_ENV",
            "default_backend",
            "require_weaviate_configured",
            "weaviate_api_key",
            "weaviate_url",
        }
        assert set(vs.__all__) == expected

    def test_private_dependency_check_is_re_exported_but_unlisted(self) -> None:
        """Six modules import it by name, so it must resolve — without being public."""
        from akgentic.tool.vector_store.vector import _check_vector_search_dependencies

        assert vs._check_vector_search_dependencies is _check_vector_search_dependencies
        assert "_check_vector_search_dependencies" not in vs.__all__
