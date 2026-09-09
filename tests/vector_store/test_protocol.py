"""Unit tests for akgentic.tool.vector_store.protocol models and protocols."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from akgentic.tool.core.params import BaseToolParam
from akgentic.tool.vector_store.protocol import (
    EMBEDDING_DIMENSIONS,
    SHARED_COLLECTIONS,
    CollectionStatus,
    EmbeddingProvider,
    SearchHit,
    SearchResult,
    VectorStoreConfig,
    VectorStoreParam,
    VectorStoreService,
    check_shared_scope,
    collection_is_team_scoped,
    require_dimension_matches,
)

# ---------------------------------------------------------------------------
# CollectionStatus enum tests (AC3)
# ---------------------------------------------------------------------------


class TestCollectionStatus:
    """Tests for CollectionStatus StrEnum."""

    def test_has_exactly_two_values(self) -> None:
        assert len(CollectionStatus) == 2

    def test_ready_value(self) -> None:
        assert CollectionStatus.READY == "ready"
        assert str(CollectionStatus.READY) == "ready"

    def test_error_value(self) -> None:
        assert CollectionStatus.ERROR == "error"
        assert str(CollectionStatus.ERROR) == "error"

    def test_values_list(self) -> None:
        values = {s.value for s in CollectionStatus}
        assert values == {"ready", "error"}

    def test_indexing_is_gone_and_error_survives(self) -> None:
        """The member nothing could assign again is deleted; the one with a meaning stays.

        ``INDEXING`` was derived from the store's open embedding requests. Those no
        longer exist, so there is no interval left for a collection to be indexing
        in. ``ERROR`` is unassigned today but still means a backend-level fault that
        invalidates a whole collection.
        """
        assert "INDEXING" not in CollectionStatus.__members__
        assert "ERROR" in CollectionStatus.__members__
        assert list(CollectionStatus) == [CollectionStatus.READY, CollectionStatus.ERROR]


# ---------------------------------------------------------------------------
# VectorStoreParam tests (AC2, AC8)
# ---------------------------------------------------------------------------


class TestVectorStoreParam:
    """Tests for VectorStoreParam serialization and defaults."""

    def test_default_values(self) -> None:
        cfg = VectorStoreParam()
        assert cfg.dimension == 1536
        assert cfg.backend == "inmemory"
        assert cfg.tenant is None

    def test_round_trip_serialization_defaults(self) -> None:
        cfg = VectorStoreParam()
        data = cfg.model_dump()
        restored = VectorStoreParam.model_validate(data)
        assert restored == cfg

    def test_non_default_values(self) -> None:
        cfg = VectorStoreParam(
            dimension=768,
            backend="weaviate",
            tenant="team-42",
        )
        assert cfg.dimension == 768
        assert cfg.backend == "weaviate"
        assert cfg.tenant == "team-42"

    def test_round_trip_serialization_non_defaults(self) -> None:
        cfg = VectorStoreParam(
            dimension=768,
            backend="weaviate",
            tenant="team-42",
        )
        data = cfg.model_dump()
        restored = VectorStoreParam.model_validate(data)
        assert restored == cfg

    def test_dimension_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            VectorStoreParam(dimension=0)

    def test_dimension_rejects_negative(self) -> None:
        with pytest.raises(ValidationError):
            VectorStoreParam(dimension=-1)

    def test_six_fields_in_order_with_their_defaults(self) -> None:
        """The whole shape: what every consumer carries when it names nothing."""
        assert list(VectorStoreParam.model_fields) == [
            "backend",
            "dimension",
            "tenant",
            "params",
            "embedding_model",
            "embedding_provider",
        ]
        param = VectorStoreParam()
        assert {name: getattr(param, name) for name in VectorStoreParam.model_fields} == {
            "backend": "inmemory",
            "dimension": 1536,
            "tenant": None,
            "params": {},
            "embedding_model": "text-embedding-3-small",
            "embedding_provider": "openai",
        }

    def test_embedding_model_round_trips(self) -> None:
        cfg = VectorStoreParam(dimension=3072, embedding_model="text-embedding-3-large")
        restored = VectorStoreParam.model_validate(cfg.model_dump())
        assert restored == cfg
        assert restored.embedding_model == "text-embedding-3-large"

    def test_embedding_provider_round_trips(self) -> None:
        cfg = VectorStoreParam(embedding_provider="azure", embedding_model="my-deployment")
        restored = VectorStoreParam.model_validate(cfg.model_dump())
        assert restored == cfg
        assert restored.embedding_provider == "azure"

    def test_every_field_non_default_round_trips(self) -> None:
        cfg = VectorStoreParam(
            backend="weaviate",
            dimension=3072,
            tenant="team-42",
            params={"distance": "cosine"},
            embedding_model="text-embedding-3-large",
            embedding_provider="azure",
        )
        assert VectorStoreParam.model_validate(cfg.model_dump()) == cfg

    def test_embedding_provider_is_closed(self) -> None:
        with pytest.raises(ValidationError):
            VectorStoreParam(embedding_provider="cohere")  # type: ignore[arg-type]

    def test_a_pre_story_payload_still_validates(self) -> None:
        """A persisted three-key record gets the two embedding defaults."""
        cfg = VectorStoreParam.model_validate(
            {"dimension": 768, "backend": "inmemory", "tenant": "t"}
        )
        assert cfg.dimension == 768
        assert cfg.embedding_model == "text-embedding-3-small"
        assert cfg.embedding_provider == "openai"

    def test_a_payload_carrying_deleted_keys_still_validates(self) -> None:
        """No ``extra="forbid"``: the deleted persistence keys are dropped, not refused."""
        cfg = VectorStoreParam.model_validate(
            {"persistence": "workspace", "workspace_path": "/tmp/x", "dimension": 8}
        )
        assert cfg.dimension == 8
        assert "persistence" not in cfg.model_dump()
        assert "workspace_path" not in cfg.model_dump()

    def test_it_is_not_a_capability_param(self) -> None:
        """Backing configuration is exposed through no channel: no expose, no instructions."""
        assert not issubclass(VectorStoreParam, BaseToolParam)
        assert "expose" not in VectorStoreParam.model_fields
        assert "instructions" not in VectorStoreParam.model_fields


# ---------------------------------------------------------------------------
# require_dimension_matches
# ---------------------------------------------------------------------------


class TestRequireDimensionMatches:
    """A known model's width is exact; an unknown model is trusted."""

    def test_the_table_names_the_three_models(self) -> None:
        assert EMBEDDING_DIMENSIONS == {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    def test_known_model_with_matching_dimension_passes(self) -> None:
        require_dimension_matches(
            VectorStoreParam(dimension=3072, embedding_model="text-embedding-3-large"),
            "PlanningTool",
        )
        require_dimension_matches(VectorStoreParam(), "PlanningTool")

    def test_known_model_with_wrong_dimension_raises_with_all_four_facts(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            require_dimension_matches(VectorStoreParam(dimension=3072), "KnowledgeGraphTool")
        message = str(excinfo.value)
        assert "KnowledgeGraphTool" in message
        assert "text-embedding-3-small" in message
        assert "dimension=3072" in message
        assert "1536" in message

    def test_unknown_model_passes_at_any_dimension(self) -> None:
        """An Azure deployment name is routinely not a model name."""
        require_dimension_matches(
            VectorStoreParam(dimension=3, embedding_model="my-azure-deployment"), "Card"
        )
        require_dimension_matches(
            VectorStoreParam(dimension=99999, embedding_model="test-embedding"), "Card"
        )


# ---------------------------------------------------------------------------
# SearchHit tests (AC4, AC8)
# ---------------------------------------------------------------------------


class TestSearchHit:
    """Tests for SearchHit construction and serialization."""

    def test_construction(self) -> None:
        hit = SearchHit(ref_type="entity", ref_id="abc-123", text="hello world", score=0.95)
        assert hit.ref_type == "entity"
        assert hit.ref_id == "abc-123"
        assert hit.text == "hello world"
        assert hit.score == 0.95

    def test_round_trip_serialization(self) -> None:
        hit = SearchHit(ref_type="entity", ref_id="abc-123", text="hello world", score=0.95)
        data = hit.model_dump()
        restored = SearchHit.model_validate(data)
        assert restored == hit

    def test_scope_path_and_ordinal_default_to_none(self) -> None:
        """The three workspace fields are additive: an existing construction is valid."""
        hit = SearchHit(ref_type="entity", ref_id="abc-123", text="hello world", score=0.95)
        assert hit.scope is None
        assert hit.path is None
        assert hit.ordinal is None

    def test_scope_path_and_ordinal_round_trip(self) -> None:
        """All three survive serialisation."""
        hit = SearchHit(
            ref_type="chunk",
            ref_id="abc-123",
            text="hello world",
            score=0.95,
            scope="ws-1",
            path="docs/report.md",
            ordinal=7,
        )
        restored = SearchHit.model_validate(hit.model_dump())
        assert restored == hit
        assert restored.scope == "ws-1"
        assert restored.path == "docs/report.md"
        assert restored.ordinal == 7


# ---------------------------------------------------------------------------
# SearchResult tests (AC5, AC8)
# ---------------------------------------------------------------------------


class TestSearchResult:
    """Tests for SearchResult construction and serialization."""

    def test_construction_with_hits(self) -> None:
        hit = SearchHit(ref_type="entity", ref_id="id-1", text="some text", score=0.9)
        result = SearchResult(hits=[hit], status=CollectionStatus.READY)
        assert len(result.hits) == 1
        assert result.status == CollectionStatus.READY

    def test_round_trip_serialization(self) -> None:
        hit = SearchHit(ref_type="relation", ref_id="r-1", text="related", score=0.85)
        result = SearchResult(hits=[hit], status=CollectionStatus.READY)
        data = result.model_dump()
        restored = SearchResult.model_validate(data)
        assert restored == result

    def test_indexing_pending_is_gone_from_the_schema(self) -> None:
        """The field nothing assigns any more is off the model."""
        assert "indexing_pending" not in SearchResult.model_fields

    def test_a_payload_still_carrying_the_key_validates(self) -> None:
        """The benign field case: a stored payload drops the key rather than failing.

        ``SearchResult`` is a return value, not a persisted model, so this matters
        only for a caller that kept one on disk. Pydantic's default ``extra="ignore"``
        drops the key and the attribute is simply absent.
        """
        restored = SearchResult.model_validate(
            {"hits": [], "status": "ready", "indexing_pending": 7}
        )
        assert restored.status == CollectionStatus.READY
        assert not hasattr(restored, "indexing_pending")

    def test_empty_hits_with_error_status(self) -> None:
        result = SearchResult(hits=[], status=CollectionStatus.ERROR)
        assert result.hits == []
        assert result.status == CollectionStatus.ERROR
        data = result.model_dump()
        restored = SearchResult.model_validate(data)
        assert restored == result


# ---------------------------------------------------------------------------
# VectorStoreConfig tests (AC7, AC8)
# ---------------------------------------------------------------------------


class TestVectorStoreConfig:
    """Tests for VectorStoreConfig defaults and serialization."""

    def test_default_values(self) -> None:
        cfg = VectorStoreConfig()
        assert cfg.weaviate_url is None
        assert cfg.weaviate_api_key is None

    def test_the_two_embedding_fields_are_gone(self) -> None:
        """They were inert once the consumers took over embedding, and their
        only writer was the deleted card."""
        assert "embedding_model" not in VectorStoreConfig.model_fields
        assert "embedding_provider" not in VectorStoreConfig.model_fields

    def test_a_checkpoint_carrying_the_removed_keys_still_loads(self) -> None:
        """A removed FIELD is dropped by ``extra='ignore'``; every survivor is intact."""
        cfg = VectorStoreConfig.model_validate(
            {
                "name": "#VectorStore",
                "role": "ToolActor",
                "embedding_model": "text-embedding-ada-002",
                "embedding_provider": "azure",
                "weaviate_url": "http://localhost:8080",
                "weaviate_api_key": "secret-key-123",
            }
        )
        assert not hasattr(cfg, "embedding_model")
        assert not hasattr(cfg, "embedding_provider")
        assert cfg.name == "#VectorStore"
        assert cfg.role == "ToolActor"
        assert cfg.weaviate_url == "http://localhost:8080"
        assert cfg.weaviate_api_key == "secret-key-123"

    def test_round_trip_serialization_defaults(self) -> None:
        cfg = VectorStoreConfig()
        data = cfg.model_dump()
        restored = VectorStoreConfig.model_validate(data)
        assert restored == cfg

    def test_all_optional_fields_populated(self) -> None:
        cfg = VectorStoreConfig(
            weaviate_url="http://localhost:8080",
            weaviate_api_key="secret-key-123",
        )
        assert cfg.weaviate_url == "http://localhost:8080"
        assert cfg.weaviate_api_key == "secret-key-123"

    def test_round_trip_serialization_all_fields(self) -> None:
        cfg = VectorStoreConfig(
            weaviate_url="http://localhost:8080",
            weaviate_api_key="secret-key-123",
        )
        data = cfg.model_dump()
        restored = VectorStoreConfig.model_validate(data)
        assert restored == cfg


# ---------------------------------------------------------------------------
# EmbeddingProvider protocol tests (AC6, AC11)
# ---------------------------------------------------------------------------


class TestEmbeddingProvider:
    """Tests for EmbeddingProvider structural subtyping."""

    def test_structural_subtyping(self) -> None:
        """A class with a matching embed method satisfies EmbeddingProvider."""

        class _FakeEmbedder:
            def embed(self, texts: list[str]) -> list[list[float]]:
                return [[0.1, 0.2] for _ in texts]

        embedder: EmbeddingProvider = _FakeEmbedder()
        result = embedder.embed(["test"])
        assert result == [[0.1, 0.2]]


# ---------------------------------------------------------------------------
# VectorStoreService protocol tests (AC1, AC11)
# ---------------------------------------------------------------------------


class TestVectorStoreService:
    """Tests for VectorStoreService structural subtyping."""

    def test_structural_subtyping(self) -> None:
        """A class implementing all 4 methods satisfies VectorStoreService."""
        from akgentic.tool.vector_store.vector import VectorEntry

        class _FakeStore:
            def create_collection(self, name: str, config: VectorStoreParam) -> None:
                pass

            def add(self, collection: str, entries: list[VectorEntry]) -> None:
                pass

            def remove(self, collection: str, ref_ids: list[str]) -> None:
                pass

            def search(
                self, collection: str, query_vector: list[float], top_k: int
            ) -> SearchResult:
                return SearchResult(hits=[], status=CollectionStatus.READY)

        store: VectorStoreService = _FakeStore()
        store.create_collection("test", VectorStoreParam())
        result = store.search("test", [0.1, 0.2], top_k=5)
        assert result.hits == []
        assert result.status == CollectionStatus.READY


# ---------------------------------------------------------------------------
# Per-collection scoping: the declaration, and the guard that replaces the boundary
# ---------------------------------------------------------------------------


class TestCollectionIsTeamScoped:
    """A collection declares its scoping by name, and the default is isolation."""

    def test_the_workspace_collection_is_shared(self) -> None:
        """Two teams over one filesystem tree must read the same rows."""
        assert collection_is_team_scoped("workspace_chunks") is False

    @pytest.mark.parametrize("collection", ["planning", "knowledge_graph"])
    def test_the_per_team_collections_stay_scoped(self, collection: str) -> None:
        """Rows that belong to one team keep the predicate that isolates them."""
        assert collection_is_team_scoped(collection) is True

    def test_an_unregistered_name_is_team_scoped(self) -> None:
        """The safe answer for a collection nobody declared is the isolating one."""
        assert collection_is_team_scoped("a_collection_nobody_declared") is True

    def test_the_declaration_is_a_frozenset(self) -> None:
        """Nothing can add to it at runtime — a mutable global is the catalog hazard again."""
        assert isinstance(SHARED_COLLECTIONS, frozenset)
        with pytest.raises(AttributeError):
            SHARED_COLLECTIONS.add("planning")  # type: ignore[attr-defined]

    def test_it_holds_exactly_the_workspace_collection(self) -> None:
        assert SHARED_COLLECTIONS == frozenset({"workspace_chunks"})


class TestTheDeclarationCannotDriftFromTheConstantsItNames:
    """The one cost of naming collections by string, closed by a spec rather than a comment.

    ``vector_store`` may not import from ``workspace``, ``planning`` or
    ``knowledge_graph``, so the shared-collection name is duplicated here. These
    specs make a rename of any of those constants redden a test instead of
    silently un-sharing — or silently un-isolating — a collection.
    """

    def test_the_workspace_rag_collection_is_the_shared_one(self) -> None:
        from akgentic.tool.workspace import RAG_COLLECTION

        assert collection_is_team_scoped(RAG_COLLECTION) is False
        assert RAG_COLLECTION in SHARED_COLLECTIONS

    def test_the_planning_collection_is_team_scoped(self) -> None:
        from akgentic.tool.planning.planning_actor import PLAN_COLLECTION

        assert collection_is_team_scoped(PLAN_COLLECTION) is True

    def test_the_knowledge_graph_collection_is_team_scoped(self) -> None:
        from akgentic.tool.knowledge_graph.kg_actor import KG_COLLECTION

        assert collection_is_team_scoped(KG_COLLECTION) is True


class TestCheckSharedScope:
    """The boundary moves rather than disappearing: a shared collection needs a scope."""

    def test_a_shared_collection_without_a_scope_is_refused(self) -> None:
        with pytest.raises(ValueError, match="workspace_chunks") as excinfo:
            check_shared_scope("workspace_chunks", None)

        assert "scope" in str(excinfo.value)

    def test_a_shared_collection_with_a_scope_passes(self) -> None:
        check_shared_scope("workspace_chunks", "user/tree")  # does not raise

    def test_an_empty_scope_is_a_scope(self) -> None:
        """Only ``None`` is the omission this guard exists for.

        ``""`` is a value the caller chose, not a value it forgot; treating it as
        missing would invent a rule the workspace's own path semantics do not have.
        """
        check_shared_scope("workspace_chunks", "")  # does not raise

    @pytest.mark.parametrize("collection", ["planning", "knowledge_graph"])
    def test_a_team_scoped_collection_needs_no_scope(self, collection: str) -> None:
        """Those collections still have their team predicate, so nothing changes for them."""
        check_shared_scope(collection, None)  # does not raise
