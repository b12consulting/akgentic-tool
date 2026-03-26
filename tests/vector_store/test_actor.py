"""Unit tests for VectorStoreActor.

Covers: actor lifecycle (on_start), all proxy method delegation, error
handling (RetriableError, catch/log/swallow), state persistence round-trip,
collection status tracking, and graceful degradation when backend is
unavailable.

Pattern: Instantiate VectorStoreActor() directly, set config, call
on_start(). Same approach as test_kg_actor.py and test_planning_actor.py.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from akgentic.tool.errors import RetriableError
from akgentic.tool.vector_store.actor import (
    VS_ACTOR_NAME,
    VS_ACTOR_ROLE,
    VectorStoreActor,
    VectorStoreState,
)
from akgentic.tool.vector_store.protocol import (
    CollectionConfig,
    CollectionStatus,
    SearchResult,
    VectorStoreConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_actor() -> VectorStoreActor:
    """Create and initialise a VectorStoreActor for testing."""
    actor = VectorStoreActor()
    actor.config = VectorStoreConfig(name=VS_ACTOR_NAME, role=VS_ACTOR_ROLE)
    actor.on_start()
    return actor


def _mock_backend() -> MagicMock:
    """Return a MagicMock that mimics InMemoryBackend."""
    backend = MagicMock()
    backend.get_state.return_value = {"collections": {}}
    return backend


# ---------------------------------------------------------------------------
# VectorStoreState (AC11)
# ---------------------------------------------------------------------------


class TestVectorStoreState:
    """AC11: VectorStoreState(BaseState) construction and serialisation."""

    def test_construction_defaults(self) -> None:
        """State has empty defaults."""
        state = VectorStoreState()
        assert state.backend_state == {}
        assert state.collection_statuses == {}

    def test_serialisation_round_trip(self) -> None:
        """State round-trips through Pydantic serialisation."""
        state = VectorStoreState(
            backend_state={"collections": {"c1": {"config": {}, "entries": []}}},
            collection_statuses={"c1": CollectionStatus.READY},
        )
        data = state.model_dump()
        restored = VectorStoreState.model_validate(data)
        assert restored.backend_state == state.backend_state
        assert restored.collection_statuses == state.collection_statuses


# ---------------------------------------------------------------------------
# Actor lifecycle (AC1, AC2, AC3, AC10)
# ---------------------------------------------------------------------------


class TestActorLifecycle:
    """AC1-3, AC10: Actor class, constants, on_start, runtime state."""

    def test_on_start_initialises_state_with_observer(self) -> None:
        """AC3: on_start sets state with observer wired."""
        actor = _make_actor()
        assert isinstance(actor.state, VectorStoreState)
        # Observer is wired — notify_state_change should not raise
        actor.state.notify_state_change()

    def test_on_start_backend_is_none(self) -> None:
        """AC10: Backend starts as None (lazy)."""
        actor = _make_actor()
        assert actor._backend is None

    def test_on_start_embedding_svc_is_none(self) -> None:
        """AC10: Embedding service starts as None (lazy)."""
        actor = _make_actor()
        assert actor._embedding_svc is None

    def test_singleton_constants(self) -> None:
        """AC2: Constants have expected values."""
        assert VS_ACTOR_NAME == "#VectorStore"
        assert VS_ACTOR_ROLE == "ToolActor"


# ---------------------------------------------------------------------------
# create_collection (AC4, AC9, AC12)
# ---------------------------------------------------------------------------


class TestCreateCollection:
    """AC4: create_collection delegates to backend and sets status."""

    def test_delegates_to_backend(self) -> None:
        """AC4: Delegation to InMemoryBackend.create_collection."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend

        config = CollectionConfig()
        actor.create_collection("test_col", config)

        backend.create_collection.assert_called_once_with("test_col", config)

    def test_sets_status_ready(self) -> None:
        """AC4: Collection status is READY after creation."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend

        actor.create_collection("test_col", CollectionConfig())
        assert actor.state.collection_statuses["test_col"] == CollectionStatus.READY

    def test_notifies_state_change(self) -> None:
        """AC12: state.notify_state_change() called after creation."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend

        with patch.object(VectorStoreState, "notify_state_change") as mock_notify:
            actor.create_collection("test_col", CollectionConfig())
            mock_notify.assert_called_once()

    def test_syncs_backend_state(self) -> None:
        """AC11: _sync_backend_state called after creation."""
        actor = _make_actor()
        backend = _mock_backend()
        backend.get_state.return_value = {"collections": {"test_col": {}}}
        actor._backend = backend

        actor.create_collection("test_col", CollectionConfig())
        assert actor.state.backend_state == {"collections": {"test_col": {}}}

    def test_idempotent_second_call(self) -> None:
        """AC4: Second create_collection for same name is no-op (via backend)."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend

        actor.create_collection("test_col", CollectionConfig())
        actor.create_collection("test_col", CollectionConfig())
        assert backend.create_collection.call_count == 2
        # Backend itself handles idempotency (no-op on existing collection)

    def test_graceful_when_backend_unavailable(self) -> None:
        """AC9: Logs warning when backend cannot be created."""
        actor = _make_actor()
        with patch.object(actor, "_get_or_create_backend", return_value=None):
            # Should not raise
            actor.create_collection("test_col", CollectionConfig())
            assert "test_col" not in actor.state.collection_statuses


# ---------------------------------------------------------------------------
# add (AC5, AC9, AC12)
# ---------------------------------------------------------------------------


class TestAdd:
    """AC5: add delegates to backend with state notification."""

    def test_delegates_to_backend(self) -> None:
        """AC5: Delegation to InMemoryBackend.add."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend
        entries = [MagicMock()]

        actor.add("col1", entries)
        backend.add.assert_called_once_with("col1", entries)

    def test_notifies_state_change(self) -> None:
        """AC12: state.notify_state_change() called after add."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend

        with patch.object(VectorStoreState, "notify_state_change") as mock_notify:
            actor.add("col1", [])
            mock_notify.assert_called_once()

    def test_nonexistent_collection_raises_retriable(self) -> None:
        """AC9: ValueError from backend becomes RetriableError."""
        actor = _make_actor()
        backend = _mock_backend()
        backend.add.side_effect = ValueError("Collection 'col1' does not exist")
        actor._backend = backend

        with pytest.raises(RetriableError, match="does not exist"):
            actor.add("col1", [])

    def test_unexpected_error_swallowed(self) -> None:
        """AC9: Unexpected errors caught/logged/swallowed."""
        actor = _make_actor()
        backend = _mock_backend()
        backend.add.side_effect = RuntimeError("unexpected")
        actor._backend = backend

        # Should not raise
        actor.add("col1", [])


# ---------------------------------------------------------------------------
# remove (AC6, AC9, AC12)
# ---------------------------------------------------------------------------


class TestRemove:
    """AC6: remove delegates to backend with state notification."""

    def test_delegates_to_backend(self) -> None:
        """AC6: Delegation to InMemoryBackend.remove."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend

        actor.remove("col1", ["id1", "id2"])
        backend.remove.assert_called_once_with("col1", ["id1", "id2"])

    def test_notifies_state_change(self) -> None:
        """AC12: state.notify_state_change() called after remove."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend

        with patch.object(VectorStoreState, "notify_state_change") as mock_notify:
            actor.remove("col1", ["id1"])
            mock_notify.assert_called_once()

    def test_nonexistent_collection_raises_retriable(self) -> None:
        """AC9: ValueError from backend becomes RetriableError."""
        actor = _make_actor()
        backend = _mock_backend()
        backend.remove.side_effect = ValueError("Collection 'col1' does not exist")
        actor._backend = backend

        with pytest.raises(RetriableError, match="does not exist"):
            actor.remove("col1", ["id1"])

    def test_unexpected_error_swallowed(self) -> None:
        """AC9: Unexpected errors caught/logged/swallowed."""
        actor = _make_actor()
        backend = _mock_backend()
        backend.remove.side_effect = RuntimeError("unexpected")
        actor._backend = backend

        actor.remove("col1", ["id1"])


# ---------------------------------------------------------------------------
# search (AC7, AC9)
# ---------------------------------------------------------------------------


class TestSearch:
    """AC7: search delegates to backend and returns SearchResult."""

    def test_delegates_to_backend(self) -> None:
        """AC7: Delegation to InMemoryBackend.search."""
        actor = _make_actor()
        backend = _mock_backend()
        expected = SearchResult(
            hits=[],
            status=CollectionStatus.READY,
            indexing_pending=0,
        )
        backend.search.return_value = expected
        actor._backend = backend

        result = actor.search("col1", [0.1, 0.2], 5)
        backend.search.assert_called_once_with("col1", [0.1, 0.2], 5)
        assert result == expected

    def test_nonexistent_collection_raises_retriable(self) -> None:
        """AC9: ValueError from backend becomes RetriableError."""
        actor = _make_actor()
        backend = _mock_backend()
        backend.search.side_effect = ValueError("Collection 'col1' does not exist")
        actor._backend = backend

        with pytest.raises(RetriableError, match="does not exist"):
            actor.search("col1", [0.1], 5)

    def test_unexpected_error_returns_empty(self) -> None:
        """AC9: Unexpected errors return empty SearchResult."""
        actor = _make_actor()
        backend = _mock_backend()
        backend.search.side_effect = RuntimeError("unexpected")
        actor._backend = backend

        result = actor.search("col1", [0.1], 5)
        assert result.hits == []

    def test_backend_unavailable_returns_empty(self) -> None:
        """AC9: No backend returns empty SearchResult."""
        actor = _make_actor()
        with patch.object(actor, "_get_or_create_backend", return_value=None):
            result = actor.search("col1", [0.1], 5)
            assert result.hits == []


# ---------------------------------------------------------------------------
# embed (AC8, AC9)
# ---------------------------------------------------------------------------


class TestEmbed:
    """AC8: embed delegates to EmbeddingService and returns vectors."""

    def test_delegates_to_embedding_service(self) -> None:
        """AC8: Delegation to EmbeddingService.embed."""
        actor = _make_actor()
        mock_svc = MagicMock()
        mock_svc.embed.return_value = [[0.1, 0.2, 0.3]]
        actor._embedding_svc = mock_svc

        result = actor.embed(["hello"])
        mock_svc.embed.assert_called_once_with(["hello"])
        assert result == [[0.1, 0.2, 0.3]]

    def test_returns_empty_when_service_unavailable(self) -> None:
        """AC8: Returns [] when embedding service is None."""
        actor = _make_actor()
        with patch.object(actor, "_get_or_create_embedding_svc", return_value=None):
            result = actor.embed(["hello"])
            assert result == []

    def test_returns_empty_on_failure(self) -> None:
        """AC9: Catch/log/swallow on embed failure."""
        actor = _make_actor()
        mock_svc = MagicMock()
        mock_svc.embed.side_effect = RuntimeError("API error")
        actor._embedding_svc = mock_svc

        result = actor.embed(["hello"])
        assert result == []


# ---------------------------------------------------------------------------
# State persistence round-trip (AC11)
# ---------------------------------------------------------------------------


class TestStatePersistence:
    """AC11: Backend state persistence round-trip via actor state."""

    def test_round_trip_through_actor_state(self) -> None:
        """Create collection, add entries, verify state round-trip."""
        actor = _make_actor()

        # Use a real-ish backend mock that tracks state
        backend = _mock_backend()
        state_snapshot: dict[str, Any] = {
            "collections": {
                "test_col": {
                    "config": CollectionConfig().model_dump(),
                    "entries": [
                        {
                            "ref_type": "test",
                            "ref_id": "e1",
                            "text": "hello",
                            "vector": [0.1, 0.2],
                        }
                    ],
                }
            }
        }
        backend.get_state.return_value = state_snapshot
        actor._backend = backend

        # Trigger a mutation to sync state
        actor.create_collection("test_col", CollectionConfig())

        # Verify actor state has the snapshot
        assert actor.state.backend_state == state_snapshot

        # Now create a new actor and verify restore
        actor2 = _make_actor()
        actor2.state.backend_state = state_snapshot

        # The lazy init should restore from state
        import akgentic.tool.vector_store.inmemory as inmemory_mod

        mock_backend2 = _mock_backend()
        original_cls = inmemory_mod.InMemoryBackend
        inmemory_mod.InMemoryBackend = MagicMock(return_value=mock_backend2)  # type: ignore[misc]
        try:
            result = actor2._get_or_create_backend()
            assert result is not None
            mock_backend2.restore_state.assert_called_once_with(state_snapshot)
        finally:
            inmemory_mod.InMemoryBackend = original_cls  # type: ignore[misc]


# ---------------------------------------------------------------------------
# collection_statuses (AC11, AC12)
# ---------------------------------------------------------------------------


class TestCollectionStatuses:
    """AC11: collection_statuses tracks per-collection status."""

    def test_tracks_multiple_collections(self) -> None:
        """Multiple collections tracked independently."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend

        actor.create_collection("col_a", CollectionConfig())
        actor.create_collection("col_b", CollectionConfig())

        assert actor.state.collection_statuses["col_a"] == CollectionStatus.READY
        assert actor.state.collection_statuses["col_b"] == CollectionStatus.READY

    def test_status_in_serialised_state(self) -> None:
        """Collection statuses survive serialisation."""
        state = VectorStoreState(
            collection_statuses={"c1": CollectionStatus.READY, "c2": CollectionStatus.INDEXING}
        )
        data = state.model_dump()
        restored = VectorStoreState.model_validate(data)
        assert restored.collection_statuses["c1"] == CollectionStatus.READY
        assert restored.collection_statuses["c2"] == CollectionStatus.INDEXING


# ---------------------------------------------------------------------------
# Lazy backend initialisation (AC3, AC10)
# ---------------------------------------------------------------------------


class TestLazyBackend:
    """AC3/AC10: Lazy backend and embedding service initialisation."""

    def test_get_or_create_backend_caches(self) -> None:
        """Backend is cached after first creation."""
        actor = _make_actor()
        mock_backend = _mock_backend()
        actor._backend = mock_backend

        result = actor._get_or_create_backend()
        assert result is mock_backend

    def test_get_or_create_backend_returns_none_on_import_error(self) -> None:
        """Returns None when vector_search deps missing."""
        actor = _make_actor()
        # Patch the inmemory module so importing InMemoryBackend raises
        import akgentic.tool.vector_store.inmemory as inmemory_mod

        original_cls = inmemory_mod.InMemoryBackend
        inmemory_mod.InMemoryBackend = MagicMock(  # type: ignore[misc]
            side_effect=ImportError("no numpy"),
        )
        try:
            result = actor._get_or_create_backend()
            assert result is None
        finally:
            inmemory_mod.InMemoryBackend = original_cls  # type: ignore[misc]

    def test_get_or_create_embedding_svc_caches(self) -> None:
        """Embedding service is cached after first creation."""
        actor = _make_actor()
        mock_svc = MagicMock()
        actor._embedding_svc = mock_svc

        result = actor._get_or_create_embedding_svc()
        assert result is mock_svc

    def test_get_or_create_embedding_svc_returns_none_on_failure(self) -> None:
        """Returns None when EmbeddingService creation fails."""
        actor = _make_actor()
        import akgentic.tool.vector as vector_mod

        original_cls = vector_mod.EmbeddingService
        vector_mod.EmbeddingService = MagicMock(  # type: ignore[misc]
            side_effect=Exception("no API key"),
        )
        try:
            result = actor._get_or_create_embedding_svc()
            assert result is None
        finally:
            vector_mod.EmbeddingService = original_cls  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Public API exports (AC13)
# ---------------------------------------------------------------------------


class TestPublicApiExports:
    """AC13: vector_store/__init__.py re-exports actor symbols."""

    def test_actor_exported(self) -> None:
        """VectorStoreActor in __all__."""
        import akgentic.tool.vector_store as vs

        assert "VectorStoreActor" in vs.__all__
        assert hasattr(vs, "VectorStoreActor")

    def test_state_exported(self) -> None:
        """VectorStoreState in __all__."""
        import akgentic.tool.vector_store as vs

        assert "VectorStoreState" in vs.__all__
        assert hasattr(vs, "VectorStoreState")

    def test_constants_exported(self) -> None:
        """VS_ACTOR_NAME and VS_ACTOR_ROLE in __all__."""
        import akgentic.tool.vector_store as vs

        assert "VS_ACTOR_NAME" in vs.__all__
        assert "VS_ACTOR_ROLE" in vs.__all__
        assert vs.VS_ACTOR_NAME == "#VectorStore"
        assert vs.VS_ACTOR_ROLE == "ToolActor"
