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
    PendingRequest,
    VectorStoreActor,
    VectorStoreState,
)
from akgentic.tool.vector_store.embedding_actor import (
    EmbeddingCompleted,
    EmbeddingError,
    EmbeddingResult,
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


def _mock_entry(
    ref_id: str = "e1",
    ref_type: str = "entity",
    text: str = "hello",
    vector: list[float] | None = None,
) -> MagicMock:
    """Return a MagicMock that mimics VectorEntry.

    Args:
        ref_id: Entry reference ID.
        ref_type: Entry reference type.
        text: Entry text.
        vector: Embedding vector; empty list means needs-embedding.
    """
    entry = MagicMock()
    entry.ref_id = ref_id
    entry.ref_type = ref_type
    entry.text = text
    entry.vector = vector if vector is not None else []
    return entry


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

    def test_collection_configs_round_trip(self) -> None:
        """collection_configs survives Pydantic serialisation."""
        state = VectorStoreState(
            collection_configs={
                "c1": {
                    "dimension": 128,
                    "backend": "weaviate",
                    "tenant": "team-42",
                }
            },
        )
        data = state.model_dump()
        restored = VectorStoreState.model_validate(data)
        assert restored.collection_configs == state.collection_configs
        assert restored.collection_configs["c1"]["tenant"] == "team-42"

    def test_collection_configs_defaults_empty(self) -> None:
        """collection_configs defaults to empty dict."""
        state = VectorStoreState()
        assert state.collection_configs == {}


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

    def test_populates_collection_configs(self) -> None:
        """create_collection stores config dict in state.collection_configs."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend

        config = CollectionConfig(dimension=128, tenant="team-42")
        actor.create_collection("test_col", config)
        assert "test_col" in actor.state.collection_configs
        cfg = actor.state.collection_configs["test_col"]
        assert cfg["dimension"] == 128
        assert cfg["tenant"] == "team-42"
        assert cfg["backend"] == "inmemory"
        # The deleted workspace-persistence mode leaves no trace in the serialised config.
        assert "persistence" not in cfg
        assert "workspace_path" not in cfg

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

    def test_pre_embedded_delegates_to_backend(self) -> None:
        """AC9: Pre-embedded entries go directly to backend.add()."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend
        entry = _mock_entry(vector=[0.1, 0.2])

        actor.add("col1", [entry])
        backend.add.assert_called_once_with("col1", [entry])

    def test_pre_embedded_notifies_state_change(self) -> None:
        """AC12: state.notify_state_change() called after pre-embedded add."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend
        entry = _mock_entry(vector=[0.1])

        with patch.object(VectorStoreState, "notify_state_change") as mock_notify:
            actor.add("col1", [entry])
            mock_notify.assert_called_once()

    def test_pre_embedded_nonexistent_collection_raises_retriable(self) -> None:
        """AC9: ValueError from backend becomes RetriableError."""
        actor = _make_actor()
        backend = _mock_backend()
        backend.add.side_effect = ValueError("Collection 'col1' does not exist")
        actor._backend = backend
        entry = _mock_entry(vector=[0.1])

        with pytest.raises(RetriableError, match="does not exist"):
            actor.add("col1", [entry])

    def test_pre_embedded_unexpected_error_swallowed(self) -> None:
        """AC9: Unexpected errors caught/logged/swallowed."""
        actor = _make_actor()
        backend = _mock_backend()
        backend.add.side_effect = RuntimeError("unexpected")
        actor._backend = backend
        entry = _mock_entry(vector=[0.1])

        # Should not raise
        actor.add("col1", [entry])

    def test_empty_entries_no_op(self) -> None:
        """Empty entries list is a no-op."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend

        actor.add("col1", [])
        backend.add.assert_not_called()

    def test_backend_unavailable_skips(self) -> None:
        """When backend is None, add() logs and returns."""
        actor = _make_actor()
        with patch.object(actor, "_get_or_create_backend", return_value=None):
            actor.add("col1", [_mock_entry(vector=[0.1])])


# ---------------------------------------------------------------------------
# Non-blocking add — needs embedding (AC2, AC3, AC8)
# ---------------------------------------------------------------------------


class TestAddNeedsEmbedding:
    """AC2/AC3: Entries without vectors spawn EmbeddingActor."""

    def test_spawns_embedding_actor(self) -> None:
        """AC2: add() with vectorless entries spawns EmbeddingActor."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend
        entry = _mock_entry(vector=[])

        mock_addr = MagicMock()
        mock_proxy = MagicMock()
        with (
            patch.object(actor, "createActor", return_value=mock_addr) as mock_create,
            patch.object(actor, "proxy_tell", return_value=mock_proxy),
        ):
            actor.add("col1", [entry])
            mock_create.assert_called_once()
            mock_proxy.receiveMsg_EmbeddingRequest.assert_called_once()

    def test_sets_status_indexing(self) -> None:
        """AC3: Collection status transitions to INDEXING on add."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend
        entry = _mock_entry(vector=[])

        with (
            patch.object(actor, "createActor", return_value=MagicMock()),
            patch.object(actor, "proxy_tell", return_value=MagicMock()),
        ):
            actor.add("col1", [entry])

        assert actor.state.collection_statuses["col1"] == CollectionStatus.INDEXING

    def test_tracks_indexing_pending_count(self) -> None:
        """AC3: indexing_pending incremented by number of entries."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend
        entries = [_mock_entry(ref_id="e1"), _mock_entry(ref_id="e2")]

        with (
            patch.object(actor, "createActor", return_value=MagicMock()),
            patch.object(actor, "proxy_tell", return_value=MagicMock()),
        ):
            actor.add("col1", entries)

        assert actor.state.indexing_pending["col1"] == 2

    def test_stores_pending_request(self) -> None:
        """One open request records its own collection, count and raw entries."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend
        entry = _mock_entry(ref_id="e1", text="hello")

        with (
            patch.object(actor, "createActor", return_value=MagicMock()),
            patch.object(actor, "proxy_tell", return_value=MagicMock()),
        ):
            actor.add("col1", [entry], request_ref="docs/a.md")

        assert len(actor.state.pending_requests) == 1
        record = next(iter(actor.state.pending_requests.values()))
        assert record.collection == "col1"
        assert record.count == 1
        assert record.request_ref == "docs/a.md"
        assert record.entries == [{"ref_type": "entity", "ref_id": "e1", "text": "hello"}]

    def test_does_not_call_backend_add(self) -> None:
        """AC2: Vectorless entries do NOT call backend.add() directly."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend
        entry = _mock_entry(vector=[])

        with (
            patch.object(actor, "createActor", return_value=MagicMock()),
            patch.object(actor, "proxy_tell", return_value=MagicMock()),
        ):
            actor.add("col1", [entry])

        backend.add.assert_not_called()

    def test_retry_from_error_resets_to_indexing(self) -> None:
        """AC8: add() on ERROR collection transitions to INDEXING."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend
        actor.state.collection_statuses["col1"] = CollectionStatus.ERROR

        entry = _mock_entry(vector=[])
        with (
            patch.object(actor, "createActor", return_value=MagicMock()),
            patch.object(actor, "proxy_tell", return_value=MagicMock()),
        ):
            actor.add("col1", [entry])

        assert actor.state.collection_statuses["col1"] == CollectionStatus.INDEXING

    def test_mixed_entries_partitioned(self) -> None:
        """AC9: Mixed entries: pre-embedded go to backend, vectorless to actor."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend

        pre = _mock_entry(ref_id="pre1", vector=[0.1, 0.2])
        needs = _mock_entry(ref_id="needs1", vector=[])

        with (
            patch.object(actor, "createActor", return_value=MagicMock()),
            patch.object(actor, "proxy_tell", return_value=MagicMock()),
        ):
            actor.add("col1", [pre, needs])

        # Pre-embedded goes to backend
        backend.add.assert_called_once_with("col1", [pre])
        # Needs-embedding tracked in pending
        assert actor.state.indexing_pending["col1"] == 1


# ---------------------------------------------------------------------------
# receiveMsg_EmbeddingResult / receiveMsg_EmbeddingError — per-request settle
# ---------------------------------------------------------------------------


def _open_request(
    actor: VectorStoreActor,
    request_id: str,
    collection: str = "col1",
    count: int = 2,
    request_ref: str | None = None,
) -> None:
    """Open one embedding request on *actor* the way ``_add_needs_embedding`` would."""
    actor.state.pending_requests[request_id] = PendingRequest(
        request_id=request_id,
        collection=collection,
        request_ref=request_ref,
        count=count,
        entries=[
            {"ref_type": "entity", "ref_id": f"{request_id}-{i}", "text": f"t{i}"}
            for i in range(count)
        ],
    )
    actor._refresh_derived(collection)


def _result_for(request_id: str, collection: str = "col1") -> EmbeddingResult:
    """Build a two-entry EmbeddingResult carrying *request_id*."""
    from akgentic.tool.vector_store.vector import VectorEntry

    entries = [
        VectorEntry(ref_type="entity", ref_id="e1", text="hi", vector=[0.1]),
        VectorEntry(ref_type="entity", ref_id="e2", text="bye", vector=[0.2]),
    ]
    return EmbeddingResult(
        collection=collection, entries=entries, request_id=request_id
    )


class TestReceiveEmbeddingResult:
    """VectorStoreActor settles the request its result names, and no other."""

    def test_inserts_into_backend(self) -> None:
        """Entries are inserted into the backend on result delivery."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend
        _open_request(actor, "req-1")

        result = _result_for("req-1")
        actor.receiveMsg_EmbeddingResult(result)

        backend.add.assert_called_once_with("col1", result.entries)

    def test_transitions_to_ready(self) -> None:
        """Status returns to READY when the last open request settles."""
        actor = _make_actor()
        actor._backend = _mock_backend()
        _open_request(actor, "req-1")

        actor.receiveMsg_EmbeddingResult(_result_for("req-1"))

        assert actor.state.collection_statuses["col1"] == CollectionStatus.READY
        assert actor.state.indexing_pending.get("col1") is None
        assert actor.state.pending_requests == {}

    def test_stays_indexing_while_another_request_is_open(self) -> None:
        """A second open request keeps the collection INDEXING at its own count."""
        actor = _make_actor()
        actor._backend = _mock_backend()
        _open_request(actor, "req-1", count=2)
        _open_request(actor, "req-2", count=3)

        actor.receiveMsg_EmbeddingResult(_result_for("req-1"))

        assert actor.state.collection_statuses["col1"] == CollectionStatus.INDEXING
        assert actor.state.indexing_pending["col1"] == 3
        assert set(actor.state.pending_requests) == {"req-2"}

    def test_unknown_request_id_settles_nothing(self) -> None:
        """A result naming no open request must not disturb another's bookkeeping."""
        actor = _make_actor()
        actor._backend = _mock_backend()
        _open_request(actor, "req-1", count=2)

        actor.receiveMsg_EmbeddingResult(_result_for("req-unknown"))

        assert set(actor.state.pending_requests) == {"req-1"}
        assert actor.state.indexing_pending["col1"] == 2
        assert actor.state.collection_statuses["col1"] == CollectionStatus.INDEXING

    def test_notifies_state_change(self) -> None:
        """state.notify_state_change() called after result delivery."""
        actor = _make_actor()
        actor._backend = _mock_backend()
        _open_request(actor, "req-1")

        with patch.object(VectorStoreState, "notify_state_change") as mock_notify:
            actor.receiveMsg_EmbeddingResult(_result_for("req-1"))
            assert mock_notify.call_count >= 1

    def test_backend_add_failure_settles_without_erroring_collection(self) -> None:
        """A failed insert closes its own request; the collection never goes ERROR."""
        actor = _make_actor()
        backend = _mock_backend()
        backend.add.side_effect = RuntimeError("disk full")
        actor._backend = backend
        _open_request(actor, "req-1", count=2)
        _open_request(actor, "req-2", count=3)

        actor.receiveMsg_EmbeddingResult(_result_for("req-1"))

        assert actor.state.collection_statuses["col1"] == CollectionStatus.INDEXING
        assert actor.state.indexing_pending["col1"] == 3
        assert set(actor.state.pending_requests) == {"req-2"}

    def test_out_of_order_results_attribute_to_their_own_request(self) -> None:
        """Two requests into one collection, settled in reverse order."""
        actor = _make_actor()
        actor._backend = _mock_backend()
        _open_request(actor, "req-1", count=2)
        _open_request(actor, "req-2", count=3)
        assert actor.state.indexing_pending["col1"] == 5

        # The SECOND request settles first.
        actor.receiveMsg_EmbeddingResult(_result_for("req-2"))
        assert set(actor.state.pending_requests) == {"req-1"}
        assert actor.state.pending_requests["req-1"].count == 2
        assert len(actor.state.pending_requests["req-1"].entries) == 2
        assert actor.state.indexing_pending["col1"] == 2
        assert actor.state.collection_statuses["col1"] == CollectionStatus.INDEXING

        actor.receiveMsg_EmbeddingResult(_result_for("req-1"))
        assert actor.state.pending_requests == {}
        assert actor.state.collection_statuses["col1"] == CollectionStatus.READY

    def test_resolves_backend_for_the_collection_not_the_inmemory_one(self) -> None:
        """A Weaviate-backed collection's async batch reaches Weaviate, not memory."""
        actor = _make_actor()
        inmemory = _mock_backend()
        weaviate = MagicMock()
        actor._backend = inmemory
        actor._weaviate_backend = weaviate
        actor.state.collection_configs["wv_col"] = {"backend": "weaviate"}
        _open_request(actor, "req-1", collection="wv_col")

        result = _result_for("req-1", collection="wv_col")
        actor.receiveMsg_EmbeddingResult(result)

        weaviate.add.assert_called_once_with("wv_col", result.entries)
        inmemory.add.assert_not_called()
        # The in-memory snapshot is not taken for a Weaviate collection either.
        inmemory.get_state.assert_not_called()
        assert actor.state.backend_state == {}

    def test_backend_unavailable_still_settles_the_request(self) -> None:
        """No backend must not leave the request open for ever."""
        actor = _make_actor()
        _open_request(actor, "req-1")

        with patch.object(actor, "_get_backend_for_collection", return_value=None):
            actor.receiveMsg_EmbeddingResult(_result_for("req-1"))

        assert actor.state.pending_requests == {}
        assert actor.state.collection_statuses["col1"] == CollectionStatus.READY


class TestReceiveEmbeddingError:
    """An embedding failure fails one request, not the collection."""

    def test_settles_only_the_failed_request(self) -> None:
        """The other request's record, count and the status are untouched."""
        actor = _make_actor()
        _open_request(actor, "req-a", count=2)
        _open_request(actor, "req-b", count=3)

        actor.receiveMsg_EmbeddingError(
            EmbeddingError(collection="col1", error="API failed", request_id="req-a")
        )

        assert set(actor.state.pending_requests) == {"req-b"}
        assert actor.state.pending_requests["req-b"].count == 3
        assert len(actor.state.pending_requests["req-b"].entries) == 3
        assert actor.state.indexing_pending["col1"] == 3
        assert actor.state.collection_statuses["col1"] == CollectionStatus.INDEXING

    def test_collection_never_reaches_error_status(self) -> None:
        """A single failed batch no longer blanks the collection."""
        actor = _make_actor()
        _open_request(actor, "req-a", count=2)

        actor.receiveMsg_EmbeddingError(
            EmbeddingError(collection="col1", error="API failed", request_id="req-a")
        )

        assert actor.state.collection_statuses["col1"] != CollectionStatus.ERROR
        assert actor.state.collection_statuses["col1"] == CollectionStatus.READY

    def test_survivor_settles_the_collection_to_ready(self) -> None:
        """After A fails, B succeeding is what returns the collection to READY."""
        actor = _make_actor()
        actor._backend = _mock_backend()
        _open_request(actor, "req-a", count=2)
        _open_request(actor, "req-b", count=3)

        actor.receiveMsg_EmbeddingError(
            EmbeddingError(collection="col1", error="API failed", request_id="req-a")
        )
        assert actor.state.collection_statuses["col1"] == CollectionStatus.INDEXING

        actor.receiveMsg_EmbeddingResult(_result_for("req-b"))
        assert actor.state.collection_statuses["col1"] == CollectionStatus.READY
        assert actor.state.pending_requests == {}

    def test_unknown_request_id_settles_nothing(self) -> None:
        """An error naming no open request leaves every record in place."""
        actor = _make_actor()
        _open_request(actor, "req-a", count=2)

        actor.receiveMsg_EmbeddingError(
            EmbeddingError(collection="col1", error="boom", request_id="ghost")
        )

        assert set(actor.state.pending_requests) == {"req-a"}
        assert actor.state.indexing_pending["col1"] == 2

    def test_notifies_state_change(self) -> None:
        """state.notify_state_change() called after error."""
        actor = _make_actor()
        _open_request(actor, "req-a")

        with patch.object(VectorStoreState, "notify_state_change") as mock_notify:
            actor.receiveMsg_EmbeddingError(
                EmbeddingError(collection="col1", error="fail", request_id="req-a")
            )
            assert mock_notify.call_count >= 1


# ---------------------------------------------------------------------------
# EmbeddingCompleted — the completion signal told back to the requester
# ---------------------------------------------------------------------------


class TestEmbeddingCompleted:
    """The requester is told how its request ended, either way."""

    def test_told_on_success_with_request_ref_echoed(self) -> None:
        """Success delivers EmbeddingCompleted with error=None and the caller's ref."""
        actor = _make_actor()
        actor._backend = _mock_backend()
        requester = MagicMock()

        with (
            patch.object(actor, "createActor", return_value=MagicMock()),
            patch.object(actor, "proxy_tell", return_value=MagicMock()),
        ):
            actor.add(
                "col1",
                [_mock_entry(ref_id="e1")],
                requester=requester,
                request_ref="docs/report.md",
            )
        request_id = next(iter(actor.state.pending_requests))

        with patch.object(actor, "send") as mock_send:
            actor.receiveMsg_EmbeddingResult(_result_for(request_id))

        mock_send.assert_called_once()
        target, message = mock_send.call_args.args
        assert target is requester
        assert isinstance(message, EmbeddingCompleted)
        assert message.request_id == request_id
        assert message.request_ref == "docs/report.md"
        assert message.collection == "col1"
        assert message.count == 1
        assert message.error is None

    def test_told_on_failure_with_the_error(self) -> None:
        """Failure delivers the same message with error populated."""
        actor = _make_actor()
        requester = MagicMock()
        _open_request(actor, "req-1", count=2, request_ref="docs/a.md")
        actor._request_requesters["req-1"] = requester

        with patch.object(actor, "send") as mock_send:
            actor.receiveMsg_EmbeddingError(
                EmbeddingError(collection="col1", error="API down", request_id="req-1")
            )

        _, message = mock_send.call_args.args
        assert message.error == "API down"
        assert message.request_ref == "docs/a.md"
        assert message.count == 2

    def test_told_when_the_backend_insert_fails(self) -> None:
        """A failed insert is reported to the caller, not just swallowed into a settle.

        This is the path that replaced the collection-wide ERROR: the failure has to
        reach somebody, and the requester is the only one left who can act on it.
        """
        actor = _make_actor()
        backend = _mock_backend()
        backend.add.side_effect = RuntimeError("disk full")
        actor._backend = backend
        requester = MagicMock()
        _open_request(actor, "req-1", count=2, request_ref="docs/a.md")
        actor._request_requesters["req-1"] = requester

        with patch.object(actor, "send") as mock_send:
            actor.receiveMsg_EmbeddingResult(_result_for("req-1"))

        _, message = mock_send.call_args.args
        assert message.error == "disk full"
        assert message.request_ref == "docs/a.md"

    def test_told_when_no_backend_is_available(self) -> None:
        """An unavailable backend settles the request AND says why."""
        actor = _make_actor()
        requester = MagicMock()
        _open_request(actor, "req-1", request_ref="docs/b.md")
        actor._request_requesters["req-1"] = requester

        with (
            patch.object(actor, "_get_backend_for_collection", return_value=None),
            patch.object(actor, "send") as mock_send,
        ):
            actor.receiveMsg_EmbeddingResult(_result_for("req-1"))

        _, message = mock_send.call_args.args
        assert message.error == "Backend unavailable"
        assert message.request_ref == "docs/b.md"

    def test_no_requester_means_no_message(self) -> None:
        """The three existing call sites pass no requester and get no delivery."""
        actor = _make_actor()
        actor._backend = _mock_backend()
        _open_request(actor, "req-1")

        with patch.object(actor, "send") as mock_send:
            actor.receiveMsg_EmbeddingResult(_result_for("req-1"))

        mock_send.assert_not_called()

    def test_delivery_failure_does_not_break_the_settle(self) -> None:
        """A stopped requester must not take the settle down with it."""
        actor = _make_actor()
        actor._backend = _mock_backend()
        _open_request(actor, "req-1")
        actor._request_requesters["req-1"] = MagicMock()

        with patch.object(actor, "send", side_effect=RuntimeError("actor gone")):
            actor.receiveMsg_EmbeddingResult(_result_for("req-1"))

        assert actor.state.pending_requests == {}
        assert actor.state.collection_statuses["col1"] == CollectionStatus.READY

    def test_requester_map_is_not_state(self) -> None:
        """The requester lives outside BaseState — an address there breaks snapshots."""
        actor = _make_actor()
        assert "_request_requesters" not in VectorStoreState.model_fields
        assert actor._request_requesters == {}


# ---------------------------------------------------------------------------
# Search during INDEXING (AC6)
# ---------------------------------------------------------------------------


class TestSearchDuringIndexing:
    """AC6: search() returns partial results with INDEXING status."""

    def test_returns_indexing_status(self) -> None:
        """AC6: Search returns status=INDEXING when collection is indexing."""
        actor = _make_actor()
        backend = _mock_backend()
        backend.search.return_value = SearchResult(
            hits=[], status=CollectionStatus.READY, indexing_pending=0
        )
        actor._backend = backend
        actor.state.collection_statuses["col1"] = CollectionStatus.INDEXING
        actor.state.indexing_pending["col1"] = 3

        result = actor.search("col1", [0.1], 5)
        assert result.status == CollectionStatus.INDEXING
        assert result.indexing_pending == 3

    def test_returns_ready_when_not_indexing(self) -> None:
        """Search returns READY status when no indexing."""
        actor = _make_actor()
        backend = _mock_backend()
        backend.search.return_value = SearchResult(
            hits=[], status=CollectionStatus.READY, indexing_pending=0
        )
        actor._backend = backend
        actor.state.collection_statuses["col1"] = CollectionStatus.READY

        result = actor.search("col1", [0.1], 5)
        assert result.status == CollectionStatus.READY
        assert result.indexing_pending == 0


# ---------------------------------------------------------------------------
# State serialisation with new fields
# ---------------------------------------------------------------------------


class TestVectorStoreStateNewFields:
    """Open requests and derived counts serialise correctly."""

    def test_pending_requests_round_trip(self) -> None:
        """pending_requests survives serialisation."""
        state = VectorStoreState(
            pending_requests={
                "r1": PendingRequest(
                    request_id="r1",
                    collection="c1",
                    request_ref="docs/a.md",
                    count=1,
                    entries=[{"ref_type": "t", "ref_id": "1", "text": "hi"}],
                )
            },
            indexing_pending={"c1": 1},
        )
        data = state.model_dump()
        restored = VectorStoreState.model_validate(data)
        assert restored.pending_requests == state.pending_requests
        assert restored.indexing_pending == state.indexing_pending

    def test_defaults_empty(self) -> None:
        """New fields default to empty dicts."""
        state = VectorStoreState()
        assert state.pending_requests == {}
        assert state.indexing_pending == {}

    def test_legacy_pending_entries_payload_still_validates(self) -> None:
        """A snapshot written before the rename loads, dropping the retired key."""
        legacy = {
            "backend_state": {},
            "collection_statuses": {},
            "pending_entries": {"c1": [{"ref_type": "t", "ref_id": "1", "text": "hi"}]},
            "indexing_pending": {"c1": 1},
            "collection_configs": {},
        }
        restored = VectorStoreState.model_validate(legacy)
        assert restored.pending_requests == {}
        assert not hasattr(restored, "pending_entries")


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
        backend.remove.assert_called_once_with(
            "col1", ["id1", "id2"], scope=None, path_prefix=None
        )

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
        backend.search.assert_called_once_with(
            "col1", [0.1, 0.2], 5, scope=None, path_prefix=None
        )
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
# scope / path_prefix pass-through
# ---------------------------------------------------------------------------


class TestScopePassThrough:
    """The actor forwards both predicates to the backend, unchanged."""

    def test_search_forwards_both_predicates(self) -> None:
        """search(scope=..., path_prefix=...) reaches the backend as given."""
        actor = _make_actor()
        backend = _mock_backend()
        backend.search.return_value = SearchResult(
            hits=[], status=CollectionStatus.READY, indexing_pending=0
        )
        actor._backend = backend

        actor.search("col1", [0.1], 5, scope="ws-1", path_prefix="docs/")

        backend.search.assert_called_once_with(
            "col1", [0.1], 5, scope="ws-1", path_prefix="docs/"
        )

    def test_remove_forwards_both_predicates(self) -> None:
        """remove(scope=..., path_prefix=...) reaches the backend as given."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend

        actor.remove("col1", ["id1"], scope="ws-1", path_prefix="docs/")

        backend.remove.assert_called_once_with(
            "col1", ["id1"], scope="ws-1", path_prefix="docs/"
        )

    def test_status_override_preserves_a_field_the_actor_never_heard_of(self) -> None:
        """search() overrides two fields by copy, never by rebuilding the result.

        An enumerated rebuild is correct on the day it is written and drops the next
        field added to ``SearchResult`` in silence. A whole-model comparison cannot
        catch that — only a field the write path has never seen can.
        """

        class _SearchResultWithExtra(SearchResult):
            extra_field: str = "sentinel"

        actor = _make_actor()
        backend = _mock_backend()
        backend.search.return_value = _SearchResultWithExtra(
            hits=[], status=CollectionStatus.READY, indexing_pending=0
        )
        actor._backend = backend
        actor.state.collection_statuses["col1"] = CollectionStatus.INDEXING
        actor.state.indexing_pending["col1"] = 4

        result = actor.search("col1", [0.1], 5)

        assert result.status == CollectionStatus.INDEXING
        assert result.indexing_pending == 4
        assert isinstance(result, _SearchResultWithExtra)
        assert result.extra_field == "sentinel"

    def test_add_does_not_forward_correlation_arguments_to_the_backend(self) -> None:
        """requester and request_ref are actor-level, never a backend concern."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend
        entry = _mock_entry(vector=[0.1, 0.2])

        actor.add("col1", [entry], requester=MagicMock(), request_ref="docs/a.md")

        backend.add.assert_called_once_with("col1", [entry])


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

    def test_get_or_create_embedding_svc_returns_none_on_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns None when EmbeddingService creation fails.

        The patch has to land in the module ``actor.py`` actually imports from
        (``vector_store.vector``). Patching the deprecated ``akgentic.tool.vector``
        façade would write into the façade's globals, shadowing its ``__getattr__``
        for the rest of the session — which is why this uses ``monkeypatch``.
        """
        actor = _make_actor()
        import akgentic.tool.vector_store.vector as vector_mod

        monkeypatch.setattr(
            vector_mod,
            "EmbeddingService",
            MagicMock(side_effect=Exception("no API key")),
        )
        result = actor._get_or_create_embedding_svc()
        assert result is None


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

    def test_embedding_actor_exported(self) -> None:
        """EmbeddingActor and message models in __all__."""
        import akgentic.tool.vector_store as vs

        assert "EmbeddingActor" in vs.__all__
        assert "EmbeddingRequest" in vs.__all__
        assert "EmbeddingResult" in vs.__all__
        assert "EmbeddingError" in vs.__all__
        assert hasattr(vs, "EmbeddingActor")
        assert hasattr(vs, "EmbeddingRequest")
        assert hasattr(vs, "EmbeddingResult")
        assert hasattr(vs, "EmbeddingError")


# ---------------------------------------------------------------------------
# Weaviate backend routing (AC11 — Story 12.1)
# ---------------------------------------------------------------------------


class TestWeaviateRouting:
    """AC11: VectorStoreActor routes weaviate collections to WeaviateBackend."""

    def test_create_collection_routes_to_weaviate(self) -> None:
        """create_collection with backend='weaviate' uses WeaviateBackend."""
        actor = _make_actor()
        actor.config = VectorStoreConfig(
            name=VS_ACTOR_NAME,
            role=VS_ACTOR_ROLE,
            weaviate_url="http://localhost:8080",
        )
        mock_wb = MagicMock()
        actor._weaviate_backend = mock_wb

        config = CollectionConfig(backend="weaviate", dimension=384)
        actor.create_collection("wv_col", config)

        mock_wb.create_collection.assert_called_once_with("wv_col", config)
        assert actor.state.collection_configs["wv_col"]["backend"] == "weaviate"
        assert actor.state.collection_statuses["wv_col"] == CollectionStatus.READY

    def test_create_collection_inmemory_still_works(self) -> None:
        """create_collection with backend='inmemory' still routes to InMemoryBackend."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend

        config = CollectionConfig(backend="inmemory")
        actor.create_collection("im_col", config)

        backend.create_collection.assert_called_once_with("im_col", config)
        assert actor.state.collection_configs["im_col"]["backend"] == "inmemory"

    def test_add_routes_to_weaviate_backend(self) -> None:
        """add() for a weaviate collection routes to WeaviateBackend."""
        actor = _make_actor()
        mock_wb = MagicMock()
        actor._weaviate_backend = mock_wb
        actor.state.collection_configs["wv_col"] = {"backend": "weaviate"}

        entry = _mock_entry(vector=[0.1, 0.2])
        actor.add("wv_col", [entry])

        mock_wb.add.assert_called_once_with("wv_col", [entry])

    def test_remove_routes_to_weaviate_backend(self) -> None:
        """remove() for a weaviate collection routes to WeaviateBackend."""
        actor = _make_actor()
        mock_wb = MagicMock()
        actor._weaviate_backend = mock_wb
        actor.state.collection_configs["wv_col"] = {"backend": "weaviate"}

        actor.remove("wv_col", ["id1"])

        mock_wb.remove.assert_called_once_with(
            "wv_col", ["id1"], scope=None, path_prefix=None
        )

    def test_search_routes_to_weaviate_backend(self) -> None:
        """search() for a weaviate collection routes to WeaviateBackend."""
        actor = _make_actor()
        mock_wb = MagicMock()
        expected = SearchResult(hits=[], status=CollectionStatus.READY, indexing_pending=0)
        mock_wb.search.return_value = expected
        actor._weaviate_backend = mock_wb
        actor.state.collection_configs["wv_col"] = {"backend": "weaviate"}

        result = actor.search("wv_col", [0.1], 5)

        mock_wb.search.assert_called_once_with(
            "wv_col", [0.1], 5, scope=None, path_prefix=None
        )
        assert result == expected

    def test_inmemory_collection_not_routed_to_weaviate(self) -> None:
        """inmemory collections still go to InMemoryBackend even when weaviate is available."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend
        mock_wb = MagicMock()
        actor._weaviate_backend = mock_wb
        actor.state.collection_configs["im_col"] = {"backend": "inmemory"}

        entry = _mock_entry(vector=[0.1])
        actor.add("im_col", [entry])

        backend.add.assert_called_once()
        mock_wb.add.assert_not_called()

    def test_weaviate_backend_unavailable_logs_warning(self) -> None:
        """create_collection logs warning when weaviate backend is unavailable."""
        actor = _make_actor()
        actor.config = VectorStoreConfig(name=VS_ACTOR_NAME, role=VS_ACTOR_ROLE)
        # No weaviate_url => _get_or_create_weaviate_backend returns None

        config = CollectionConfig(backend="weaviate")
        actor.create_collection("wv_col", config)

        # Should not crash, just skip
        assert "wv_col" not in actor.state.collection_statuses

    def test_weaviate_no_sync_backend_state(self) -> None:
        """Weaviate collections should NOT call _sync_backend_state."""
        actor = _make_actor()
        mock_wb = MagicMock()
        actor._weaviate_backend = mock_wb
        actor.config = VectorStoreConfig(
            name=VS_ACTOR_NAME,
            role=VS_ACTOR_ROLE,
            weaviate_url="http://localhost:8080",
        )

        config = CollectionConfig(backend="weaviate")
        actor.create_collection("wv_col", config)

        # backend_state should still be empty (not synced for weaviate)
        assert actor.state.backend_state == {}

    def test_get_backend_for_collection_defaults_to_inmemory(self) -> None:
        """Unknown collections default to inmemory backend."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend

        result = actor._get_backend_for_collection("unknown")
        assert result is backend


# ---------------------------------------------------------------------------
# Weaviate team_id propagation
# ---------------------------------------------------------------------------


class TestWeaviateTeamIdPropagation:
    """The actor's own team_id reaches the WeaviateBackend it builds."""

    def test_backend_built_with_actor_team_id(self) -> None:
        """_get_or_create_weaviate_backend passes str(self.team_id)."""
        actor = _make_actor()
        actor.config = VectorStoreConfig(
            name=VS_ACTOR_NAME,
            role=VS_ACTOR_ROLE,
            weaviate_url="http://localhost:8080",
            weaviate_api_key="secret",
        )

        with patch(
            "akgentic.tool.vector_store.weaviate.WeaviateBackend"
        ) as mock_cls:
            actor._get_or_create_weaviate_backend()

        assert mock_cls.call_args[1]["team_id"] == str(actor.team_id)
        assert mock_cls.call_args[1]["url"] == "http://localhost:8080"
        assert mock_cls.call_args[1]["api_key"] == "secret"

    def test_team_id_is_not_configuration(self) -> None:
        """team_id is propagated by the actor system, never a VectorStoreConfig field."""
        assert "team_id" not in VectorStoreConfig.model_fields

    def test_two_actors_stamp_distinct_team_ids(self) -> None:
        """Each team's actor builds a backend carrying its own id."""
        first, second = _make_actor(), _make_actor()
        for actor in (first, second):
            actor.config = VectorStoreConfig(
                name=VS_ACTOR_NAME,
                role=VS_ACTOR_ROLE,
                weaviate_url="http://localhost:8080",
            )

        with patch("akgentic.tool.vector_store.weaviate.WeaviateBackend") as mock_cls:
            first._get_or_create_weaviate_backend()
            second._get_or_create_weaviate_backend()

        stamped = [c[1]["team_id"] for c in mock_cls.call_args_list]
        assert stamped == [str(first.team_id), str(second.team_id)]
        assert stamped[0] != stamped[1]


# ---------------------------------------------------------------------------
# A request that cannot be started, and the metadata that must survive the trip
# ---------------------------------------------------------------------------


class TestASpawnFailureSettlesItsOwnRequest:
    """The record is written before the child exists, so a failure must close it.

    Latent until story 45-7: the record's requester was never told, so nothing
    waited on the signal and a collection pinned at ``INDEXING`` for ever was
    invisible. The workspace indexer is the first caller that waits.
    """

    def test_a_failing_spawn_leaves_no_open_request(self) -> None:
        """Otherwise the collection reports ``INDEXING`` with nothing in flight."""
        actor = _make_actor()
        actor._backend = _mock_backend()

        with patch.object(actor, "createActor", side_effect=RuntimeError("no thread")):
            actor.add("col1", [_mock_entry(vector=[])])

        assert actor.state.pending_requests == {}
        assert actor.state.collection_statuses["col1"] == CollectionStatus.READY
        assert actor.state.indexing_pending.get("col1") is None

    def test_the_requester_is_told_the_failure(self) -> None:
        """A caller that waits on ``EmbeddingCompleted`` must never wait for ever."""
        actor = _make_actor()
        actor._backend = _mock_backend()
        requester = MagicMock()
        delivered: list[Any] = []

        with (
            patch.object(actor, "createActor", side_effect=RuntimeError("no thread")),
            patch.object(actor, "send", side_effect=lambda to, msg: delivered.append(msg)),
        ):
            actor.add("col1", [_mock_entry(vector=[])], requester=requester, request_ref="a.md")

        [completed] = delivered
        assert isinstance(completed, EmbeddingCompleted)
        assert completed.request_ref == "a.md"
        assert completed.error is not None and "no thread" in completed.error

    def test_a_failing_tell_settles_the_request_too(self) -> None:
        """The child exists but never got its payload, so nothing will report it."""
        actor = _make_actor()
        actor._backend = _mock_backend()

        with (
            patch.object(actor, "createActor", return_value=MagicMock()),
            patch.object(actor, "proxy_tell", side_effect=RuntimeError("proxy gone")),
        ):
            actor.add("col1", [_mock_entry(vector=[])])

        assert actor.state.pending_requests == {}

    def test_a_successful_spawn_leaves_the_request_open(self) -> None:
        """The guard above must not settle the healthy path."""
        actor = _make_actor()
        actor._backend = _mock_backend()

        with (
            patch.object(actor, "createActor", return_value=MagicMock()),
            patch.object(actor, "proxy_tell", return_value=MagicMock()),
        ):
            actor.add("col1", [_mock_entry(vector=[])])

        assert len(actor.state.pending_requests) == 1
        assert actor.state.collection_statuses["col1"] == CollectionStatus.INDEXING


class TestScopeSurvivesTheEmbeddingRoundTrip:
    """``EmbeddingRequest`` carries three of ``VectorEntry``'s seven fields.

    The result is rebuilt from exactly those three, so ``scope``, ``path`` and
    ``ordinal`` arrive back as ``None`` — and those are the three every scoped
    removal and every scoped search filters on. An entry stored without them is
    findable by nobody and removable by nobody.
    """

    def _round_trip(self, actor: VectorStoreActor, backend: MagicMock) -> list[Any]:
        """Issue one scoped ``add`` and deliver its embedded result back."""
        from akgentic.tool.vector_store.vector import VectorEntry

        original = VectorEntry(
            ref_type="workspace_chunk",
            ref_id="chunk-1",
            text="hello",
            vector=[],
            scope="team-7",
            path="docs/a.md",
            ordinal=3,
        )
        with (
            patch.object(actor, "createActor", return_value=MagicMock()),
            patch.object(actor, "proxy_tell", return_value=MagicMock()),
        ):
            actor.add("col1", [original], request_ref="docs/a.md")
        request_id = next(iter(actor.state.pending_requests))

        actor.receiveMsg_EmbeddingResult(
            EmbeddingResult(
                collection="col1",
                # Exactly what ``EmbeddingActor`` rebuilds: the three carried
                # fields plus the vector it produced.
                entries=[
                    VectorEntry(
                        ref_type="workspace_chunk",
                        ref_id="chunk-1",
                        text="hello",
                        vector=[0.1, 0.2],
                    )
                ],
                request_id=request_id,
            )
        )
        return list(backend.add.call_args[0][1])

    def test_scope_path_and_ordinal_reach_the_backend(self) -> None:
        """Without this the workspace's scoped removal would match nothing."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend

        [stored] = self._round_trip(actor, backend)

        assert (stored.scope, stored.path, stored.ordinal) == ("team-7", "docs/a.md", 3)
        assert stored.vector == [0.1, 0.2]

    def test_the_originals_are_dropped_when_the_request_settles(self) -> None:
        """They are held for one round trip, not for the life of the team."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend

        self._round_trip(actor, backend)

        assert actor._request_entries == {}

    def test_a_result_that_does_not_line_up_is_passed_through_untouched(self) -> None:
        """Mis-attributing metadata would be worse than not restoring it."""
        from akgentic.tool.vector_store.vector import VectorEntry

        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend
        actor._request_entries["req-1"] = [
            VectorEntry(ref_type="t", ref_id="a", text="a", vector=[], scope="team-7"),
            VectorEntry(ref_type="t", ref_id="b", text="b", vector=[], scope="team-7"),
        ]
        _open_request(actor, "req-1", count=2)

        actor.receiveMsg_EmbeddingResult(
            EmbeddingResult(
                collection="col1",
                entries=[VectorEntry(ref_type="t", ref_id="a", text="a", vector=[0.1])],
                request_id="req-1",
            )
        )

        [stored] = backend.add.call_args[0][1]
        assert stored.scope is None
