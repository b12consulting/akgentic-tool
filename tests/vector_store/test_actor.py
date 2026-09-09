"""Unit tests for VectorStoreActor.

Covers: actor lifecycle (on_start), all proxy method delegation, error
handling (RetriableError, catch/log/swallow), state persistence round-trip,
collection status tracking, and graceful degradation when backend is
unavailable.

Pattern: Instantiate VectorStoreActor() directly, set config, call
on_start(). Same approach as test_kg_actor.py and test_planning_actor.py.
"""

from __future__ import annotations

import logging
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
    CollectionStatus,
    SearchResult,
    VectorStoreConfig,
    VectorStoreParam,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_ACTOR_LOGGER = "akgentic.tool.vector_store.actor"


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """The WARNING records the actor's logger emitted."""
    return [
        record
        for record in caplog.records
        if record.name == _ACTOR_LOGGER and record.levelno == logging.WARNING
    ]


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
        assert state.backend_states == {}
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

    def test_the_embedding_and_request_slots_are_gone(self) -> None:
        """AC 8: the store neither embeds nor keeps per-request bookkeeping."""
        actor = _make_actor()
        for name in (
            "_add_needs_embedding",
            "receiveMsg_EmbeddingResult",
            "receiveMsg_EmbeddingError",
            "_restore_metadata",
            "_settle_request",
            "_tell_completed",
            "_refresh_derived",
            "embed",
            "_get_or_create_embedding_svc",
            "_warn_if_embedding_disagrees",
            "_embedding_svc",
            "_request_requesters",
            "_request_entries",
        ):
            assert not hasattr(actor, name), name

    def test_add_takes_exactly_a_collection_and_entries(self) -> None:
        """AC 8: no ``requester`` and no ``request_ref`` survive on the signature."""
        import inspect

        parameters = list(inspect.signature(VectorStoreActor.add).parameters)
        assert parameters == ["self", "collection", "entries"]

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

        config = VectorStoreParam()
        actor.create_collection("test_col", config)

        backend.create_collection.assert_called_once_with("test_col", config)

    def test_sets_status_ready(self) -> None:
        """AC4: Collection status is READY after creation."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend

        actor.create_collection("test_col", VectorStoreParam())
        assert actor.state.collection_statuses["test_col"] == CollectionStatus.READY

    def test_populates_collection_configs(self) -> None:
        """create_collection stores config dict in state.collection_configs."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend

        config = VectorStoreParam(dimension=128, tenant="team-42", embedding_model="test-embedding")
        actor.create_collection("test_col", config)
        assert "test_col" in actor.state.collection_configs
        cfg = actor.state.collection_configs["test_col"]
        assert cfg["dimension"] == 128
        assert cfg["tenant"] == "team-42"
        assert cfg["backend"] == "inmemory"
        assert cfg["embedding_model"] == "test-embedding"
        # The deleted workspace-persistence mode leaves no trace in the serialised config.
        assert "persistence" not in cfg
        assert "workspace_path" not in cfg

    def test_refuses_a_dimension_its_model_cannot_produce(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A configuration rule is not a backend fault: it raises, it is not swallowed."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend
        caplog.clear()

        with (
            caplog.at_level(logging.WARNING, logger=_ACTOR_LOGGER),
            pytest.raises(ValueError, match="dimension=3072"),
        ):
            actor.create_collection("c", VectorStoreParam(dimension=3072))

        backend.create_collection.assert_not_called()
        assert "c" not in actor.state.collection_configs
        assert "c" not in actor.state.collection_statuses
        assert _warnings(caplog) == []

    def test_a_disagreeing_embedding_model_is_recorded_without_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """AC 11: the store embeds nothing, so there is nothing to disagree with."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend
        caplog.clear()

        with caplog.at_level(logging.WARNING, logger=_ACTOR_LOGGER):
            actor.create_collection(
                "c", VectorStoreParam(dimension=3072, embedding_model="text-embedding-3-large")
            )

        assert _warnings(caplog) == []
        backend.create_collection.assert_called_once()
        assert actor.state.collection_configs["c"]["embedding_model"] == "text-embedding-3-large"
        assert actor.state.collection_configs["c"]["dimension"] == 3072
        assert actor.state.collection_statuses["c"] == CollectionStatus.READY

    def test_the_record_is_the_whole_model_never_an_enumeration(self) -> None:
        """A field the write path has never heard of must survive into state."""

        class _ParamWithExtra(VectorStoreParam):
            extra_field: str = "sentinel"

        actor = _make_actor()
        actor._backend = _mock_backend()

        actor.create_collection("c", _ParamWithExtra(embedding_model="test-embedding"))

        assert actor.state.collection_configs["c"]["extra_field"] == "sentinel"
        assert actor.state.collection_configs["c"]["params"] == {}

    def test_the_record_is_a_plain_dict_that_survives_the_state_round_trip(self) -> None:
        """The serializer's class tag is stripped, so the state's own copy validates.

        ``Agent.notify_state_change`` re-validates the dumped state; a tagged record
        would be re-hydrated into a model and refused by the per-collection dict.
        """
        actor = _make_actor()
        actor._backend = _mock_backend()

        actor.create_collection("c", VectorStoreParam(embedding_model="test-embedding"))

        record = actor.state.collection_configs["c"]
        assert "__model__" not in record
        assert type(actor.state).model_validate(actor.state.model_dump()).collection_configs == {
            "c": record
        }

    def test_notifies_state_change(self) -> None:
        """AC12: state.notify_state_change() called after creation."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend

        with patch.object(VectorStoreState, "notify_state_change") as mock_notify:
            actor.create_collection("test_col", VectorStoreParam())
            mock_notify.assert_called_once()

    def test_syncs_backend_state(self) -> None:
        """AC11: _sync_backend_state called after creation."""
        actor = _make_actor()
        backend = _mock_backend()
        backend.get_state.return_value = {"collections": {"test_col": {}}}
        actor._backend = backend

        actor.create_collection("test_col", VectorStoreParam())
        assert actor.state.backend_state == {"collections": {"test_col": {}}}

    def test_idempotent_second_call(self) -> None:
        """AC4: Second create_collection for same name is no-op (via backend)."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend

        actor.create_collection("test_col", VectorStoreParam())
        actor.create_collection("test_col", VectorStoreParam())
        assert backend.create_collection.call_count == 2
        # Backend itself handles idempotency (no-op on existing collection)

    def test_a_backend_that_cannot_be_built_raises_rather_than_skipping(self) -> None:
        """The cause is now as loud as the consequence ``add`` already refuses to hide.

        Skipping it silently was what let a consumer keep an optimistic binding
        whose every later write went nowhere. The caller's own ``try`` turns this
        into the same one-WARNING degraded mode a cluster consumer enters when
        its factory raises.
        """
        from akgentic.tool.errors import RetriableError

        actor = _make_actor()
        with patch.object(actor, "_get_or_create_backend", return_value=None):
            with pytest.raises(RetriableError, match="inmemory"):
                actor.create_collection("test_col", VectorStoreParam())
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

    def test_an_unexpected_backend_fault_warns_once_and_raises_retriable(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """AC 10: a write that could not land is the caller's problem, not a log line."""
        actor = _make_actor()
        backend = _mock_backend()
        backend.add.side_effect = RuntimeError("disk")
        actor._backend = backend
        caplog.clear()

        with (
            caplog.at_level(logging.WARNING, logger=_ACTOR_LOGGER),
            pytest.raises(RetriableError, match="disk"),
        ):
            actor.add("col1", [_mock_entry(vector=[0.1])])

        assert len(_warnings(caplog)) == 1

    def test_an_empty_vector_is_refused_before_the_backend_is_touched(self) -> None:
        """AC 9: a retry cannot invent a vector the caller never supplied."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend

        with (
            patch.object(VectorStoreState, "notify_state_change") as mock_notify,
            pytest.raises(ValueError, match="e-empty") as excinfo,
        ):
            actor.add("c", [_mock_entry(ref_id="e-empty", vector=[])])

        assert "'c'" in str(excinfo.value)
        backend.add.assert_not_called()
        mock_notify.assert_not_called()

    def test_an_empty_vector_is_not_retriable(self) -> None:
        """AC 9: ``ValueError``, deliberately not ``RetriableError``."""
        actor = _make_actor()
        actor._backend = _mock_backend()

        with pytest.raises(ValueError) as excinfo:
            actor.add("c", [_mock_entry(ref_id="e-empty", vector=[])])

        assert not isinstance(excinfo.value, RetriableError)

    def test_a_batch_mixing_one_empty_vector_is_refused_whole(self) -> None:
        """AC 9: the populated half is not written behind the caller's back."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend

        with pytest.raises(ValueError, match="e-empty"):
            actor.add(
                "c",
                [_mock_entry(ref_id="e-full", vector=[0.1]), _mock_entry(ref_id="e-empty")],
            )

        backend.add.assert_not_called()

    def test_an_empty_batch_reaches_the_backend_unchanged(self) -> None:
        """An empty list carries no empty vector, so nothing refuses it."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend

        actor.add("col1", [])
        backend.add.assert_called_once_with("col1", [])

    def test_a_backend_that_cannot_be_built_is_a_failure_not_a_silent_skip(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The last way a write could still land nowhere and report success.

        ``create_collection`` now refuses an unbuildable backend too, so a
        consumer no longer binds over a store that has no collection. This is the
        other half of the same contract: if ``add`` returned quietly, the
        workspace would count the batch and take the file to ``EMBEDDED`` with
        nothing behind it — the swallow this method exists not to do.
        """
        actor = _make_actor()
        caplog.clear()

        with (
            patch.object(actor, "_get_or_create_backend", return_value=None),
            patch.object(VectorStoreState, "notify_state_change") as mock_notify,
            caplog.at_level(logging.WARNING, logger=_ACTOR_LOGGER),
            pytest.raises(RetriableError, match="col1"),
        ):
            actor.add("col1", [_mock_entry(vector=[0.1])])

        assert len(_warnings(caplog)) == 1
        mock_notify.assert_not_called()

    def test_an_unavailable_backend_still_degrades_on_the_read_paths(self) -> None:
        """Only the write raises: a miss costs a read, a lost write costs the truth."""
        actor = _make_actor()
        with patch.object(actor, "_get_or_create_backend", return_value=None):
            actor.remove("col1", ["e1"])  # must not raise
            assert actor.search("col1", [0.1], 5).hits == []


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

    def test_the_backends_result_is_returned_unchanged(self) -> None:
        """AC 12: no override — there is no progress the actor knows and the backend does not."""

        class _SearchResultWithExtra(SearchResult):
            extra_field: str = "sentinel"

        actor = _make_actor()
        backend = _mock_backend()
        backend.search.return_value = _SearchResultWithExtra(
            hits=[], status=CollectionStatus.READY, indexing_pending=0
        )
        actor._backend = backend
        actor.state.collection_statuses["col1"] = CollectionStatus.READY

        result = actor.search("col1", [0.1], 5)

        assert result is backend.search.return_value
        assert result.status == CollectionStatus.READY
        assert result.indexing_pending == 0
        assert isinstance(result, _SearchResultWithExtra)
        assert result.extra_field == "sentinel"


# ---------------------------------------------------------------------------
# The state fields that left, and the checkpoint that must still load
# ---------------------------------------------------------------------------


class TestTheRetiredStateFields:
    """AC 13 and AC 17: two fields gone, and an old snapshot that still loads."""

    def test_the_two_derived_fields_are_gone(self) -> None:
        assert "pending_requests" not in VectorStoreState.model_fields
        assert "indexing_pending" not in VectorStoreState.model_fields

    def test_the_surviving_fields_are_unchanged(self) -> None:
        assert "collection_statuses" in VectorStoreState.model_fields
        assert "collection_configs" in VectorStoreState.model_fields
        state = VectorStoreState()
        assert state.collection_statuses == {}
        assert state.collection_configs == {}

    def test_a_checkpoint_taken_mid_index_still_loads(self) -> None:
        """AC 17: the tagged record is imported, rebuilt, and then dropped.

        The serializer's before-validator resolves every ``__model__`` tag by
        import **before** Pydantic ever sees the keys, so a tag whose class no
        longer exists raises out of the deserializer and the whole state fails to
        load. This is why the tombstone class survives.
        """
        snapshot = {
            "backend_state": {},
            "collection_statuses": {"c": CollectionStatus.READY},
            "pending_requests": {
                "r": {
                    "__model__": "akgentic.tool.vector_store.actor.PendingRequest",
                    "request_id": "r",
                    "collection": "c",
                    "request_ref": "docs/a.md",
                    "count": 2,
                    "entries": [{"ref_type": "t", "ref_id": "1", "text": "hi"}],
                }
            },
            "indexing_pending": {"c": 2},
            "collection_configs": {"c": {"dimension": 1536, "backend": "inmemory"}},
        }

        restored = VectorStoreState.model_validate(snapshot)

        assert not hasattr(restored, "pending_requests")
        assert not hasattr(restored, "indexing_pending")
        assert restored.collection_configs == {"c": {"dimension": 1536, "backend": "inmemory"}}
        assert restored.collection_statuses == {"c": CollectionStatus.READY}

    def test_a_bare_dict_snapshot_of_the_previous_rename_still_loads(self) -> None:
        """The 49-2 shape: an untagged retired key is simply dropped."""
        legacy = {
            "backend_state": {},
            "collection_statuses": {},
            "pending_entries": {"c1": [{"ref_type": "t", "ref_id": "1", "text": "hi"}]},
            "indexing_pending": {"c1": 1},
            "collection_configs": {},
        }
        restored = VectorStoreState.model_validate(legacy)
        assert not hasattr(restored, "pending_entries")
        assert not hasattr(restored, "indexing_pending")


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
                    "config": VectorStoreParam().model_dump(),
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
        actor.create_collection("test_col", VectorStoreParam())

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

        actor.create_collection("col_a", VectorStoreParam())
        actor.create_collection("col_b", VectorStoreParam())

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

    def test_embedding_worker_exported(self) -> None:
        """AC 21: the worker, the builder, the name function, the two reports."""
        import akgentic.tool.vector_store as vs

        for name in (
            "EmbeddingWorker",
            "EmbeddingRequest",
            "EmbeddingResult",
            "EmbeddingError",
            "build_embedding_service",
            "embedding_worker_name",
        ):
            assert name in vs.__all__, name
            assert hasattr(vs, name), name

    def test_the_retired_names_are_not_exported(self) -> None:
        """AC 21: the store's own embedding surface is gone from the façade."""
        import akgentic.tool.vector_store as vs

        for name in ("EmbeddingActor", "EmbeddingCompleted", "PendingRequest"):
            assert name not in vs.__all__, name
            assert not hasattr(vs, name), name


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

        config = VectorStoreParam(
            backend="weaviate", dimension=384, embedding_model="test-embedding"
        )
        actor.create_collection("wv_col", config)

        mock_wb.create_collection.assert_called_once_with("wv_col", config)
        assert actor.state.collection_configs["wv_col"]["backend"] == "weaviate"
        assert actor.state.collection_statuses["wv_col"] == CollectionStatus.READY

    def test_create_collection_inmemory_still_works(self) -> None:
        """create_collection with backend='inmemory' still routes to InMemoryBackend."""
        actor = _make_actor()
        backend = _mock_backend()
        actor._backend = backend

        config = VectorStoreParam(backend="inmemory")
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

    def test_weaviate_backend_unavailable_raises_and_creates_nothing(self) -> None:
        """No cluster URL anywhere means no backend, and that is not skipped quietly."""
        from akgentic.tool.errors import RetriableError

        actor = _make_actor()
        actor.config = VectorStoreConfig(name=VS_ACTOR_NAME, role=VS_ACTOR_ROLE)
        # No weaviate_url => _get_or_create_weaviate_backend returns None

        config = VectorStoreParam(backend="weaviate")
        with pytest.raises(RetriableError, match="weaviate"):
            actor.create_collection("wv_col", config)

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

        config = VectorStoreParam(backend="weaviate")
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
    """The actor's own team_id reaches the WeaviateBackend it builds from the shared client."""

    def test_backend_built_with_actor_team_id(self) -> None:
        """The accessor resolves the process's client and stamps str(self.team_id)."""
        actor = _make_actor()
        actor.config = VectorStoreConfig(
            name=VS_ACTOR_NAME,
            role=VS_ACTOR_ROLE,
            weaviate_url="http://localhost:8080",
            weaviate_api_key="secret",
        )

        with (
            patch("akgentic.tool.vector_store.weaviate._weaviate_client") as get_client,
            patch("akgentic.tool.vector_store.weaviate.WeaviateBackend") as mock_cls,
        ):
            actor._get_or_create_weaviate_backend()

        get_client.assert_called_once_with("http://localhost:8080", "secret")
        assert mock_cls.call_args[1] == {
            "client": get_client.return_value,
            "team_id": str(actor.team_id),
        }

    def test_team_id_is_not_configuration(self) -> None:
        """team_id is propagated by the actor system, never a VectorStoreConfig field."""
        assert "team_id" not in VectorStoreConfig.model_fields

    def test_two_actors_stamp_distinct_team_ids(self) -> None:
        """Each team's actor builds a backend carrying its own id, on the one client."""
        first, second = _make_actor(), _make_actor()
        for actor in (first, second):
            actor.config = VectorStoreConfig(
                name=VS_ACTOR_NAME,
                role=VS_ACTOR_ROLE,
                weaviate_url="http://localhost:8080",
            )

        with (
            patch("akgentic.tool.vector_store.weaviate._weaviate_client") as get_client,
            patch("akgentic.tool.vector_store.weaviate.WeaviateBackend") as mock_cls,
        ):
            first._get_or_create_weaviate_backend()
            second._get_or_create_weaviate_backend()

        stamped = [c[1]["team_id"] for c in mock_cls.call_args_list]
        assert stamped == [str(first.team_id), str(second.team_id)]
        assert stamped[0] != stamped[1]
        assert all(c[1]["client"] is get_client.return_value for c in mock_cls.call_args_list)


class TestStoppingAnActorLeavesTheSharedClientOpen:
    """Invariant 3 of the thread-safety answer: a team stopping must not disconnect the others.

    The client is shared across every team in the process, so ``VectorStoreActor``
    gains no ``on_stop`` and the only closer is ``close_all()`` at process exit.
    """

    def test_on_stop_does_not_close_the_client(self) -> None:
        """Stopping an actor that holds a Weaviate backend calls no close() on the client."""
        actor = _make_actor()
        client = MagicMock(name="shared-client")
        backend = MagicMock(name="weaviate-backend")
        backend._client = client
        actor._weaviate_backend = backend
        actor._backends["weaviate"] = backend

        actor.on_stop()

        client.close.assert_not_called()
        backend.close.assert_not_called()


# ---------------------------------------------------------------------------
# Registry-driven routing and query passthrough
# ---------------------------------------------------------------------------


class TestQueryPassthrough:
    """The optional VectorQuery reaches the backend only when supplied."""

    def test_none_query_forwards_none_to_backend(self) -> None:
        """No query still routes through the scoped call with ``query=None``."""
        actor = _make_actor()
        backend = _mock_backend()
        expected = SearchResult(hits=[], status=CollectionStatus.READY, indexing_pending=0)
        backend.search.return_value = expected
        actor._backend = backend

        actor.search("col1", [0.1], 5)

        backend.search.assert_called_once_with(
            "col1", [0.1], 5, scope=None, path_prefix=None
        )

    def test_query_is_forwarded_when_supplied(self) -> None:
        from akgentic.tool.vector_store.protocol import VectorQuery

        actor = _make_actor()
        backend = _mock_backend()
        backend.search.return_value = SearchResult(
            hits=[], status=CollectionStatus.READY, indexing_pending=0
        )
        actor._backend = backend
        query = VectorQuery(filters={"ref_type": "entity"})

        actor.search("col1", [0.1], 5, query=query)

        backend.search.assert_called_once_with(
            "col1", [0.1], 5, scope=None, path_prefix=None, query=query
        )


class TestRegistryRouting:
    """A registered third-party backend routes without any actor edits."""

    def test_routes_to_registered_custom_backend_and_skips_state_sync(self) -> None:
        from akgentic.tool.vector_store.registry import (
            BackendSpec,
            register_backend,
            unregister_backend,
        )

        actor = _make_actor()
        custom = MagicMock()
        register_backend(
            BackendSpec(
                name="custom",
                factory=lambda _ctx: custom,
                persists_in_actor_state=False,
            )
        )
        try:
            config = VectorStoreParam(backend="custom")
            actor.create_collection("cc", config)

            custom.create_collection.assert_called_once_with("cc", config)
            assert actor.state.collection_configs["cc"]["backend"] == "custom"
            # External backend: nothing snapshotted into actor state.
            assert actor.state.backend_state == {}

            entry = _mock_entry(vector=[0.1])
            actor.add("cc", [entry])
            custom.add.assert_called_once_with("cc", [entry])
        finally:
            unregister_backend("custom")

    def test_stateful_custom_backend_is_snapshotted_and_restored(self) -> None:
        from akgentic.tool.vector_store.registry import (
            BackendSpec,
            register_backend,
            unregister_backend,
        )

        first = _mock_backend()
        first.get_state.return_value = {"value": "saved"}
        second = _mock_backend()
        backends = iter([first, second])
        register_backend(
            BackendSpec(
                name="stateful",
                factory=lambda _ctx: next(backends),
                persists_in_actor_state=True,
            )
        )
        try:
            actor = _make_actor()
            config = VectorStoreParam(backend="stateful")
            actor.create_collection("cc", config)
            assert actor.state.backend_states["stateful"] == {"value": "saved"}

            restored = _make_actor()
            restored.state.backend_states = actor.state.backend_states
            assert restored._get_backend("stateful") is second
            second.restore_state.assert_called_once_with({"value": "saved"})
        finally:
            unregister_backend("stateful")

    def test_stateful_custom_backend_restores_empty_snapshot(self) -> None:
        from akgentic.tool.vector_store.registry import (
            BackendSpec,
            register_backend,
            unregister_backend,
        )

        backend = _mock_backend()
        register_backend(
            BackendSpec(
                name="empty_state",
                factory=lambda _ctx: backend,
                persists_in_actor_state=True,
            )
        )
        try:
            actor = _make_actor()
            actor.state.backend_states["empty_state"] = {}
            assert actor._get_backend("empty_state") is backend
            backend.restore_state.assert_called_once_with({})
        finally:
            unregister_backend("empty_state")

    def test_replacing_builtin_name_routes_to_replacement_factory(self) -> None:
        from akgentic.tool.vector_store.registry import (
            BackendSpec,
            get_backend_spec,
            register_backend,
        )

        original = get_backend_spec("inmemory")
        replacement = _mock_backend()
        register_backend(
            BackendSpec(name="inmemory", factory=lambda _ctx: replacement),
            replace=True,
        )
        try:
            actor = _make_actor()
            assert actor._get_backend("inmemory") is replacement
        finally:
            register_backend(original, replace=True)

    def test_stateful_inmemory_replacement_uses_named_snapshot(self) -> None:
        from akgentic.tool.vector_store.registry import (
            BackendSpec,
            get_backend_spec,
            register_backend,
        )

        original = get_backend_spec("inmemory")
        first = _mock_backend()
        first.get_state.return_value = {"value": "replacement"}
        second = _mock_backend()
        backends = iter([first, second])
        register_backend(
            BackendSpec(
                name="inmemory",
                factory=lambda _ctx: next(backends),
                persists_in_actor_state=True,
                selectable_as_default=False,
            ),
            replace=True,
        )
        try:
            actor = _make_actor()
            actor.create_collection("cc", VectorStoreParam(backend="inmemory"))
            assert actor.state.backend_state == {}
            assert actor.state.backend_states["inmemory"] == {"value": "replacement"}

            restored = _make_actor()
            restored.state.backend_states = actor.state.backend_states
            assert restored._get_backend("inmemory") is second
            second.restore_state.assert_called_once_with({"value": "replacement"})
        finally:
            register_backend(original, replace=True)


# ---------------------------------------------------------------------------
# The store refuses a collection it could not create (AC 25, 26, 28)
# ---------------------------------------------------------------------------


class TestCreateCollectionRefusesInsteadOfDegrading:
    """Both ways this method used to swallow now raise.

    The setup path told the truth only after 49-3's fixup made the *write* path
    tell it; this closes the other half.
    """

    def test_an_unbuildable_backend_raises_and_writes_nothing(self) -> None:
        from akgentic.tool.errors import RetriableError

        actor = _make_actor()
        with patch.object(actor, "_get_backend", return_value=None):
            with pytest.raises(RetriableError) as excinfo:
                actor.create_collection("c", VectorStoreParam(backend="inmemory"))

        message = str(excinfo.value)
        assert "c" in message
        assert "inmemory" in message
        # Nothing is written before the raise, so there is no partial state.
        assert actor.state.collection_configs == {}
        assert actor.state.collection_statuses == {}

    def test_a_backend_that_raises_is_re_raised_as_retriable_with_one_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from akgentic.tool.errors import RetriableError

        actor = _make_actor()
        backend = _mock_backend()
        backend.create_collection.side_effect = RuntimeError("schema")

        with patch.object(actor, "_get_backend", return_value=backend):
            with caplog.at_level(logging.WARNING, logger="akgentic.tool.vector_store.actor"):
                with pytest.raises(RetriableError, match="schema"):
                    actor.create_collection("c", VectorStoreParam(backend="inmemory"))

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert actor.state.collection_configs == {}
        assert actor.state.collection_statuses == {}

    def test_a_contradictory_dimension_is_still_a_value_error(self) -> None:
        """``require_dimension_matches`` runs outside the try, so it is not retriable."""
        from akgentic.tool.errors import RetriableError

        actor = _make_actor()
        param = VectorStoreParam(dimension=3072, embedding_model="text-embedding-3-small")

        with pytest.raises(ValueError) as excinfo:
            actor.create_collection("c", param)

        assert not isinstance(excinfo.value, RetriableError)

    def test_remove_and_search_still_degrade(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The asymmetry is the decision: a read that answers empty costs a miss."""
        actor = _make_actor()

        with patch.object(actor, "_get_backend", return_value=None):
            with caplog.at_level(logging.WARNING, logger="akgentic.tool.vector_store.actor"):
                actor.remove("c", ["1"])  # must not raise
                result = actor.search("c", [0.1], 3)  # must not raise

        assert result.hits == []
        assert result.status == CollectionStatus.READY

    def test_the_in_memory_extra_being_absent_degrades_at_the_consumer(self) -> None:
        """The reason the loud option was refused.

        A missing ``[vector_search]`` extra is a legitimate build failure for the
        in-memory backend, and the package's documented contract is to degrade.
        The store refuses; it does not fail the team's build.
        """
        from akgentic.tool.errors import RetriableError

        actor = _make_actor()
        with patch(
            "akgentic.tool.vector_store.inmemory._check_vector_search_dependencies",
            side_effect=ImportError("numpy"),
        ):
            # _get_or_create_backend catches the ImportError and answers None,
            # which is the same shape as an unreachable cluster from here.
            with pytest.raises(RetriableError):
                actor.create_collection("c", VectorStoreParam(backend="inmemory"))

        assert actor.state.collection_statuses == {}
