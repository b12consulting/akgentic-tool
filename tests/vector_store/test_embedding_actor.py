"""Unit tests for EmbeddingWorker — the consumer-owned embedding worker.

Covers: the ``DeferredWorker`` inheritance and its budget, the three message
models, ``produce`` filling vectors in place, the vector-count guard, the report
path through the consumer's two handlers rather than ``deliver`` / ``fail``, the
budget reaching the OpenAI client, and the worker name.

Pattern: instantiate ``EmbeddingWorker()`` directly, set config, call
``on_start()``. Mock the parent address and ``proxy_tell`` to capture reports.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from akgentic.core.agent_config import BaseConfig
from akgentic.core.orchestrator import STOP_TIMEOUT

from akgentic.tool.core.deferred import DEFAULT_WORKER_TIMEOUT_S, WORKER_ROLE, DeferredWorker
from akgentic.tool.vector_store.embedding_actor import (
    EMBED_WORKER_NAME_PREFIX,
    EmbeddingError,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingWorker,
    build_embedding_service,
    embedding_worker_name,
)
from akgentic.tool.vector_store.vector import VectorEntry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _EntryWithExtra(VectorEntry):
    """A ``VectorEntry`` carrying a field ``produce`` has never heard of.

    The strong-form Golden Rule 12 guard: an enumerated rebuild naming every
    field that exists today returns a plain ``VectorEntry`` and loses this one.
    """

    extra_field: str = "sentinel"


def _make_worker() -> EmbeddingWorker:
    """Create and initialise an EmbeddingWorker for testing."""
    worker = EmbeddingWorker()
    worker.config = BaseConfig(name="#embed-test_col-abc123", role=WORKER_ROLE)
    worker._parent = MagicMock()
    worker.on_start()
    return worker


def _entries() -> list[VectorEntry]:
    """Two entries carrying every field the workspace sets."""
    return [
        VectorEntry(
            ref_type="workspace_chunk",
            ref_id="e1",
            text="hello world",
            vector=[],
            scope="ws-1",
            path="docs/report.md",
            ordinal=0,
        ),
        VectorEntry(
            ref_type="workspace_chunk",
            ref_id="e2",
            text="goodbye",
            vector=[],
            scope="ws-1",
            path="docs/report.md",
            ordinal=1,
        ),
    ]


def _make_request(
    collection: str = "test_col",
    entries: list[VectorEntry] | None = None,
    request_ref: str | None = "docs/report.md",
) -> EmbeddingRequest:
    """Create a test EmbeddingRequest."""
    return EmbeddingRequest(
        deferred_key="test-req-1",
        collection=collection,
        entries=_entries() if entries is None else entries,
        request_ref=request_ref,
        embedding_model="text-embedding-3-small",
        embedding_provider="openai",
    )


class _Recorder:
    """A double for the parent proxy that records which handler was called."""

    def __init__(self) -> None:
        self.results: list[EmbeddingResult] = []
        self.errors: list[EmbeddingError] = []
        self.delivered: list[tuple[object, object]] = []
        self.failed: list[tuple[object, str]] = []

    def receiveMsg_EmbeddingResult(self, msg: EmbeddingResult) -> None:  # noqa: N802
        self.results.append(msg)

    def receiveMsg_EmbeddingError(self, msg: EmbeddingError) -> None:  # noqa: N802
        self.errors.append(msg)

    def deliver(self, key: object, value: object) -> None:
        self.delivered.append((key, value))

    def fail(self, key: object, error: str) -> None:
        self.failed.append((key, error))


def _run(worker: EmbeddingWorker, request: EmbeddingRequest) -> tuple[_Recorder, MagicMock]:
    """Drive one payload through the worker, capturing reports and the stop."""
    recorder = _Recorder()
    with (
        patch.object(worker, "proxy_tell", return_value=recorder),
        patch.object(worker, "stop") as mock_stop,
    ):
        worker.receiveMsg_DeferredPayload(request)
    return recorder, mock_stop


# ---------------------------------------------------------------------------
# AC 1 — the worker is a DeferredWorker with the worker budget
# ---------------------------------------------------------------------------


class TestWorkerIsADeferredWorker:
    """AC 1: inheritance, and a budget strictly below the stop backstop."""

    def test_embedding_worker_is_a_deferred_worker(self) -> None:
        assert issubclass(EmbeddingWorker, DeferredWorker)

    def test_timeout_is_the_worker_default(self) -> None:
        assert EmbeddingWorker.timeout_s == DEFAULT_WORKER_TIMEOUT_S

    def test_timeout_is_strictly_below_the_stop_backstop(self) -> None:
        assert EmbeddingWorker.timeout_s < STOP_TIMEOUT

    def test_the_old_name_is_gone(self) -> None:
        import akgentic.tool.vector_store.embedding_actor as module

        assert not hasattr(module, "EmbeddingActor")
        assert not hasattr(module, "EmbeddingCompleted")


# ---------------------------------------------------------------------------
# AC 2 — the message models
# ---------------------------------------------------------------------------


class TestMessageModels:
    """AC 2: field sets and round trips."""

    def test_request_is_a_deferred_payload(self) -> None:
        from akgentic.tool.core.deferred import DeferredPayload

        assert issubclass(EmbeddingRequest, DeferredPayload)

    def test_request_round_trips_with_every_field_non_default(self) -> None:
        request = EmbeddingRequest(
            deferred_key="req-9",
            collection="c9",
            entries=_entries(),
            request_ref="some/path.md",
            embedding_model="text-embedding-3-large",
            embedding_provider="azure",
        )
        restored = EmbeddingRequest.model_validate(request.model_dump())
        assert restored.deferred_key == "req-9"
        assert restored.collection == "c9"
        assert restored.request_ref == "some/path.md"
        assert restored.embedding_model == "text-embedding-3-large"
        assert restored.embedding_provider == "azure"
        assert [entry.ref_id for entry in restored.entries] == ["e1", "e2"]
        assert restored.entries[0].scope == "ws-1"
        assert restored.entries[0].ordinal == 0

    def test_result_round_trips(self) -> None:
        result = EmbeddingResult(
            collection="c1", entries=_entries(), request_id="r1", request_ref="p1"
        )
        restored = EmbeddingResult.model_validate(result.model_dump())
        assert restored.request_id == "r1"
        assert restored.request_ref == "p1"
        assert len(restored.entries) == 2

    def test_error_round_trips(self) -> None:
        err = EmbeddingError(collection="c1", error="boom", request_id="r1", request_ref="p1")
        restored = EmbeddingError.model_validate(err.model_dump())
        assert restored.error == "boom"
        assert restored.request_ref == "p1"

    def test_retired_names_are_not_exported(self) -> None:
        import akgentic.tool.vector_store as package

        assert "EmbeddingCompleted" not in package.__all__
        assert "PendingRequest" not in package.__all__
        assert "EmbeddingActor" not in package.__all__


# ---------------------------------------------------------------------------
# AC 3 — produce fills the vectors in place
# ---------------------------------------------------------------------------


class TestProduce:
    """AC 3: Golden Rule 12 — the one field that was produced."""

    def test_the_other_six_fields_survive(self) -> None:
        worker = _make_worker()
        service = MagicMock()
        service.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
        with patch(
            "akgentic.tool.vector_store.embedding_actor.build_embedding_service",
            return_value=service,
        ):
            produced = worker.produce(_make_request())
        assert [entry.vector for entry in produced] == [[0.1, 0.2], [0.3, 0.4]]
        assert [entry.ref_id for entry in produced] == ["e1", "e2"]
        assert [entry.scope for entry in produced] == ["ws-1", "ws-1"]
        assert [entry.path for entry in produced] == ["docs/report.md", "docs/report.md"]
        assert [entry.ordinal for entry in produced] == [0, 1]
        assert [entry.text for entry in produced] == ["hello world", "goodbye"]
        assert [entry.ref_type for entry in produced] == ["workspace_chunk", "workspace_chunk"]

    def test_a_field_the_worker_has_never_heard_of_survives(self) -> None:
        """The strong-form guard: an enumerated rebuild would drop this."""
        worker = _make_worker()
        entry = _EntryWithExtra(
            ref_type="workspace_chunk", ref_id="e1", text="hello", vector=[]
        )
        service = MagicMock()
        service.embed.return_value = [[0.5]]
        with patch(
            "akgentic.tool.vector_store.embedding_actor.build_embedding_service",
            return_value=service,
        ):
            produced = worker.produce(_make_request(entries=[entry]))
        assert isinstance(produced[0], _EntryWithExtra)
        assert produced[0].extra_field == "sentinel"
        assert produced[0].vector == [0.5]

    def test_the_texts_are_what_is_embedded(self) -> None:
        worker = _make_worker()
        service = MagicMock()
        service.embed.return_value = [[0.1], [0.2]]
        with patch(
            "akgentic.tool.vector_store.embedding_actor.build_embedding_service",
            return_value=service,
        ):
            worker.produce(_make_request())
        service.embed.assert_called_once_with(["hello world", "goodbye"])

    def test_an_alien_payload_raises(self) -> None:
        from akgentic.tool.core.deferred import DeferredPayload

        worker = _make_worker()
        with pytest.raises(TypeError):
            worker.produce(DeferredPayload(deferred_key="k"))


# ---------------------------------------------------------------------------
# AC 4 — a vector count mismatch is a failure, not a truncation
# ---------------------------------------------------------------------------


class TestVectorCountMismatch:
    """AC 4: one vector for two entries fails the batch."""

    def test_one_vector_for_two_entries_reports_an_error(self) -> None:
        worker = _make_worker()
        service = MagicMock()
        service.embed.return_value = [[0.1]]
        with patch(
            "akgentic.tool.vector_store.embedding_actor.build_embedding_service",
            return_value=service,
        ):
            recorder, _ = _run(worker, _make_request())
        assert recorder.results == []
        assert len(recorder.errors) == 1
        assert recorder.errors[0].request_id == "test-req-1"
        assert recorder.errors[0].request_ref == "docs/report.md"


# ---------------------------------------------------------------------------
# AC 5 — the report path
# ---------------------------------------------------------------------------


class TestTheReportPath:
    """AC 5: the consumer's two handlers, never ``deliver`` / ``fail``."""

    def test_success_calls_the_result_handler_once(self) -> None:
        worker = _make_worker()
        service = MagicMock()
        service.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
        with patch(
            "akgentic.tool.vector_store.embedding_actor.build_embedding_service",
            return_value=service,
        ):
            recorder, mock_stop = _run(worker, _make_request())
        assert len(recorder.results) == 1
        result = recorder.results[0]
        assert result.collection == "test_col"
        assert result.request_id == "test-req-1"
        assert result.request_ref == "docs/report.md"
        assert [entry.vector for entry in result.entries] == [[0.1, 0.2], [0.3, 0.4]]
        assert recorder.errors == []
        mock_stop.assert_called_once()

    def test_success_never_calls_deliver_or_fail(self) -> None:
        worker = _make_worker()
        service = MagicMock()
        service.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
        with patch(
            "akgentic.tool.vector_store.embedding_actor.build_embedding_service",
            return_value=service,
        ):
            recorder, _ = _run(worker, _make_request())
        assert recorder.delivered == []
        assert recorder.failed == []

    def test_failure_calls_the_error_handler_once(self) -> None:
        worker = _make_worker()
        service = MagicMock()
        service.embed.side_effect = RuntimeError("API timeout")
        with patch(
            "akgentic.tool.vector_store.embedding_actor.build_embedding_service",
            return_value=service,
        ):
            recorder, mock_stop = _run(worker, _make_request())
        assert len(recorder.errors) == 1
        err = recorder.errors[0]
        assert err.collection == "test_col"
        assert "API timeout" in err.error
        assert err.request_id == "test-req-1"
        assert err.request_ref == "docs/report.md"
        assert recorder.results == []
        mock_stop.assert_called_once()

    def test_failure_never_calls_deliver_or_fail(self) -> None:
        worker = _make_worker()
        service = MagicMock()
        service.embed.side_effect = RuntimeError("boom")
        with patch(
            "akgentic.tool.vector_store.embedding_actor.build_embedding_service",
            return_value=service,
        ):
            recorder, _ = _run(worker, _make_request())
        assert recorder.delivered == []
        assert recorder.failed == []

    def test_no_parent_tells_nothing_logs_one_warning_and_still_stops(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        worker = _make_worker()
        worker._parent = None
        service = MagicMock()
        service.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
        recorder = _Recorder()
        with (
            patch(
                "akgentic.tool.vector_store.embedding_actor.build_embedding_service",
                return_value=service,
            ),
            patch.object(worker, "proxy_tell", return_value=recorder),
            patch.object(worker, "stop") as mock_stop,
            caplog.at_level("WARNING"),
        ):
            worker.receiveMsg_DeferredPayload(_make_request())
        assert recorder.results == []
        assert recorder.errors == []
        assert len(caplog.records) == 1
        mock_stop.assert_called_once()

    def test_an_alien_payload_stops_without_telling_anything(self) -> None:
        from akgentic.tool.core.deferred import DeferredPayload

        worker = _make_worker()
        recorder = _Recorder()
        with (
            patch.object(worker, "proxy_tell", return_value=recorder),
            patch.object(worker, "stop") as mock_stop,
        ):
            worker.receiveMsg_DeferredPayload(DeferredPayload(deferred_key="k"))
        assert recorder.results == []
        assert recorder.errors == []
        mock_stop.assert_called_once()


# ---------------------------------------------------------------------------
# AC 6 — the budget reaches the client
# ---------------------------------------------------------------------------


class TestTheBudgetReachesTheClient:
    """AC 6: ``timeout=`` on the one call the worker makes."""

    def test_built_service_passes_the_worker_budget(self) -> None:
        client = MagicMock()
        client.embeddings.create.return_value = MagicMock(data=[MagicMock(embedding=[0.1])])
        with patch("openai.OpenAI", return_value=client):
            service = build_embedding_service("m", "openai")
            service.embed(["test"])
        client.embeddings.create.assert_called_once_with(
            input=["test"], model="m", timeout=EmbeddingWorker.timeout_s
        )

    def test_a_service_without_a_budget_passes_no_timeout(self) -> None:
        from akgentic.tool.vector_store.vector import EmbeddingService

        client = MagicMock()
        client.embeddings.create.return_value = MagicMock(data=[MagicMock(embedding=[0.1])])
        with patch("openai.OpenAI", return_value=client):
            EmbeddingService(model="m", provider="openai").embed(["test"])
        client.embeddings.create.assert_called_once_with(input=["test"], model="m")

    def test_the_built_service_carries_the_model_and_provider(self) -> None:
        with patch(
            "akgentic.tool.vector_store.vector.EmbeddingService"
        ) as service_cls:
            build_embedding_service("text-embedding-3-large", "azure")
        service_cls.assert_called_once_with(
            model="text-embedding-3-large",
            provider="azure",
            timeout_s=EmbeddingWorker.timeout_s,
        )


# ---------------------------------------------------------------------------
# AC 7 — the worker name
# ---------------------------------------------------------------------------


class TestWorkerName:
    """AC 7: the ``#embed-`` prefix and the id suffix."""

    def test_name_carries_the_prefix_the_collection_and_the_id(self) -> None:
        name = embedding_worker_name("c", "0123456789abcdef-more")
        assert name.startswith("#embed-c-")
        assert "0123456789ab" in name

    def test_the_prefix_starts_with_a_hash(self) -> None:
        assert EMBED_WORKER_NAME_PREFIX.startswith("#")

    def test_worker_role_is_the_deferred_role(self) -> None:
        assert WORKER_ROLE == "ToolActor"
