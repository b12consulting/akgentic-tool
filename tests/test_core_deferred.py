"""Unit tests for the deferred-result mechanism (``akgentic.tool.core.deferred``).

Cheap and mock-based: actors are instantiated directly and ``createActor`` /
``proxy_tell`` are patched, following ``tests/vector_store/test_embedding_actor.py``.
The two tests that genuinely need a running actor system (``get`` while a worker is
in flight, and team teardown) live in ``tests/test_core_deferred_teardown.py``.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.core.messages.message import Message
from akgentic.core.orchestrator import STOP_TIMEOUT
from akgentic.core.utils.serializer import SerializableBaseModel

from akgentic.tool.core.deferred import (
    DeferredPayload,
    DeferredResultActor,
    DeferredWorker,
    poll_deferred,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubWorker(DeferredWorker):
    """Worker that produces a fixed value without touching anything external."""

    def produce(self, payload: DeferredPayload) -> Any:
        return f"value-for-{payload.deferred_key}"


class _RaisingWorker(DeferredWorker):
    """Worker whose production always fails."""

    def produce(self, payload: DeferredPayload) -> Any:
        raise RuntimeError("boom")


class _NoneWorker(DeferredWorker):
    """Worker that produces nothing — reported as a failure, not cached."""

    def produce(self, payload: DeferredPayload) -> Any:
        return None


class _StubCache(DeferredResultActor[BaseConfig, BaseState, str, str]):
    """Minimal concrete cache actor over ``str`` keys and ``str`` values."""

    def worker_class(self) -> type[DeferredWorker]:
        return _StubWorker


def _make_cache(capacity: int = 128, ttl: float = 60.0) -> _StubCache:
    """Build an initialised cache actor without starting a real actor system."""
    actor = _StubCache()
    actor.config = BaseConfig(name="#StubCache", role="ToolActor")
    actor.cache_capacity = capacity
    actor.negative_ttl_s = ttl
    actor.on_start()
    return actor


def _make_worker(worker_class: type[DeferredWorker] = _StubWorker) -> DeferredWorker:
    """Build an initialised worker with a mocked parent address."""
    worker = worker_class()
    worker.config = BaseConfig(name="#defer-k1", role="ToolActor")
    worker._parent = MagicMock()
    worker.on_start()
    return worker


def _payload(key: str = "k1") -> DeferredPayload:
    return DeferredPayload(deferred_key=key)


# ---------------------------------------------------------------------------
# AC10 — payloads are SerializableBaseModel, never Message
# ---------------------------------------------------------------------------


class TestPayloadBaseClass:
    """AC10: the payload must stay out of the orchestrator telemetry path."""

    def test_payload_is_serializable_base_model(self) -> None:
        assert issubclass(DeferredPayload, SerializableBaseModel)

    def test_payload_is_not_a_message(self) -> None:
        """A ``Message`` payload would emit Received/Processed telemetry, which is
        exactly what makes a transient worker look like a busy team member."""
        assert not issubclass(DeferredPayload, Message)

    def test_payload_carries_its_key(self) -> None:
        assert _payload("k9").deferred_key == "k9"


# ---------------------------------------------------------------------------
# AC6 — poll_deferred
# ---------------------------------------------------------------------------


class TestPollDeferred:
    """AC6: bounded caller-side poll with a degraded answer."""

    def test_returns_first_non_none_immediately(self) -> None:
        results: list[str | None] = [None, "ready", "never-reached"]
        calls: list[int] = []

        def fetch() -> str | None:
            calls.append(1)
            return results[len(calls) - 1]

        with patch("akgentic.tool.core.deferred.time.sleep") as mock_sleep:
            assert poll_deferred(fetch, attempts=3, delay=0.01) == "ready"
        assert len(calls) == 2
        assert mock_sleep.call_count == 1

    def test_returns_none_when_budget_exhausted(self) -> None:
        fetch = MagicMock(return_value=None)
        with patch("akgentic.tool.core.deferred.time.sleep") as mock_sleep:
            assert poll_deferred(fetch, attempts=3, delay=0.01) is None
        assert fetch.call_count == 3
        # Sleeps BETWEEN attempts only: never before the first, never after the last.
        assert mock_sleep.call_count == 2

    def test_zero_attempts_performs_no_call(self) -> None:
        """A deployment opting out of polling entirely."""
        fetch = MagicMock(return_value="ready")
        assert poll_deferred(fetch, attempts=0) is None
        fetch.assert_not_called()

    def test_signature_defaults(self) -> None:
        fetch = MagicMock(return_value=None)
        with patch("akgentic.tool.core.deferred.time.sleep") as mock_sleep:
            assert poll_deferred(fetch) is None
        assert fetch.call_count == 5
        assert mock_sleep.call_args_list == [((0.4,), {})] * 4


# ---------------------------------------------------------------------------
# AC1/AC3 — the four-method contract and de-duplication
# ---------------------------------------------------------------------------


class TestRequestSpawnsOneWorker:
    """AC1/AC3: ``request`` spawns exactly one worker per key."""

    def test_three_requests_spawn_one_worker(self) -> None:
        actor = _make_cache()
        with (
            patch.object(actor, "createActor", return_value=MagicMock()) as mock_create,
            patch.object(actor, "proxy_tell", return_value=MagicMock()),
        ):
            for _ in range(3):
                actor.request("k1", _payload())
        assert mock_create.call_count == 1

    def test_worker_name_starts_with_hash(self) -> None:
        """AC5: the teardown invariant — a worker must read as a tool actor."""
        actor = _make_cache()
        with (
            patch.object(actor, "createActor", return_value=MagicMock()) as mock_create,
            patch.object(actor, "proxy_tell", return_value=MagicMock()),
        ):
            actor.request("k1", _payload())
        config = mock_create.call_args.kwargs["config"]
        assert config.name.startswith("#")

    def test_request_forwards_the_payload_to_the_worker(self) -> None:
        actor = _make_cache()
        proxy = MagicMock()
        payload = _payload()
        with (
            patch.object(actor, "createActor", return_value=MagicMock()),
            patch.object(actor, "proxy_tell", return_value=proxy),
        ):
            actor.request("k1", payload)
        proxy.receiveMsg_DeferredPayload.assert_called_once_with(payload)

    def test_request_binds_the_payload_key_to_the_cache_key(self) -> None:
        """A payload whose key drifted from the request key would wedge the key:
        the worker would clear a different in-flight mark and this one would stay
        set for the life of the actor."""
        actor = _make_cache()
        proxy = MagicMock()
        with (
            patch.object(actor, "createActor", return_value=MagicMock()),
            patch.object(actor, "proxy_tell", return_value=proxy),
        ):
            actor.request("k1", _payload("some-other-key"))
        forwarded = proxy.receiveMsg_DeferredPayload.call_args.args[0]
        assert forwarded.deferred_key == "k1"

    def test_spawn_failure_does_not_wedge_the_key(self) -> None:
        """A worker that never starts can never report, so ``request`` itself has
        to clear the in-flight mark — otherwise the key is dead for ever."""
        actor = _make_cache(ttl=0.05)
        with (
            patch.object(actor, "createActor", side_effect=RuntimeError("no thread")),
            patch.object(actor, "proxy_tell", return_value=MagicMock()),
        ):
            actor.request("k1", _payload())
        assert "k1" not in actor._in_flight
        time.sleep(0.08)  # the failure is negatively cached, so wait out its TTL
        with (
            patch.object(actor, "createActor", return_value=MagicMock()) as mock_create,
            patch.object(actor, "proxy_tell", return_value=MagicMock()),
        ):
            actor.request("k1", _payload())
        assert mock_create.call_count == 1

    def test_cached_key_spawns_no_worker(self) -> None:
        actor = _make_cache()
        actor.deliver("k1", "cached")
        with (
            patch.object(actor, "createActor", return_value=MagicMock()) as mock_create,
            patch.object(actor, "proxy_tell", return_value=MagicMock()),
        ):
            actor.request("k1", _payload())
        mock_create.assert_not_called()

    def test_delivery_clears_the_in_flight_mark(self) -> None:
        """A second production is allowed once the first key has been evicted."""
        actor = _make_cache(capacity=1)
        with (
            patch.object(actor, "createActor", return_value=MagicMock()) as mock_create,
            patch.object(actor, "proxy_tell", return_value=MagicMock()),
        ):
            actor.request("k1", _payload())
            actor.deliver("k1", "v1")
            actor.deliver("k2", "v2")  # evicts k1 at capacity 1
            actor.request("k1", _payload())
        assert mock_create.call_count == 2


class TestGet:
    """AC1: ``get`` is an O(1) lookup that degrades to ``None``."""

    def test_get_returns_delivered_value(self) -> None:
        actor = _make_cache()
        actor.deliver("k1", "v1")
        assert actor.get("k1") == "v1"

    def test_get_unknown_key_returns_none(self) -> None:
        assert _make_cache().get("nope") is None

    def test_get_in_flight_key_returns_none(self) -> None:
        actor = _make_cache()
        with (
            patch.object(actor, "createActor", return_value=MagicMock()),
            patch.object(actor, "proxy_tell", return_value=MagicMock()),
        ):
            actor.request("k1", _payload())
        assert actor.get("k1") is None

    def test_get_negatively_cached_key_returns_none(self) -> None:
        """The caller degrades exactly as it does for a key still in flight; it is
        ``request`` that refuses to respawn."""
        actor = _make_cache()
        actor.fail("k1", "boom")
        assert actor.get("k1") is None


# ---------------------------------------------------------------------------
# AC2 — capped LRU cache
# ---------------------------------------------------------------------------


class TestCacheEviction:
    """AC2: the cache is a capped LRU, negative entries included."""

    def test_evicts_least_recently_used_at_cap(self) -> None:
        actor = _make_cache(capacity=2)
        actor.deliver("k1", "v1")
        actor.deliver("k2", "v2")
        actor.deliver("k3", "v3")
        assert actor.get("k1") is None
        assert actor.get("k2") == "v2"
        assert actor.get("k3") == "v3"

    def test_get_refreshes_recency(self) -> None:
        actor = _make_cache(capacity=2)
        actor.deliver("k1", "v1")
        actor.deliver("k2", "v2")
        assert actor.get("k1") == "v1"  # k1 becomes the most recent
        actor.deliver("k3", "v3")
        assert actor.get("k1") == "v1"
        assert actor.get("k2") is None

    def test_negative_entries_occupy_cache_slots(self) -> None:
        """A failure costs a slot, so failures alone can evict a live value."""
        actor = _make_cache(capacity=2)
        actor.deliver("k1", "v1")
        actor.fail("k2", "boom")
        actor.fail("k3", "boom")
        assert actor.get("k1") is None
        with (
            patch.object(actor, "createActor", return_value=MagicMock()) as mock_create,
            patch.object(actor, "proxy_tell", return_value=MagicMock()),
        ):
            actor.request("k3", _payload("k3"))  # still negatively cached
            actor.request("k1", _payload())  # evicted, so it may be produced again
        assert mock_create.call_count == 1


# ---------------------------------------------------------------------------
# AC4 — negative entries and their TTL
# ---------------------------------------------------------------------------


class TestNegativeEntryTtl:
    """AC4: a failure suppresses respawns for its TTL, then stops suppressing."""

    def test_no_respawn_inside_the_ttl(self) -> None:
        actor = _make_cache(ttl=60.0)
        actor.fail("k1", "boom")
        with (
            patch.object(actor, "createActor", return_value=MagicMock()) as mock_create,
            patch.object(actor, "proxy_tell", return_value=MagicMock()),
        ):
            actor.request("k1", _payload())
            actor.request("k1", _payload())
        mock_create.assert_not_called()

    def test_respawn_after_the_ttl_expires(self) -> None:
        actor = _make_cache(ttl=0.05)
        actor.fail("k1", "boom")
        time.sleep(0.08)
        with (
            patch.object(actor, "createActor", return_value=MagicMock()) as mock_create,
            patch.object(actor, "proxy_tell", return_value=MagicMock()),
        ):
            actor.request("k1", _payload())
        assert mock_create.call_count == 1

    def test_failure_clears_the_in_flight_mark(self) -> None:
        actor = _make_cache(ttl=0.05)
        with (
            patch.object(actor, "createActor", return_value=MagicMock()) as mock_create,
            patch.object(actor, "proxy_tell", return_value=MagicMock()),
        ):
            actor.request("k1", _payload())
            actor.fail("k1", "boom")
            time.sleep(0.08)
            actor.request("k1", _payload())
        assert mock_create.call_count == 2

    def test_delivery_clears_a_previous_failure(self) -> None:
        actor = _make_cache()
        actor.fail("k1", "boom")
        actor.deliver("k1", "v1")
        assert actor.get("k1") == "v1"


# ---------------------------------------------------------------------------
# AC5 — the worker
# ---------------------------------------------------------------------------


class TestWorkerTimeoutBudget:
    """AC5: the worker's default budget is below the orchestrator's backstop."""

    def test_default_budget_below_stop_backstop(self) -> None:
        assert DeferredWorker.timeout_s < STOP_TIMEOUT


class TestWorkerDelivery:
    """AC5: produce once, report to the parent, stop — on both paths."""

    def test_delivers_value_to_parent_and_stops(self) -> None:
        worker = _make_worker()
        proxy = MagicMock()
        with (
            patch.object(worker, "proxy_tell", return_value=proxy),
            patch.object(worker, "stop") as mock_stop,
        ):
            worker.receiveMsg_DeferredPayload(_payload())
        proxy.deliver.assert_called_once_with("k1", "value-for-k1")
        mock_stop.assert_called_once()

    def test_failing_produce_reports_failure_and_stops(self) -> None:
        worker = _make_worker(_RaisingWorker)
        proxy = MagicMock()
        with (
            patch.object(worker, "proxy_tell", return_value=proxy),
            patch.object(worker, "stop") as mock_stop,
        ):
            worker.receiveMsg_DeferredPayload(_payload())
        proxy.deliver.assert_not_called()
        assert proxy.fail.call_args.args[0] == "k1"
        assert "boom" in proxy.fail.call_args.args[1]
        mock_stop.assert_called_once()

    def test_produce_returning_none_reports_failure(self) -> None:
        """``None`` is the caller's "no answer" signal, so it is never cached as a
        value — it becomes a negative entry that expires."""
        worker = _make_worker(_NoneWorker)
        proxy = MagicMock()
        with (
            patch.object(worker, "proxy_tell", return_value=proxy),
            patch.object(worker, "stop"),
        ):
            worker.receiveMsg_DeferredPayload(_payload())
        proxy.deliver.assert_not_called()
        proxy.fail.assert_called_once()

    def test_missing_parent_is_logged_and_the_worker_still_stops(self) -> None:
        worker = _make_worker()
        worker._parent = None
        with (
            patch.object(worker, "proxy_tell") as mock_proxy_tell,
            patch.object(worker, "stop") as mock_stop,
        ):
            worker.receiveMsg_DeferredPayload(_payload())
        mock_proxy_tell.assert_not_called()
        mock_stop.assert_called_once()

    def test_missing_parent_on_the_failure_path_still_stops(self) -> None:
        worker = _make_worker(_RaisingWorker)
        worker._parent = None
        with (
            patch.object(worker, "proxy_tell") as mock_proxy_tell,
            patch.object(worker, "stop") as mock_stop,
        ):
            worker.receiveMsg_DeferredPayload(_payload())
        mock_proxy_tell.assert_not_called()
        mock_stop.assert_called_once()

    def test_a_failing_report_does_not_escape_the_handler(self) -> None:
        worker = _make_worker(_RaisingWorker)
        proxy = MagicMock()
        proxy.fail.side_effect = RuntimeError("parent unreachable")
        with (
            patch.object(worker, "proxy_tell", return_value=proxy),
            patch.object(worker, "stop") as mock_stop,
        ):
            worker.receiveMsg_DeferredPayload(_payload())
        mock_stop.assert_called_once()


class TestCacheActorContract:
    """AC1: the cache actor exposes exactly the four methods plus the factory."""

    def test_worker_class_is_declared_by_the_subclass(self) -> None:
        assert _make_cache().worker_class() is _StubWorker

    def test_worker_name_echoes_the_key_without_guaranteeing_uniqueness(self) -> None:
        actor = _make_cache()
        assert actor._worker_name("abc") == "#defer-abc"
