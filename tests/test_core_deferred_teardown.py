"""Real-actor tests for the deferred-result mechanism.

These are the only two tests in this package that start an ``ActorSystem``; every
other deferred-result test is mock-based and lives in ``test_core_deferred.py``.
Both assertions here are impossible to make against mocks:

* ``get`` while a worker is in flight is a WALL-CLOCK claim — a mock-based version
  passes even when the cache actor is fully blocked, which is the exact failure it
  exists to catch.
* the teardown test depends on the orchestrator's real phase-2 stop gate, which
  reads the live, telemetry-derived team roster.

Both are threading-based, never asyncio, so the package's ``asyncio_mode``
divergence between CI (strict) and local runs (auto) cannot bite here.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pykka
import pytest
from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.agent import Akgent
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.core.orchestrator import Orchestrator

from akgentic.tool.core.deferred import (
    DeferredPayload,
    DeferredResultActor,
    DeferredWorker,
    poll_deferred,
)
from akgentic.tool.vector import VectorEntry
from akgentic.tool.vector_store.actor import VectorStoreActor
from akgentic.tool.vector_store.embedding_actor import EmbeddingActor
from akgentic.tool.vector_store.protocol import VectorStoreConfig

# How long the deliberately slow production takes. Everything else is measured
# against it, so the margins stay structural rather than a timing race.
SLOW_PRODUCE_S = 2.0

# A `get` must return in a small fraction of that — a blocked cache actor makes it
# wait the FULL production, not slightly longer.
GET_BUDGET_S = 0.5

_worker_entered = threading.Event()
_worker_left = threading.Event()


@pytest.fixture(autouse=True)
def cleanup_actors() -> Generator[None, None, None]:
    """Stop leaked actors after each test so a failure cannot bleed into the suite."""
    _worker_entered.clear()
    _worker_left.clear()
    yield
    pykka.ActorRegistry.stop_all()


# ---------------------------------------------------------------------------
# AC7 — `get` returns while a worker is in flight (wall-clock)
# ---------------------------------------------------------------------------


class _SlowWorker(DeferredWorker):
    """Production that takes ``SLOW_PRODUCE_S``, signalling entry and exit."""

    def produce(self, payload: DeferredPayload) -> Any:
        _worker_entered.set()
        time.sleep(SLOW_PRODUCE_S)
        _worker_left.set()
        return f"value-for-{payload.deferred_key}"


class _SlowCache(DeferredResultActor[BaseConfig, BaseState, str, str]):
    def worker_class(self) -> type[DeferredWorker]:
        return _SlowWorker


def test_get_returns_while_a_worker_is_in_flight() -> None:
    """The cache actor's thread must stay free while a worker does the slow call.

    Timed only once the worker is provably inside ``produce`` (via the entry
    event), so the assertion carries no start-up race and no weakening.
    """
    system = ActorSystem()
    try:
        cache_addr = system.createActor(
            _SlowCache, config=BaseConfig(name="#SlowCache", role="ToolActor")
        )
        system.proxy_tell(cache_addr, _SlowCache).request("k1", DeferredPayload(deferred_key="k1"))
        assert _worker_entered.wait(timeout=10.0), "worker never entered produce()"

        started = time.monotonic()
        value = system.proxy_ask(cache_addr, _SlowCache).get("k1")
        elapsed = time.monotonic() - started

        assert value is None, "value cannot be ready while the worker is still working"
        assert not _worker_left.is_set(), "the worker finished too early to prove anything"
        assert elapsed < GET_BUDGET_S, f"get() blocked behind the worker ({elapsed:.2f}s)"

        # And the value does land once production completes.
        cache_proxy = system.proxy_ask(cache_addr, _SlowCache)
        assert poll_deferred(lambda: cache_proxy.get("k1"), attempts=10, delay=0.4) == "value-for-k1"
    finally:
        system.shutdown(timeout=10)


def test_second_request_while_in_flight_spawns_no_second_worker() -> None:
    """AC3 on a live system: the in-flight set survives a real concurrent ask."""
    system = ActorSystem()
    try:
        cache_addr = system.createActor(
            _SlowCache, config=BaseConfig(name="#SlowCache", role="ToolActor")
        )
        cache_tell = system.proxy_tell(cache_addr, _SlowCache)
        cache_tell.request("k1", DeferredPayload(deferred_key="k1"))
        assert _worker_entered.wait(timeout=10.0)
        cache_tell.request("k1", DeferredPayload(deferred_key="k1"))

        workers = [
            ref
            for ref in pykka.ActorRegistry.get_by_class(_SlowWorker)
            if ref.is_alive()
        ]
        assert len(workers) == 1
    finally:
        system.shutdown(timeout=10)


# ---------------------------------------------------------------------------
# AC9 — team teardown with a worker in flight
# ---------------------------------------------------------------------------

_embed_entered = threading.Event()
_embed_left = threading.Event()
_idle_tool_stopped = threading.Event()

EMBED_SLEEP_S = 2.0
GRACE_S = 8.0

# The unrelated tool must be down well before the in-flight worker finishes.
TOOL_TEARDOWN_BUDGET_S = 1.0


class _IdleTool(Akgent):
    """A second, unrelated tool actor. It does nothing but record its own stop.

    It is what makes this test discriminating: with a worker whose name lacks the
    ``#`` prefix, the orchestrator's phase-2 gate sees a live non-tool member and
    refuses to stop ANY tool actor — including this one, which has nothing to do
    with the work in flight.
    """

    def on_stop(self) -> None:
        _idle_tool_stopped.set()
        super().on_stop()


def _slow_embedding_service(self: EmbeddingActor, model: str, provider: str) -> None:
    """Stand-in for the embedding call: slow, then reports the service as absent.

    Returning ``None`` sends the worker down its ``EmbeddingError`` path, so the
    test needs neither credentials nor the ``[vector_search]`` extras.
    """
    _embed_entered.set()
    time.sleep(EMBED_SLEEP_S)
    _embed_left.set()
    return None


def _build_vector_store_team(system: ActorSystem) -> tuple[Any, Any]:
    """Start an orchestrator with two tool children: a vector store and an idle tool."""
    orch_addr = system.createActor(
        Orchestrator, config=BaseConfig(name="@Orchestrator", role="Orchestrator")
    )
    orch_proxy = system.proxy_ask(orch_addr, Orchestrator)
    vs_addr = orch_proxy.createActor(
        VectorStoreActor,
        config=VectorStoreConfig(name="#VectorStore", role="ToolActor"),
    )
    orch_proxy.createActor(_IdleTool, config=BaseConfig(name="#IdleTool", role="ToolActor"))
    return orch_addr, vs_addr


def test_team_stop_with_an_embedding_worker_in_flight_does_not_ride_the_backstop() -> None:
    """A spawned worker must not count as a non-tool member of the team.

    Written to fail against a worker name without the ``#`` prefix: the phase-2
    gate would then keep ``#IdleTool`` alive until the embedding finished, and the
    whole teardown would serialise behind an unrelated slow call.
    """
    system = ActorSystem()
    with (
        patch.object(EmbeddingActor, "_get_or_create_embedding_svc", _slow_embedding_service),
        patch.object(VectorStoreActor, "_get_backend_for_collection", return_value=MagicMock()),
    ):
        try:
            orch_addr, vs_addr = _build_vector_store_team(system)
            entry = VectorEntry(ref_type="entity", ref_id="e1", text="hello", vector=[])
            system.proxy_tell(vs_addr, VectorStoreActor).add("col1", [entry])
            assert _embed_entered.wait(timeout=10.0), "embedding worker never started"

            stopped: threading.Event = system.proxy_ask(orch_addr, Orchestrator).stop(GRACE_S)

            # The discriminating assertion: an unrelated tool actor tears down while
            # the worker is STILL inside its slow call.
            assert _idle_tool_stopped.wait(timeout=TOOL_TEARDOWN_BUDGET_S), (
                "tool teardown was deferred behind the in-flight worker"
            )
            assert not _embed_left.is_set(), "the worker finished too early to prove anything"

            # And the team stop completes gracefully, well inside the grace period —
            # it never reaches the orchestrator's backstop.
            assert stopped.wait(timeout=GRACE_S - 2.0), "team stop rode the stop backstop"
            assert not orch_addr.is_alive()
        finally:
            system.shutdown(timeout=10)
