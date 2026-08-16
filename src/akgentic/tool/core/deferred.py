"""Deferred-result mechanism — a cache actor that never performs the slow call.

A tool actor is a team singleton with one thread. Slow external work performed
inside a method callers reach through ``proxy_ask`` occupies that thread for the
whole call, and every other member queuing on it blocks. A Pykka ``timeout=`` on
the ask does not help: it abandons the future *without cancelling the work*, so
the actor stays occupied and its mailbox backs up.

This module splits that shape into three pieces:

* :class:`DeferredResultActor` — the cache actor. Spawns, caches, answers ``get``.
  It NEVER blocks on external work.
* :class:`DeferredWorker` — a short-lived, single-shot, ``#``-named child that
  performs exactly one unit of work, reports it to its parent, and stops itself.
* :func:`poll_deferred` — a bounded caller-side poll. ``None`` means the caller
  degrades and proceeds; it never waits without a budget.

Seven rules the mechanism enforces (a partial adoption is not an adoption):

1. The cache actor never performs the slow call.
2. One worker per key, short-lived, self-stopping.
3. The worker's actor name starts with ``#`` — a teardown invariant, see below.
4. ``request`` de-duplicates through the in-flight set: three callers asking for
   one key produce one worker and one external call.
5. Failures are cached negatively, with a TTL — never an uncapped respawn.
6. The cache is capped (LRU), including its negative entries.
7. Callers poll with a bounded budget and always have a degraded answer.

**The ``#`` prefix is load-bearing.** Every actor emits ``StartMessage`` from
``Akgent.__init__``, so every spawned worker shows up in
``Orchestrator.get_team()`` — which is telemetry-derived over the whole tree. The
orchestrator's phase-2 stop gate refuses to stop tool actors while any *non-tool*
member is alive, and "tool actor" means ``name.startswith("#")``. A worker named
``summarize-abc123`` therefore blocks tool-actor teardown until it finishes; for a
call on a long HTTP timeout, that makes every team stop ride the orchestrator's
30 s backstop.

**Second-order, and independent of the name:** a worker is a *child* of the cache
actor and ``Akgent.stop()`` stops its children blocking, so a worker mid-call holds
its parent's stop open whatever it is called. That is why every worker carries a
timeout budget below the orchestrator backstop and MUST hand it to its I/O client.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

from pydantic import Field

from akgentic.core.agent import Akgent
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.core.utils.serializer import SerializableBaseModel

logger = logging.getLogger(__name__)

DEFAULT_CACHE_CAPACITY: int = 128
"""LRU capacity of a cache actor, values and negative entries together.

Conservative on purpose: every distinct key also costs a dead ``ActorAddress`` in
the cache actor's ``_children`` list for the lifetime of the team (nothing prunes
it before teardown), so a large cache is not free even after eviction.
"""

DEFAULT_NEGATIVE_TTL_S: float = 60.0
"""How long a failure suppresses a respawn for its key.

Long enough that a failing dependency is not hammered once per poll, short enough
that a transient failure heals inside a single conversation.
"""

DEFAULT_WORKER_TIMEOUT_S: float = 20.0
"""Worker budget, strictly below the orchestrator's stop backstop (30 s).

A Python thread cannot be cancelled, so this is a *budget*, not a cancel: it is
only real once ``produce`` hands it to whatever client it calls.
"""

WORKER_NAME_PREFIX: str = "#defer-"
"""Worker actor-name prefix. Only the leading ``#`` is load-bearing."""

WORKER_ROLE: str = "ToolActor"
"""Role reported by spawned workers."""

_WORKER_NAME_KEY_CHARS: int = 12
"""How much of the key is echoed into the worker name, for readability only."""


class DeferredPayload(SerializableBaseModel):
    """One unit of work handed to a :class:`DeferredWorker`.

    A ``SerializableBaseModel`` and deliberately **not** a ``Message``:
    ``Akgent.on_receive`` emits the ``ReceivedMessage`` / ``ProcessedMessage``
    telemetry sandwich only for ``Message`` instances. Consumers derive "who is
    working" from exactly those two types, so a ``Message`` payload would make
    every transient worker surface as a busy team member — the mechanism has to
    stay invisible to whatever is built on top of it.

    Subclasses carry the actual work description, and may narrow ``deferred_key``
    to the concrete key type they use (``uuid.UUID``, ``str``, …).
    """

    deferred_key: Any = Field(
        ...,
        description="Cache key this unit of work produces. Hashable; narrowed by subclasses.",
    )


@dataclass(slots=True)
class _CacheSlot[V]:
    """One LRU slot: either a produced value, or a failure with an expiry.

    Runtime bookkeeping local to the cache actor — it never crosses an actor
    boundary, which is why it is a dataclass rather than a serializable model.
    Values and failures share one slot type so they share one capacity budget.
    """

    value: V | None = None
    error: str | None = None
    expires_at: float | None = None


def poll_deferred[V](
    fetch: Callable[[], V | None],
    attempts: int = 5,
    delay: float = 0.4,
) -> V | None:
    """Call *fetch* up to *attempts* times, sleeping *delay* between attempts.

    Returns the first non-``None`` result immediately. Returns ``None`` when the
    budget is exhausted — the caller then degrades and proceeds; it must never
    wait unbounded for a deferred result.

    ``attempts=0`` performs zero calls and returns ``None``, which is how a
    deployment opts out of polling entirely.

    Args:
        fetch: Zero-argument lookup, typically ``lambda: proxy.get(key)``.
        attempts: Maximum number of calls to *fetch*.
        delay: Seconds slept *between* attempts (never before the first, never
            after the last).

    Returns:
        The first non-``None`` value, or ``None`` when the budget is exhausted.
    """
    for attempt in range(attempts):
        value = fetch()
        if value is not None:
            return value
        if attempt < attempts - 1:
            time.sleep(delay)
    return None


class DeferredWorker(Akgent[BaseConfig, BaseState], ABC):
    """Short-lived, single-shot actor that performs one blocking call.

    Spawned by a :class:`DeferredResultActor` with a ``#``-prefixed name, handed
    exactly one :class:`DeferredPayload`, and stopped by itself the moment the
    result — or the failure — has been reported to its parent. It is never
    reused and never accumulates state.
    """

    timeout_s: float = DEFAULT_WORKER_TIMEOUT_S
    """Budget for the external call. Subclasses may lower it, never raise it
    above the orchestrator's stop backstop."""

    def on_start(self) -> None:  # noqa: ANN201
        """Initialise state and attach the observer."""
        self.state = BaseState()
        self.state.observer(self)

    @abstractmethod
    def produce(self, payload: DeferredPayload) -> Any:
        """Perform the one blocking call this worker exists for.

        Implementations MUST hand :attr:`timeout_s` to whatever client they use
        (httpx timeout, model settings, …). A timeout that does not reach the I/O
        client is decoration: a Python thread cannot be cancelled, so the call
        keeps running and the worker keeps its parent's teardown open.

        Args:
            payload: The unit of work, a :class:`DeferredPayload` subclass.

        Returns:
            The produced value, or ``None`` to report failure for this key.

        Raises:
            Exception: Any failure is caught and reported to the parent as a
                negative cache entry.
        """

    def receiveMsg_DeferredPayload(self, msg: DeferredPayload) -> None:
        """Produce, report to the parent, and stop — always.

        ``self.stop()`` runs in a ``finally`` so the worker also goes away on the
        exception path; a worker that survives its unit of work is a leak and,
        while it lives, a member of the team roster.
        """
        try:
            value = self.produce(msg)
            if value is None:
                self._report_failure(msg.deferred_key, "produce() returned None")
            else:
                self._report_value(msg.deferred_key, value)
        except Exception as exc:  # noqa: BLE001
            self._report_failure(msg.deferred_key, str(exc))
        finally:
            self.stop()

    def _report_value(self, key: Any, value: Any) -> None:
        """Tell the parent cache actor to cache *value* under *key*."""
        if self._parent is None:
            logger.warning("[%s] No parent address, cannot deliver result", self.config.name)
            return
        parent_proxy = self.proxy_tell(self._parent, DeferredResultActor)
        parent_proxy.deliver(key, value)

    def _report_failure(self, key: Any, error: str) -> None:
        """Tell the parent cache actor to record a negative entry for *key*."""
        if self._parent is None:
            logger.warning("[%s] No parent address, cannot deliver failure", self.config.name)
            return
        try:
            parent_proxy = self.proxy_tell(self._parent, DeferredResultActor)
            parent_proxy.fail(key, error)
        except Exception as send_exc:  # noqa: BLE001
            logger.warning("[%s] Failed to send failure to parent: %s", self.config.name, send_exc)


class DeferredResultActor[
    ConfigType: BaseConfig,
    StateType: BaseState,
    K: Hashable,
    V,
](Akgent[ConfigType, StateType], ABC):
    """Keyed result cache with de-duplicated, off-thread production.

    The actor's own thread is occupied only by dict lookups, so N members can
    query it concurrently while a production is in flight. Slow work happens in
    :class:`DeferredWorker` children; nothing on this class may block on I/O.

    ``ConfigType`` and ``StateType`` come first in the parameter list because
    ``Akgent`` already declares them; ``K`` (the hashable cache key — ``dict`` and
    ``set`` membership depend on it) and ``V`` (the produced value) follow.
    """

    cache_capacity: int = DEFAULT_CACHE_CAPACITY
    """LRU capacity. Values and negative entries share this one budget."""

    negative_ttl_s: float = DEFAULT_NEGATIVE_TTL_S
    """Seconds a recorded failure suppresses a respawn for its key."""

    def on_start(self) -> None:  # noqa: ANN201
        """Attach the state observer and initialise the deferred bookkeeping.

        A subclass with its own state assigns ``self.state`` first and then calls
        ``super().on_start()``.
        """
        self.state.observer(self)
        self._slots: OrderedDict[K, _CacheSlot[V]] = OrderedDict()
        self._in_flight: set[K] = set()

    @abstractmethod
    def worker_class(self) -> type[DeferredWorker]:
        """Return the worker class this cache actor spawns."""

    def get(self, key: K) -> V | None:
        """ASK. O(1) lookup — never blocks on external work.

        Returns ``None`` for a key that is unknown, still in flight, or
        negatively cached: all three mean "no answer yet, degrade". It is
        :meth:`request` — not this method — that refuses to respawn a worker for
        a key whose failure is still within its TTL.

        A hit counts as a use and refreshes LRU recency.

        Args:
            key: Cache key.

        Returns:
            The cached value, or ``None``.
        """
        slot = self._slots.get(key)
        if slot is None or slot.value is None:
            return None
        self._slots.move_to_end(key)
        return slot.value

    def request(self, key: K, payload: DeferredPayload) -> None:
        """TELL. Spawn exactly one worker for *key*, or do nothing.

        A no-op when the key is already cached, already in flight, or holds a
        live negative entry. The in-flight set — not the worker's actor name — is
        what de-duplicates: three callers asking for one key produce one worker
        and one external call.

        ``payload.deferred_key`` is overwritten with *key*: the worker reports
        back under the payload's key while the in-flight mark is held under this
        one, so a caller that lets the two drift would clear a different mark and
        leave this key in flight for the lifetime of the actor — silently, and
        for ever. Binding them here makes that unrepresentable.

        Args:
            key: Cache key the worker will produce.
            payload: The unit of work; its ``deferred_key`` is set from *key*.
        """
        if key in self._in_flight or self._is_known(key):
            return
        self._in_flight.add(key)
        payload = payload.model_copy(update={"deferred_key": key})
        # createActor, NOT getChildrenOrCreate: the singleton rule applies to the
        # cache actor itself, never to per-key workers.
        try:
            worker = self.createActor(
                self.worker_class(),
                config=BaseConfig(name=self._worker_name(key), role=WORKER_ROLE),
            )
            self.proxy_tell(worker, DeferredWorker).receiveMsg_DeferredPayload(payload)
        except Exception as exc:  # noqa: BLE001
            # A worker that never started can never report, so nothing else would
            # ever clear the mark. Record it as a failure instead: the TTL then
            # governs the retry, exactly as it does for a production that failed.
            logger.warning(
                "[%s] Deferred worker spawn failed for %s: %s", self.config.name, key, exc
            )
            self.fail(key, f"worker spawn failed: {exc}")

    def deliver(self, key: K, value: V) -> None:
        """TELL, from the worker. Cache *value* and clear the in-flight mark."""
        self._in_flight.discard(key)
        self._store(key, _CacheSlot(value=value))

    def fail(self, key: K, error: str) -> None:
        """TELL, from the worker. Record a negative entry, clear the in-flight mark.

        The entry expires after :attr:`negative_ttl_s`, and until then
        :meth:`request` will not spawn another worker for this key. That TTL is
        the whole retry policy — an uncapped respawn would turn one broken
        dependency into one external call per poll.
        """
        self._in_flight.discard(key)
        self._store(
            key,
            _CacheSlot(error=error, expires_at=time.monotonic() + self.negative_ttl_s),
        )

    def _worker_name(self, key: K) -> str:
        """Build the worker's actor name.

        Only the leading ``#`` is load-bearing (it keeps the worker out of the
        orchestrator's non-tool roster). The key suffix is a debugging aid, NOT a
        uniqueness mechanism: de-duplication lives in the in-flight set, so two
        live workers may legally share a name.
        """
        return f"{WORKER_NAME_PREFIX}{str(key)[:_WORKER_NAME_KEY_CHARS]}"

    def _is_known(self, key: K) -> bool:
        """True when *key* holds a value or a still-live negative entry.

        An expired negative entry is dropped here, which frees its LRU slot and
        lets the next :meth:`request` spawn a worker again.
        """
        slot = self._slots.get(key)
        if slot is None:
            return False
        if slot.value is not None:
            return True
        if slot.expires_at is not None and slot.expires_at > time.monotonic():
            return True
        del self._slots[key]
        return False

    def _store(self, key: K, slot: _CacheSlot[V]) -> None:
        """Insert *slot* as the most recently used entry and evict at the cap."""
        self._slots[key] = slot
        self._slots.move_to_end(key)
        while len(self._slots) > self.cache_capacity:
            evicted, _ = self._slots.popitem(last=False)
            logger.debug("[%s] Evicted deferred cache entry %s", self.config.name, evicted)
