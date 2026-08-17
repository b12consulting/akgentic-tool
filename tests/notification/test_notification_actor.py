"""The ``#NotificationTool`` actor: delivery, races, resume, teardown (AC5–8).

Nothing here waits out a real delay. Entries are made due by rewriting ``fire_at``
into the past, and the one test that genuinely exercises the tick thread shortens
the interval instead of sleeping through it.
"""

from __future__ import annotations

import gc
import queue
import threading
import weakref
from collections.abc import Callable, Generator
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from akgentic.core.actor_address import ActorAddress
from akgentic.core.actor_address_impl import ActorAddressImpl, ActorAddressProxy
from akgentic.core.agent import Akgent
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.core.messages.message import Message
from akgentic.tool.notification import actor as actor_module
from akgentic.tool.notification.actor import (
    NOTIFICATION_ACTOR_NAME,
    NotificationActor,
    _tick_loop,
)
from akgentic.tool.notification.models import (
    NotificationConfig,
    NotificationState,
    PendingNotification,
    Tick,
)
from pykka import ActorDeadError, ActorRef

from tests.notification.conftest import FAKE_MESSAGE_PATH, FakeNotificationMessage

_DELIVERED: queue.Queue[Message] = queue.Queue()


class RecordingAgent(Akgent[BaseConfig, BaseState]):
    """Owner stand-in that publishes whatever notification reaches it."""

    def receiveMsg_FakeNotificationMessage(self, msg: FakeNotificationMessage) -> None:
        _DELIVERED.put(msg)


class StateChangeSpy:
    """Counts ``notify_state_change`` calls, which the actor cannot observe itself."""

    def __init__(self) -> None:
        self.calls = 0

    def notify_state_change(self, state: BaseState) -> None:
        self.calls += 1


def _new_actor() -> NotificationActor:
    return NotificationActor(
        config=NotificationConfig(
            name=NOTIFICATION_ACTOR_NAME,
            message_class=FAKE_MESSAGE_PATH,
        )
    )


def _make_due(actor: NotificationActor, notification_id: int) -> None:
    """Rewrite one entry's due time into the past, changing nothing else."""
    entry = actor.state.pending[notification_id]
    actor.state.pending[notification_id] = entry.model_copy(
        update={"fire_at": datetime.now(UTC) - timedelta(seconds=1)}
    )


@pytest.fixture(autouse=True)
def drain_delivered() -> Generator[None, None, None]:
    """Keep one test's deliveries out of the next one's assertions."""
    while not _DELIVERED.empty():
        _DELIVERED.get_nowait()
    yield
    while not _DELIVERED.empty():
        _DELIVERED.get_nowait()


@pytest.fixture
def notifier() -> Generator[NotificationActor, None, None]:
    """A started (but not thread-run) actor, torn down through ``on_stop``."""
    actor = _new_actor()
    actor.on_start()
    yield actor
    actor.on_stop()


@pytest.fixture
def address_factory() -> Generator[Callable[[str], ActorAddress], None, None]:
    """Mint real addresses, keeping their agents alive for the test's duration.

    A stub address will not do here: the actor persists its state on every
    mutation, and only a real address survives that serialization round trip.
    """
    agents: list[Akgent[BaseConfig, BaseState]] = []

    def make(name: str) -> ActorAddress:
        agent = RecordingAgent(config=BaseConfig(name=name, role="tester"))
        agents.append(agent)
        return ActorAddressImpl(agent.actor_ref)

    yield make
    agents.clear()


@pytest.fixture
def live_owner() -> Generator[tuple[ActorRef[RecordingAgent], ActorAddress], None, None]:
    """A genuinely running owner, so a delivery can be observed end to end."""
    ref = RecordingAgent.start(config=BaseConfig(name="alice", role="tester"))
    yield ref, ActorAddressImpl(ref)
    ref.stop()


class TestDelivery:
    def test_a_due_entry_is_delivered_once_to_its_owner(
        self,
        notifier: NotificationActor,
        live_owner: tuple[ActorRef[RecordingAgent], ActorAddress],
    ) -> None:
        _, address = live_owner
        notification_id = notifier.schedule(address, "check the build", 30)
        _make_due(notifier, notification_id)

        notifier.receiveMsg_Tick(Tick())

        delivered = _DELIVERED.get(timeout=2.0)
        assert isinstance(delivered, FakeNotificationMessage)
        assert delivered.content == "check the build"
        assert delivered.type == "notification"
        assert delivered.sender == address, "the notification comes from the owner itself"
        assert notifier.state.pending == {}

        notifier.receiveMsg_Tick(Tick())
        with pytest.raises(queue.Empty):
            _DELIVERED.get(timeout=0.2)

    def test_an_entry_that_is_not_due_stays_pending(
        self,
        notifier: NotificationActor,
        live_owner: tuple[ActorRef[RecordingAgent], ActorAddress],
    ) -> None:
        _, address = live_owner
        notifier.schedule(address, "later", 300)

        notifier.receiveMsg_Tick(Tick())

        assert list(notifier.state.pending) == [1]
        with pytest.raises(queue.Empty):
            _DELIVERED.get(timeout=0.2)

    def test_the_tick_thread_drives_delivery_end_to_end(
        self,
        monkeypatch: pytest.MonkeyPatch,
        live_owner: tuple[ActorRef[RecordingAgent], ActorAddress],
    ) -> None:
        """The one test that uses the real thread — with a short interval, not a sleep."""
        monkeypatch.setattr(actor_module, "TICK_INTERVAL_S", 0.05)
        _, address = live_owner
        ref = NotificationActor.start(
            config=NotificationConfig(name=NOTIFICATION_ACTOR_NAME, message_class=FAKE_MESSAGE_PATH)
        )
        try:
            ref.proxy().schedule(address, "self-driven", 0).get(timeout=2.0)
            delivered = _DELIVERED.get(timeout=2.0)
            assert delivered.content == "self-driven"
        finally:
            ref.stop()

    def test_a_dead_owner_drops_the_entry_without_raising(
        self, notifier: NotificationActor, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An agent fired before its notification came due: log it, drop it, move on."""
        ref = RecordingAgent.start(config=BaseConfig(name="gone", role="tester"))
        address = ActorAddressImpl(ref)
        ref.stop()

        notification_id = notifier.schedule(address, "into the void", 30)
        _make_due(notifier, notification_id)

        notifier.receiveMsg_Tick(Tick())  # must not raise

        assert notifier.state.pending == {}
        assert "could not be delivered" in caplog.text

    def test_an_unreachable_owner_address_is_also_dropped(
        self, notifier: NotificationActor, address_factory: Callable[[str], ActorAddress]
    ) -> None:
        """A restored address carries no live ref; the tick must survive it anyway."""
        restored = ActorAddressProxy(address_factory("restored").serialize())
        notification_id = notifier.schedule(restored, "orphan", 30)
        _make_due(notifier, notification_id)

        notifier.receiveMsg_Tick(Tick())

        assert notifier.state.pending == {}


class TestCancelVersusTick:
    def test_an_entry_cancelled_before_its_tick_never_delivers(
        self,
        notifier: NotificationActor,
        live_owner: tuple[ActorRef[RecordingAgent], ActorAddress],
    ) -> None:
        _, address = live_owner
        notification_id = notifier.schedule(address, "cancelled in time", 30)
        _make_due(notifier, notification_id)

        assert notifier.cancel(notification_id, address) is True
        notifier.receiveMsg_Tick(Tick())

        with pytest.raises(queue.Empty):
            _DELIVERED.get(timeout=0.2)

    def test_a_tick_over_an_empty_scan_notifies_nothing(self, notifier: NotificationActor) -> None:
        """A stale tick is a no-op — not a state change, and not an exception."""
        spy = StateChangeSpy()
        notifier.state.observer(spy)
        spy.calls = 0

        notifier.receiveMsg_Tick(Tick())

        assert spy.calls == 0

    def test_a_firing_tick_notifies_the_state_change_once(
        self,
        notifier: NotificationActor,
        live_owner: tuple[ActorRef[RecordingAgent], ActorAddress],
    ) -> None:
        _, address = live_owner
        first = notifier.schedule(address, "one", 30)
        second = notifier.schedule(address, "two", 30)
        _make_due(notifier, first)
        _make_due(notifier, second)

        spy = StateChangeSpy()
        notifier.state.observer(spy)
        spy.calls = 0

        notifier.receiveMsg_Tick(Tick())

        assert spy.calls == 1, "one notification per tick, not one per entry"
        assert notifier.state.pending == {}


class TestOwnershipOnTheActor:
    def test_list_for_returns_only_that_owners_entries(
        self, notifier: NotificationActor, address_factory: Callable[[str], ActorAddress]
    ) -> None:
        alice, bob = address_factory("alice"), address_factory("bob")
        notifier.schedule(alice, "alice's", 30)
        notifier.schedule(bob, "bob's", 30)

        assert [e.content for e in notifier.list_for(alice)] == ["alice's"]
        assert [e.content for e in notifier.list_for(bob)] == ["bob's"]

    def test_cancel_refuses_another_owners_entry_and_keeps_it(
        self, notifier: NotificationActor, address_factory: Callable[[str], ActorAddress]
    ) -> None:
        alice, bob = address_factory("alice"), address_factory("bob")
        notification_id = notifier.schedule(bob, "bob's", 30)

        assert notifier.cancel(notification_id, alice) is False
        assert notifier.state.pending[notification_id].content == "bob's"

    def test_cancel_of_an_unknown_id_is_false(
        self, notifier: NotificationActor, address_factory: Callable[[str], ActorAddress]
    ) -> None:
        assert notifier.cancel(404, address_factory("alice")) is False

    def test_owners_are_matched_by_agent_id_not_by_address_object(
        self, notifier: NotificationActor
    ) -> None:
        """A resumed team holds a different address object for the same agent."""
        agent = RecordingAgent(config=BaseConfig(name="alice", role="tester"))
        scheduled_with = ActorAddressImpl(agent.actor_ref)
        looked_up_with = ActorAddressImpl(agent.actor_ref)
        assert scheduled_with is not looked_up_with

        notifier.schedule(scheduled_with, "same agent", 30)
        assert len(notifier.list_for(looked_up_with)) == 1


class TestResume:
    def test_a_restored_entry_whose_delay_expired_fires_on_the_first_tick(
        self,
        notifier: NotificationActor,
        live_owner: tuple[ActorRef[RecordingAgent], ActorAddress],
    ) -> None:
        """No re-arm code exists, and none is needed: ``fire_at`` is absolute."""
        _, address = live_owner
        long_ago = datetime.now(UTC) - timedelta(hours=1)
        notifier.init_state(
            NotificationState(
                next_id=8,
                pending={
                    7: PendingNotification(
                        notification_id=7,
                        owner=address,
                        content="due while the team was down",
                        created_at=long_ago,
                        fire_at=long_ago + timedelta(seconds=30),
                    )
                },
            )
        )

        notifier.receiveMsg_Tick(Tick())

        delivered = _DELIVERED.get(timeout=2.0)
        assert delivered.content == "due while the team was down"
        assert notifier.state.pending == {}
        assert notifier.state.next_id == 8, "the restored counter is not reset"


class TestTeardownAndRetention:
    def test_the_tick_thread_is_not_alive_after_on_stop(self) -> None:
        actor = _new_actor()
        actor.on_start()
        thread = actor._tick_thread
        assert thread.is_alive()

        actor.on_stop()

        assert not thread.is_alive()

    def test_on_stop_still_tears_down_after_a_failed_on_start(self) -> None:
        """A config this deployment cannot resolve must not cost the stop telemetry.

        Pykka logs a failing ``on_start`` and keeps the actor running, so
        ``on_stop`` has to cope with an actor that never got its tick thread. If
        it raises on the missing thread instead, ``Akgent.on_stop`` never runs
        and the ``#``-prefixed singleton goes without announcing its stop.
        """
        actor = NotificationActor(
            config=NotificationConfig(
                name=NOTIFICATION_ACTOR_NAME,
                message_class="tests.notification.conftest.NotAMessage",
            )
        )
        with pytest.raises(ValueError, match="not a Message subclass"):
            actor.on_start()

        actor.on_stop()  # must not raise

    def test_the_loop_exits_when_the_actor_is_dead(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A dead actor ends the loop; it never ticks a corpse forever."""
        monkeypatch.setattr(actor_module, "TICK_INTERVAL_S", 0.01)
        ref = MagicMock()
        ref.tell.side_effect = ActorDeadError("gone")
        stop_event = threading.Event()

        thread = threading.Thread(target=_tick_loop, args=(stop_event, ref), daemon=True)
        thread.start()
        thread.join(timeout=2.0)

        assert not thread.is_alive()
        assert ref.tell.call_count == 1
        assert not stop_event.is_set(), "it exited on the error, not on a stop"

    def test_the_tick_thread_does_not_root_the_actor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The regression guard for the one real leak: a ``self``-capturing target.

        The thread is still running here, holding the loop's arguments. If those
        arguments included the actor — a bound method as the target would do it —
        the weakref below would still resolve.
        """
        monkeypatch.setattr(actor_module, "TICK_INTERVAL_S", 0.01)
        actor = _new_actor()
        actor.on_start()
        stop_event, thread = actor._stop_event, actor._tick_thread
        actor_ref = weakref.ref(actor)

        del actor
        gc.collect()

        try:
            assert actor_ref() is None, "the tick thread is holding the actor alive"
            assert thread.is_alive()
        finally:
            stop_event.set()
            thread.join(timeout=2.0)

    def test_a_pending_entry_does_not_pin_its_owner(self, notifier: NotificationActor) -> None:
        """``owner`` is data, not a proxy, so an entry cannot keep an agent alive."""
        agent = RecordingAgent(config=BaseConfig(name="alice", role="tester"))
        notifier.schedule(ActorAddressImpl(agent.actor_ref), "outlives its owner", 300)
        agent_ref = weakref.ref(agent)

        del agent
        gc.collect()

        assert agent_ref() is None
        assert len(notifier.state.pending) == 1
