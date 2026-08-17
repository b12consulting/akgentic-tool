"""The ``#NotificationTool`` actor: delivery, races, resume, teardown (AC5–8).

Nothing here waits out a real delay. Entries are made due by rewriting ``fire_at``
into the past, and the one test that genuinely exercises the tick thread shortens
the interval instead of sleeping through it.
"""

from __future__ import annotations

import gc
import logging
import queue
import threading
import time
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
)
from pykka import ActorDeadError, ActorRef

from tests.conftest import MockActorAddress
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


class FakeRoster:
    """``Orchestrator`` stand-in: hands back a team roster and counts its asks."""

    def __init__(self, *members: ActorAddress) -> None:
        self.members = list(members)
        self.calls = 0

    def get_team(self) -> list[ActorAddress]:
        self.calls += 1
        return list(self.members)


class OrchestratorAddress(MockActorAddress):
    """A mock address that can absorb the telemetry an agent tells its orchestrator.

    ``_notify_orchestrator`` reaches straight for ``orchestrator._actor_ref``, which a
    bare ``MockActorAddress`` does not have — construction, delivery and ``on_stop``
    would each raise ``AttributeError`` without it.
    """

    def __init__(self) -> None:
        super().__init__("#Orchestrator", "orchestrator")
        self._actor_ref = MagicMock()


def _give_roster(actor: NotificationActor, roster: FakeRoster) -> None:
    """Point *actor* at an orchestrator whose ask returns *roster*."""
    actor._orchestrator = OrchestratorAddress()
    actor.proxy_ask = lambda *_args, **_kwargs: roster  # type: ignore[method-assign]


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

        notifier.deliver_due()

        delivered = _DELIVERED.get(timeout=2.0)
        assert isinstance(delivered, FakeNotificationMessage)
        assert delivered.content == "check the build"
        assert delivered.type == "notification"
        assert delivered.sender is not None
        assert delivered.sender.name == NOTIFICATION_ACTOR_NAME, (
            "the actor sends as itself — no owner-proxy hop to spoof the sender"
        )
        assert notifier.state.pending == {}

        notifier.deliver_due()
        with pytest.raises(queue.Empty):
            _DELIVERED.get(timeout=0.2)

    def test_an_entry_that_is_not_due_stays_pending(
        self,
        notifier: NotificationActor,
        live_owner: tuple[ActorRef[RecordingAgent], ActorAddress],
    ) -> None:
        _, address = live_owner
        notifier.schedule(address, "later", 300)

        notifier.deliver_due()

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

        notifier.deliver_due()  # must not raise

        assert notifier.state.pending == {}
        assert "could not be delivered" in caplog.text

    def test_an_unreachable_owner_address_is_also_dropped(
        self, notifier: NotificationActor, address_factory: Callable[[str], ActorAddress]
    ) -> None:
        """A restored address carries no live ref; the tick must survive it anyway."""
        restored = ActorAddressProxy(address_factory("restored").serialize())
        notification_id = notifier.schedule(restored, "orphan", 30)
        _make_due(notifier, notification_id)

        notifier.deliver_due()

        assert notifier.state.pending == {}


class TestAvailabilityGuard:
    """Delivery waits for an owner that is off the team, for a bounded while."""

    def test_a_due_entry_whose_owner_is_on_the_roster_is_delivered(
        self,
        notifier: NotificationActor,
        live_owner: tuple[ActorRef[RecordingAgent], ActorAddress],
    ) -> None:
        _, address = live_owner
        _give_roster(notifier, FakeRoster(MockActorAddress("alice")))
        notification_id = notifier.schedule(address, "the build is red", 30)
        _make_due(notifier, notification_id)

        notifier.deliver_due()

        assert _DELIVERED.get(timeout=2.0).content == "the build is red"
        assert notifier.state.pending == {}

    def test_a_due_entry_whose_owner_is_absent_is_postponed_untouched(
        self,
        notifier: NotificationActor,
        live_owner: tuple[ActorRef[RecordingAgent], ActorAddress],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Postponing costs nothing: no send, no mutation, no checkpoint, no log line."""
        _, address = live_owner
        _give_roster(notifier, FakeRoster(MockActorAddress("someone-else")))
        notification_id = notifier.schedule(address, "waiting for alice", 30)
        _make_due(notifier, notification_id)
        before = notifier.state.pending[notification_id]

        spy = StateChangeSpy()
        notifier.state.observer(spy)
        spy.calls = 0

        with caplog.at_level(logging.WARNING):
            notifier.deliver_due()

        assert notifier.state.pending[notification_id] == before, "postponing mutates nothing"
        assert spy.calls == 0, "a scan that only postpones has changed no state"
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == [], (
            "a per-tick warning would be one line a second, for every absent owner"
        )
        with pytest.raises(queue.Empty):
            _DELIVERED.get(timeout=0.2)

    def test_an_owner_that_returns_within_the_grace_window_is_delivered(
        self,
        notifier: NotificationActor,
        live_owner: tuple[ActorRef[RecordingAgent], ActorAddress],
    ) -> None:
        """The postponed entry is re-examined every tick, so a returning owner gets it."""
        _, address = live_owner
        roster = FakeRoster()
        _give_roster(notifier, roster)
        notification_id = notifier.schedule(address, "welcome back", 30)
        _make_due(notifier, notification_id)

        notifier.deliver_due()
        assert notification_id in notifier.state.pending

        roster.members.append(MockActorAddress("alice"))
        notifier.deliver_due()

        assert _DELIVERED.get(timeout=2.0).content == "welcome back"
        assert notifier.state.pending == {}
        assert roster.calls == 2, "one ask per firing scan, and both scans found work"

    def test_an_absent_owner_past_the_grace_window_is_dropped_with_one_warning(
        self,
        notifier: NotificationActor,
        live_owner: tuple[ActorRef[RecordingAgent], ActorAddress],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The bound is what keeps ``pending`` self-draining, and it is loud."""
        _, address = live_owner
        _give_roster(notifier, FakeRoster())
        long_ago = datetime.now(UTC) - timedelta(hours=1)
        notifier.init_state(
            NotificationState(
                next_id=2,
                pending={
                    1: PendingNotification(
                        notification_id=1,
                        owner=address,
                        content="nobody came back",
                        created_at=long_ago,
                        fire_at=long_ago,
                    )
                },
            )
        )
        spy = StateChangeSpy()
        notifier.state.observer(spy)
        spy.calls = 0

        with caplog.at_level(logging.WARNING):
            notifier.deliver_due()

        assert notifier.state.pending == {}
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1, "one line for the drop, not one a tick for the wait"
        assert "1" in warnings[0].getMessage()
        assert "alice" in warnings[0].getMessage()
        assert spy.calls == 1
        with pytest.raises(queue.Empty):  # no speculative send to an absent owner
            _DELIVERED.get(timeout=0.2)

    def test_one_firing_scan_asks_the_orchestrator_exactly_once(
        self, notifier: NotificationActor
    ) -> None:
        first = RecordingAgent.start(config=BaseConfig(name="alice", role="tester"))
        second = RecordingAgent.start(config=BaseConfig(name="bob", role="tester"))
        try:
            roster = FakeRoster(MockActorAddress("alice"), MockActorAddress("bob"))
            _give_roster(notifier, roster)
            for ref, content in ((first, "alice's"), (second, "bob's")):
                _make_due(notifier, notifier.schedule(ActorAddressImpl(ref), content, 30))

            notifier.deliver_due()

            assert roster.calls == 1, "the roster is read once a scan, never once an entry"
            assert notifier.state.pending == {}
        finally:
            first.stop()
            second.stop()

    def test_an_idle_scan_never_touches_the_orchestrator(
        self,
        notifier: NotificationActor,
        live_owner: tuple[ActorRef[RecordingAgent], ActorAddress],
    ) -> None:
        """On an idle team the orchestrator sees nothing from this actor, ever."""
        _, address = live_owner
        roster = FakeRoster(MockActorAddress("alice"))
        _give_roster(notifier, roster)

        notifier.deliver_due()
        assert roster.calls == 0, "nothing pending, nothing asked"

        notifier.schedule(address, "not yet", 300)
        notifier.deliver_due()

        assert roster.calls == 0, "pending but nothing due is still an idle scan"

    def test_without_an_orchestrator_every_due_entry_is_delivered(
        self,
        notifier: NotificationActor,
        live_owner: tuple[ActorRef[RecordingAgent], ActorAddress],
    ) -> None:
        """The harness path: no roster to consult means deliver, not withhold."""
        _, address = live_owner
        roster = FakeRoster()
        notifier.proxy_ask = lambda *_args, **_kwargs: roster  # type: ignore[method-assign]
        assert notifier.orchestrator is None

        _make_due(notifier, notifier.schedule(address, "harness path", 30))

        notifier.deliver_due()

        assert _DELIVERED.get(timeout=2.0).content == "harness path"
        assert notifier.state.pending == {}
        assert roster.calls == 0, "no orchestrator, no ask"


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
        notifier.deliver_due()

        with pytest.raises(queue.Empty):
            _DELIVERED.get(timeout=0.2)

    def test_a_tick_over_an_empty_scan_notifies_nothing(self, notifier: NotificationActor) -> None:
        """A stale tick is a no-op — not a state change, and not an exception."""
        spy = StateChangeSpy()
        notifier.state.observer(spy)
        spy.calls = 0

        notifier.deliver_due()

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

        notifier.deliver_due()

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

    def test_list_for_none_returns_every_owners_entries_in_id_order(
        self, notifier: NotificationActor, address_factory: Callable[[str], ActorAddress]
    ) -> None:
        """``None`` is the widening the card's ``all=True`` passes down."""
        alice, bob = address_factory("alice"), address_factory("bob")
        notifier.schedule(bob, "bob's", 30)
        notifier.schedule(alice, "alice's", 30)

        assert [e.content for e in notifier.list_for(None)] == ["bob's", "alice's"]

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

        notifier.deliver_due()

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
        ref.proxy.side_effect = ActorDeadError("gone")
        stop_event = threading.Event()

        thread = threading.Thread(target=_tick_loop, args=(stop_event, ref), daemon=True)
        thread.start()
        thread.join(timeout=2.0)

        assert not thread.is_alive()
        assert ref.proxy.call_count == 1
        assert not stop_event.is_set(), "it exited on the error, not on a stop"

    def test_a_failing_tick_is_logged_and_retried_rather_than_ending_the_loop(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Only a dead actor retires the thread; anything else ticks again.

        Building the proxy runs pykka's attribute introspection on the timer
        thread, so the failure surface is wider than a bare tell's. An escaping
        exception would stop delivery for the team's lifetime with nothing to
        report it — the bounded join in ``on_stop`` would still succeed.
        """
        monkeypatch.setattr(actor_module, "TICK_INTERVAL_S", 0.01)
        ref = MagicMock()
        ref.proxy.side_effect = RuntimeError("dict mutated during update")
        stop_event = threading.Event()

        thread = threading.Thread(target=_tick_loop, args=(stop_event, ref), daemon=True)
        with caplog.at_level(logging.WARNING):
            thread.start()
            time.sleep(0.1)
            stop_event.set()
            thread.join(timeout=2.0)

        assert not thread.is_alive()
        assert ref.proxy.call_count > 1, "it kept ticking after the failure"
        assert "Notification tick failed" in caplog.text

    def test_the_tick_thread_does_not_root_the_actor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The regression guard for the one real leak: a ``self``-capturing target.

        The thread is still running here, holding the loop's arguments. If those
        arguments included the actor — a bound method as the target would do it —
        the weakref below would still resolve.

        The interval is long on purpose: the thread must be parked in its wait
        when the collection happens, so it provably still holds those arguments.
        A short one lets it tick, notice the collected actor and exit first,
        which would leave the guard asserting over a thread that had already
        dropped everything it held.
        """
        monkeypatch.setattr(actor_module, "TICK_INTERVAL_S", 30.0)
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

    def test_a_collected_actor_also_ends_the_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A collected actor retires its thread, not only a stopped one.

        The old tell went on filling the inbox of an actor nobody could reach;
        the proxy call raises ``ActorDeadError`` for a deallocated actor as
        readily as for a stopped one, so the thread ends itself.
        """
        monkeypatch.setattr(actor_module, "TICK_INTERVAL_S", 0.01)
        actor = _new_actor()
        actor.on_start()
        stop_event, thread = actor._stop_event, actor._tick_thread

        del actor
        gc.collect()
        thread.join(timeout=2.0)

        assert not thread.is_alive()
        assert not stop_event.is_set(), "it exited on the dead actor, not on a stop"

    def test_a_pending_entry_does_not_pin_its_owner(self, notifier: NotificationActor) -> None:
        """``owner`` is data, not a proxy, so an entry cannot keep an agent alive."""
        agent = RecordingAgent(config=BaseConfig(name="alice", role="tester"))
        notifier.schedule(ActorAddressImpl(agent.actor_ref), "outlives its owner", 300)
        agent_ref = weakref.ref(agent)

        del agent
        gc.collect()

        assert agent_ref() is None
        assert len(notifier.state.pending) == 1
