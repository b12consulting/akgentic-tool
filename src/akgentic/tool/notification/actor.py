"""``#NotificationTool``: the team singleton that holds pending timers.

The actor owns a dict of :class:`PendingNotification` and one daemon thread that
sends the actor a :class:`NotificationTick` about once a second. The tick is
mailbox-routed, so the scan runs on the actor thread where ``schedule`` and
``cancel`` run: every mutation of ``pending`` happens on one thread, and no lock
is needed (ADR-035 §3).

Delivery of an entry whose ``fire_at`` has passed needs no re-arm logic — a
restored entry is simply due on the first tick after a resume. It does need its
owner to be on the team: a scan that finds work reads the roster once and
postpones, rather than delivers, an entry whose owner is absent, for up to
``DELIVERY_GRACE_S`` (ADR-035 §3).
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime, timedelta

from akgentic.core.actor_address import ActorAddress
from akgentic.core.agent import Akgent
from akgentic.core.messages.message import Message
from akgentic.core.orchestrator import Orchestrator
from akgentic.tool.notification.models import (
    NOTIFICATION_TYPE,
    NotificationConfig,
    NotificationState,
    PendingNotification,
    resolve_message_class,
)

logger = logging.getLogger(__name__)

NOTIFICATION_ACTOR_NAME = "#NotificationTool"
"""Singleton actor name. The ``#`` prefix is the orchestrator's teardown invariant."""

NOTIFICATION_ACTOR_ROLE = "ToolActor"

TICK_INTERVAL_S = 1.0
"""Tick period. Delivery granularity is ±1 s, accepted against a 300 s cap."""

TICK_ASK_TIMEOUT_S = 10.0
"""How long a tick waits for the scan to be handled.

Bounded so a wedged actor cannot park the timer thread forever. Generous against
a scan that is a dict walk plus one cached roster read — a timeout here means
something is wrong, not merely busy, so it is logged rather than absorbed."""

TICK_JOIN_TIMEOUT_S = 2.0
"""Join budget in ``on_stop`` — bounded, because the loop observes the stop
event within one tick."""

DELIVERY_GRACE_S = 300.0
"""How long a due entry waits for an owner that is off the team.

The bound is what keeps ``pending`` self-draining: an owner that never comes
back would otherwise pin its entry for the team's lifetime, and unbounded
postponement turns the wait into the leak the scan was designed to avoid."""


def _tick_loop(stop_event: threading.Event, address: ActorAddress) -> None:
    """Tick the actor every ``TICK_INTERVAL_S`` until *stop_event* is set.

    Module-level, and capturing only its two arguments: a bound-method target
    would root the actor instance for as long as the thread lives, which is the
    one real leak this design could have. An ``ActorAddress`` holds the actor
    weakly, so the parked thread never keeps it alive.

    The tick is a **message, not a proxied method call**. Building a Pykka proxy
    takes a strong reference to the actor and introspects its attributes, so a
    retained proxy is exactly the leak above and a per-tick one costs a
    quarter-millisecond of introspection to send one small message.

    ``ask`` rather than ``tell``, and bounded: waiting for the scan to be handled
    is what keeps a slow actor from being handed ticks faster than it drains
    them, and ``TICK_ASK_TIMEOUT_S`` is what keeps a wedged one from parking this
    thread for the team's lifetime.

    Only a dead actor ends the loop. Anything else — a timeout, a transient
    failure inside the scan — is logged and retried on the next tick: a single
    escaping exception would retire the thread for the team's lifetime, every
    pending notification silently undelivered, with the bounded join in
    ``on_stop`` still succeeding so nothing would report it.

    Args:
        stop_event: Set by ``on_stop`` to end the loop.
        address: Address of the notification actor (holds it weakly).
    """
    while not stop_event.wait(TICK_INTERVAL_S):
        try:
            address.ask(NotificationTick(), timeout=TICK_ASK_TIMEOUT_S)
        except Exception:  # noqa: BLE001 — a failed tick must not retire the thread
            if not address.is_alive():
                break
            logger.warning("Notification tick failed; retrying on the next tick", exc_info=True)


def _same_owner(left: ActorAddress, right: ActorAddress) -> bool:
    """Compare two addresses by ``agent_id``.

    Identity is the agent, not the address object: a resumed team holds a
    deserialized address for the same agent, and the two classes do not compare
    equal to each other.
    """
    return left.agent_id == right.agent_id


class NotificationTick:
    """The timer thread's request for a delivery scan.

    Deliberately not a :class:`~akgentic.core.messages.message.Message`: the
    orchestrator's telemetry sandwich fires on those, so a tick modelled as one
    would emit a Received/Processed pair every second for every team. A plain
    object still routes to ``receiveMsg_NotificationTick`` through the MRO
    dispatcher, and carries no conversational meaning to anything else.
    """


class NotificationActor(Akgent[NotificationConfig, NotificationState]):
    """Team singleton holding pending notifications and delivering them on time.

    Every ask-reachable method is a dict operation, and the one cross-actor call
    — the roster read a firing scan makes — is an in-memory lookup on the
    orchestrator's cached team. Nothing slow ever runs on this actor's thread, so
    the deferred-result pattern stays disengaged.
    """

    _stop_event: threading.Event | None = None
    _tick_thread: threading.Thread | None = None

    def on_start(self) -> None:
        """Initialise state, resolve the delivery class, and start the tick thread."""
        super().on_start()
        self.state = NotificationState()
        self.state.observer(self)
        self._message_cls: type[Message] = resolve_message_class(self.config.message_class)
        self._stop_event = threading.Event()
        self._tick_thread = threading.Thread(
            target=_tick_loop,
            args=(self._stop_event, self.myAddress),
            name=f"{NOTIFICATION_ACTOR_NAME}-tick",
            daemon=True,
        )
        self._tick_thread.start()

    def on_stop(self) -> None:
        """Stop the tick thread, then run the base teardown.

        The thread may not exist: pykka logs a failing ``on_start`` and runs the
        actor anyway, so a config naming a delivery class this deployment cannot
        import leaves the actor short of its thread. The base teardown — the
        state checkpoint and the stop telemetry the ``#`` prefix relies on — has
        to run in that case too, which it would not if this raised first.
        """
        if self._stop_event is not None:
            self._stop_event.set()
        if self._tick_thread is not None:
            self._tick_thread.join(timeout=TICK_JOIN_TIMEOUT_S)
        super().on_stop()

    def receiveMsg_NotificationTick(self, msg: NotificationTick) -> None:
        """Run one delivery scan on the actor thread.

        The indirection is what puts :meth:`deliver_due` on the mailbox thread
        without the timer thread holding a proxy to this actor.

        Single-parameter on purpose: the dispatcher passes a sender only to
        handlers that declare a parameter named exactly ``sender``, and a tick
        has none to speak of.

        Args:
            msg: The tick itself, which carries nothing.
        """
        self.deliver_due()

    ##
    ## Ask-path methods — dict operations only
    ##
    def schedule(self, owner: ActorAddress, content: str, delay_seconds: int) -> int:
        """Store a notification for *owner*, due *delay_seconds* from now.

        Args:
            owner: Address of the scheduling agent, and of the recipient.
            content: Content to deliver.
            delay_seconds: Delay before delivery. Range checking belongs to the
                card, which owns the configured cap.

        Returns:
            The new notification id.
        """
        now = datetime.now(UTC)
        notification_id = self.state.next_id
        self.state.next_id += 1
        self.state.pending[notification_id] = PendingNotification(
            notification_id=notification_id,
            owner=owner,
            content=content,
            created_at=now,
            fire_at=now + timedelta(seconds=delay_seconds),
        )
        self.state.notify_state_change()
        return notification_id

    def list_for(self, owner: ActorAddress | None) -> list[PendingNotification]:
        """Return pending entries, oldest id first.

        Args:
            owner: Whose entries to return, or ``None`` for every owner's. The
                parameter is required and positional on purpose — this actor is
                ask-reachable from every card in the team, so widening the
                listing must always be something a caller wrote deliberately
                rather than something a forgotten argument grants.

        Returns:
            The matching entries, sorted by ``notification_id`` ascending.
        """
        return [
            entry
            for entry in sorted(self.state.pending.values(), key=lambda e: e.notification_id)
            if owner is None or _same_owner(entry.owner, owner)
        ]

    def cancel(self, notification_id: int, owner: ActorAddress) -> bool:
        """Remove *owner*'s own pending entry.

        Returns:
            True when the entry existed and belonged to *owner*. False for an
            unknown id and for another agent's id alike — the caller turns that
            into a ``RetriableError``, and another agent's entry stays pending.
        """
        entry = self.state.pending.get(notification_id)
        if entry is None or not _same_owner(entry.owner, owner):
            return False
        del self.state.pending[notification_id]
        self.state.notify_state_change()
        return True

    ##
    ## Delivery
    ##
    def deliver_due(self) -> None:
        """Deliver every entry whose owner is on the team, and postpone the rest.

        Called once per tick by the timer thread through this actor's own proxy,
        so it runs on the mailbox thread and cannot interleave with ``schedule``
        or ``cancel``. A scan that finds nothing due is a no-op: no roster read,
        no message, and no state-change notification.

        A scan that does find work reads the roster **once**, then classifies
        every due entry against it. An owner that is off the team has its entry
        left exactly where it is — the ``fire_at <= now`` scan is already a retry
        loop, so the next tick re-examines it and delivers as soon as the owner
        is back. That wait is bounded by ``DELIVERY_GRACE_S`` (ADR-035 §3).
        """
        now = datetime.now(UTC)
        due = [entry for entry in self.state.pending.values() if entry.fire_at <= now]
        if not due:
            return
        present = self._present_member_names()
        mutated = False
        for entry in due:
            if present is not None and entry.owner.name not in present:
                mutated |= self._postpone_or_drop(entry, now)
                continue
            del self.state.pending[entry.notification_id]
            mutated = True
            self._deliver(entry)
        if mutated:
            self.state.notify_state_change()

    def _present_member_names(self) -> set[str] | None:
        """Names of the agents currently on the team, or ``None`` for "no roster".

        Matching is by **name** rather than by ``agent_id``: an entry restored
        with a team resume carries the id the owner had before the restart,
        while the roster holds the freshly started agent. The ids differ, the
        names do not — which is why ``_same_owner`` is deliberately not used
        here.

        ``None`` means there is nothing to check against, and the caller reads
        it as deliver-everything. That is the harness path — in production the
        singleton is reached through ``getChildrenOrCreate`` and always has an
        orchestrator — and ``kg_actor.py`` degrades on the same condition.

        The ask is an in-memory read of a cached roster, so it stays well inside
        what may run on this actor's thread, and the orchestrator never blocking-
        asks a tool actor, so there is no cycle to deadlock on.
        """
        if self.orchestrator is None:
            return None
        orchestrator = self.proxy_ask(self.orchestrator, Orchestrator)
        return {member.name for member in orchestrator.get_team()}

    def _postpone_or_drop(self, entry: PendingNotification, now: datetime) -> bool:
        """Leave *entry* pending for its absent owner, unless the grace has run out.

        Postponing is silent and free by design: no send, no ``fire_at`` bump, no
        state-change notification. A postponed entry is re-examined once a
        second, so anything logged here at WARNING — or persisted here — would
        repeat for as long as the owner stays away.

        Returns:
            True when the entry was dropped, which the caller counts as the
            mutation behind its single end-of-scan notification.
        """
        waited_s = (now - entry.fire_at).total_seconds()
        if waited_s < DELIVERY_GRACE_S:
            return False
        del self.state.pending[entry.notification_id]
        logger.warning(
            "Notification %s waited %.0f s for absent owner %s — dropped",
            entry.notification_id,
            waited_s,
            entry.owner.name,
        )
        return True

    def _deliver(self, entry: PendingNotification) -> None:
        """Send one due entry to its owner.

        This actor sends as itself, so the delivered ``sender`` is
        ``#NotificationTool`` — ``Akgent.send`` stamps the sender of whichever
        actor performs the send. The call is a tell underneath, so a busy owner
        never blocks this actor's mailbox.

        A failure — most often a restored address with no live ref — is logged
        and the entry dropped. It is never retried and never raised: one dead
        recipient must not break the scan for everyone else. An owner *fired*
        before its notification came due no longer arrives here: it leaves the
        roster with its stop, so the availability guard postpones its entry and
        drops it on the grace expiry instead.
        """
        try:
            message = self._message_cls.model_validate(
                {"content": entry.content, "type": NOTIFICATION_TYPE}
            )
            self.send(entry.owner, message)
        except Exception:  # noqa: BLE001 — a dead or unreachable owner drops the entry
            logger.warning(
                "Notification %s could not be delivered to %s — dropped",
                entry.notification_id,
                entry.owner,
                exc_info=True,
            )
