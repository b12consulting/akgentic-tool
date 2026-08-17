"""``#NotificationTool``: the team singleton that holds pending timers.

The actor owns a dict of :class:`PendingNotification` and one daemon thread that
tells it a ``Tick`` about once a second. Every mutation happens on the mailbox
thread, so there is no lock here and none is needed; the thread's only action is
the tell (ADR-035 §3).

Delivery of an entry whose ``fire_at`` has passed needs no re-arm logic — a
restored entry is simply due on the first tick after a resume.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

from pykka import ActorDeadError, ActorRef

from akgentic.core.actor_address import ActorAddress
from akgentic.core.agent import Akgent
from akgentic.core.messages.message import Message
from akgentic.tool.notification.models import (
    NotificationConfig,
    NotificationState,
    PendingNotification,
    Tick,
    resolve_message_class,
)

logger = logging.getLogger(__name__)

NOTIFICATION_ACTOR_NAME = "#NotificationTool"
"""Singleton actor name. The ``#`` prefix is the orchestrator's teardown invariant."""

NOTIFICATION_ACTOR_ROLE = "ToolActor"

TICK_INTERVAL_S = 1.0
"""Tick period. Delivery granularity is ±1 s, accepted against a 300 s cap."""

TICK_JOIN_TIMEOUT_S = 2.0
"""Join budget in ``on_stop`` — bounded, because the loop observes the stop
event within one tick."""

NOTIFICATION_TYPE = "notification"
"""Value written to the delivered message's ``type`` field."""


def _tick_loop(stop_event: threading.Event, actor_ref: ActorRef[Any]) -> None:
    """Tell *actor_ref* a ``Tick`` every ``TICK_INTERVAL_S`` until *stop_event* is set.

    Module-level, and capturing only its two arguments: a bound-method target
    would root the actor instance for as long as the thread lives, which is the
    one real leak this design could have. A dead actor breaks the loop rather
    than leaving it ticking a corpse forever.

    Args:
        stop_event: Set by ``on_stop`` to end the loop.
        actor_ref: Reference to the notification actor (holds it weakly).
    """
    while not stop_event.wait(TICK_INTERVAL_S):
        try:
            actor_ref.tell(Tick())
        except ActorDeadError:
            break


def _same_owner(left: ActorAddress, right: ActorAddress) -> bool:
    """Compare two addresses by ``agent_id``.

    Identity is the agent, not the address object: a resumed team holds a
    deserialized address for the same agent, and the two classes do not compare
    equal to each other.
    """
    return left.agent_id == right.agent_id


class NotificationActor(Akgent[NotificationConfig, NotificationState]):
    """Team singleton holding pending notifications and delivering them on time.

    Every ask-reachable method is a dict operation, so nothing slow ever runs on
    this actor's thread and the deferred-result pattern is not engaged.
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
            args=(self._stop_event, self.actor_ref),
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

    def list_for(self, owner: ActorAddress) -> list[PendingNotification]:
        """Return *owner*'s pending entries, oldest id first."""
        return [
            entry
            for entry in sorted(self.state.pending.values(), key=lambda e: e.notification_id)
            if _same_owner(entry.owner, owner)
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
    def receiveMsg_Tick(self, msg: Tick) -> None:
        """Deliver every entry that has come due, and drop it.

        A tick that finds nothing due is a no-op: no message, and no state-change
        notification for an empty scan.
        """
        now = datetime.now(UTC)
        due = [entry for entry in self.state.pending.values() if entry.fire_at <= now]
        if not due:
            return
        for entry in due:
            del self.state.pending[entry.notification_id]
            self._deliver(entry)
        self.state.notify_state_change()

    def _deliver(self, entry: PendingNotification) -> None:
        """Send one due entry to its owner, as a message from that owner to itself.

        The owner is asked to send the message so the delivered ``sender`` is the
        owner rather than this actor — ``Akgent.send`` stamps the sender of
        whichever actor performs the send. The call is a tell, so a busy owner
        never blocks this actor's mailbox.

        A failure — most often an owner fired before its notification came due —
        is logged and the entry dropped. It is never retried and never raised:
        one dead recipient must not break the tick for everyone else.
        """
        try:
            message = self._message_cls.model_validate(
                {"content": entry.content, "type": NOTIFICATION_TYPE}
            )
            self.proxy_tell(entry.owner, Akgent).send(entry.owner, message)
        except Exception:  # noqa: BLE001 — a dead or unreachable owner drops the entry
            logger.warning(
                "Notification %s could not be delivered to %s — dropped",
                entry.notification_id,
                entry.owner,
                exc_info=True,
            )
