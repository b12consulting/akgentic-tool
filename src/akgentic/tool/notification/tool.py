"""``NotificationTool``: let an agent schedule a delayed message to itself.

The card carries exactly two fields — the dotted path of the delivery message
class and the maximum schedulable delay. Everything else is behaviour: the three
capabilities are ownership-scoped by the caller's ``myAddress``, captured once at
bind time, and they reach the ``#NotificationTool`` singleton through the ask
proxy (ADR-035 §2).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast

from pydantic import Field, PrivateAttr

from akgentic.core.actor_address import ActorAddress
from akgentic.core.orchestrator import Orchestrator
from akgentic.tool.core import BaseToolParam, ToolCard
from akgentic.tool.core.observer import ActorToolObserver, ToolObserver
from akgentic.tool.errors import RetriableError
from akgentic.tool.notification.actor import (
    NOTIFICATION_ACTOR_NAME,
    NOTIFICATION_ACTOR_ROLE,
    NotificationActor,
)
from akgentic.tool.notification.models import (
    DEFAULT_MESSAGE_CLASS,
    NotificationConfig,
    resolve_message_class,
)

DEFAULT_MAX_DELAY_SECONDS = 300


class ListPendingNotifications(BaseToolParam):
    """Registry key for ``list_pending_notifications``. Carries no configuration.

    Per-capability parameter classes are out of scope for this tool: the card's
    two fields are its whole configuration, and channel placement comes from
    which card method returns the callable — ``get_tools()`` for ``TOOL_CALL``,
    ``get_commands()`` for ``COMMAND``. This class exists only because
    ``ToolCard.get_commands()`` is keyed by ``type[BaseToolParam]``.
    """


class CancelNotification(BaseToolParam):
    """Registry key for ``cancel_notification``. Carries no configuration.

    See :class:`ListPendingNotifications` for why it exists.
    """


def _require_live_agent(observer_or_none: Callable[[], ActorToolObserver | None]) -> None:
    """Raise a retriable error when the owning agent has already stopped.

    Raises:
        RetriableError: If the weakly-held observer is gone.
    """
    if observer_or_none() is None:
        raise RetriableError("Notifications are unavailable; the agent is shutting down.")


class NotificationTool(ToolCard):
    """Self-scheduled reminders for the agent carrying the card.

    Attributes:
        message_class: Dotted import path of the class delivered when a
            notification comes due. It must resolve to a ``Message`` subclass
            declaring ``content`` and ``type`` fields; any violation raises
            ``ValueError`` at ``observer()`` bind time, never at fire time.
            Naming the class by path rather than importing it is what keeps this
            package free of any dependency on the package that owns it.
        max_delay_seconds: Upper bound on a schedulable delay.
    """

    message_class: str = Field(
        default=DEFAULT_MESSAGE_CLASS,
        description=(
            "Dotted import path of the message class delivered on fire. Must be a "
            "Message subclass declaring 'content' and 'type' fields."
        ),
    )
    max_delay_seconds: int = Field(
        default=DEFAULT_MAX_DELAY_SECONDS,
        description="Maximum delay, in seconds, that an agent may schedule.",
    )

    _notification_proxy: NotificationActor | None = PrivateAttr(default=None)

    ##
    ## Wiring
    ##
    def observer(self, observer: ToolObserver) -> NotificationTool:
        """Validate the delivery class, then bind the ``#NotificationTool`` singleton.

        The parameter keeps the base ``ToolObserver`` type so the override stays
        substitutable — ``ToolFactory`` attaches one observer to every card
        uniformly — and :meth:`_actor_observer` applies the narrower type.

        ``message_class`` is resolved **before** the actor is bound, so a
        misconfigured card cannot leave a live singleton behind.

        Args:
            observer: The owning agent, held weakly by the base class.

        Returns:
            Self, enabling method chaining.

        Raises:
            ValueError: If ``message_class`` does not resolve to a usable
                message class, or if the observer has no orchestrator.
        """
        super().observer(observer)  # store the observer weakly via the base setter
        resolve_message_class(self.message_class)

        actor_observer = self._actor_observer()
        if actor_observer.orchestrator is None:
            raise ValueError("NotificationTool requires access to the orchestrator.")

        orchestrator_proxy = actor_observer.proxy_ask(actor_observer.orchestrator, Orchestrator)
        notification_addr = orchestrator_proxy.getChildrenOrCreate(
            NotificationActor,
            config=NotificationConfig(
                name=NOTIFICATION_ACTOR_NAME,
                role=NOTIFICATION_ACTOR_ROLE,
                message_class=self.message_class,
            ),
        )
        self._notification_proxy = actor_observer.proxy_ask(notification_addr, NotificationActor)
        return self

    def _actor_observer(self) -> ActorToolObserver:
        """Live observer typed as the actor protocol. Raises once the agent stops."""
        return cast(ActorToolObserver, self._observer)

    def _actor_observer_or_none(self) -> ActorToolObserver | None:
        """Live observer typed as the actor protocol; ``None`` once the agent stops."""
        return cast("ActorToolObserver | None", self._observer_or_none())

    def _actor_proxy(self) -> NotificationActor:
        """Return the bound actor proxy.

        Raises:
            RuntimeError: If the card was never wired — a programming error, not
                something an LLM can correct.
        """
        if self._notification_proxy is None:
            raise RuntimeError("NotificationTool.observer() must be called before use.")
        return self._notification_proxy

    ##
    ## Capability surface
    ##
    def get_tools(self) -> list[Callable[..., Any]]:
        """Return the three LLM-callable capabilities."""
        return [
            self._send_factory(),
            self._list_factory(),
            self._cancel_factory(),
        ]

    def get_commands(self) -> dict[type[BaseToolParam], Callable[..., Any]]:
        """Return the two capabilities that are also programmatic commands."""
        return {
            ListPendingNotifications: self._list_factory(),
            CancelNotification: self._cancel_factory(),
        }

    ##
    ## Closure factories — owner captured as data, observer captured weakly
    ##
    def _send_factory(self) -> Callable[..., Any]:
        proxy = self._actor_proxy()
        owner: ActorAddress = self._actor_observer().myAddress
        observer_or_none = self._actor_observer_or_none  # bound method -> weak edge to agent
        max_delay = self.max_delay_seconds

        def send_notification_message(content: str, delay_seconds: int) -> str:
            """Schedule a message to yourself, delivered after a delay.

            Use it to defer your own attention — check a result later, or nudge
            yourself if nothing has happened by then.

            Args:
                content: The text you want to receive.
                delay_seconds: Delay before delivery, at least 1 second.

            Returns:
                A confirmation carrying the notification id, which
                cancel_notification takes.
            """
            _require_live_agent(observer_or_none)
            if not 1 <= delay_seconds <= max_delay:
                raise RetriableError(
                    f"delay_seconds must be between 1 and {max_delay} seconds, got {delay_seconds}."
                )
            notification_id = proxy.schedule(owner, content, delay_seconds)
            return (
                f"Notification {notification_id} scheduled, delivered in {delay_seconds} seconds."
            )

        # The cap is configuration, so it reaches the LLM through the schema
        # rather than through a literal in the source docstring.
        send_notification_message.__doc__ = (
            f"{send_notification_message.__doc__}\n"
            f"The maximum delay is {max_delay} seconds; a larger one is rejected."
        )
        return send_notification_message

    def _list_factory(self) -> Callable[..., Any]:
        proxy = self._actor_proxy()
        owner: ActorAddress = self._actor_observer().myAddress
        observer_or_none = self._actor_observer_or_none

        def list_pending_notifications() -> str:
            """List your own pending notifications, with the time left on each."""
            _require_live_agent(observer_or_none)
            entries = proxy.list_for(owner)
            if not entries:
                return "No pending notifications."
            now = datetime.now(UTC)
            lines = [
                f"- id {entry.notification_id}: {entry.content} "
                f"(in {max(0, round((entry.fire_at - now).total_seconds()))} seconds)"
                for entry in entries
            ]
            return "\n".join(["Pending notifications:", *lines])

        return list_pending_notifications

    def _cancel_factory(self) -> Callable[..., Any]:
        proxy = self._actor_proxy()
        owner: ActorAddress = self._actor_observer().myAddress
        observer_or_none = self._actor_observer_or_none

        def cancel_notification(notification_id: int) -> str:
            """Cancel one of your own pending notifications by id."""
            _require_live_agent(observer_or_none)
            if not proxy.cancel(notification_id, owner):
                raise RetriableError(
                    f"No pending notification with id {notification_id}; "
                    f"use list_pending_notifications to see your own."
                )
            return f"Notification {notification_id} cancelled."

        return cancel_notification
