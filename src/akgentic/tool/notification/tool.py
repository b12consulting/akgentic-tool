"""``NotificationTool``: let an agent schedule a delayed message to itself.

Two fields configure delivery — the dotted path of the message class and the
maximum schedulable delay — and one field per capability decides whether it
exists and which channels it reaches, through the ``expose`` set of its own
``BaseToolParam``, like every other card in this package. All three capabilities
capture the caller's ``myAddress`` once at bind time and reach the
``#NotificationTool`` singleton through the ask proxy (ADR-035 §2). That capture
fixes what a caller may **cancel**, and the default scope of what it lists;
``pending_notification(all=True)`` widens the listing to the whole team without
widening the authority.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from inspect import cleandoc
from typing import Any, cast

from pydantic import Field, PrivateAttr

from akgentic.core.actor_address import ActorAddress
from akgentic.core.orchestrator import Orchestrator
from akgentic.tool.core import (
    COMMAND,
    TOOL_CALL,
    BaseToolParam,
    Channels,
    ToolCard,
    _resolve,
)
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
    PendingNotification,
    resolve_message_class,
)

DEFAULT_MAX_DELAY_SECONDS = 300


class RegisterNotification(BaseToolParam):
    """Schedule a delayed message to yourself — on both channels by default."""

    expose: set[Channels] = {TOOL_CALL, COMMAND}


class PendingNotifications(BaseToolParam):
    """Configuration of the listing capability — on both channels by default.

    Lists the caller's own pending entries by default, and every team member's
    when called with ``all=True``. Plural because the singular already names the
    persisted entry this capability lists — :class:`PendingNotification`, a state
    model rather than a configuration one.
    """

    expose: set[Channels] = {TOOL_CALL, COMMAND}


class CancelNotification(BaseToolParam):
    """Cancel one of your own pending notifications — on both channels by default."""

    expose: set[Channels] = {TOOL_CALL, COMMAND}


def _with_instructions(params: BaseToolParam, doc: str | None) -> str | None:
    """Append *params*' instructions to *doc*, keeping the docstring parseable.

    ``format_docstring`` appends a flush-left block, and the schema builder
    parses the result with griffe, which dedents only when every line past the
    first shares one margin. Appending to a still-indented docstring therefore
    leaves the ``Args:`` section unparsed and every parameter undescribed —
    ``cleandoc`` first is what keeps the two compatible.
    """
    return params.format_docstring(cleandoc(doc) if doc else doc)


def _owner_marker(entry: PendingNotification, show_owner: bool) -> str:
    """Return the ``"@name "`` prefix for a listing line, or ``""`` when scoped.

    The name comes off the entry's stored address and nothing else: both
    ``ActorAddress`` implementations answer ``name`` without a live actor, so the
    marker survives a resume and an owner fired before its notification came due.
    """
    return f"@{entry.owner.name} " if show_owner else ""


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
            declaring ``content`` and ``type`` fields and able to carry
            ``type="notification"``; any violation raises ``ValueError`` at
            ``observer()`` bind time, never at fire time. Naming the class by
            path rather than importing it is what keeps this package free of any
            dependency on the package that owns it.
        max_delay_seconds: Upper bound on a schedulable delay.
        register_notification: The scheduling capability. ``False`` removes it
            entirely; a param instance narrows its channels or adds
            ``instructions``.
        pending_notification: The listing capability, same shape. It reports the
            caller's own entries, and every member's when asked.
        cancel_notification: The cancellation capability, same shape.
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

    register_notification: RegisterNotification | bool = True
    pending_notification: PendingNotifications | bool = True
    cancel_notification: CancelNotification | bool = True

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
        """Return the enabled capabilities that reach the ``TOOL_CALL`` channel."""
        tools: list[Callable[..., Any]] = []

        rn = _resolve(self.register_notification, RegisterNotification)
        if rn and TOOL_CALL in rn.expose:
            tools.append(self._register_factory(rn))

        pn = _resolve(self.pending_notification, PendingNotifications)
        if pn and TOOL_CALL in pn.expose:
            tools.append(self._pending_factory(pn))

        cn = _resolve(self.cancel_notification, CancelNotification)
        if cn and TOOL_CALL in cn.expose:
            tools.append(self._cancel_factory(cn))

        return tools

    def get_commands(self) -> dict[type[BaseToolParam], Callable[..., Any]]:
        """Return the enabled capabilities that reach the ``COMMAND`` channel."""
        commands: dict[type[BaseToolParam], Callable[..., Any]] = {}

        rn = _resolve(self.register_notification, RegisterNotification)
        if rn and COMMAND in rn.expose:
            commands[RegisterNotification] = self._register_factory(rn)

        pn = _resolve(self.pending_notification, PendingNotifications)
        if pn and COMMAND in pn.expose:
            commands[PendingNotifications] = self._pending_factory(pn)

        cn = _resolve(self.cancel_notification, CancelNotification)
        if cn and COMMAND in cn.expose:
            commands[CancelNotification] = self._cancel_factory(cn)

        return commands

    ##
    ## Closure factories — owner captured as data, observer captured weakly
    ##
    def _register_factory(self, params: RegisterNotification) -> Callable[..., Any]:
        proxy = self._actor_proxy()
        owner: ActorAddress = self._actor_observer().myAddress
        observer_or_none = self._actor_observer_or_none  # bound method -> weak edge to agent
        max_delay = self.max_delay_seconds

        def register_notification(content: str, delay_seconds: int) -> str:
            """Schedule a message to yourself after a delay.

            Use this as a reminder or follow-up prompt: for example, remind
            yourself to check a result later, ask another teammate something
            later, or nudge yourself if nothing has happened by then. The delay
            must be between 1 and {max_delay} seconds; larger values are
            rejected.

            Args:
                content: The text you want to receive.
                delay_seconds: Delay before delivery, from 1 to {max_delay}
                    seconds.

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
        # rather than through a literal in the source docstring. Substituted into
        # the docstring rather than appended after it: the schema builder parses
        # the docstring with griffe, which dedents only when every line shares
        # one margin — a flush-left line appended to an indented docstring leaves
        # the Args section unparsed and both parameters undescribed.
        #
        # The order is load-bearing. Substituting first keeps that dedent intact;
        # appending first would both re-break it and feed user-supplied
        # `instructions` to str.format, where a single brace raises.
        substituted = (register_notification.__doc__ or "").format(max_delay=max_delay)
        register_notification.__doc__ = _with_instructions(params, substituted)
        return register_notification

    def _pending_factory(self, params: PendingNotifications) -> Callable[..., Any]:
        proxy = self._actor_proxy()
        owner: ActorAddress = self._actor_observer().myAddress
        observer_or_none = self._actor_observer_or_none

        def pending_notification(all: bool = False) -> str:
            """List pending notifications, with the time left on each.

            Args:
                all: List every team member's pending notifications instead of
                    only your own, each line naming its owner. It widens what you
                    can see, never what you can cancel: another agent's entry
                    still refuses to be cancelled.

            Returns:
                One line per pending notification, or a note that there are none.
            """
            _require_live_agent(observer_or_none)
            entries = proxy.list_for(None if all else owner)
            if not entries:
                return "No pending notifications."
            now = datetime.now(UTC)
            lines = [
                f"- {_owner_marker(entry, all)}id {entry.notification_id}: {entry.content} "
                f"(in {max(0, round((entry.fire_at - now).total_seconds()))} seconds)"
                for entry in entries
            ]
            return "\n".join(["Pending notifications:", *lines])

        pending_notification.__doc__ = _with_instructions(params, pending_notification.__doc__)
        return pending_notification

    def _cancel_factory(self, params: CancelNotification) -> Callable[..., Any]:
        proxy = self._actor_proxy()
        owner: ActorAddress = self._actor_observer().myAddress
        observer_or_none = self._actor_observer_or_none

        def cancel_notification(notification_id: int) -> str:
            """Cancel one of your own pending notifications by id.

            Args:
                notification_id: Id of the notification to cancel, as reported by
                    register_notification or pending_notification.

            Returns:
                A confirmation that the notification was cancelled.
            """
            _require_live_agent(observer_or_none)
            if not proxy.cancel(notification_id, owner):
                raise RetriableError(
                    f"No pending notification with id {notification_id}; "
                    f"use pending_notification to see your own."
                )
            return f"Notification {notification_id} cancelled."

        cancel_notification.__doc__ = _with_instructions(params, cancel_notification.__doc__)
        return cancel_notification
