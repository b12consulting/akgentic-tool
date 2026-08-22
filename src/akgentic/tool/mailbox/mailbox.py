"""``MailboxTool``: the mailbox on three channels (ADR-040 §7).

One card, three capabilities, one channel each: live status on
``LLM_CONTEXT`` (the model sees mail arriving), ``read_mailbox`` on
``TOOL_CALL`` (a non-consuming peek at what is waiting), and ``stop`` on
``COMMAND`` (a human or program can request cancellation). Every capability
reads ``observer.get_mailbox()`` — a local peek; the card creates no actor
and performs no proxy round trip (NFR1).

The card owns the vocabulary only. Cancellation *enforcement* — matching
pending messages, raising into a run — is the agent's, in another package
(ADR-040 §5): a hook here would make cancellation de-configurable by
omitting the card.
"""

from __future__ import annotations

from collections.abc import Callable
from inspect import cleandoc
from typing import Any, cast

from akgentic.core.messages import Message
from akgentic.tool.core import (
    COMMAND,
    LLM_CONTEXT,
    TOOL_CALL,
    BaseToolParam,
    ContextState,
    ToolCard,
    _resolve,
)
from akgentic.tool.errors import ToolObserverGone
from akgentic.tool.mailbox.observer import MailboxToolObserver
from akgentic.tool.mailbox.params import MailboxStatus, ReadMailbox, Stop
from akgentic.tool.mailbox.state import make_mailbox_state_provider, sender_name

_EMPTY_MAILBOX = "Your mailbox is empty — nothing is pending."


def _content_of(message: Message) -> str:
    """The message's full content, string-rendered; ``""`` when it has none."""
    content = getattr(message, "content", "")
    if content is None:
        return ""
    return content if isinstance(content, str) else str(content)


def _render_mailbox(messages: list[Message]) -> str:
    """Render sender, type and full content of each pending message.

    Args:
        messages: The pending messages, oldest first.

    Returns:
        One numbered block per message, or the empty-mailbox sentence — never
        ``""`` (an empty tool return reads as a malfunction to the model).
    """
    if not messages:
        return _EMPTY_MAILBOX
    count = len(messages)
    noun = "message" if count == 1 else "messages"
    blocks = [
        f"{index}. From {sender_name(message)} ({type(message).__name__}):\n{_content_of(message)}"
        for index, message in enumerate(messages, start=1)
    ]
    return "\n\n".join([f"{count} pending {noun}:", *blocks])


class MailboxTool(ToolCard):
    """The mailbox capability: status, on-demand peek, and cancellation surface.

    Attributes:
        mailbox_status: The pending-mailbox snapshot on ``LLM_CONTEXT``.
            ``True`` (the default) enables it with the param's defaults; a
            ``MailboxStatus`` instance may narrow the channels; ``False``
            removes exactly this capability and nothing else.
        read_mailbox: The non-consuming ``read_mailbox`` tool on ``TOOL_CALL``.
            Same ``Param | bool`` convention.
        stop: The ``stop`` command (``/stop`` surface) on ``COMMAND``.
            Same ``Param | bool`` convention.
    """

    mailbox_status: MailboxStatus | bool = True
    read_mailbox: ReadMailbox | bool = True
    stop: Stop | bool = True

    def _mailbox_observer_or_none(self) -> MailboxToolObserver | None:
        """Live observer typed as the mailbox protocol; ``None`` once the agent stops.

        Conformance is a documented precondition of ``observer()``, not a
        runtime gate — observers are duck-typed, so a non-conforming one fails
        at first use (the ``TeamTool`` convention).
        """
        return cast(MailboxToolObserver | None, self._observer_or_none())

    def get_context_states(self) -> list[Callable[[], ContextState | None]]:
        """Return the mailbox provider when the status capability serves ``LLM_CONTEXT``.

        The provider is 34-1's ``make_mailbox_state_provider``, handed the
        card's bound ``None``-returning accessor — it never raises and returns
        ``None`` once the observer is collected (ADR-030).
        """
        providers: list[Callable[[], ContextState | None]] = []
        status = _resolve(self.mailbox_status, MailboxStatus)
        if status and LLM_CONTEXT in status.expose:
            providers.append(make_mailbox_state_provider(self._mailbox_observer_or_none))
        return providers

    def get_tools(self) -> list[Callable[..., Any]]:
        """Return ``read_mailbox`` when its capability serves ``TOOL_CALL``."""
        tools: list[Callable[..., Any]] = []
        read = _resolve(self.read_mailbox, ReadMailbox)
        if read and TOOL_CALL in read.expose:
            tools.append(self._read_mailbox_factory(read))
        return tools

    def get_commands(self) -> dict[type[BaseToolParam], Callable[..., Any]]:
        """Return ``stop`` when its capability serves ``COMMAND``."""
        commands: dict[type[BaseToolParam], Callable[..., Any]] = {}
        stop = _resolve(self.stop, Stop)
        if stop and COMMAND in stop.expose:
            commands[Stop] = self._stop_factory(stop)
        return commands

    def _read_mailbox_factory(self, params: ReadMailbox) -> Callable[..., Any]:
        """Create the ``read_mailbox`` callable.

        The closure captures the bound accessor, never the observer, so it
        cannot pin a stopped agent (ADR-030). This is in-life code, so the
        raising form applies: ``ToolObserverGone`` is a defined outcome.

        Args:
            params: Configuration for the read capability.

        Returns:
            A zero-argument callable named ``read_mailbox``.
        """
        observer_or_none = self._mailbox_observer_or_none  # bound method -> weak edge to agent

        def read_mailbox() -> str:
            """Peek at the messages waiting in your mailbox.

            Reading does NOT consume them: every message listed here will still
            be delivered to you as its own turn after the current run ends. Use
            this to decide whether to wrap up early or how to prioritise —
            never to answer a pending message from inside the current run.

            Returns:
                Sender, message type and full content of every pending message,
                or a sentence saying the mailbox is empty.
            """
            observer = observer_or_none()
            if observer is None:
                raise ToolObserverGone("read_mailbox used after its owning agent was stopped")
            return _render_mailbox(observer.get_mailbox())

        read_mailbox.__doc__ = params.format_docstring(cleandoc(read_mailbox.__doc__ or ""))
        return read_mailbox

    def _stop_factory(self, params: Stop) -> Callable[..., Any]:
        """Create the ``stop`` command callable (``/stop`` string surface).

        The handler only replies: commands dispatch while the agent is idle,
        so there is nothing to cancel by the time it runs. The mid-run effect
        of a ``/stop`` message is the agent's enforcement, in another package —
        nothing here raises, tracks, or interrupts. Nothing observer-shaped is
        captured, so the closure outlives its agent without pinning it.

        Args:
            params: Configuration for the stop capability.

        Returns:
            A zero-argument callable named ``stop``.
        """

        def stop() -> str:
            """Cancel the agent's current run.

            Sent while the agent is busy, the request interrupts the run at the
            next step boundary. Dispatched while the agent is idle, there is
            nothing to cancel and this only replies.

            Returns:
                A confirmation that nothing is running.
            """
            return "nothing is running — there is no active run to cancel."

        stop.__doc__ = params.format_docstring(cleandoc(stop.__doc__ or ""))
        return stop
