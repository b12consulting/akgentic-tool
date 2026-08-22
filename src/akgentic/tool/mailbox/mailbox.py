"""``MailboxTool``: the mailbox on two channels (ADR-040 §7, ADR-019 §4).

One card, two capabilities, one channel each: ``read_mailbox`` on
``TOOL_CALL`` — a *consuming* read, which absorbs the mail it shows — and
``stop`` on ``COMMAND``, so a human or program can request cancellation. The
card creates no actor and performs no proxy round trip (NFR1).

The card no longer serves ``LLM_CONTEXT``. Mailbox awareness reaches the model
through the agent's mid-run arrival notice alone (ADR-019 §4b); a second,
card-side carrier only narrated the same arrivals twice.

The card owns the ``/stop`` *registration* only. Both the cancellation
vocabulary (recognising the string) and its *enforcement* — matching pending
messages, raising into a run — are the agent's, in another package
(ADR-040 §5): ``BaseAgent`` builds its cancel capability unconditionally, so
an agent configured without this card is still interruptible and has no card
to borrow a predicate from. The private predicate below is not a second
vocabulary; it is this card declining to swallow the agent's input.
"""

from __future__ import annotations

from collections.abc import Callable
from inspect import cleandoc
from typing import Any, cast

from akgentic.core.messages import CancelMessage, Message
from akgentic.tool.core import (
    COMMAND,
    TOOL_CALL,
    BaseToolParam,
    ToolCard,
    _resolve,
)
from akgentic.tool.errors import ToolObserverGone
from akgentic.tool.mailbox.observer import MailboxToolObserver
from akgentic.tool.mailbox.params import ReadMailbox, Stop

_EMPTY_MAILBOX = "Your mailbox is empty — nothing is pending."

_NOTHING_TO_CANCEL = "There is no run to cancel."

_REPLY_PROTOCOLS: dict[str, str] = {
    "request": "A reply is expected: respond to {sender} with the result.",
    "response": "This is a reply to something you asked. Take it into account and continue.",
    "instruction": "Carry it out; acknowledge to {sender} only if asked to.",
    "notification": "Informational message. No reply is expected.",
    "acknowledgment": "Receipt confirmed. No further action needed.",
}
"""What each message type asks of its recipient, mirroring the agent's table.

The canonical table is ``REPLY_PROTOCOLS`` in ``akgentic.agent.output_models``,
alongside ``AgentMessage.type``. Both are out of reach here — ``akgentic-tool``
may import from ``akgentic-core`` only — so the domain carries its own copy and
reads the type duck-typed. Nothing keeps the two in sync; a reworded protocol
line in the agent diverges here silently.

These strings describe **message mechanics only**: what arrived, and whether a
reply is owed. No team policy, nothing about who should do the work or whether
to delegate — that belongs in the agents' prompts, where it can differ per role.
"""


def _is_cancel_envelope(message: Message) -> bool:
    """Whether this card must leave ``message`` in the mailbox untouched.

    A *defensive exclusion*, not a cancellation predicate. The canonical one is
    the agent's ``is_cancel`` (``akgentic.agent.capabilities.mailbox_capability``),
    which is unreachable from here, so this mirrors its two branches from what
    this package legitimately owns: ``CancelMessage`` is core vocabulary, and
    ``/stop`` is this card's own registered command surface.

    It must stay **at least as broad** as the agent's. If the two ever disagree,
    over-excluding costs one message its place in a consuming read — it still
    arrives as its own turn — while under-excluding consumes a cancel and
    silently kills the interrupt.
    """
    if isinstance(message, CancelMessage):
        return True
    content = getattr(message, "content", "")
    if not isinstance(content, str):
        return False
    tokens = content.split(maxsplit=1)
    return bool(tokens) and tokens[0] == "/stop"


def sender_name(message: Message) -> str:
    """The sender's display name, or ``"unknown"`` when the message has none."""
    sender = getattr(message, "sender", None)
    name = getattr(sender, "name", None)
    return name if isinstance(name, str) and name else "unknown"


def _content_of(message: Message) -> str:
    """The message's full content, string-rendered; ``""`` when it has none."""
    content = getattr(message, "content", "")
    if content is None:
        return ""
    return content if isinstance(content, str) else str(content)


def _protocol_line(message: Message) -> str:
    """The framing ``receiveMsg_AgentMessage`` would have applied; ``""`` if none.

    The type is read duck-typed: the base ``Message`` declares no ``type`` at
    all, so a bare ``UserMessage`` or a ``CancelMessage`` simply gets no line
    rather than an error. A type outside the table is treated the same way.
    """
    message_type = getattr(message, "type", None)
    if not isinstance(message_type, str) or not message_type:
        return ""
    protocol = _REPLY_PROTOCOLS.get(message_type)
    if protocol is None:
        return ""
    article = "an" if message_type[0] in "aeiou" else "a"
    sender = sender_name(message)
    return f"You received {article} {message_type} from {sender}. " + protocol.format(sender=sender)


def _render_message(index: int, message: Message) -> str:
    """One numbered block: header, the reply protocol when known, full content."""
    header = f"{index}. From {sender_name(message)} ({type(message).__name__}):"
    protocol = _protocol_line(message)
    body = _content_of(message)
    return f"{header}\n{protocol}\n\n{body}" if protocol else f"{header}\n{body}"


def _render_mailbox(messages: list[Message]) -> str:
    """Render sender, type, reply protocol and full content of each message.

    Args:
        messages: The messages the read absorbed, oldest first.

    Returns:
        One numbered block per message, or the empty-mailbox sentence — never
        ``""`` (an empty tool return reads as a malfunction to the model).
    """
    if not messages:
        return _EMPTY_MAILBOX
    count = len(messages)
    noun = "message" if count == 1 else "messages"
    blocks = [_render_message(index, message) for index, message in enumerate(messages, start=1)]
    head = f"{count} {noun} taken from your mailbox — they will not be delivered again:"
    return "\n\n".join([head, *blocks])


class MailboxTool(ToolCard):
    """The mailbox capability: a consuming read and a cancellation surface.

    Attributes:
        read_mailbox: The consuming ``read_mailbox`` tool on ``TOOL_CALL``.
            ``True`` (the default) enables it with the param's defaults; a
            ``ReadMailbox`` instance may narrow the channels; ``False``
            removes exactly this capability and nothing else.
        stop: The ``stop`` command (``/stop`` surface) on ``COMMAND``.
            Same ``Param | bool`` convention.
    """

    read_mailbox: ReadMailbox | bool = True
    stop: Stop | bool = True

    def _mailbox_observer_or_none(self) -> MailboxToolObserver | None:
        """Live observer typed as the mailbox protocol; ``None`` once the agent stops.

        Conformance is a documented precondition of ``observer()``, not a
        runtime gate — observers are duck-typed, so a non-conforming one fails
        at first use (the ``TeamTool`` convention).
        """
        return cast(MailboxToolObserver | None, self._observer_or_none())

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

        The read consumes: it renders what ``consume_mailbox`` *returned*,
        never the peeked list. The peek is a superset — the primitive skips
        ``reply_to`` envelopes and ignores ids dequeued in between — and
        rendering the superset would show the model a message it did not
        absorb, which is the double answer this whole capability removes.
        The telemetry for each removal is the primitive's own; nothing is
        emitted here.

        Args:
            params: Configuration for the read capability.

        Returns:
            A zero-argument callable named ``read_mailbox``.
        """
        observer_or_none = self._mailbox_observer_or_none  # bound method -> weak edge to agent

        def read_mailbox() -> str:
            """Read the messages waiting in your mailbox and take them on.

            Reading ABSORBS them: everything listed below has been removed from
            your mailbox and will NOT be delivered to you again as its own
            turn. Deal with it in this run — answer it, act on it, or fold it
            into what you are already doing. Anything you leave unread stays
            queued and arrives as its own turn later, so read only when you
            mean to handle what you find.

            Returns:
                Sender, message type, the reply protocol each message expects
                and its full content, or a sentence saying the mailbox is empty.
            """
            observer = observer_or_none()
            if observer is None:
                raise ToolObserverGone("read_mailbox used after its owning agent was stopped")
            pending = observer.get_mailbox()
            consumable = [message for message in pending if not _is_cancel_envelope(message)]
            absorbed = observer.consume_mailbox([message.id for message in consumable])
            return _render_mailbox(absorbed)

        read_mailbox.__doc__ = params.format_docstring(cleandoc(read_mailbox.__doc__ or ""))
        return read_mailbox

    def _stop_factory(self, params: Stop) -> Callable[..., Any]:
        """Create the ``stop`` command callable (``/stop`` string surface).

        The handler answers, because by the time it runs the answer is known.
        Commands dispatch while the agent is idle, and a cancel that reaches a
        handler is by construction the idle case — the agent purges a mid-run
        cancel at the moment it recognises it, so one can never be dequeued
        into a handler (ADR-019 §3). There is therefore exactly one thing this
        can mean, and saying it is better than silence: a human who typed
        ``/stop`` and heard nothing back cannot tell a no-op from a failure.

        The mid-run effect of a ``/stop`` message is the agent's enforcement,
        in another package — nothing here raises, tracks, or interrupts.
        Nothing observer-shaped is captured, so the closure outlives its agent
        without pinning it.

        The callable's docstring is user-facing surface: ``descriptors()``
        builds each ``CommandDescriptor.description`` from it, and those are
        announced to every frontend's command palette.

        Args:
            params: Configuration for the stop capability.

        Returns:
            A zero-argument callable named ``stop``.
        """

        def stop() -> str:
            """Cancel the agent's current run.

            Sent while the agent is busy, the request interrupts the run at the
            next step boundary. Dispatched while the agent is idle there is
            nothing to cancel, and it says so.
            """
            return _NOTHING_TO_CANCEL

        stop.__doc__ = params.format_docstring(cleandoc(stop.__doc__ or ""))
        return stop
