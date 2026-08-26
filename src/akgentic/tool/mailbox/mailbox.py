"""``MailboxTool``: the mailbox on two channels (ADR-010, ADR-040 §7).

One card, two capabilities, one channel each: ``read_mailbox`` on ``TOOL_CALL``
— a *signal*, which names a message id and acknowledges it — and ``stop`` on
``COMMAND``, so a human or program can request cancellation. The card creates no
actor and performs no proxy round trip (NFR1).

``read_mailbox`` no longer reads, consumes or renders anything. It used to do
all three, and the renderer took each message's body as
``getattr(message, "content", "")`` — correct for exactly the one message class
it was written against, and silently empty for every other. A message class that
declares its own fields instead of ``content`` was consumed, rendered blank, and
never reached its own handler. Rendering a message is the message's job, and
delivering one is the agent's; this card's job is to carry the id across.

The card no longer serves ``LLM_CONTEXT``. Mailbox awareness reaches the model
through the agent's mid-run arrival notice alone (ADR-019 §4b); a second,
card-side carrier only narrated the same arrivals twice.

The card owns the ``/stop`` *registration* only. The cancellation vocabulary and
its enforcement are the agent's, in another package (ADR-040 §5): ``BaseAgent``
builds its cancel capability unconditionally, so an agent configured without this
card is still interruptible. Excluding a cancel from a mid-run read is likewise
the agent's offer rule now (ADR-010 §7), not a filter here.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from inspect import cleandoc
from typing import Any

from akgentic.core.messages import Message
from akgentic.tool.core import (
    COMMAND,
    TOOL_CALL,
    BaseToolParam,
    ToolCard,
    _resolve,
)
from akgentic.tool.core.observer import ToolObserver
from akgentic.tool.errors import ToolObserverGone
from akgentic.tool.mailbox.params import ReadMailbox, Stop

_ACKNOWLEDGED = (
    "Acknowledged — that message is yours for this run and will not arrive again as its own turn."
)

_NOTHING_TO_CANCEL = "There is no run to cancel."


def _resolve_message_class(dotted_path: str) -> type[Message]:
    """Import *dotted_path* and check it names a ``Message`` subclass.

    Deliberately **not** ``akgentic.tool.notification.models.resolve_message_class``,
    and deliberately not shared with it. That one additionally requires the class
    to declare ``content`` and ``type``, because a notification's payload is
    written into those two fields. Here there is no payload: an entry only names
    a handler whose runs show the mailbox preview, and a message class is under no
    obligation to have either field — the classes this whitelist exists to name
    are precisely the ones carrying their own fields instead. Do not deduplicate
    the two.

    No check for ``mailbox_preview()`` either: that Protocol belongs to
    ``akgentic-agent`` (ADR-010 §2), which this package may not import and whose
    vocabulary it is not entitled to.

    Args:
        dotted_path: Dotted import path of a handler's message class.

    Returns:
        The resolved class.

    Raises:
        ValueError: When the path carries no module part, names a module that is
            not importable, names an attribute the module does not have, or
            resolves to something that is not a ``Message`` subclass. All four
            are configuration defects and surface at wiring time.
    """
    module_path, _, class_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(
            f"mailbox_preview_handlers entry {dotted_path!r} is not a dotted path to a class."
        )
    try:
        module = import_module(module_path)
    except ImportError as exc:
        raise ValueError(
            f"mailbox_preview_handlers entry {dotted_path!r} is not importable: {exc}"
        ) from exc

    resolved = getattr(module, class_name, None)
    if resolved is None:
        raise ValueError(
            f"mailbox_preview_handlers entry {dotted_path!r} is not importable: "
            f"module {module_path!r} has no attribute {class_name!r}."
        )
    if not (isinstance(resolved, type) and issubclass(resolved, Message)):
        raise ValueError(
            f"mailbox_preview_handlers entry {dotted_path!r} resolves to {resolved!r}, "
            f"which is not a Message subclass."
        )
    return resolved


class MailboxTool(ToolCard):
    """The mailbox capability: an acknowledgement signal and a cancellation surface.

    Attributes:
        read_mailbox: The ``read_mailbox`` signal on ``TOOL_CALL``. ``True``
            (the default) enables it with the param's defaults; a ``ReadMailbox``
            instance may narrow the channels; ``False`` removes exactly this
            capability and nothing else.
        stop: The ``stop`` command (``/stop`` surface) on ``COMMAND``.
            Same ``Param | bool`` convention.
        mailbox_preview_handlers: Dotted paths of the *handler* message classes
            whose runs show the mailbox preview — the message the agent is
            currently handling, not the mail waiting in the box. ``None`` (the
            default) means every handler shows it; ``[]`` means none does, which
            is a different value and is never coerced back to ``None``. Every
            entry is resolved when the card is wired to an observer, so a typo
            raises at agent init rather than going quiet for the agent's life.
        absorbed_prefix: What a message absorbed through ``read_mailbox`` is
            prefixed with when it is injected into the run. The default is the
            wording shipped today, reproduced here so a catalog entry shows an
            operator what the agent is currently being told rather than a
            ``null``. **A deployment editing this can delete the clause "It does
            NOT replace what you were already asked to do"** — the sentence that
            stops the failure it was written for, where an agent that had just
            finished a report answered only the newer mid-run question and the
            report reached nobody. That is inherent to making the wording
            configurable, and it is stated here rather than hidden. **This card
            never reads the field**: it carries it for ``akgentic-agent``'s
            mailbox capability to pick up at capability construction.
        arrival_closing: The mid-run arrival notice's closing line, for a
            listing that offers at least one message id. Default as above — the
            wording shipped today. There is deliberately **no** field for the
            no-id closing: a listing carrying no id offers no read, so there is
            no timing to advise on. **This card never reads the field** either;
            rendering the notice is ``akgentic-agent``'s, and it reads both
            fields defensively, so an older card still works.
    """

    read_mailbox: ReadMailbox | bool = True
    stop: Stop | bool = True
    mailbox_preview_handlers: list[str] | None = None
    absorbed_prefix: str = (
        "Additional work, taken on mid-run. It does NOT replace what you were already asked "
        "to do. It may be a separate request, in which case answer both before this run ends, "
        "one message each in your output; or it may add to or correct the request already in "
        "flight, in which case one message answers both. When unsure, answer separately."
    )
    arrival_closing: str = (
        "Call `read_mailbox` with one of the ids above to take that message on now — worth doing "
        "if it may add to or change what you are working on, since a correction only helps before "
        "the work is finished. Otherwise finish your current work first — you will get them just "
        "after."
    )

    def observer(self, observer: ToolObserver) -> MailboxTool:
        """Store the observer, then resolve every whitelisted handler class.

        This is agent init: ``ToolFactory.__init__`` calls this hook for every
        card it holds, and ``BaseAgent`` builds its factory with ``observer=self``
        — so an unresolvable entry stops the agent before its first run.

        Validation lives here rather than in a Pydantic validator on purpose. A
        field validator would raise at *construction*, making a card carrying a
        perfectly valid entry impossible to deserialize in any process where that
        class happens not to be importable — a catalog reader, a serialization
        round trip. It does not live in ``get_tools`` either: that returns early
        when ``read_mailbox`` is disabled, so a typo on a disabled capability
        would never be seen.

        The parameter keeps the base ``ToolObserver`` type so the override stays
        substitutable — ``ToolFactory`` attaches one observer to every card
        uniformly.

        Args:
            observer: The owning agent, held weakly by the base class.

        Returns:
            Self, enabling method chaining.

        Raises:
            ValueError: If any ``mailbox_preview_handlers`` entry does not
                resolve to a ``Message`` subclass. The message names the
                offending path.
        """
        super().observer(observer)
        for dotted_path in self.mailbox_preview_handlers or []:
            _resolve_message_class(dotted_path)
        return self

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

        The call is a signal and nothing else: it takes an id, does not look it
        up, does not touch the mailbox, and returns an acknowledgement. What
        makes the acknowledgement true is the agent's capability, which reads the
        id back off the completed tool call and absorbs that one message
        (ADR-010 §4). The liveness check is therefore load-bearing rather than
        defensive — an acknowledgement from a card whose agent has stopped is a
        false one, because the capability that would act on it went with the
        agent.

        Args:
            params: Configuration for the read capability.

        Returns:
            A one-argument callable named ``read_mailbox``.
        """
        observer_or_none = self._observer_or_none  # bound method -> weak edge to agent

        def read_mailbox(message_id: str) -> str:
            """Take on a message waiting in your mailbox, by its id.

            Naming a message here ABSORBS it: you take it on in THIS run and it
            will NOT be delivered to you again as its own turn. Deal with it
            now — answer it, act on it, or fold it into what you are already
            doing. Anything you leave unnamed stays queued and arrives as its own
            turn later, so name a message only when you mean to handle it.

            Args:
                message_id: The id of the message to take on, exactly as it was
                    given to you in the notice announcing that message's arrival.

            Returns:
                A short confirmation that the message is yours for this run. The
                message itself is not returned here — you already have its
                arrival notice.
            """
            observer = observer_or_none()
            if observer is None:
                raise ToolObserverGone("read_mailbox used after its owning agent was stopped")
            return _ACKNOWLEDGED

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
