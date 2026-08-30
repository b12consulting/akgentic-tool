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
from inspect import cleandoc
from typing import Any

from akgentic.tool.core import (
    COMMAND,
    TOOL_CALL,
    BaseToolParam,
    ToolCard,
    _resolve,
)
from akgentic.tool.errors import ToolObserverGone
from akgentic.tool.mailbox.params import ReadMailbox, Stop

_ACKNOWLEDGED = (
    "Acknowledged — that message is yours for this run and will not arrive again as its own turn."
)

_NOTHING_TO_CANCEL = "There is no run to cancel."


class MailboxTool(ToolCard):
    """The mailbox capability: an acknowledgement signal and a cancellation surface.

    Two switches and nothing else. The wording the mailbox injects mid-run — the
    absorbed-message prefix and the arrival notice's closing line — belongs to the
    capability in ``akgentic.tool.mailbox.capability``, which takes both as
    keyword-only constructor parameters defaulted from the constants beside them.
    It briefly lived here as two string fields; because the catalog dumps a card
    with no ``exclude_defaults``, every persisted team froze its own copy of prose
    that is expected to keep changing, so an improvement reached only the teams
    created after it.

    Attributes:
        read_mailbox: The ``read_mailbox`` signal on ``TOOL_CALL``. ``True``
            (the default) enables it with the param's defaults; a ``ReadMailbox``
            instance may narrow the channels; ``False`` removes exactly this
            capability and nothing else.
        stop: The ``stop`` command (``/stop`` surface) on ``COMMAND``.
            Same ``Param | bool`` convention.
    """

    read_mailbox: ReadMailbox | bool = True
    stop: Stop | bool = True

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
