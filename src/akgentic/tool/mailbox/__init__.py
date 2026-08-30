"""The mailbox capability's domain vocabulary and card (ADR-010, ADR-040).

A signal over the owning agent's inbox: ``read_mailbox`` names one pending
message by id and acknowledges it, so that message is taken on in the current
run rather than arriving again as its own turn. The card reads nothing, consumes
nothing and renders nothing — absorbing the named message is
:class:`MailboxCapability`'s job, and rendering it is the message's own.
``MailboxTool`` wires the signal onto two channels — ``read_mailbox`` on
``TOOL_CALL`` and ``stop`` on ``COMMAND``.

**Card, capability and contracts are one subject and ship together.** The
capability lived in ``akgentic.agent.capabilities`` until every mailbox change
cost two stories, two PRs and a release in between. ``akgentic-agent`` still
*builds* it for every agent and catches ``RunInterruptedError`` in ``act()`` —
that is the enforcement, and it is what keeps cancellation impossible to
de-configure — but it no longer holds the implementation and no longer carries
an ``akgentic.agent.capabilities`` package: these names are imported from here.
"""

from akgentic.tool.mailbox.capability import (
    ABSORBED_PREFIX,
    ARRIVAL_CLOSING,
    PREVIEW_LIMIT,
    MailboxAccess,
    MailboxCapability,
    MailboxRenderError,
    RunInterruptedError,
    is_cancel,
    render_arrival_notice,
)
from akgentic.tool.mailbox.mailbox import MailboxTool
from akgentic.tool.mailbox.message import MailboxMessage
from akgentic.tool.mailbox.observer import MailboxToolObserver
from akgentic.tool.mailbox.params import ReadMailbox, Stop

__all__ = [
    "ABSORBED_PREFIX",
    "ARRIVAL_CLOSING",
    "PREVIEW_LIMIT",
    "MailboxAccess",
    "MailboxCapability",
    "MailboxMessage",
    "MailboxRenderError",
    "MailboxTool",
    "MailboxToolObserver",
    "ReadMailbox",
    "RunInterruptedError",
    "Stop",
    "is_cancel",
    "render_arrival_notice",
]
