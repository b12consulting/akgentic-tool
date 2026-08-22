"""The mailbox capability's domain vocabulary and card (ADR-040, ADR-019 §4).

A consuming read over the owning agent's inbox: ``read_mailbox`` absorbs the
messages it shows, so they never also arrive as their own turn — everything
except a cancel, which the card leaves queued because it is the cancellation's
single source of truth. ``MailboxTool`` wires that onto two channels —
``read_mailbox`` on ``TOOL_CALL`` and ``stop`` on ``COMMAND``.
"""

from akgentic.tool.mailbox.mailbox import MailboxTool
from akgentic.tool.mailbox.observer import MailboxToolObserver
from akgentic.tool.mailbox.params import ReadMailbox, Stop

__all__ = [
    "MailboxTool",
    "MailboxToolObserver",
    "ReadMailbox",
    "Stop",
]
