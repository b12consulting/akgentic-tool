"""The mailbox capability's domain vocabulary and card (ADR-040).

A non-consuming peek over the owning agent's inbox, structured for diffing:
every message listed is still delivered as its own turn. ``MailboxTool``
wires that vocabulary onto three channels — status on ``LLM_CONTEXT``,
``read_mailbox`` on ``TOOL_CALL``, and ``stop`` on ``COMMAND``.
"""

from akgentic.tool.mailbox.cancel import is_cancel, render_arrival_notice
from akgentic.tool.mailbox.mailbox import MailboxTool
from akgentic.tool.mailbox.observer import MailboxToolObserver
from akgentic.tool.mailbox.params import MailboxStatus, ReadMailbox, Stop
from akgentic.tool.mailbox.state import (
    MailboxRow,
    MailboxState,
    make_mailbox_state_provider,
)

__all__ = [
    "MailboxRow",
    "MailboxState",
    "MailboxStatus",
    "MailboxTool",
    "MailboxToolObserver",
    "ReadMailbox",
    "Stop",
    "is_cancel",
    "make_mailbox_state_provider",
    "render_arrival_notice",
]
