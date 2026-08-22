"""The mailbox capability's domain vocabulary (ADR-040).

A non-consuming peek over the owning agent's inbox, structured for diffing:
every message listed is still delivered as its own turn. The card that wires
this vocabulary onto channels (``MailboxTool``) is story 34-2.
"""

from akgentic.tool.mailbox.cancel import is_cancel, render_arrival_notice
from akgentic.tool.mailbox.observer import MailboxToolObserver
from akgentic.tool.mailbox.state import (
    MailboxRow,
    MailboxState,
    make_mailbox_state_provider,
)

__all__ = [
    "MailboxRow",
    "MailboxState",
    "MailboxToolObserver",
    "is_cancel",
    "make_mailbox_state_provider",
    "render_arrival_notice",
]
