"""The mailbox capability's domain vocabulary and card (ADR-010, ADR-040).

A signal over the owning agent's inbox: ``read_mailbox`` names one pending
message by id and acknowledges it, so that message is taken on in the current
run rather than arriving again as its own turn. The card reads nothing, consumes
nothing and renders nothing — absorbing the named message is the agent's
capability, and rendering it is the message's own job. ``MailboxTool`` wires that
onto two channels — ``read_mailbox`` on ``TOOL_CALL`` and ``stop`` on
``COMMAND``.
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
