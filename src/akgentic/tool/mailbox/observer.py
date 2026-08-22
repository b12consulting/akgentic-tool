"""Observer protocol for the mailbox tool.

``MailboxToolObserver`` is one tool's contract, not the package's: the mailbox
capability is its only consumer. It lives beside that tool rather than in
``core/`` so the global surface stays limited to what more than one domain
actually needs — the same placement as ``TeamManagementToolObserver``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from akgentic.core.messages import Message
from akgentic.tool.core.observer import ToolObserver


@runtime_checkable
class MailboxToolObserver(ToolObserver, Protocol):
    """Observer protocol for the mailbox capability (ADR-040 §1).

    Extends ToolObserver with the single mailbox-specific method the
    capability needs: a non-consuming peek over the owning agent's inbox.
    """

    def get_mailbox(self) -> list[Message]:
        """Peek at the pending messages without consuming them.

        The peek is non-consuming: every message listed here will still be
        delivered to the agent as its own turn (ADR-040 §3).

        Returns:
            Messages currently pending in the agent's inbox, oldest first.
        """
        ...
