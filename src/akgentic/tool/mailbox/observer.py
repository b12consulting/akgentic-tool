"""Observer protocol for the mailbox tool.

``MailboxToolObserver`` is one tool's contract, not the package's: the mailbox
capability is its only consumer. It lives beside that tool rather than in
``core/`` so the global surface stays limited to what more than one domain
actually needs — the same placement as ``TeamManagementToolObserver``.
"""

from __future__ import annotations

import uuid
from typing import Protocol, runtime_checkable

from akgentic.core.messages import Message
from akgentic.tool.core.observer import ToolObserver


@runtime_checkable
class MailboxToolObserver(ToolObserver, Protocol):
    """Observer protocol for the mailbox capability (ADR-040 §1, ADR-019 §4).

    Extends ToolObserver with the two mailbox methods the capability needs: a
    peek over the owning agent's inbox, and the removal that makes
    ``read_mailbox`` a consuming read. ``Akgent`` provides both, so no
    agent-side change is needed to satisfy the widened protocol.
    """

    def get_mailbox(self) -> list[Message]:
        """Peek at the pending messages without dequeuing them.

        The peek itself removes nothing. It is no longer a promise of
        redelivery, though: ``read_mailbox`` goes on to consume most of what a
        peek returns, so a message seen here may well never get its own turn.

        Returns:
            Messages currently pending in the agent's inbox, oldest first.
        """
        ...

    def consume_mailbox(self, message_ids: list[uuid.UUID]) -> list[Message]:
        """Remove the named messages from the mailbox without delivering them.

        Each removed message loses its own turn — this is how a consuming read
        absorbs its mail. Caller-idempotent: an id that is no longer queued is
        ignored silently, so a message dequeued between a peek and this call
        simply does not come back. Messages whose envelope carries a
        ``reply_to`` are left alone, so the return is a subset of what was
        asked for.

        Emitting the removal telemetry is the primitive's own duty, never the
        caller's — no call site can forget it, and none can double it.

        Args:
            message_ids: Ids of the queued messages to remove.

        Returns:
            The messages actually removed, in the order they sat in the queue.
        """
        ...
