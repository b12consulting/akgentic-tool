"""Structured context state for the mailbox domain (ADR-040 §2).

``MailboxState`` carries the pending-message snapshot the capability exposes on
the ``LLM_CONTEXT`` channel. The snapshot is a non-consuming peek: every row
listed here is still delivered to the agent as its own turn (ADR-040 §3), so
the delta narrates arrivals only — departures ARE the turns.
"""

import logging
from collections import Counter
from collections.abc import Callable, Iterable
from typing import Self

from akgentic.core.messages import Message
from akgentic.core.utils import SerializableBaseModel
from akgentic.tool.core import ContextState
from akgentic.tool.mailbox.observer import MailboxToolObserver

logger = logging.getLogger(__name__)

_PREVIEW_LIMIT = 120


class MailboxRow(SerializableBaseModel):
    """One pending message as the mailbox state carries it.

    ``preview`` is the first line of the message content, truncated to
    ~120 characters. Rows deliberately carry no message id (ADR-040 §2);
    arrival identity is the multiset difference over these three fields.
    """

    sender: str
    message_type: str
    preview: str


class MailboxState(ContextState):
    """The pending mailbox at one point in time, diffable row by row.

    The state is a non-consuming peek over the agent's inbox: every row will
    still arrive as its own turn, so renderers announce without ever draining.
    """

    rows: list[MailboxRow]

    @classmethod
    def from_messages(cls, messages: list[Message]) -> Self:
        """Build a snapshot from pending messages (oldest first)."""
        return cls(rows=[_row_from_message(message) for message in messages])

    def render_full(self) -> str:
        """The whole pending mailbox: count and senders, or ``""`` when empty."""
        if not self.rows:
            return ""
        count = len(self.rows)
        noun = "message" if count == 1 else "messages"
        senders = ", ".join(_unique_ordered(row.sender for row in self.rows))
        return f"{count} {noun} pending from {senders}, consider wrapping up the current thread"

    def render_delta(self, previous: Self) -> str | None:
        """Arrivals since ``previous``; ``None`` when nothing arrived.

        Arrivals are the multiset row difference: rows whose occurrence count
        now exceeds their count in ``previous`` (the inbox only grows at the
        back and shrinks at the front). Departures are never narrated — those
        messages became their own turns, and narrating them would be double
        delivery (ADR-040 §2).
        """
        remaining = Counter(_row_key(row) for row in previous.rows)
        arrived: list[MailboxRow] = []
        for row in self.rows:
            key = _row_key(row)
            if remaining[key] > 0:
                remaining[key] -= 1
            else:
                arrived.append(row)
        if not arrived:
            return None
        return "\n".join(_arrival_line(row) for row in arrived)


def make_mailbox_state_provider(
    accessor: Callable[[], MailboxToolObserver | None],
) -> Callable[[], MailboxState | None]:
    """Create the mailbox context-state provider from a weak observer accessor.

    Args:
        accessor: The card's bound ``None``-returning observer accessor. The
            provider captures this callable, never the observer itself, so it
            cannot pin a stopped agent (ADR-030).

    Returns:
        Zero-arg provider producing a mailbox snapshot, or ``None`` when the
        observer has been collected. Never raises.
    """

    def mailbox_state() -> MailboxState | None:
        try:
            observer = accessor()
            if observer is None:
                return None  # agent gone -> state unavailable
            return MailboxState.from_messages(observer.get_mailbox())
        except Exception:
            logger.error("Failed to get mailbox state", exc_info=True)
            return None

    return mailbox_state


def _row_key(row: MailboxRow) -> tuple[str, str, str]:
    """Hashable identity of a row for the multiset difference."""
    return (row.sender, row.message_type, row.preview)


def _arrival_line(row: MailboxRow) -> str:
    """One narrated arrival."""
    line = f"New message pending from {row.sender} ({row.message_type})"
    return f"{line}: {row.preview}" if row.preview else line


def _row_from_message(message: Message) -> MailboxRow:
    """Map a pending message to its state row (sender, type name, preview)."""
    return MailboxRow(
        sender=sender_name(message),
        message_type=type(message).__name__,
        preview=_preview(message),
    )


def sender_name(message: Message) -> str:
    """The sender's display name, or ``"unknown"`` when the message has none."""
    sender = getattr(message, "sender", None)
    name = getattr(sender, "name", None)
    return name if isinstance(name, str) and name else "unknown"


def _preview(message: Message) -> str:
    """First line of the message content, truncated to ~120 characters."""
    content = getattr(message, "content", "")
    if not isinstance(content, str) or not content:
        return ""
    first_line = content.splitlines()[0]
    return first_line[:_PREVIEW_LIMIT]


def _unique_ordered(values: Iterable[str]) -> list[str]:
    """Deduplicate while preserving first-seen order."""
    return list(dict.fromkeys(values))
