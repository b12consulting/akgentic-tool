"""Run-cancellation vocabulary for the mailbox capability (ADR-040 §4, §5).

``is_cancel`` recognises both spellings of one intent — the typed
``CancelMessage`` (programmatic senders) and the ``/stop`` string surface
(human / frontend Esc). ``render_arrival_notice`` is the wording of the
ephemeral mid-run doorbell. Vocabulary only: announce-growth tracking, tail
injection and the capability hook live in ``akgentic-agent`` (Epic 20), which
imports these instead of restating them.
"""

from akgentic.core.messages import CancelMessage, Message
from akgentic.tool.mailbox.state import _unique_ordered, sender_name


def is_cancel(msg: Message) -> bool:
    """Whether ``msg`` asks the recipient to abandon its current run.

    ``True`` for a ``CancelMessage`` instance, or for a message whose content
    strips to a string whose first whitespace-delimited token is exactly
    ``/stop`` — so ``"  /stop now"`` cancels and ``"/stopwatch"`` does not.
    A message without usable string content is simply ``False``.
    """
    if isinstance(msg, CancelMessage):
        return True
    content = getattr(msg, "content", "")
    if not isinstance(content, str):
        return False
    tokens = content.split(maxsplit=1)
    return bool(tokens) and tokens[0] == "/stop"


def render_arrival_notice(new_messages: list[Message]) -> str:
    """One-line doorbell for messages that arrived mid-run (ADR-040 §5).

    Returns ``""`` for an empty list. Defensive on message shapes: a message
    without a usable sender or content still renders (as ``unknown``).
    """
    if not new_messages:
        return ""
    count = len(new_messages)
    noun = "message" if count == 1 else "messages"
    senders = ", ".join(_unique_ordered(sender_name(message) for message in new_messages))
    return (
        f"{count} new {noun} arrived (from {senders}) — "
        "call `read_mailbox` to see them, or finish your current work first."
    )
