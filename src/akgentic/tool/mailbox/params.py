"""Capability parameters for the mailbox card (ADR-040).

Three capabilities, one channel each by default: live status on
``LLM_CONTEXT``, a non-consuming ``read_mailbox`` on ``TOOL_CALL``, and
``stop`` on ``COMMAND``. None of them needs a field beyond ``expose`` —
configuration read at factory bind time, never tool-call schema.
"""

from akgentic.tool.core import COMMAND, LLM_CONTEXT, TOOL_CALL, BaseToolParam, Channels


class MailboxStatus(BaseToolParam):
    """Expose the pending mailbox as structured context state."""

    expose: set[Channels] = {LLM_CONTEXT}


class ReadMailbox(BaseToolParam):
    """Peek at pending messages on demand, without consuming them."""

    expose: set[Channels] = {TOOL_CALL}


class Stop(BaseToolParam):
    """Request cancellation of the current run (``/stop``)."""

    expose: set[Channels] = {COMMAND}
