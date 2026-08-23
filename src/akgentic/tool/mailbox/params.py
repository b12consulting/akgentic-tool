"""Capability parameters for the mailbox card (ADR-010, ADR-040).

Two capabilities, one channel each by default: the ``read_mailbox`` signal on
``TOOL_CALL``, and ``stop`` on ``COMMAND``. Neither needs a field beyond
``expose`` — configuration read at factory bind time, never tool-call schema.
"""

from akgentic.tool.core import COMMAND, TOOL_CALL, BaseToolParam, Channels


class ReadMailbox(BaseToolParam):
    """Take on a pending message by id, absorbing it into the current run."""

    expose: set[Channels] = {TOOL_CALL}


class Stop(BaseToolParam):
    """Request cancellation of the current run (``/stop``)."""

    expose: set[Channels] = {COMMAND}
