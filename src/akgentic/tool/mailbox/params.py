"""Capability parameters for the mailbox card (ADR-040, ADR-019 §4b).

Two capabilities, one channel each by default: a consuming ``read_mailbox`` on
``TOOL_CALL``, and ``stop`` on ``COMMAND``. Neither needs a field beyond
``expose`` — configuration read at factory bind time, never tool-call schema.
"""

from akgentic.tool.core import COMMAND, TOOL_CALL, BaseToolParam, Channels


class ReadMailbox(BaseToolParam):
    """Read pending messages on demand, absorbing the ones it shows."""

    expose: set[Channels] = {TOOL_CALL}


class Stop(BaseToolParam):
    """Request cancellation of the current run (``/stop``)."""

    expose: set[Channels] = {COMMAND}
