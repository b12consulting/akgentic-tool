"""Channel names for capability exposure, plus their module-level aliases."""

from enum import StrEnum


class Channels(StrEnum):
    """Valid channel names for capability exposure."""

    SYSTEM_PROMPT = "system_prompt"
    """Expose as a system prompt injected into the LLM context."""

    TOOL_CALL = "tool_call"
    """Expose as a callable tool for the LLM."""

    COMMAND = "command"
    """Expose as a programmatic command for inter-agent orchestration."""


# Backward-compatible module-level aliases
SYSTEM_PROMPT = Channels.SYSTEM_PROMPT
TOOL_CALL = Channels.TOOL_CALL
COMMAND = Channels.COMMAND
