"""Channel names for capability exposure, plus their module-level aliases."""

from enum import StrEnum


class Channels(StrEnum):
    """Valid channel names for capability exposure."""

    SYSTEM_PROMPT = "system_prompt"
    """Expose as a system prompt injected into the LLM context.

    Rendered once into the frozen system block — part of the cached prefix.
    Static content only; volatile state belongs on ``LLM_CONTEXT``.
    """

    TOOL_CALL = "tool_call"
    """Expose as a callable tool for the LLM."""

    COMMAND = "command"
    """Expose as a programmatic command for inter-agent orchestration."""

    LLM_CONTEXT = "llm_context"
    """Expose as structured context state pushed into the context tail.

    Delivered per turn, as a delta (ADR-037 §4) — the channel for volatile
    tool state that must not invalidate the cached system-prompt prefix.
    """


# Backward-compatible module-level aliases
SYSTEM_PROMPT = Channels.SYSTEM_PROMPT
TOOL_CALL = Channels.TOOL_CALL
COMMAND = Channels.COMMAND
LLM_CONTEXT = Channels.LLM_CONTEXT
