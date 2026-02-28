"""akgentic-tool public API."""

# Submodules with their own __init__ files
from . import mcp, planning, search, team
from .core import (
    COMMAND,
    SYSTEM_PROMPT,
    TOOL_CALL,
    BaseToolParam,
    Channels,
    ToolCard,
    ToolFactory,
)
from .errors import RetriableError
from .event import (
    ActorToolObserver,
    TeamManagementToolObserver,
    ToolCallEvent,
    ToolObserver,
)

__all__ = [
    # Core abstractions
    "BaseToolParam",
    "ToolCard",
    "ToolFactory",
    # Expose channel constants
    "COMMAND",
    "SYSTEM_PROMPT",
    "TOOL_CALL",
    "Channels",
    # Errors
    "RetriableError",
    # Events and observers
    "ToolCallEvent",
    "ToolObserver",
    "ActorToolObserver",
    "TeamManagementToolObserver",
    # Submodules
    "mcp",
    "planning",
    "search",
    "team",
]
