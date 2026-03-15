"""akgentic-tool public API."""

from __future__ import annotations

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

try:
    from .vector import EmbeddingService, VectorEntry, VectorIndex

    _VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    _VECTOR_SEARCH_AVAILABLE = False

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
    # Vector infrastructure (requires akgentic-tool[vector_search])
    "VectorEntry",
    "EmbeddingService",
    "VectorIndex",
]
