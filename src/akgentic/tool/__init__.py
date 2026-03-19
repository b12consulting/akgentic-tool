"""akgentic-tool public API."""

from __future__ import annotations

# Submodules with their own __init__ files
from . import mcp, planning, sandbox, search, team, workspace  # noqa: F401
from .core import (  # noqa: F401
    COMMAND,
    SYSTEM_PROMPT,
    TOOL_CALL,
    BaseToolParam,
    Channels,
    ToolCard,
    ToolFactory,
)
from .errors import RetriableError  # noqa: F401
from .event import (  # noqa: F401
    ActorToolObserver,
    TeamManagementToolObserver,
    ToolCallEvent,
    ToolObserver,
)
from .sandbox.tool import ExecTool  # noqa: F401
from .workspace.tool import WorkspaceTool  # noqa: F401

try:
    from .vector import EmbeddingService, VectorEntry, VectorIndex  # noqa: F401

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
    "sandbox",
    "search",
    "team",
    "workspace",
    "ExecTool",
    "WorkspaceTool",
]

if _VECTOR_SEARCH_AVAILABLE:
    __all__ += ["VectorEntry", "EmbeddingService", "VectorIndex"]
