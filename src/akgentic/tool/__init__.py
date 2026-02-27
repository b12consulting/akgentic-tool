"""akgentic-tool public API."""

# Submodules with their own __init__ files
from . import mcp, planning, search
from .core import BaseToolParam, ToolCard, ToolFactory
from .errors import ToolError
from .event import ActorToolObserver, ToolCallEvent, ToolObserver

__all__ = [
    # Core abstractions
    "BaseToolParam",
    "ToolCard",
    "ToolFactory",
    # Errors
    "ToolError",
    # Events and observers
    "ToolCallEvent",
    "ToolObserver",
    "ActorToolObserver",
    # Submodules
    "mcp",
    "planning",
    "search",
]
