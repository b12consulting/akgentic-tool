"""akgentic-tool public API."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .knowledge_graph.models import KnowledgeGraphStateEvent as KnowledgeGraphStateEvent
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
    ToolObserver,
    ToolStateEvent,
    ToolStatePayload,
)
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
    "ToolObserver",
    "ActorToolObserver",
    "TeamManagementToolObserver",
    "ToolStateEvent",
    "ToolStatePayload",
    "KnowledgeGraphStateEvent",
    # Submodules
    "mcp",
    "planning",
    "sandbox",
    "search",
    "team",
    "workspace",
    "WorkspaceTool",
    "BwrapSandboxActor",
    "ExecTool",
    "SeatbeltSandboxActor",
]

if _VECTOR_SEARCH_AVAILABLE:
    __all__ += ["VectorEntry", "EmbeddingService", "VectorIndex"]


_LAZY_SUBMODULES = {"mcp", "planning", "sandbox", "search", "team", "workspace"}
_LAZY_ATTRS = {
    "BwrapSandboxActor": (".sandbox.bwrap", "BwrapSandboxActor"),
    "ExecTool": (".sandbox.tool", "ExecTool"),
    "SeatbeltSandboxActor": (".sandbox.seatbelt", "SeatbeltSandboxActor"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module

    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        value = getattr(import_module(module_name, __name__), attr_name)
        globals()[name] = value
        return value

    if name == "KnowledgeGraphStateEvent":
        from .knowledge_graph.models import KnowledgeGraphStateEvent

        return KnowledgeGraphStateEvent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
