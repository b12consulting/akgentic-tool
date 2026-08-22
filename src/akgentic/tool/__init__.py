"""akgentic-tool public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .knowledge_graph.models import KnowledgeGraphStateEvent as KnowledgeGraphStateEvent

# Submodules with their own __init__ files
from . import mcp, planning, sandbox, search, team, workspace  # noqa: F401
from .core import (  # noqa: F401
    COMMAND,
    LLM_CONTEXT,
    SYSTEM_PROMPT,
    TOOL_CALL,
    BaseToolParam,
    Channels,
    CommandRegistry,
    ContextState,
    ToolCard,
    ToolFactory,
    ToolState,
    normalize_system_prompt_to_llm_context,
)
from .core.event import (  # noqa: F401
    CommandArg,
    CommandDescriptor,
    CommandsAnnouncedEvent,
    ToolStateEvent,
)
from .core.observer import ActorToolObserver, ToolObserver, ToolStateCarrier  # noqa: F401
from .errors import CommandNotRecognized, RetriableError, ToolObserverGone  # noqa: F401
from .metadata.tool import MetadataTool  # noqa: F401
from .notification.tool import NotificationTool  # noqa: F401
from .sandbox.bwrap import BwrapSandboxActor  # noqa: F401
from .sandbox.seatbelt import SeatbeltSandboxActor  # noqa: F401
from .sandbox.tool import ExecTool  # noqa: F401
from .skill.tool import SkillTool  # noqa: F401
from .team.observer import TeamManagementToolObserver  # noqa: F401
from .workspace.tool import WorkspaceTool  # noqa: F401

try:
    from .vector_store.vector import EmbeddingService, VectorEntry, VectorIndex  # noqa: F401

    _VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    _VECTOR_SEARCH_AVAILABLE = False

__all__ = [
    # Core abstractions
    "BaseToolParam",
    "ContextState",
    "ToolCard",
    "ToolFactory",
    "ToolState",
    "CommandRegistry",
    "normalize_system_prompt_to_llm_context",
    # Expose channel constants
    "COMMAND",
    "LLM_CONTEXT",
    "SYSTEM_PROMPT",
    "TOOL_CALL",
    "Channels",
    # Errors
    "RetriableError",
    "CommandNotRecognized",
    "ToolObserverGone",
    # Events and observers
    "ToolObserver",
    "ActorToolObserver",
    "ToolStateCarrier",
    "TeamManagementToolObserver",
    "ToolStateEvent",
    "KnowledgeGraphStateEvent",
    # Command discovery models
    "CommandArg",
    "CommandDescriptor",
    "CommandsAnnouncedEvent",
    # Submodules
    "mcp",
    "planning",
    "sandbox",
    "search",
    "team",
    "workspace",
    "BwrapSandboxActor",
    "ExecTool",
    "MetadataTool",
    "NotificationTool",
    "SeatbeltSandboxActor",
    "SkillTool",
    "WorkspaceTool",
]

if _VECTOR_SEARCH_AVAILABLE:
    __all__ += ["VectorEntry", "EmbeddingService", "VectorIndex"]


def __getattr__(name: str) -> Any:
    """Lazy re-export of the KG delta payload (Story 17.1).

    ``KnowledgeGraphStateEvent`` lives in ``akgentic.tool.knowledge_graph.models``
    and pulls the ``[vector_search]`` optional dependency chain when imported.
    Exposing it via module ``__getattr__`` keeps the bare ``akgentic.tool``
    import cheap (see ``test_tool_import_does_not_trigger_kg_import``) while
    still honoring AC #5 of Story 17.1.
    """
    if name == "KnowledgeGraphStateEvent":
        from .knowledge_graph.models import KnowledgeGraphStateEvent

        return KnowledgeGraphStateEvent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
