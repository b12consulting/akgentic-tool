"""Knowledge Graph subsystem for akgentic-tool.

Provides in-memory knowledge graph with entity/relation CRUD,
backed by a Pykka actor (KnowledgeGraphActor) for thread safety.

Requires the ``[kg]`` optional dependency group::

    pip install akgentic-tool[kg]

Public API
----------
Models:
    Entity, Relation, KnowledgeGraph, KnowledgeGraphState,
    EntityCreate, EntityUpdate, RelationCreate, RelationDelete, ManageGraph

ToolCard:
    KnowledgeGraphTool, GetGraph, UpdateGraph, SearchGraph

Actor:
    KnowledgeGraphActor, KG_ACTOR_NAME, KG_ACTOR_ROLE

Utilities:
    _check_kg_dependencies — validates optional deps are installed.
"""

from __future__ import annotations

from akgentic.tool.knowledge_graph.kg_actor import (
    KG_ACTOR_NAME,
    KG_ACTOR_ROLE,
    KnowledgeGraphActor,
)
from akgentic.tool.knowledge_graph.kg_tool import (
    GetGraph,
    KnowledgeGraphTool,
    SearchGraph,
    UpdateGraph,
)
from akgentic.tool.knowledge_graph.models import (
    Entity,
    EntityCreate,
    EntityUpdate,
    GetGraphQuery,
    GraphView,
    KnowledgeGraph,
    KnowledgeGraphState,
    ManageGraph,
    Relation,
    RelationCreate,
    RelationDelete,
    SearchHit,
    SearchQuery,
    SearchResult,
)


def _check_kg_dependencies() -> None:
    """Validate that ``[kg]`` optional dependencies are installed.

    Raises:
        ImportError: With install instructions when ``numpy`` or ``openai`` is missing.
    """
    missing: list[str] = []
    try:
        import numpy as _  # noqa: F811, F401
    except ImportError:
        missing.append("numpy")
    try:
        import openai as _  # noqa: F811, F401
    except ImportError:
        missing.append("openai")
    if missing:
        msg = (
            f"The knowledge_graph module requires extra dependencies ({', '.join(missing)}). "
            "Install them with: pip install akgentic-tool[kg]"
        )
        raise ImportError(msg)


__all__ = [
    # Core domain models
    "Entity",
    "Relation",
    "KnowledgeGraph",
    "KnowledgeGraphState",
    # CRUD request models
    "EntityCreate",
    "EntityUpdate",
    "RelationCreate",
    "RelationDelete",
    "ManageGraph",
    # Query & search models (Story 1.2)
    "GetGraphQuery",
    "GraphView",
    "SearchQuery",
    "SearchHit",
    "SearchResult",
    # ToolCard (Story 1.3)
    "KnowledgeGraphTool",
    "GetGraph",
    "UpdateGraph",
    "SearchGraph",
    # Actor
    "KnowledgeGraphActor",
    "KG_ACTOR_NAME",
    "KG_ACTOR_ROLE",
    # Utilities
    "_check_kg_dependencies",
]
