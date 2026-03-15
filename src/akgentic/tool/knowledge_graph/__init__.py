"""Knowledge Graph subsystem for akgentic-tool.

Provides in-memory knowledge graph with entity/relation CRUD,
backed by a Pykka actor (KnowledgeGraphActor) for thread safety.

Requires the ``[vector_search]`` optional dependency group::

    pip install akgentic-tool[vector_search]

Public API
----------
Models:
    Entity, Relation, KnowledgeGraph, KnowledgeGraphState,
    EntityCreate, EntityUpdate, RelationCreate, RelationDelete, ManageGraph,
    PathStep, GetGraphQuery, GraphView, SearchQuery, SearchHit, SearchResult

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
    KnowledgeGraphConfig,
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
    PathStep,
    Relation,
    RelationCreate,
    RelationDelete,
    SearchHit,
    SearchQuery,
    SearchResult,
    VectorEntry,
)
from akgentic.tool.knowledge_graph.vector_index import EmbeddingService, VectorIndex


def _check_kg_dependencies() -> None:
    """Validate that ``[vector_search]`` optional dependencies are installed.

    Raises:
        ImportError: With install instructions when ``numpy`` or ``openai`` is missing.
    """
    missing: list[str] = []
    try:
        import numpy as _np  # noqa: F401
    except ImportError:
        missing.append("numpy")
    try:
        import openai as _openai  # noqa: F401
    except ImportError:
        missing.append("openai")
    if missing:
        msg = (
            f"The knowledge_graph module requires extra dependencies ({', '.join(missing)}). "
            "Install them with: pip install akgentic-tool[vector_search]"
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
    # Query & search models (Story 1.2 + 3.1)
    "GetGraphQuery",
    "GraphView",
    "PathStep",
    "SearchQuery",
    "SearchHit",
    "SearchResult",
    # Vector index models (Story 2.1)
    "VectorEntry",
    # ToolCard (Story 1.3)
    "KnowledgeGraphTool",
    "GetGraph",
    "UpdateGraph",
    "SearchGraph",
    # Actor (Story 2.1: KnowledgeGraphConfig added)
    "KnowledgeGraphActor",
    "KnowledgeGraphConfig",
    "KG_ACTOR_NAME",
    "KG_ACTOR_ROLE",
    # Vector index services (Story 2.1)
    "EmbeddingService",
    "VectorIndex",
    # Utilities
    "_check_kg_dependencies",
]
