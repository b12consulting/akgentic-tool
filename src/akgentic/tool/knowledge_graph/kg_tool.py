"""KnowledgeGraph ToolCard integration.

Provides ``KnowledgeGraphTool`` — a ``ToolCard`` that exposes graph
operations (get_graph, update_graph, search) through configurable channels,
with read-only mode support.

Follows the same pattern as ``PlanningTool`` in akgentic.tool.planning.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from pydantic import Field, field_validator

from akgentic.core.orchestrator import Orchestrator
from akgentic.tool.core import (
    COMMAND,
    LLM_CONTEXT,
    TOOL_CALL,
    BaseToolParam,
    Channels,
    ContextState,
    ToolCard,
    _resolve,
    normalize_system_prompt_to_llm_context,
)
from akgentic.tool.core.observer import ActorToolObserver
from akgentic.tool.knowledge_graph.kg_actor import (
    KG_ACTOR_NAME,
    KG_ACTOR_ROLE,
    KnowledgeGraphActor,
    KnowledgeGraphConfig,
)
from akgentic.tool.knowledge_graph.models import (
    GetGraphQuery,
    GraphView,
    ManageGraph,
    SearchQuery,
    SearchResult,
)
from akgentic.tool.knowledge_graph.state import KnowledgeGraphSummaryState, RootRow
from akgentic.tool.vector_store.protocol import CollectionConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# BaseToolParam subclasses (Task 1)
# ---------------------------------------------------------------------------


class GetGraph(BaseToolParam):
    """Get the full knowledge graph — as structured context state and/or command."""

    expose: set[Channels] = {LLM_CONTEXT, COMMAND}
    prompt_include_schema: bool = Field(
        default=True,
        description="Include entity/relation type schema in the graph summary context state.",
    )
    prompt_include_roots: bool = Field(
        default=True,
        description="Include root entities listing in the graph summary context state.",
    )

    _normalize_expose = field_validator("expose", mode="after")(
        normalize_system_prompt_to_llm_context
    )


class UpdateGraph(BaseToolParam):
    """Update the knowledge graph (create/update/delete entities & relations)."""

    expose: set[Channels] = {TOOL_CALL}


class SearchGraph(BaseToolParam):
    """Search the knowledge graph by keyword, vector, or hybrid mode."""

    expose: set[Channels] = {TOOL_CALL, COMMAND}


def _build_kg_summary_state(
    view: GraphView, include_schema: bool, include_roots: bool
) -> KnowledgeGraphSummaryState:
    """Snapshot the compact graph summary from a ``GraphView`` (ADR-037 §5).

    Carries the full type/root data regardless of the flags — the flags only
    gate rendering. Output scales as O(types + roots), never O(entities).

    Args:
        view: The graph view to summarize.
        include_schema: Resolved ``GetGraph.prompt_include_schema`` flag.
        include_roots: Resolved ``GetGraph.prompt_include_roots`` flag.

    Returns:
        The summary state; an empty graph is a real state whose
        ``render_full()`` is the sentinel line.
    """
    root_entities = sorted((e for e in view.entities if e.is_root), key=lambda e: e.name)
    return KnowledgeGraphSummaryState(
        entity_count=len(view.entities),
        relation_count=len(view.relations),
        entity_types=sorted({e.entity_type for e in view.entities}),
        relation_types=sorted({r.relation_type for r in view.relations}),
        roots=[
            RootRow(name=e.name, entity_type=e.entity_type, description=e.description)
            for e in root_entities
        ],
        include_schema=include_schema,
        include_roots=include_roots,
    )


# ---------------------------------------------------------------------------
# KnowledgeGraphTool ToolCard (Task 2)
# ---------------------------------------------------------------------------


class KnowledgeGraphTool(ToolCard):
    """Knowledge graph tool exposing graph operations through configurable channels.

    Follows the same actor-based pattern as ``PlanningTool``: a singleton
    ``KnowledgeGraphActor`` is created/retrieved via the orchestrator,
    and tool factories delegate to the actor proxy. The ``VectorStoreActor``
    singleton is owned by ``VectorStoreTool`` (declared via ``depends_on``);
    this tool only looks it up at actor-start time.
    """

    vector_store: bool | str = Field(
        default=True,
        description=(
            "False disables vector store wiring; True uses the default VectorStoreActor; "
            "str names a specific VectorStoreActor to look up."
        ),
    )

    collection: CollectionConfig = Field(
        default_factory=CollectionConfig,
        description=(
            "Vector collection configuration (backend, persistence, dimension, tenant). "
            "Propagated to KnowledgeGraphConfig and used by "
            "KnowledgeGraphActor._acquire_vs_proxy when calling create_collection on the "
            "VectorStoreActor."
        ),
    )

    search_top_k: int = Field(
        default=10,
        description=(
            "Default maximum number of search hits to return. "
            "Can be overridden per-call via SearchQuery.top_k."
        ),
    )
    search_score_threshold: float = Field(
        default=0.3,
        description=(
            "Default minimum cosine similarity score for vector/hybrid search results. "
            "Hits below this threshold are filtered out. "
            "Can be overridden per-call via SearchQuery.score_threshold."
        ),
    )

    @property
    def depends_on(self) -> list[str]:
        """Runtime dependency on VectorStoreTool, conditional on vector_store.

        When ``vector_store`` is ``False`` this tool is in degraded mode and
        does not need VectorStoreActor — the factory must not require a
        ``VectorStoreTool`` in the team config. Any other value (``True`` or a
        name ``str``) requires VectorStoreTool to be wired first so the
        KG actor can look up the VectorStoreActor during ``on_start``.
        """
        return ["VectorStoreTool"] if self.vector_store is not False else []

    get_graph: GetGraph | bool = Field(
        default=True,
        description=(
            "Get the full graph — exposed as structured context state and command by default"
        ),
    )
    update_graph: UpdateGraph | bool = Field(
        default=True,
        description="Update graph — TOOL_CALL by default",
    )
    search: SearchGraph | bool = Field(
        default=True,
        description="Search graph — TOOL_CALL + COMMAND by default",
    )

    read_only: bool = False

    # ------------------------------------------------------------------
    # Observer / actor wiring (2.2)
    # ------------------------------------------------------------------

    def observer(self, observer: ActorToolObserver) -> None:  # type: ignore[override]
        """Attach observer and set up the KG actor proxy.

        Assumes ``VectorStoreTool.observer()`` has already created the
        ``VectorStoreActor`` singleton (ordering enforced by
        ``ToolFactory`` topological sort via ``depends_on``). The
        ``KnowledgeGraphActor`` looks that actor up by name during its own
        ``on_start``.
        """
        from akgentic.tool.knowledge_graph import _check_kg_dependencies

        _check_kg_dependencies()
        super().observer(observer)  # store the observer weakly via the base setter

        if observer.orchestrator is None:
            raise ValueError("KnowledgeGraphTool requires access to the orchestrator.")

        orchestrator_proxy = observer.proxy_ask(observer.orchestrator, Orchestrator)

        # Create/retrieve KnowledgeGraphActor singleton. VectorStoreActor creation
        # is owned by VectorStoreTool (depends_on enforces ordering).
        kg_addr = orchestrator_proxy.getChildrenOrCreate(
            KnowledgeGraphActor,
            config=KnowledgeGraphConfig(
                name=KG_ACTOR_NAME,
                role=KG_ACTOR_ROLE,
                vector_store=self.vector_store,
                collection=self.collection,
                search_top_k=self.search_top_k,
                search_score_threshold=self.search_score_threshold,
            ),
        )

        self._kg_proxy = observer.proxy_ask(kg_addr, KnowledgeGraphActor)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_graph_view(view: GraphView) -> str:
        """Format a ``GraphView`` as a human-readable string."""
        if not view.entities:
            return "Knowledge graph is empty."
        lines = ["Knowledge Graph:"]
        lines.append("Entities:")
        for e in view.entities:
            lines.append(f"  - {e.name} ({e.entity_type}): {e.description}")
        lines.append("Relations:")
        for r in view.relations:
            desc = f" — {r.description}" if r.description else ""
            lines.append(f"  - {r.from_entity} --[{r.relation_type}]--> {r.to_entity}{desc}")
        return "\n".join(lines)

    @staticmethod
    def _format_search_result(result: SearchResult) -> str:
        """Format a ``SearchResult`` as a human-readable string."""
        if not result.hits:
            return "No results found."
        lines = ["Search Results:"]
        for hit in result.hits:
            if hit.entity:
                lines.append(
                    f"  - [entity] {hit.entity.name} ({hit.entity.entity_type}): "
                    f"{hit.entity.description} (score: {hit.score:.2f})"
                )
            elif hit.relation:
                r = hit.relation
                lines.append(
                    f"  - [relation] {r.from_entity} --[{r.relation_type}]--> "
                    f"{r.to_entity} (score: {hit.score:.2f})"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Context states (2.3)
    # ------------------------------------------------------------------

    def get_context_states(self) -> list[Callable[[], ContextState | None]]:
        """Get context-state providers for the graph summary (ADR-037 §5).

        Returns:
            List with the graph-summary provider, or empty when disabled or
            not exposed on ``LLM_CONTEXT``.
        """
        gp = _resolve(self.get_graph, GetGraph)
        if gp and LLM_CONTEXT in gp.expose:
            return [self._kg_summary_state_factory(gp)]
        return []

    def _kg_summary_state_factory(self, params: GetGraph) -> Callable[[], ContextState | None]:
        """Create the graph-summary context-state provider.

        Args:
            params: Configuration for the get_graph capability

        Returns:
            Zero-arg provider producing a summary snapshot, or ``None`` when
            the state is unavailable. Never raises.
        """
        kg_proxy = self._kg_proxy
        include_schema = params.prompt_include_schema
        include_roots = params.prompt_include_roots

        def knowledge_graph_summary_state() -> KnowledgeGraphSummaryState | None:
            try:
                view = kg_proxy.get_graph(GetGraphQuery())
                return _build_kg_summary_state(view, include_schema, include_roots)
            except Exception:
                logger.error("Failed to get knowledge graph summary state", exc_info=True)
                return None

        return knowledge_graph_summary_state

    # ------------------------------------------------------------------
    # Tools (2.4)
    # ------------------------------------------------------------------

    def get_tools(self) -> list[Callable[..., Any]]:
        """Return callable tool functions for LLM agents."""
        tools: list[Callable[..., Any]] = []

        gp = _resolve(self.get_graph, GetGraph)
        if gp and TOOL_CALL in gp.expose:
            tools.append(self._get_graph_factory(gp))

        if not self.read_only:
            up = _resolve(self.update_graph, UpdateGraph)
            if up and TOOL_CALL in up.expose:
                tools.append(self._update_graph_factory(up))

        sp = _resolve(self.search, SearchGraph)
        if sp and TOOL_CALL in sp.expose:
            tools.append(self._search_factory(sp))

        return tools

    # ------------------------------------------------------------------
    # Commands (2.5)
    # ------------------------------------------------------------------

    def get_commands(self) -> dict[type[BaseToolParam], Callable[..., Any]]:
        """Return command mappings for inter-agent orchestration."""
        commands: dict[type[BaseToolParam], Callable[..., Any]] = {}

        gp = _resolve(self.get_graph, GetGraph)
        if gp and COMMAND in gp.expose:
            commands[GetGraph] = self._get_graph_factory(gp)

        sp = _resolve(self.search, SearchGraph)
        if sp and COMMAND in sp.expose:
            commands[SearchGraph] = self._search_factory(sp)

        return commands

    # ------------------------------------------------------------------
    # Factory closures (2.6, 2.7, 2.8)
    # ------------------------------------------------------------------

    def _get_graph_factory(self, params: GetGraph) -> Callable[..., Any]:
        """Return a closure that fetches and formats the graph."""
        kg_proxy = self._kg_proxy
        format_view = self._format_graph_view

        def get_graph() -> str:
            """Get the current knowledge graph."""
            view = kg_proxy.get_graph(GetGraphQuery())
            return format_view(view)

        get_graph.__doc__ = params.format_docstring(get_graph.__doc__)
        return get_graph

    def _update_graph_factory(self, params: UpdateGraph) -> Callable[..., Any]:
        """Return a closure that applies graph mutations."""
        kg_proxy = self._kg_proxy

        def update_graph(update: ManageGraph) -> str:
            """Update the knowledge graph (create/update/delete entities & relations).

            Use this tool to add new entities, update existing ones, create or
            remove relations, and delete entities from the knowledge graph."""
            return kg_proxy.update_graph(update)

        update_graph.__doc__ = params.format_docstring(update_graph.__doc__)
        return update_graph

    def _search_factory(self, params: SearchGraph) -> Callable[..., Any]:
        """Return a closure that searches the graph."""
        kg_proxy = self._kg_proxy
        format_result = self._format_search_result

        def search_graph(query: SearchQuery) -> str:
            """Search the knowledge graph by keyword, vector, or hybrid mode."""
            result = kg_proxy.search(query)
            return format_result(result)

        search_graph.__doc__ = params.format_docstring(search_graph.__doc__)
        return search_graph
