from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal, cast

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
from akgentic.tool.core.observer import ActorToolObserver, ToolObserver
from akgentic.tool.errors import RetriableError
from akgentic.tool.planning.planning_actor import (
    PlanActor,
    PlanConfig,
    Task,
    TaskStatus,
    UpdatePlan,
)
from akgentic.tool.planning.state import PlanningState, TaskRow
from akgentic.tool.vector_store.hybrid import DEFAULT_ALPHA
from akgentic.tool.vector_store.protocol import CollectionConfig, require_weaviate_configured

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PLANNING_ACTOR_NAME = "#PlanningTool"
PLANNING_ACTOR_ROLE = "ToolActor"


class GetPlanning(BaseToolParam):
    """Get the full team plan — as structured context state and/or tool."""

    expose: set[Channels] = {LLM_CONTEXT, COMMAND}
    filter_by_agent: bool = Field(
        default=True,
        description=(
            "When True (default), the planning context state shows only tasks owned or created "
            "by the calling agent. The team summary (totals + owner breakdown) is always shown. "
            "Set False to list all tasks."
        ),
    )

    _normalize_expose = field_validator("expose", mode="after")(
        normalize_system_prompt_to_llm_context
    )


class GetPlanningTask(BaseToolParam):
    """Get a single task by ID."""

    expose: set[Channels] = {TOOL_CALL, COMMAND}


class UpdatePlanning(BaseToolParam):
    """Update tasks."""


class SearchPlanning(BaseToolParam):
    """Search tasks by status, owner, creator, and/or natural-language description."""

    expose: set[Channels] = {TOOL_CALL, COMMAND}


def _build_planning_state(
    tasks: list[Task], agent_name: str, filter_by_agent: bool
) -> PlanningState:
    """Snapshot the planning board, shaped for one agent (ADR-037 §5).

    Per-agent shaping happens here, at production time: when ``filter_by_agent``
    is on, only tasks the agent owns — or created and are assigned — enter the
    rows (an unassigned task never appears even if the agent created it), while
    ``total`` and ``owner_counts`` always cover the whole board.

    Args:
        tasks: All tasks on the board.
        agent_name: The observing agent's own actor name.
        filter_by_agent: Whether ``tasks`` is narrowed to the agent's own tasks.

    Returns:
        The planning state; an empty board is a real state whose
        ``render_full()`` is the sentinel line.
    """
    owner_counts: dict[str, int] = {}
    for task in tasks:
        key = task.owner if task.owner else "unassigned"
        owner_counts[key] = owner_counts.get(key, 0) + 1

    visible = (
        [t for t in tasks if t.owner == agent_name or (t.owner and t.creator == agent_name)]
        if filter_by_agent
        else tasks
    )
    rows = [
        TaskRow(
            id=t.id,
            status=t.status,
            description=t.description,
            owner=t.owner,
            creator=t.creator,
            output=t.output,
        )
        for t in visible
    ]
    return PlanningState(
        total=len(tasks),
        owner_counts=owner_counts,
        tasks=rows,
        agent_name=agent_name,
        filter_by_agent=filter_by_agent,
    )


class PlanningTool(ToolCard):
    """Team planning management via actor-based plan store.

    The ``VectorStoreActor`` singleton is owned by ``VectorStoreTool`` and
    declared as a dependency here; this tool only looks it up at actor-start
    time.
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
            "Propagated to PlanConfig and used by PlanActor._acquire_vs_proxy when calling "
            "create_collection on the VectorStoreActor."
        ),
    )

    search_top_k: int = Field(
        default=10,
        description="Default top-k for semantic search in search_planning.",
    )
    search_score_threshold: float = Field(
        default=0.5,
        description="Default minimum cosine similarity score for semantic results.",
    )
    hybrid_alpha: float = Field(
        default=DEFAULT_ALPHA,
        ge=0.0,
        le=1.0,
        description=(
            "Weight of the vector leg when fusing hybrid search results; the keyword leg "
            "gets 1 - alpha. Matches Weaviate's hybrid() alpha parameter. Below 0.5 a "
            "keyword match outranks a strong semantic hit."
        ),
    )

    @property
    def depends_on(self) -> list[str]:
        """Runtime dependency on VectorStoreTool, conditional on vector_store.

        When ``vector_store`` is ``False`` this tool is in degraded mode and
        does not need VectorStoreActor — the factory must not require a
        ``VectorStoreTool`` in the team config. Any other value (``True`` or a
        name ``str``) requires VectorStoreTool to be wired first so the
        PlanActor can look up the VectorStoreActor during ``on_start``.
        """
        return ["VectorStoreTool"] if self.vector_store is not False else []

    get_planning: GetPlanning | bool = Field(
        default=True,
        description="By default the plan is exposed as structured context state and as a command",
    )
    get_planning_task: GetPlanningTask | bool = True
    update_planning: UpdatePlanning | bool = True
    search_planning: SearchPlanning | bool = True

    def observer(self, observer: ToolObserver) -> PlanningTool:
        """Attach observer and set up the planning actor proxy.

        Assumes ``VectorStoreTool.observer()`` has already created the
        ``VectorStoreActor`` singleton (ordering enforced by
        ``ToolFactory`` topological sort via ``depends_on``). The
        ``PlanActor`` looks that actor up by name during its own ``on_start``.

        Requires an ActorToolObserver for actor system access; the parameter keeps
        the base ``ToolObserver`` type so the override stays substitutable, and
        :meth:`_actor_observer` applies the narrower type.

        Raises:
            ValueError: If observer.orchestrator is None.
        """
        require_weaviate_configured(self.collection, "PlanningTool")
        super().observer(observer)  # store the observer weakly via the base setter
        actor_observer = self._actor_observer()
        if actor_observer.orchestrator is None:
            raise ValueError("PlanningTool requires access to the orchestrator.")

        orchestrator_proxy = actor_observer.proxy_ask(actor_observer.orchestrator, Orchestrator)

        # Create/retrieve PlanActor singleton. VectorStoreActor creation is owned
        # by VectorStoreTool (depends_on enforces ordering).
        planning_tool_addr = orchestrator_proxy.getChildrenOrCreate(
            PlanActor,
            config=PlanConfig(
                name=PLANNING_ACTOR_NAME,
                role=PLANNING_ACTOR_ROLE,
                vector_store=self.vector_store,
                collection=self.collection,
                search_top_k=self.search_top_k,
                search_score_threshold=self.search_score_threshold,
                hybrid_alpha=self.hybrid_alpha,
            ),
        )

        self._planning_proxy = actor_observer.proxy_ask(planning_tool_addr, PlanActor)
        return self

    def _actor_observer(self) -> ActorToolObserver:
        """Live observer typed as the actor protocol. Raises once the agent stops.

        Conformance is a documented precondition of :meth:`observer`, not a runtime
        gate — observers are duck-typed, so a non-conforming one fails at first use
        just as it did before.
        """
        return cast(ActorToolObserver, self._observer)

    def _actor_observer_or_none(self) -> ActorToolObserver | None:
        """Live observer typed as the actor protocol; ``None`` once the agent stops."""
        return cast(ActorToolObserver | None, self._observer_or_none())

    def get_context_states(self) -> list[Callable[[], ContextState | None]]:
        """Get context-state providers for planning context (ADR-037 §5).

        Returns:
            List with the planning-state provider, or empty when disabled or
            not exposed on ``LLM_CONTEXT``.
        """
        gp = _resolve(self.get_planning, GetPlanning)
        if gp and LLM_CONTEXT in gp.expose:
            return [self._planning_state_factory(gp)]
        return []

    def get_tools(self) -> list[Callable[..., Any]]:
        tools: list[Callable[..., Any]] = []

        gp = _resolve(self.get_planning, GetPlanning)
        if gp and TOOL_CALL in gp.expose:
            tools.append(self._planning_prompt_factory(gp))

        gpi = _resolve(self.get_planning_task, GetPlanningTask)
        if gpi and TOOL_CALL in gpi.expose:
            tools.append(self._get_planning_task_factory(gpi))

        up = _resolve(self.update_planning, UpdatePlanning)
        if up and TOOL_CALL in up.expose:
            tools.append(self._update_planning_factory(up))

        sp = _resolve(self.search_planning, SearchPlanning)
        if sp and TOOL_CALL in sp.expose:
            tools.append(self._search_planning_factory(sp))

        return tools

    def get_commands(self) -> dict[type[BaseToolParam], Callable[..., Any]]:
        commands: dict[type[BaseToolParam], Callable[..., Any]] = {}

        gp = _resolve(self.get_planning, GetPlanning)
        if gp and COMMAND in gp.expose:
            commands[GetPlanning] = self._planning_prompt_factory(gp)

        gpi = _resolve(self.get_planning_task, GetPlanningTask)
        if gpi and COMMAND in gpi.expose:
            commands[GetPlanningTask] = self._get_planning_task_factory(gpi)

        sp = _resolve(self.search_planning, SearchPlanning)
        if sp and COMMAND in sp.expose:
            commands[SearchPlanning] = self._search_planning_factory(sp)

        return commands

    def _planning_state_factory(self, params: GetPlanning) -> Callable[[], ContextState | None]:
        """Create the planning context-state provider.

        Args:
            params: Configuration for the get_planning capability

        Returns:
            Zero-arg provider producing a planning snapshot, or ``None`` when
            the state is unavailable. Never raises.
        """
        planning_proxy = self._planning_proxy
        observer_or_none = self._actor_observer_or_none  # bound method -> weak edge to agent
        filter_by_agent = params.filter_by_agent

        def planning_state() -> PlanningState | None:
            try:
                observer = observer_or_none()
                if observer is None:
                    return None  # agent gone -> state unavailable
                return _build_planning_state(
                    planning_proxy.get_planning(), observer.myAddress.name, filter_by_agent
                )
            except Exception:
                logger.error("Failed to get planning state", exc_info=True)
                return None

        return planning_state

    def _planning_prompt_factory(self, params: GetPlanning) -> Callable[..., Any]:
        planning_proxy = self._planning_proxy
        # Capture agent identity and filter setting at bind time — stable for actor's lifetime.
        agent_name = self._actor_observer().myAddress.name
        filter_by_agent = params.filter_by_agent

        def planning_summary() -> str:
            """Summarize the team planning: task totals, per-owner breakdown, and
            the task list (all tasks, or only yours when ``filter_by_agent``)."""
            tasks = planning_proxy.get_planning()
            return _build_planning_state(tasks, agent_name, filter_by_agent).render_full()

        return planning_summary

    def _get_planning_task_factory(self, params: GetPlanningTask) -> Callable[..., Any]:
        planning_proxy = self._planning_proxy

        def get_planning_task(task_id: int) -> Task | str:
            """Get a single team task by its integer ID."""
            return planning_proxy.get_planning_task(task_id)

        get_planning_task.__doc__ = params.format_docstring(get_planning_task.__doc__)
        return get_planning_task

    def _update_planning_factory(self, params: UpdatePlanning) -> Callable[..., Any]:
        planning_proxy = self._planning_proxy
        observer_or_none = self._actor_observer_or_none  # bound method -> weak edge to agent

        def update_planning(update: UpdatePlan) -> str:
            """Update team tasks (create, update, delete).

            Field constraints (violating them causes a validation error):
            - description: max 300 characters — keep it concise.
            - output: max 150 characters — will be truncated automatically if exceeded.
            """
            observer = observer_or_none()
            if observer is None:
                raise RetriableError("Plan is unavailable; the agent is shutting down.")
            ## observer.myAddress is used to set the creator of any new tasks in the plan.
            return planning_proxy.update_planning(update, observer.myAddress)

        update_planning.__doc__ = params.format_docstring(update_planning.__doc__)
        return update_planning

    def _search_planning_factory(self, params: SearchPlanning) -> Callable[..., Any]:
        planning_proxy = self._planning_proxy

        def search_planning(
            query: str | None = None,
            mode: Literal["hybrid", "vector", "keyword"] = "hybrid",
            status: TaskStatus | None = None,
            owner: str | None = None,
            creator: str | None = None,
            top_k: int | None = None,
            score_threshold: float | None = None,
        ) -> list[str]:
            """Search tasks. All filters are AND-combined; omit all for full list.

            Args:
                query: Search text for keyword and/or semantic matching.
                mode: "hybrid" (default) = keyword + semantic,
                    "keyword" = substring only, "vector" = semantic only.
                status: Filter by status.
                owner: Filter by owner.
                creator: Filter by creator.
                top_k: Max semantic hits (default 10).
                score_threshold: Min cosine similarity (default 0.5).

            Returns scored results: "(semantic: 0.85)", "(keyword match)", "(hybrid: 0.90)".
            """
            return planning_proxy.search_planning(
                status=status,
                owner=owner,
                creator=creator,
                query=query,
                mode=mode,
                top_k=top_k,
                score_threshold=score_threshold,
            )

        search_planning.__doc__ = params.format_docstring(search_planning.__doc__)
        return search_planning
