from __future__ import annotations

import logging
from typing import Callable, Literal

from pydantic import Field

from akgentic.core.orchestrator import Orchestrator
from akgentic.tool.core import (
    COMMAND,
    SYSTEM_PROMPT,
    TOOL_CALL,
    BaseToolParam,
    Channels,
    ToolCard,
    _resolve,
)
from akgentic.tool.event import ActorToolObserver, ToolCallEvent
from akgentic.tool.planning.planning_actor import PlanActor, PlanConfig, Task, UpdatePlan

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PLANNING_ACTOR_NAME = "#PlanningTool"
PLANNING_ACTOR_ROLE = "ToolActor"


class GetPlanning(BaseToolParam):
    """Get the full team plan — as system prompt and/or tool."""

    expose: set[Channels] = {SYSTEM_PROMPT, COMMAND}
    filter_by_agent: bool = Field(
        default=True,
        description=(
            "When True (default), the system prompt shows only tasks owned or created by the "
            "calling agent. The team summary (totals + owner breakdown) is always shown. "
            "Set False to list all tasks."
        ),
    )


class GetPlanningTask(BaseToolParam):
    """Get a single task by ID."""

    expose: set[Channels] = {TOOL_CALL, COMMAND}


class UpdatePlanning(BaseToolParam):
    """Update tasks."""


class PlanningTool(ToolCard):
    """Team planning management via actor-based plan store."""

    name: str = "Planning"
    description: str = "Planning tool to manage team plans and tasks"

    get_planning: GetPlanning | bool = Field(
        default=True, description="By default the plan in included in the system prompt"
    )
    get_planning_task: GetPlanningTask | bool = True
    update_planning: UpdatePlanning | bool = True
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model passed through to PlanConfig for semantic task search",
    )
    embedding_provider: Literal["openai", "azure"] = Field(
        default="openai",
        description="Embedding provider passed through to PlanConfig for semantic task search",
    )

    def observer(self, observer: ActorToolObserver) -> None:  # type: ignore[override]
        """Attach observer and set up the planning actor proxy.

        Requires an ActorToolObserver for actor system access.
        """
        self._observer = observer
        if observer.orchestrator is None:
            raise ValueError("PlanningTool requires access to the orchestrator.")

        orchestrator_proxy_ask = observer.proxy_ask(observer.orchestrator, Orchestrator)
        planning_tool_addr = orchestrator_proxy_ask.get_team_member(PLANNING_ACTOR_NAME)

        if planning_tool_addr is None:
            logger.info(f"PlanningTool: create {PLANNING_ACTOR_NAME}.")
            config = PlanConfig(
                name=PLANNING_ACTOR_NAME,
                role=PLANNING_ACTOR_ROLE,
                embedding_model=self.embedding_model,
                embedding_provider=self.embedding_provider,
            )
            planning_tool_addr = orchestrator_proxy_ask.createActor(PlanActor, config=config)

        self._planning_proxy = observer.proxy_ask(planning_tool_addr, PlanActor)

    def get_system_prompts(self) -> list[Callable]:
        gp = _resolve(self.get_planning, GetPlanning)
        if gp and SYSTEM_PROMPT in gp.expose:
            return [self._planning_prompt_factory(gp)]
        return []

    def get_tools(self) -> list[Callable]:
        tools: list[Callable] = []

        gp = _resolve(self.get_planning, GetPlanning)
        if gp and TOOL_CALL in gp.expose:
            tools.append(self._planning_prompt_factory(gp))

        gpi = _resolve(self.get_planning_task, GetPlanningTask)
        if gpi and TOOL_CALL in gpi.expose:
            tools.append(self._get_planning_task_factory(gpi))

        up = _resolve(self.update_planning, UpdatePlanning)
        if up and TOOL_CALL in up.expose:
            tools.append(self._update_planning_factory(up))

        return tools

    def get_commands(self) -> dict[type[BaseToolParam], Callable]:
        commands: dict[type[BaseToolParam], Callable] = {}

        gp = _resolve(self.get_planning, GetPlanning)
        if gp and COMMAND in gp.expose:
            commands[GetPlanning] = self._planning_prompt_factory(gp)

        gpi = _resolve(self.get_planning_task, GetPlanningTask)
        if gpi and COMMAND in gpi.expose:
            commands[GetPlanningTask] = self._get_planning_task_factory(gpi)

        return commands

    def _planning_prompt_factory(self, params: GetPlanning) -> Callable:
        planning_proxy = self._planning_proxy
        # Capture agent identity and filter setting at bind time — stable for actor's lifetime.
        agent_name = self._observer.myAddress.name
        filter_by_agent = params.filter_by_agent

        def planning_prompt() -> str:
            """Get the full team planning."""
            tasks = planning_proxy.get_planning()
            if not tasks:
                return "No current team planning."

            total = len(tasks)

            # --- Build per-owner breakdown ---
            owner_counts: dict[str, int] = {}
            for task in tasks:
                key = task.owner if task.owner else "unassigned"
                owner_counts[key] = owner_counts.get(key, 0) + 1

            named = sorted((k, v) for k, v in owner_counts.items() if k != "unassigned")
            unassigned_count = owner_counts.get("unassigned", 0)
            breakdown_parts = [f"{name}: {count}" for name, count in named]
            if unassigned_count:
                breakdown_parts.append(f"unassigned: {unassigned_count}")
            breakdown = " | ".join(breakdown_parts)

            lines = [f"**Team planning:** {total} task{'s' if total != 1 else ''} total"]
            lines.append(f"Owners: {breakdown}")

            if filter_by_agent:
                # Only include tasks where the calling agent is owner or creator.
                # Unassigned tasks (empty owner) never appear here even if creator matches.
                own_tasks = [
                    t
                    for t in tasks
                    if t.owner == agent_name or (t.owner and t.creator == agent_name)
                ]
                if own_tasks:
                    lines.append(f"\n**Your tasks** (owner or creator: {agent_name}):")
                    for task in own_tasks:
                        role_tag = ""
                        if task.owner != agent_name and task.creator == agent_name:
                            role_tag = " [created by you]"
                        output_part = f" — Output: {task.output}" if task.output else ""
                        lines.append(
                            f"- ID {task.id} [{task.status}] {task.description}"
                            f"{output_part}{role_tag}"
                        )
                else:
                    lines.append(
                        f"\nNo tasks assigned to or created by {agent_name} yet."
                    )
            else:
                lines.append("\n**All tasks:**")
                for task in tasks:
                    output_part = f" — Output: {task.output}" if task.output else ""
                    owner_label = task.owner or "unassigned"
                    lines.append(
                        f"- ID {task.id} [{task.status}] {task.description}"
                        f"{output_part} (Owner: {owner_label}, Creator: {task.creator})"
                    )

            lines.append(
                "\nUse get_planning_task(id) for exact lookup or "
                "get_planning_task(query) for semantic search."
            )
            return "\n".join(lines)

        return planning_prompt

    def _get_planning_task_factory(self, params: GetPlanningTask) -> Callable:
        planning_proxy = self._planning_proxy
        observer = self._observer

        def get_planning_task(task_id: int | str) -> Task | str:
            """Get a single team task by its integer ID or semantic query string."""
            if observer is not None:
                observer.notify_event(
                    ToolCallEvent(tool_name="Get task", args=[task_id], kwargs={})
                )
            return planning_proxy.get_planning_task(task_id)

        get_planning_task.__doc__ = params.format_docstring(get_planning_task.__doc__)
        return get_planning_task

    def _update_planning_factory(self, params: UpdatePlanning) -> Callable:
        planning_proxy = self._planning_proxy
        observer = self._observer

        def update_planning(update: UpdatePlan) -> str:
            """Update team tasks (create, update, delete)."""
            observer.notify_event(
                ToolCallEvent(tool_name="Update planning", args=[update], kwargs={})
            )
            ## Then observer.myAddress is used to set the creator of any new tasks in the plan.
            return planning_proxy.update_planning(update, observer.myAddress)

        update_planning.__doc__ = params.format_docstring(update_planning.__doc__)
        return update_planning
