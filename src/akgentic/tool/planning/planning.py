import logging
from typing import Callable

from pydantic import Field

from akgentic.core.agent_config import BaseConfig
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
from akgentic.tool.planning.planning_actor import PlanActor, Task, UpdatePlan

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PLANNING_ACTOR_NAME = "#PlanningTool"
PLANNING_ACTOR_ROLE = "ToolActor"


class GetPlanning(BaseToolParam):
    """Get the full team plan — as system prompt and/or tool."""

    expose: set[Channels] = {SYSTEM_PROMPT, COMMAND}


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

    def observer(self, observer: ActorToolObserver):
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
            planning_tool_addr = orchestrator_proxy_ask.createActor(
                PlanActor, config=BaseConfig(name=PLANNING_ACTOR_NAME, role=PLANNING_ACTOR_ROLE)
            )

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

        def planning_prompt() -> str:
            """Get the full team planning."""

            planning = planning_proxy.get_planning()
            if not planning:
                return "No current Team planning."
            return "Team planning:\n" + "\n".join(
                [
                    f"- ID {task.id} [{task.status}] {task.description} "
                    f"{task.output and f'\u2014 Output: {task.output} '}"
                    f"(Owner: {task.owner}, Creator: {task.creator})"
                    for task in planning
                ]
            )

        return planning_prompt

    def _get_planning_task_factory(self, params: GetPlanningTask) -> Callable:
        planning_proxy = self._planning_proxy
        observer = self._observer

        def get_planning_task(task_id: int) -> Task | str:
            """Get a single team task by its ID."""
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
            """Update team tasks (create, update, delete).

            Keep the plan up to date: update task status when you make progress, record outputs when you complete work, and create sub-tasks when you delegate."""
            observer.notify_event(
                ToolCallEvent(tool_name="Update planning", args=[update], kwargs={})
            )
            ## Then observer.myAddress is used to set the creator of any new tasks in the plan.
            return planning_proxy.update_planning(update, observer.myAddress)

        update_planning.__doc__ = params.format_docstring(update_planning.__doc__)
        return update_planning
