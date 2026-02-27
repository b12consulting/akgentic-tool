import logging
from typing import Callable

from pydantic import Field

from akgentic.core.agent_config import BaseConfig
from akgentic.core.orchestrator import Orchestrator
from akgentic.tool.core import BaseToolParam, ToolCard, _resolve
from akgentic.tool.event import ActorToolObserver, ToolCallEvent
from akgentic.tool.planning.planning_actor import PlanActor, PlanItem, UpdatePlan

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PLANNING_ACTOR_NAME = "#PlanningTool"
PLANNING_ACTOR_ROLE = "ToolActor"


class GetPlanning(BaseToolParam):
    """Get the full team plan — as system prompt and/or tool."""

    system_prompt: bool = True
    llm_tool: bool = False


class GetPlanningItem(BaseToolParam):
    """Get a single planning item by ID."""

    pass


class UpdatePlanning(BaseToolParam):
    """Update planning items."""

    pass


class PlanningTool(ToolCard):
    """Team planning management via actor-based plan store."""

    name: str = "Planning"
    description: str = "Planning tool to manage team plans and tasks"

    get_planning: GetPlanning | bool = Field(
        default=True, description="By default the plan in included in the system prompt"
    )
    get_planning_item: GetPlanningItem | bool = True
    update_planning: UpdatePlanning | bool = True


    def observer(self, observer: ActorToolObserver) -> "PlanningTool":
        """Attach observer and set up the planning actor proxy.

        Requires an ActorToolObserver for actor system access.
        """
        super().observer(observer)
        if observer is None:
            raise ValueError("PlanningTool requires an ActorToolObserver to function.")
        if observer.orchestrator is None:
            raise ValueError("PlanningTool requires access to the orchestrator.")

        orchestrator_proxy_ask = observer.proxy_ask(observer.orchestrator, Orchestrator)
        planning_tool_addr = orchestrator_proxy_ask.get_team_member(PLANNING_ACTOR_NAME)

        if planning_tool_addr is None:
            logger.info(
                f"PlanningTool actor not found in team, creating new one at {PLANNING_ACTOR_NAME}."
            )
            planning_tool_addr = orchestrator_proxy_ask.createActor(
                PlanActor, config=BaseConfig(name=PLANNING_ACTOR_NAME, role=PLANNING_ACTOR_ROLE)
            )

        self._planning_proxy = observer.proxy_ask(planning_tool_addr, PlanActor)

    def get_system_prompts(self) -> list[Callable]:
        gp = _resolve(self.get_planning, GetPlanning)
        if gp and gp.system_prompt:
            planning_proxy = self._planning_proxy

            def team_planning() -> str:
                planning = planning_proxy.get_planning()
                if not planning:
                    return "No current Team planning."
                return "Team planning:\n" + "\n".join(
                    [
                        f"- ID {item.id} [{item.status}] {item.description} "
                        f"{item.output and f'— Output: {item.output} '}"
                        f"(Owner: {item.owner}, Creator: {item.creator})"
                        for item in planning
                    ]
                )

            if gp.description:
                team_planning.__doc__ = gp.description
            return [team_planning]
        return []

    def get_tools(self) -> list[Callable]:
        tools: list[Callable] = []

        gp = _resolve(self.get_planning, GetPlanning)
        if gp and gp.llm_tool:
            tools.append(self._get_planning_factory(gp))

        gpi = _resolve(self.get_planning_item, GetPlanningItem)
        if gpi and gpi.llm_tool:
            tools.append(self._get_planning_item_factory(gpi))

        up = _resolve(self.update_planning, UpdatePlanning)
        if up and up.llm_tool:
            tools.append(self._update_planning_factory(up))

        return tools

    def _get_planning_factory(self, params: GetPlanning) -> Callable:
        planning_proxy = self._planning_proxy
        observer = self._observer

        def get_planning() -> list[PlanItem]:
            """Get the full team planning."""
            if observer is not None:
                observer.notify_event(ToolCallEvent(tool_name="Get planning", args=[], kwargs={}))
            return planning_proxy.get_planning()

        if params.description:
            get_planning.__doc__ = params.description
        return get_planning

    def _get_planning_item_factory(self, params: GetPlanningItem) -> Callable:
        planning_proxy = self._planning_proxy
        observer = self._observer

        def get_planning_item(item_id: int) -> PlanItem | str:
            """Get a single team planning item by its ID."""
            if observer is not None:
                observer.notify_event(
                    ToolCallEvent(tool_name="Get planning item", args=[item_id], kwargs={})
                )
            return planning_proxy.get_planning_item(item_id)

        if params.description:
            get_planning_item.__doc__ = params.description
        return get_planning_item

    def _update_planning_factory(self, params: UpdatePlanning) -> Callable:
        planning_proxy = self._planning_proxy
        observer = self._observe

        def update_planning(update: UpdatePlan) -> str:
            """Update team planning items (create, update, delete).

            When you start or complete a task from the planning,
            do not forget to update the plan with the new status
            and output of the task.
            """
            if observer is not None:
                observer.notify_event(
                    ToolCallEvent(tool_name="Update planning", args=[update], kwargs={})
                )
            ## Then observer.myAddress is used to set the creator of any new items in the plan.
            return planning_proxy.update_planning(update, observer.myAddress)

        if params.description:
            update_planning.__doc__ = params.description
        return update_planning
