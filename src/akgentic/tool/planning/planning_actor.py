import datetime
from typing import Literal

from pydantic import BaseModel, Field

from akgentic.core.actor_address import ActorAddress
from akgentic.core.agent import Akgent, BaseConfig, BaseState
from akgentic.core.utils.serializer import SerializableBaseModel

PlanStatus = Literal["pending", "started", "completed", "abort"]


class PlanItemCreate(SerializableBaseModel):
    id: int = Field(..., description="Unique identifier of the plan item.")
    status: PlanStatus = Field(..., description="Status of the task.")
    description: str = Field(..., max_length=300, description="Short description of the task.")
    owner: str = Field(..., description="Assigned team member name; empty if not yet assigned.")
    dependencies: list[int] = Field(
        default_factory=list,
        description="List of plan item IDs that must be completed before this one.",
    )


class PlanItemUpdate(SerializableBaseModel):
    id: int = Field(..., description="Unique identifier of the plan item.")
    status: PlanStatus | None = Field(default=None, description="New status of the task.")
    description: str | None = Field(
        default=None, max_length=300, description="New description of the task."
    )
    output: str | None = Field(
        default=None, max_length=150, description="New output or result of the task."
    )
    owner: str | None = Field(default=None, description="New assigned team member name;")
    dependencies: list[int] | None = Field(
        default=None,
        description="New list of plan item IDs that must be completed first.",
    )


class PlanItem(PlanItemCreate):
    output: str = Field(default="", description="Output or result of the task.")
    creator: str = Field(default="", description="Team member name who creates the plan item.")
    updated_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="ISO timestamp of the last update.",
    )


class UpdatePlan(BaseModel):
    create_items: list[PlanItemCreate] = Field(
        default_factory=list, description="Items to add to the plan."
    )
    update_items: list[PlanItemUpdate] = Field(
        default_factory=list, description="Items to update in the plan."
    )
    delete_items: list[int] = Field(
        default_factory=list, description="Items to remove from the plan."
    )


class PlanManagerState(BaseState):
    item_list: list[PlanItem] = Field(default_factory=list)


class PlanActor(Akgent[BaseConfig, PlanManagerState]):
    """Actor responsible for managing the execution of a plan.

    The PlanManager oversees the execution of a plan, coordinating
    between different agents and tools as needed. It maintains the
    state of the plan execution and handles any necessary communication
    with other actors.

    Attributes:
        config: Configuration for the PlanManager.
        state: Current state of the plan execution.
    """

    def init(self):
        self.state = PlanManagerState()
        self.state.observer(self)

    def _create_item(self, item_create: PlanItemCreate, actor_address: ActorAddress) -> None:
        new_item = PlanItem(**item_create.__dict__, creator=actor_address.name)
        self.state.item_list.append(new_item)

    def _update_item(self, item_update: PlanItemUpdate) -> None:
        # Use __dict__ to get raw values, filter out None to only apply explicitly set fields
        updates = {k: v for k, v in item_update.__dict__.items() if v is not None}
        for idx, item in enumerate(self.state.item_list):
            if item.id == item_update.id:
                self.state.item_list[idx] = item.model_copy(update=updates)
                return
        # FIXME - handle case where item to update doesn't exist

    ##
    ## Tools to expose to agents:
    ##
    def get_planning(self) -> list[PlanItem]:
        """Get the current plan items."""
        return self.state.item_list

    def get_planning_item(self, item_id: int) -> PlanItem | str:
        """Get a specific plan item by ID."""
        item_list = self.state.item_list
        return next((item for item in item_list if item.id == item_id), "No item with that ID.")

    def update_planning(self, update: UpdatePlan, actor_address: ActorAddress) -> str:
        """Update the plan with new, updated, or deleted items."""

        # Handle item creation
        for item_create in update.create_items:
            self._create_item(item_create, actor_address)

        # Handle item updates
        for item_update in update.update_items:
            self._update_item(item_update)

        # Handle item deletions
        item_list = self.state.item_list
        for item_id in update.delete_items:
            self.state.item_list = [item for item in item_list if item.id != item_id]

        self.state.notify_state_change()
        return "Done"
