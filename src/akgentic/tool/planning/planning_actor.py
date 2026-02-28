import datetime
from typing import Literal

from pydantic import BaseModel, Field

from akgentic.core.actor_address import ActorAddress
from akgentic.core.agent import Akgent, BaseConfig, BaseState
from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.tool.errors import RetriableError

TaskStatus = Literal["pending", "started", "completed", "abort"]


class TaskCreate(SerializableBaseModel):
    id: int = Field(..., description="Unique identifier of the task.")
    status: TaskStatus = Field(..., description="Status of the task.")
    description: str = Field(..., max_length=300, description="Short description of the task.")
    owner: str = Field(..., description="Assigned team member name; empty if not yet assigned.")
    dependencies: list[int] = Field(
        default_factory=list,
        description="List of task IDs that must be completed before this one.",
    )


class TaskUpdate(SerializableBaseModel):
    id: int = Field(..., description="Unique identifier of the task.")
    status: TaskStatus | None = Field(default=None, description="New status of the task.")
    description: str | None = Field(
        default=None, max_length=300, description="New description of the task."
    )
    output: str | None = Field(
        default=None, max_length=150, description="New output or result of the task."
    )
    owner: str | None = Field(default=None, description="New assigned team member name;")
    dependencies: list[int] | None = Field(
        default=None,
        description="New list of task IDs that must be completed first.",
    )


class Task(TaskCreate):
    output: str = Field(default="", description="Output or result of the task.")
    creator: str = Field(default="", description="Team member name who creates the task.")
    updated_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC),
        description="ISO timestamp of the last update.",
    )


class UpdatePlan(BaseModel):
    create_tasks: list[TaskCreate] = Field(
        default_factory=list, description="Tasks to add to the plan."
    )
    update_tasks: list[TaskUpdate] = Field(
        default_factory=list, description="Tasks to update in the plan."
    )
    delete_tasks: list[int] = Field(
        default_factory=list, description="Tasks to remove from the plan."
    )


class PlanManagerState(BaseState):
    task_list: list[Task] = Field(default_factory=list)


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

    def on_start(self):
        self.state = PlanManagerState()
        self.state.observer(self)

    def _create_task(self, task: TaskCreate, actor_address: ActorAddress) -> None:
        new_task = Task(**task.__dict__, creator=actor_address.name)
        self.state.task_list.append(new_task)

    def _update_task(self, task_update: TaskUpdate) -> None | str:
        # Use __dict__ to get raw values, filter out None to only apply explicitly set fields
        updates = {k: v for k, v in task_update.__dict__.items() if v is not None}
        for idx, task in enumerate(self.state.task_list):
            if task.id == task_update.id:
                self.state.task_list[idx] = task.model_copy(update=updates)
                return
        return f"Update error - no task with ID {task_update.id} found."

    ##
    ## Tools to expose to agents:
    ##
    def get_planning(self) -> list[Task]:
        """Get the current plan tasks."""
        return self.state.task_list

    def get_planning_task(self, task_id: int) -> Task | str:
        """Get a specific plan task by ID."""
        task_list = self.state.task_list
        return next((task for task in task_list if task.id == task_id), "No task with that ID.")

    def update_planning(self, update: UpdatePlan, actor_address: ActorAddress) -> str:
        """Update the plan with new, updated, or deleted task."""

        errors = []

        # Handle task creation
        for task_create in update.create_tasks:
            self._create_task(task_create, actor_address)

        # Handle task updates
        for task_update in update.update_tasks:
            error = self._update_task(task_update)
            if error is not None:
                errors.append(error)

        # Handle task deletions
        for task_id in update.delete_tasks:
            if not any(task.id == task_id for task in self.state.task_list):
                errors.append(f"Delete error - no task with ID {task_id} found.")
            else:
                self.state.task_list = [task for task in self.state.task_list if task.id != task_id]

        self.state.notify_state_change()

        if errors:
            raise RetriableError("Update errors: " + "; ".join(errors))
        return "Done"
