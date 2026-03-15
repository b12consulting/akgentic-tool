import datetime
import logging
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from akgentic.core.actor_address import ActorAddress
from akgentic.core.agent import Akgent, BaseConfig, BaseState
from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.tool.errors import RetriableError

if TYPE_CHECKING:
    from akgentic.tool.vector import EmbeddingService, VectorIndex

logger = logging.getLogger(__name__)

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


class PlanConfig(BaseConfig):
    """Configuration for PlanActor with optional semantic search support."""

    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_provider: Literal["openai", "azure"] = Field(default="openai")
    semantic_search: bool = Field(
        default=True,
        description=(
            "When True, task descriptions are embedded on create/update/delete "
            "for semantic search. Falls back gracefully if [vector_search] deps are not installed."
        ),
    )


class PlanActor(Akgent[PlanConfig, PlanManagerState]):
    """Actor responsible for managing the execution of a plan.

    The PlanManager oversees the execution of a plan, coordinating
    between different agents and tools as needed. It maintains the
    state of the plan execution and handles any necessary communication
    with other actors.

    Attributes:
        config: Configuration for the PlanManager.
        state: Current state of the plan execution.
    """

    def on_start(self) -> None:
        self.state = PlanManagerState()
        self.state.observer(self)
        # Coerce BaseConfig → PlanConfig if the actor was started with a plain BaseConfig
        # (e.g., from existing PlanningTool wiring prior to Story 2.2 update).
        # This preserves backward compatibility while giving PlanActor typed config access.
        if not isinstance(self.config, PlanConfig):
            self.config = PlanConfig(
                name=self.config.name,
                role=self.config.role,
            )
        # Only instantiate VectorIndex (which imports numpy) when semantic_search is enabled.
        # This preserves the graceful fallback behaviour: if numpy is absent and
        # semantic_search=False, on_start must not crash (AC#7, AC#8).
        self._vector_index: "VectorIndex | None" = None
        if self.config.semantic_search:
            from akgentic.tool.vector import VectorIndex

            self._vector_index = VectorIndex()
        self._embedding_svc: EmbeddingService | None = None

    def _get_or_create_embedding_svc(self) -> "EmbeddingService | None":
        """Return (or lazily create) the EmbeddingService.

        Returns None when semantic_search is disabled or [vector_search]
        optional dependencies are not installed.
        """
        if not self.config.semantic_search:
            return None
        if self._embedding_svc is None:
            try:
                from akgentic.tool.vector import EmbeddingService, _check_vector_search_dependencies

                _check_vector_search_dependencies()
            except ImportError:
                return None
            self._embedding_svc = EmbeddingService(
                model=self.config.embedding_model,
                provider=self.config.embedding_provider,
            )
        return self._embedding_svc

    def _embed_task(self, task: Task) -> None:
        """Embed a task's description and store the resulting VectorEntry.

        Called after task create or update. Does nothing when embedding service
        is unavailable (semantic_search=False or missing deps). Any embedding
        error (import, network, auth) is logged and swallowed so that task CRUD
        is never interrupted by a transient embedding failure.
        """
        try:
            svc = self._get_or_create_embedding_svc()
            if svc is None or self._vector_index is None:
                return
            from akgentic.tool.vector import VectorEntry

            vector = svc.embed([task.description])[0]
            entry = VectorEntry(
                ref_type="task",
                ref_id=str(task.id),
                text=task.description,
                vector=vector,
            )
            self._vector_index.add(entry)
        except ImportError:
            logger.warning("Vector search deps missing — skipping embedding for task %s", task.id)
        except Exception:
            logger.warning(
                "Embedding failed for task %s — semantic index not updated", task.id, exc_info=True
            )

    def _create_task(self, task: TaskCreate, actor_address: ActorAddress) -> None:
        new_task = Task(**task.__dict__, creator=actor_address.name)
        self.state.task_list.append(new_task)
        if self.config.semantic_search:
            self._embed_task(new_task)

    def _update_task(self, task_update: TaskUpdate) -> None | str:
        # Use __dict__ to get raw values, filter out None to only apply explicitly set fields
        updates = {k: v for k, v in task_update.__dict__.items() if v is not None}
        for idx, task in enumerate(self.state.task_list):
            if task.id == task_update.id:
                updated_task = task.model_copy(update=updates)
                self.state.task_list[idx] = updated_task
                # Re-index only when the description actually changed (AC#5).
                description_changed = (
                    task_update.description is not None
                    and task_update.description != task.description
                )
                should_reindex = (
                    self.config.semantic_search
                    and description_changed
                    and self._vector_index is not None
                )
                if should_reindex:
                    self._vector_index.remove({str(task.id)})
                    self._embed_task(updated_task)
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
                if self.config.semantic_search and self._vector_index is not None:
                    self._vector_index.remove({str(task_id)})
                self.state.task_list = [task for task in self.state.task_list if task.id != task_id]

        self.state.notify_state_change()

        if errors:
            raise RetriableError("Update errors: " + "; ".join(errors))
        return "Done"
