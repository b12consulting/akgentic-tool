from __future__ import annotations

import datetime
import logging
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from akgentic.core.actor_address import ActorAddress
from akgentic.core.agent import Akgent, BaseConfig, BaseState
from akgentic.core.orchestrator import Orchestrator
from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.tool.errors import RetriableError
from akgentic.tool.vector import VectorEntry
from akgentic.tool.vector_store.actor import VS_ACTOR_NAME, VectorStoreActor
from akgentic.tool.vector_store.protocol import CollectionConfig

logger = logging.getLogger(__name__)

TaskStatus = Literal["pending", "started", "completed", "abort"]


class TaskCreate(SerializableBaseModel):
    id: int = Field(..., description="Unique identifier of the task.")
    status: TaskStatus = Field(..., description="Status of the task.")
    description: str = Field(
        ..., max_length=300, description="Short description of the task (max 300 chars)."
    )
    owner: str = Field(..., description="Assigned team member name; empty if not yet assigned.")
    dependencies: list[int] = Field(
        default_factory=list,
        description="List of task IDs that must be completed before this one.",
    )


class TaskUpdate(SerializableBaseModel):
    id: int = Field(..., description="Unique identifier of the task.")
    status: TaskStatus | None = Field(default=None, description="New status of the task.")
    description: str | None = Field(
        default=None, max_length=300, description="New description of the task (max 300 chars)."
    )
    output: str | None = Field(
        default=None,
        max_length=150,
        description="New output or result of the task (max 150 chars; truncated automatically).",
    )
    owner: str | None = Field(default=None, description="New assigned team member name;")
    dependencies: list[int] | None = Field(
        default=None,
        description="New list of task IDs that must be completed first.",
    )

    @field_validator("output", mode="before")
    @classmethod
    def truncate_output(cls, v: object) -> object:
        """Silently truncate output to 150 chars to avoid validation errors."""
        if isinstance(v, str) and len(v) > 150:
            return v[:147] + "..."
        return v


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


PLAN_COLLECTION: str = "planning"
"""Collection name used in VectorStoreActor."""


class PlanConfig(BaseConfig):
    """Configuration for PlanActor with optional semantic search support."""

    semantic_search: bool = Field(
        default=True,
        description=(
            "When True, task descriptions are embedded on create/update/delete "
            "for semantic search. Falls back gracefully if VectorStoreActor is unavailable."
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
        self._vs_proxy: VectorStoreActor | None = None
        if self.config.semantic_search:
            self._acquire_vs_proxy()

    def _acquire_vs_proxy(self) -> None:
        """Acquire the VectorStoreActor proxy and create the planning collection.

        Operates in degraded mode (no vector search) if the proxy cannot
        be acquired.
        """
        try:
            if self.orchestrator is None:
                logger.warning(
                    "[%s] No orchestrator; operating in degraded mode",
                    self.config.name,
                )
                return
            orch_proxy = self.proxy_ask(self.orchestrator, Orchestrator)
            vs_addr = orch_proxy.get_team_member(VS_ACTOR_NAME)
            if vs_addr is None:
                logger.warning(
                    "[%s] VectorStoreActor not found; operating in degraded mode",
                    self.config.name,
                )
                return
            self._vs_proxy = self.proxy_ask(vs_addr, VectorStoreActor)
            self._vs_proxy.create_collection(PLAN_COLLECTION, CollectionConfig())
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] Failed to acquire VectorStoreActor proxy: %s",
                self.config.name,
                exc,
            )

    def _embed_task(self, task: Task) -> None:
        """Embed a task's description and store the resulting VectorEntry.

        Called after task create or update. Does nothing when VectorStoreActor
        proxy is unavailable (semantic_search=False or proxy not acquired).
        Any embedding error is logged and swallowed so that task CRUD
        is never interrupted by a transient embedding failure.
        """
        if self._vs_proxy is None:
            return
        try:
            vectors = self._vs_proxy.embed([task.description])
            if not vectors:
                return
            entry = VectorEntry(
                ref_type="task",
                ref_id=str(task.id),
                text=task.description,
                vector=vectors[0],
            )
            self._vs_proxy.add(PLAN_COLLECTION, [entry])
        except Exception:  # noqa: BLE001
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
                if (
                    self.config.semantic_search
                    and description_changed
                    and self._vs_proxy is not None
                ):
                    self._vs_proxy.remove(PLAN_COLLECTION, [str(task.id)])
                    self._embed_task(updated_task)
                return None
        return f"Update error - no task with ID {task_update.id} found."

    ##
    ## Tools to expose to agents:
    ##
    def get_planning(self) -> list[Task]:
        """Get the current plan tasks."""
        return self.state.task_list

    def get_planning_task(self, task_id: int) -> Task | str:
        """Look up a task by exact integer ID.

        Args:
            task_id: The integer ID of the task to retrieve.

        Returns:
            The matching ``Task`` if found, or ``"No task with that ID."`` if no
            task with that ID exists.
        """
        return next(
            (t for t in self.state.task_list if t.id == task_id), "No task with that ID."
        )

    def search_planning(
        self,
        status: TaskStatus | None = None,
        owner: str | None = None,
        creator: str | None = None,
        query: str | None = None,
    ) -> list[Task]:
        """Search tasks with optional multi-criteria filters (AND logic).

        Args:
            status: Exact match on task.status. None means no filter.
            owner: Exact match on task.owner. Empty string matches unassigned tasks.
                   None means no filter.
            creator: Exact match on task.creator. None means no filter.
            query: Case-insensitive substring match on task.description (keyword phase).
                   When [vector_search] deps are available and the index is non-empty,
                   also runs a semantic phase (cosine ≥ 0.5, top_k=20). Keyword and
                   semantic hits are unioned before other filters apply.
                   Degrades to keyword-only without raising when vector deps absent.
                   None means no filter.

        Returns:
            Tasks matching ALL provided filters. When all parameters are None,
            returns the full task list.
        """
        tasks: list[Task] = list(self.state.task_list)

        # Query phase: build candidate set (keyword UNION semantic), then intersect with rest
        if query is not None:
            q_lower = query.lower()
            keyword_ids = {t.id for t in tasks if q_lower in t.description.lower()}

            semantic_ids: set[int] = set()
            if self._vs_proxy is not None:
                try:
                    vectors = self._vs_proxy.embed([query])
                    if vectors:
                        query_vector = vectors[0]
                        result = self._vs_proxy.search(PLAN_COLLECTION, query_vector, 20)
                        for hit in result.hits:
                            if hit.score >= 0.5:
                                semantic_ids.add(int(hit.ref_id))
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "Semantic search failed for query — falling back to keyword only",
                        exc_info=True,
                    )

            candidate_ids = keyword_ids | semantic_ids
            tasks = [t for t in tasks if t.id in candidate_ids]

        # Apply remaining AND filters
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        if owner is not None:
            tasks = [t for t in tasks if t.owner == owner]
        if creator is not None:
            tasks = [t for t in tasks if t.creator == creator]

        return tasks

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
                if self.config.semantic_search and self._vs_proxy is not None:
                    self._vs_proxy.remove(PLAN_COLLECTION, [str(task_id)])
                self.state.task_list = [task for task in self.state.task_list if task.id != task_id]

        self.state.notify_state_change()

        if errors:
            raise RetriableError("Update errors: " + "; ".join(errors))
        return "Done"
