from __future__ import annotations

import datetime
import logging
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from akgentic.core.actor_address import ActorAddress
from akgentic.core.agent import Akgent
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.core.orchestrator import Orchestrator
from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.tool.errors import RetriableError
from akgentic.tool.vector_store.actor import (
    VS_ACTOR_NAME,
    VS_ACTOR_ROLE,
    VectorStoreActor,
)
from akgentic.tool.vector_store.embedding_actor import build_embedding_service
from akgentic.tool.vector_store.hybrid import DEFAULT_ALPHA, hybrid_search
from akgentic.tool.vector_store.protocol import (
    EmbeddingProvider,
    VectorStoreConfig,
    VectorStoreParam,
    VectorStoreService,
    needs_store_actor,
)
from akgentic.tool.vector_store.registry import BackendContext, get_backend_spec
from akgentic.tool.vector_store.vector import VectorEntry

logger = logging.getLogger(__name__)

TaskStatus = Literal["pending", "started", "completed", "abort"]


def _as_task_id(ref_id: str) -> int | None:
    """Parse a vector-store ``ref_id`` back to a task id, or ``None`` when it is not one.

    A ``planning`` collection outlives the actor that wrote it, and on a shared cluster
    it can outlive the id format too. One foreign entry must not fail every search, so
    it is skipped with a warning — matching how the knowledge graph drops a ``ref_id``
    that no longer resolves.

    Args:
        ref_id: Reference id as the vector store returned it.

    Returns:
        The task id, or ``None`` when *ref_id* is not an integer.
    """
    try:
        return int(ref_id)
    except ValueError:
        logger.warning("Skipping vector store entry with a non-numeric ref_id: %r", ref_id)
        return None


def _score_label(task_id: int, keyword_ids: set[int], vector_scores: Mapping[int, float]) -> str:
    """Describe how *task_id* was found, for the rendered search line."""
    if task_id in vector_scores:
        prefix = "hybrid" if task_id in keyword_ids else "semantic"
        return f"{prefix}: {vector_scores[task_id]:.2f}"
    return "keyword match"


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
    """Configuration for PlanActor with vector-backed semantic search.

    The binding-to-an-actor field is gone: ``vector_store`` now carries the
    storage configuration itself, and the backend it names is what decides
    whether an actor is involved at all.
    """

    vector_store: VectorStoreParam = Field(
        default_factory=VectorStoreParam,
        description=(
            "Vector store configuration for the planning collection: backend, "
            "dimension, tenant, embedding model and provider. The backend decides "
            "how the actor resolves its storage engine — an actor-state backend "
            "through the store actor, a cluster one through the backend's own client."
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
            "Weight of the vector leg in hybrid search; the keyword leg gets "
            "1 - alpha. Matches Weaviate's hybrid() alpha parameter."
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
        self._vs_proxy: VectorStoreService | None = None
        self._embedder: EmbeddingProvider | None = None
        self._acquire_vs_proxy()

    def _acquire_vs_proxy(self) -> None:
        """Resolve this actor's storage engine and create the planning collection.

        **The slot holds a ``VectorStoreService``, not necessarily a proxy.** The
        four methods this actor calls — ``create_collection``, ``add``,
        ``remove``, ``search`` — are exactly that protocol, and the store actor
        and every backend satisfy it with identical signatures, so which one is
        behind the slot changes nothing below this method. (The attribute keeps
        the name ``_vs_proxy``: renaming it to ``_store`` is ~200 mechanical
        private sites and is routed to its own follow-up.)

        Which one it is comes from the param's backend:

        - **An actor-state backend** — the in-memory index, whose data *is* the
          actor's state — is reached through the store actor, looked up by name.
          A missing one is still a ``RuntimeError``: for a planning tool whose
          whole purpose is the vector store, that is a configuration error.
        - **A cluster backend** builds its own engine through the registered
          factory. There is no lookup to fail, so nothing here can raise
          ``RuntimeError``; what can fail is the connect, and that degrades
          exactly as a failed ``create_collection`` does — one WARNING and
          ``_vs_proxy`` left ``None``.

        ``self.orchestrator is None`` (a test harness) still logs a WARNING and
        returns, and only the actor path needs one at all.

        The embedder is built here too, from **this actor's own**
        ``config.vector_store``: the store embeds nothing, so the model this
        actor's ``VectorStoreParam`` names is the model that embeds its tasks.
        """
        param = self.config.vector_store
        store = self._resolve_store(param)
        if store is None:
            return
        try:
            store.create_collection(PLAN_COLLECTION, param)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] create_collection on the vector store failed: %s — degraded mode",
                self.config.name,
                exc,
            )
            return
        self._vs_proxy = store
        self._embedder = build_embedding_service(
            param.embedding_model,
            param.embedding_provider,
        )

    def _resolve_store(self, param: VectorStoreParam) -> VectorStoreService | None:
        """Return the storage engine *param* names, or ``None`` to stay degraded.

        Args:
            param: This actor's vector store configuration.

        Returns:
            The store actor's proxy, a freshly built backend, or ``None`` when
            retrieval must stay off.

        Raises:
            RuntimeError: When the backend needs a store actor and none is
                registered with the team.
        """
        if needs_store_actor(param):
            if self.orchestrator is None:
                logger.warning(
                    "[%s] No orchestrator; operating in degraded mode",
                    self.config.name,
                )
                return None
            orch_proxy = self.proxy_ask(self.orchestrator, Orchestrator)
            vs_addr = orch_proxy.get_team_member(VS_ACTOR_NAME)
            if vs_addr is None:
                raise RuntimeError(
                    f"{self.config.name} requires the vector store actor "
                    f"'{VS_ACTOR_NAME}' but it was not found."
                )
            return self.proxy_ask(vs_addr, VectorStoreActor)
        try:
            return get_backend_spec(param.backend).factory(
                BackendContext(
                    config=VectorStoreConfig(name=self.config.name, role=VS_ACTOR_ROLE),
                    team_id=str(self.team_id),
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] could not build the '%s' vector store backend: %s — degraded mode",
                self.config.name,
                param.backend,
                exc,
            )
            return None

    def _embed_task(self, task: Task) -> None:
        """Embed a task's description and store the resulting VectorEntry.

        Called after task create or update. Does nothing when the proxy or the
        embedder is unavailable (vector_store=False, or the proxy was not
        acquired). Any embedding error is logged and swallowed so that task CRUD
        is never interrupted by a transient embedding failure.

        **Synchronous on purpose.** One entry per call, on a call site that already
        blocks — a worker would add a spawn, a report and an in-flight map to
        preserve behaviour that exists today without any of them. What the move to
        an owned embedder buys instead is a **budget**: the blocking call now
        carries the embedding worker's ``timeout_s``, so this actor's mailbox turn
        is bounded.
        """
        if self._vs_proxy is None or self._embedder is None:
            return
        try:
            vectors = self._embedder.embed([task.description])
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
        if self._vs_proxy is not None:
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
                if description_changed and self._vs_proxy is not None:
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
        mode: Literal["hybrid", "vector", "keyword"] = "hybrid",
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[str]:
        """Search tasks with optional multi-criteria filters (AND logic).

        Args:
            status: Exact match on task.status. None means no filter.
            owner: Exact match on task.owner. Empty string matches unassigned tasks.
                   None means no filter.
            creator: Exact match on task.creator. None means no filter.
            query: Case-insensitive substring match on task.description (keyword phase).
                   When vector deps are available, also runs a semantic phase.
                   Keyword and semantic hits are unioned before other filters apply.
                   Degrades to keyword-only without raising when vector deps absent.
                   None means no filter.
            mode: Search mode — ``"hybrid"`` (default) runs both keyword and
                   semantic phases; ``"keyword"`` skips embedding/vector search;
                   ``"vector"`` skips keyword substring matching. When
                   ``_vs_proxy is None`` and ``mode="vector"``, returns empty
                   results; ``mode="hybrid"`` falls back to keyword-only with
                   a warning.
            top_k: Maximum number of semantic search hits. When None, uses
                   ``config.search_top_k`` (default 10).
            score_threshold: Minimum cosine similarity score for semantic results.
                   When None, uses ``config.search_score_threshold`` (default 0.5).

        Returns:
            Formatted strings for each matching task, ordered by score (highest
            first). Each string includes the task ID, description, status, owner,
            and a score label: ``(semantic: 0.85)`` for vector hits,
            ``(keyword match)`` for keyword-only hits, or ``(hybrid: 0.90)``
            for hits found by both keyword and semantic.
            When all parameters are None, returns the full task list (unscored).
        """
        # Filter first, so the query phase only ever scores viable candidates and
        # the top_k cut is never spent on tasks the AND filters would discard.
        tasks = self._apply_task_filters(list(self.state.task_list), status, owner, creator)

        if query is None:
            return [self._format_task_line(t, {}) for t in tasks]

        scores = self._score_query_matches(tasks, query, mode, top_k, score_threshold)
        matched = [t for t in tasks if t.id in scores]
        matched.sort(key=lambda t: scores[t.id][0], reverse=True)

        effective_top_k = top_k if top_k is not None else self.config.search_top_k
        return [self._format_task_line(t, scores) for t in matched[:effective_top_k]]

    def _score_query_matches(
        self,
        tasks: list[Task],
        query: str,
        mode: Literal["hybrid", "vector", "keyword"],
        top_k: int | None,
        score_threshold: float | None,
    ) -> dict[int, tuple[float, str]]:
        """Return ``{task_id: (score, label)}`` for tasks matching *query*.

        Runs the keyword phase, the semantic phase, or both per *mode*, then
        combines them with the shared fusion rule so that ranking matches every
        other hybrid search in the framework.
        """
        keyword_ids: set[int] = set()
        if mode in ("hybrid", "keyword"):
            q_lower = query.lower()
            keyword_ids = {t.id for t in tasks if q_lower in t.description.lower()}

        result = hybrid_search(
            [str(task_id) for task_id in sorted(keyword_ids)],
            self._vs_proxy,
            self._embedder,
            PLAN_COLLECTION,
            query,
            top_k=top_k if top_k is not None else self.config.search_top_k,
            score_threshold=(
                score_threshold
                if score_threshold is not None
                else self.config.search_score_threshold
            ),
            alpha=self.config.hybrid_alpha,
            semantic=mode != "keyword",
        )

        # The vector store keys by ref_id string; a plan keys by task id. A collection
        # can outlive the id format that wrote it — a shared cluster especially — so a
        # foreign entry is skipped rather than allowed to fail the whole search.
        resolved = [
            (task_id, ref_id, score)
            for ref_id, score in result.ranked
            if (task_id := _as_task_id(ref_id)) is not None
        ]
        vector_scores = {
            task_id: result.vector_scores[ref_id]
            for task_id, ref_id, _ in resolved
            if ref_id in result.vector_scores
        }
        return {
            task_id: (score, _score_label(task_id, keyword_ids, vector_scores))
            for task_id, _, score in resolved
        }

    @staticmethod
    def _apply_task_filters(
        tasks: list[Task],
        status: TaskStatus | None,
        owner: str | None,
        creator: str | None,
    ) -> list[Task]:
        """Return *tasks* narrowed by every non-``None`` criterion (AND logic)."""
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        if owner is not None:
            tasks = [t for t in tasks if t.owner == owner]
        if creator is not None:
            tasks = [t for t in tasks if t.creator == creator]
        return tasks

    @staticmethod
    def _format_task_line(task: Task, scores: dict[int, tuple[float, str]]) -> str:
        """Render one search result, appending its score label when scored."""
        owner_label = task.owner or "unassigned"
        line = (
            f"Task {task.id}: {task.description} [{task.status}] "
            f"(Owner: {owner_label}, Creator: {task.creator})"
        )
        if task.id in scores:
            line += f" ({scores[task.id][1]})"
        return line

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
                if self._vs_proxy is not None:
                    self._vs_proxy.remove(PLAN_COLLECTION, [str(task_id)])
                self.state.task_list = [task for task in self.state.task_list if task.id != task_id]

        self.state.notify_state_change()

        if errors:
            raise RetriableError("Update errors: " + "; ".join(errors))
        return "Done"
