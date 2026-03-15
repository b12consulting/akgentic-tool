"""Tests for PlanActor.get_planning_task — int and semantic (str) lookup paths.

Covers AC#1–#7 (get_planning_task method) and AC#8–#9 (PlanningTool wiring),
satisfying AC#10 (test file completeness).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from akgentic.tool.planning.planning_actor import (
    PlanActor,
    PlanConfig,
    Task,
)
from tests.conftest import MockActorAddress


def _make_actor(semantic_search: bool = True) -> PlanActor:
    """Construct a bare PlanActor with no Pykka runtime — calls on_start directly."""
    actor = PlanActor()
    actor.config = PlanConfig(
        name="test-plan",
        role="ToolActor",
        semantic_search=semantic_search,
    )
    actor.on_start()
    return actor


def _add_task(actor: PlanActor, task_id: int, description: str) -> Task:
    """Helper: add a task to actor state without triggering embedding.

    Directly appends to task_list so tests can control embedding lifecycle
    independently.
    """
    addr = MockActorAddress()
    actor.state.task_list.append(
        Task(
            id=task_id,
            status="pending",
            description=description,
            owner="Alice",
            creator=addr.name,
        )
    )
    return actor.state.task_list[-1]


# ---------------------------------------------------------------------------
# AC#1 — int lookup: task found (no embedding)
# ---------------------------------------------------------------------------


class TestGetPlanningTaskIntFound:
    """AC1: int task_id → exact match returned, no embedding call."""

    def test_returns_task_when_id_matches(self) -> None:
        actor = _make_actor()
        _add_task(actor, task_id=3, description="Auth module")

        with patch.object(actor, "_get_or_create_embedding_svc") as mock_svc:
            result = actor.get_planning_task(3)

        assert isinstance(result, Task)
        assert result.id == 3
        assert result.description == "Auth module"
        mock_svc.assert_not_called()


# ---------------------------------------------------------------------------
# AC#2 — int lookup: task not found
# ---------------------------------------------------------------------------


class TestGetPlanningTaskIntNotFound:
    """AC2: int task_id with no matching task → 'No task with that ID.' string."""

    def test_returns_not_found_string_when_id_missing(self) -> None:
        actor = _make_actor()
        _add_task(actor, task_id=1, description="Some task")

        result = actor.get_planning_task(99)

        assert result == "No task with that ID."


# ---------------------------------------------------------------------------
# AC#3 — str lookup: semantic search found
# ---------------------------------------------------------------------------


class TestGetPlanningTaskStrFound:
    """AC3: str task_id with populated index → EmbeddingService called, Task returned."""

    def test_semantic_search_returns_matching_task(self) -> None:
        actor = _make_actor(semantic_search=True)
        _add_task(actor, task_id=5, description="Auth module implementation")

        # Seed the vector index with a pre-computed entry for task id=5
        task = actor.state.task_list[0]
        unit_vector = [1.0, 0.0, 0.0]

        from akgentic.tool.vector import VectorEntry

        entry = VectorEntry(
            ref_type="task",
            ref_id=str(task.id),
            text=task.description,
            vector=unit_vector,
        )
        assert actor._vector_index is not None
        actor._vector_index.add(entry)

        # Mock embed to return the same unit vector → cosine similarity of 1.0
        mock_svc = MagicMock()
        mock_svc.embed.return_value = [[1.0, 0.0, 0.0]]

        with patch.object(actor, "_get_or_create_embedding_svc", return_value=mock_svc):
            result = actor.get_planning_task("auth module")

        assert isinstance(result, Task)
        assert result.id == 5
        mock_svc.embed.assert_called_once_with(["auth module"])


# ---------------------------------------------------------------------------
# AC#4 — str lookup: empty vector index
# ---------------------------------------------------------------------------


class TestGetPlanningTaskStrEmptyIndex:
    """AC4: str task_id with empty vector index → 'Semantic search unavailable.'"""

    def test_returns_unavailable_when_index_empty(self) -> None:
        actor = _make_actor(semantic_search=True)
        _add_task(actor, task_id=1, description="Some task")
        # Index is empty — no entries added

        mock_svc = MagicMock()

        with patch.object(actor, "_get_or_create_embedding_svc", return_value=mock_svc):
            result = actor.get_planning_task("something")

        assert result == "Semantic search unavailable."
        mock_svc.embed.assert_not_called()


# ---------------------------------------------------------------------------
# AC#5 — str lookup: semantic_search=False
# ---------------------------------------------------------------------------


class TestGetPlanningTaskStrSemanticDisabled:
    """AC5: str task_id with semantic_search=False → 'Semantic search unavailable.'"""

    def test_returns_unavailable_when_semantic_search_disabled(self) -> None:
        actor = _make_actor(semantic_search=False)
        _add_task(actor, task_id=1, description="Some task")

        with patch.object(actor, "_get_or_create_embedding_svc", return_value=None) as mock_fn:
            result = actor.get_planning_task("some query")

        assert result == "Semantic search unavailable."
        # _get_or_create_embedding_svc is called but returns None — no embed call
        mock_fn.assert_called_once()


# ---------------------------------------------------------------------------
# AC#6 — str lookup: missing [vector_search] deps
# ---------------------------------------------------------------------------


class TestGetPlanningTaskStrMissingDeps:
    """AC6: str task_id with missing vector deps → 'Semantic search unavailable.'"""

    def test_returns_unavailable_when_deps_missing(self) -> None:
        actor = _make_actor(semantic_search=True)
        _add_task(actor, task_id=1, description="Some task")

        # Simulate missing deps: _get_or_create_embedding_svc returns None
        with patch.object(actor, "_get_or_create_embedding_svc", return_value=None):
            result = actor.get_planning_task("auth module")

        assert result == "Semantic search unavailable."


# ---------------------------------------------------------------------------
# AC#7 — str lookup: orphaned ref_id
# ---------------------------------------------------------------------------


class TestGetPlanningTaskStrOrphanedRefId:
    """AC7: top-1 ref_id doesn't match any task (deleted) → 'No task with that ID.'"""

    def test_returns_no_task_when_ref_id_orphaned(self) -> None:
        actor = _make_actor(semantic_search=True)

        # Seed index with ref_id=99 (task deleted from task_list)
        unit_vector = [1.0, 0.0, 0.0]

        from akgentic.tool.vector import VectorEntry

        entry = VectorEntry(
            ref_type="task",
            ref_id="99",  # orphaned — task 99 not in task_list
            text="deleted task description",
            vector=unit_vector,
        )
        assert actor._vector_index is not None
        actor._vector_index.add(entry)

        # task_list has task id=1, NOT id=99
        _add_task(actor, task_id=1, description="Surviving task")

        mock_svc = MagicMock()
        mock_svc.embed.return_value = [[1.0, 0.0, 0.0]]

        with patch.object(actor, "_get_or_create_embedding_svc", return_value=mock_svc):
            result = actor.get_planning_task("deleted task description")

        assert result == "No task with that ID."


# ---------------------------------------------------------------------------
# AC#8 — PlanningTool has embedding_model and embedding_provider fields
# ---------------------------------------------------------------------------


class TestPlanningToolFields:
    """AC8: PlanningTool exposes embedding_model and embedding_provider Pydantic fields."""

    def test_default_embedding_model(self) -> None:
        from akgentic.tool.planning.planning import PlanningTool

        tool = PlanningTool()
        assert tool.embedding_model == "text-embedding-3-small"

    def test_default_embedding_provider(self) -> None:
        from akgentic.tool.planning.planning import PlanningTool

        tool = PlanningTool()
        assert tool.embedding_provider == "openai"

    def test_custom_embedding_model(self) -> None:
        from akgentic.tool.planning.planning import PlanningTool

        tool = PlanningTool(embedding_model="text-embedding-ada-002")
        assert tool.embedding_model == "text-embedding-ada-002"

    def test_custom_embedding_provider_azure(self) -> None:
        from akgentic.tool.planning.planning import PlanningTool

        tool = PlanningTool(embedding_provider="azure")
        assert tool.embedding_provider == "azure"


# ---------------------------------------------------------------------------
# AC#9 — PlanningTool.observer() passes PlanConfig with embedding fields
# ---------------------------------------------------------------------------


class TestPlanningToolObserverWiring:
    """AC9: observer() wires PlanConfig(embedding_model, embedding_provider) to PlanActor."""

    def test_observer_creates_plan_actor_with_plan_config(self) -> None:
        """observer() must pass PlanConfig (not BaseConfig) with embedding fields to PlanActor."""
        from akgentic.tool.planning.planning import PlanningTool

        tool = PlanningTool(
            embedding_model="text-embedding-ada-002",
            embedding_provider="azure",
        )

        # Capture the config passed to createActor
        captured_config: list[PlanConfig] = []

        mock_proxy_ask = MagicMock()
        mock_proxy_ask.get_team_member.return_value = None  # Force actor creation path

        def capture_create_actor(actor_cls: type, config: PlanConfig) -> MagicMock:
            captured_config.append(config)
            return MagicMock()

        mock_proxy_ask.createActor.side_effect = capture_create_actor

        mock_observer = MagicMock()
        mock_observer.orchestrator = MagicMock()
        mock_observer.proxy_ask.return_value = mock_proxy_ask

        tool.observer(mock_observer)

        assert len(captured_config) == 1
        config = captured_config[0]
        assert isinstance(config, PlanConfig), (
            f"Expected PlanConfig, got {type(config).__name__}. "
            "observer() must not pass a plain BaseConfig."
        )
        assert config.embedding_model == "text-embedding-ada-002"
        assert config.embedding_provider == "azure"
