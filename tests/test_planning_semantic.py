"""Tests for PlanActor.get_planning_task — integer ID lookup path.

Covers AC#1 (int found) and AC#2 (int not found).
Also covers AC#8–#9 (PlanningTool embedding field wiring).
"""

from __future__ import annotations

from unittest.mock import MagicMock

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

        result = actor.get_planning_task(3)

        assert isinstance(result, Task)
        assert result.id == 3
        assert result.description == "Auth module"


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
    """AC9/AC10: observer() creates VectorStoreActor and PlanActor with correct configs."""

    def test_observer_creates_vectorstore_and_plan_actors(self) -> None:
        """observer() must create VectorStoreActor with embedding fields, then PlanActor."""
        from akgentic.tool.planning.planning import PlanningTool
        from akgentic.tool.vector_store.actor import VectorStoreActor
        from akgentic.tool.vector_store.protocol import VectorStoreConfig

        tool = PlanningTool(
            embedding_model="text-embedding-ada-002",
            embedding_provider="azure",
        )

        # Capture configs passed to createActor calls
        captured_configs: list[object] = []

        mock_proxy_ask = MagicMock()
        mock_proxy_ask.get_team_member.return_value = None  # Force actor creation path

        def capture_create_actor(actor_cls: type, config: object) -> MagicMock:
            captured_configs.append(config)
            return MagicMock()

        mock_proxy_ask.createActor.side_effect = capture_create_actor

        mock_observer = MagicMock()
        mock_observer.orchestrator = MagicMock()
        mock_observer.proxy_ask.return_value = mock_proxy_ask

        tool.observer(mock_observer)

        # Two actors created: VectorStoreActor first, PlanActor second
        assert len(captured_configs) == 2

        # First: VectorStoreConfig with embedding fields
        vs_config = captured_configs[0]
        assert isinstance(vs_config, VectorStoreConfig)
        assert vs_config.embedding_model == "text-embedding-ada-002"
        assert vs_config.embedding_provider == "azure"

        # Second: PlanConfig without embedding fields
        plan_config = captured_configs[1]
        assert isinstance(plan_config, PlanConfig)
        assert "embedding_model" not in PlanConfig.model_fields
