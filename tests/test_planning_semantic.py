"""Tests for PlanActor.get_planning_task — integer ID lookup path.

Covers AC#1 (int found) and AC#2 (int not found).
Also covers AC#8–#9 (PlanningTool embedding field wiring).
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
# AC#4 — _get_planning_task_factory: ToolCallEvent emitted when observer present
# ---------------------------------------------------------------------------


class TestGetPlanningTaskToolCallEvent:
    """AC4 (Task 2.3): _get_planning_task_factory emits ToolCallEvent via observer."""

    def test_notify_event_called_with_task_id(self) -> None:
        """observer.notify_event fires with ToolCallEvent(tool_name='Get task', args=[task_id])."""
        from unittest.mock import MagicMock

        from akgentic.tool.event import ToolCallEvent
        from akgentic.tool.planning.planning import GetPlanningTask, PlanningTool

        tool = PlanningTool()

        # Wire up a mock observer and planning proxy
        mock_observer = MagicMock()
        tool._observer = mock_observer

        mock_proxy = MagicMock()
        mock_proxy.get_planning_task.return_value = "No task with that ID."
        tool._planning_proxy = mock_proxy

        fn = tool._get_planning_task_factory(GetPlanningTask())
        fn(42)

        mock_observer.notify_event.assert_called_once()
        call_args = mock_observer.notify_event.call_args[0][0]
        assert isinstance(call_args, ToolCallEvent)
        assert call_args.tool_name == "Get task"
        assert call_args.args == [42]
        assert call_args.kwargs == {}

    def test_no_notify_when_observer_is_none(self) -> None:
        """When observer is None, notify_event is not called (no AttributeError)."""
        from unittest.mock import MagicMock

        from akgentic.tool.planning.planning import GetPlanningTask, PlanningTool

        tool = PlanningTool()
        tool._observer = None  # type: ignore[assignment]

        mock_proxy = MagicMock()
        mock_proxy.get_planning_task.return_value = "No task with that ID."
        tool._planning_proxy = mock_proxy

        fn = tool._get_planning_task_factory(GetPlanningTask())
        result = fn(99)

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
