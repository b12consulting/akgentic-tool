"""Tests for KnowledgeGraphTool ToolCard.

Covers:
- BaseToolParam subclass defaults and channel exposure (Task 1)
- KnowledgeGraphTool config, read_only, get_graph=False (Task 2)
- Observer wiring, singleton actor proxy (Task 2)
- Factory closures: get_graph, update_graph, search (Task 2)
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from akgentic.core.actor_address import ActorAddress
from akgentic.core.agent import AkgentType
from akgentic.core.orchestrator import Orchestrator

from akgentic.tool.core import COMMAND, SYSTEM_PROMPT, TOOL_CALL, BaseToolParam, _resolve
from akgentic.tool.event import ToolCallEvent
from akgentic.tool.knowledge_graph.kg_actor import (
    KG_ACTOR_NAME,
    KG_ACTOR_ROLE,
    KnowledgeGraphActor,
)
from akgentic.tool.knowledge_graph.kg_tool import (
    GetGraph,
    KnowledgeGraphTool,
    SearchGraph,
    UpdateGraph,
)
from akgentic.tool.knowledge_graph.models import (
    EntityCreate,
    ManageGraph,
    RelationCreate,
    SearchQuery,
)

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockActorAddress(ActorAddress):
    """Mock ActorAddress for ToolCard tests."""

    def __init__(self, name: str = "test-agent", role: str = "test-role") -> None:
        self._name = name
        self._role = role
        self._agent_id = uuid.uuid4()

    @property
    def agent_id(self) -> uuid.UUID:
        return self._agent_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def role(self) -> str:
        return self._role

    @property
    def team_id(self) -> uuid.UUID | None:
        return None

    @property
    def squad_id(self) -> uuid.UUID | None:
        return None

    def send(self, recipient: Any, message: Any) -> None:
        pass

    def is_alive(self) -> bool:
        return True

    def stop(self) -> None:
        pass

    def handle_user_message(self) -> bool:
        return False

    def serialize(self) -> dict[str, Any]:
        return {"name": self._name, "role": self._role, "agent_id": str(self._agent_id)}

    def __repr__(self) -> str:
        return f"MockActorAddress(name={self._name})"


class MockActorToolObserver:
    """Mock implementing ActorToolObserver protocol for ToolCard tests."""

    def __init__(self) -> None:
        self.events: list[ToolCallEvent] = []
        self._address = MockActorAddress("test-agent")
        self._orchestrator = MockActorAddress("orchestrator")
        self._kg_actor: KnowledgeGraphActor | None = None
        self._orchestrator_proxy = MagicMock(spec=Orchestrator)

    @property
    def myAddress(self) -> ActorAddress:  # noqa: N802
        return self._address

    @property
    def orchestrator(self) -> ActorAddress:
        return self._orchestrator

    def notify_event(self, event: object) -> None:
        if isinstance(event, ToolCallEvent):
            self.events.append(event)

    def proxy_ask(
        self,
        actor: ActorAddress,
        actor_type: type[AkgentType] | None = None,
        timeout: int | None = None,
    ) -> Any:
        if actor == self._orchestrator:
            return self._orchestrator_proxy
        # Return the real KG actor for KG actor address
        if self._kg_actor is not None:
            return self._kg_actor
        return MagicMock()

    def setup_kg_actor(self) -> KnowledgeGraphActor:
        """Create a real KG actor and wire it for proxy_ask."""
        actor = KnowledgeGraphActor()
        actor.on_start()
        self._kg_actor = actor
        kg_addr = MockActorAddress(KG_ACTOR_NAME, KG_ACTOR_ROLE)
        self._orchestrator_proxy.get_team_member.return_value = kg_addr
        return actor


# ===========================================================================
# Task 1: BaseToolParam subclasses — channel exposure defaults
# ===========================================================================


class TestGetGraphParam:
    """GetGraph exposes on SYSTEM_PROMPT + COMMAND by default."""

    def test_default_channels(self) -> None:
        param = GetGraph()
        assert param.expose == {SYSTEM_PROMPT, COMMAND}

    def test_is_base_tool_param(self) -> None:
        assert issubclass(GetGraph, BaseToolParam)


class TestUpdateGraphParam:
    """UpdateGraph exposes on TOOL_CALL by default."""

    def test_default_channels(self) -> None:
        param = UpdateGraph()
        assert param.expose == {TOOL_CALL}

    def test_is_base_tool_param(self) -> None:
        assert issubclass(UpdateGraph, BaseToolParam)


class TestSearchGraphParam:
    """SearchGraph exposes on TOOL_CALL + COMMAND by default."""

    def test_default_channels(self) -> None:
        param = SearchGraph()
        assert param.expose == {TOOL_CALL, COMMAND}

    def test_is_base_tool_param(self) -> None:
        assert issubclass(SearchGraph, BaseToolParam)


class TestParamResolve:
    """_resolve() works with KG param types."""

    def test_resolve_true_returns_default_instance(self) -> None:
        result = _resolve(True, GetGraph)
        assert isinstance(result, GetGraph)
        assert result.expose == {SYSTEM_PROMPT, COMMAND}

    def test_resolve_false_returns_none(self) -> None:
        result = _resolve(False, GetGraph)
        assert result is None

    def test_resolve_instance_returns_same(self) -> None:
        custom = GetGraph(expose={COMMAND})
        result = _resolve(custom, GetGraph)
        assert result is custom
        assert result is not None
        assert result.expose == {COMMAND}


# ===========================================================================
# Task 2: KnowledgeGraphTool — default config, observer, factories
# ===========================================================================


class TestKnowledgeGraphToolDefaults:
    """KnowledgeGraphTool field defaults (2.1)."""

    def test_default_name(self) -> None:
        tool = KnowledgeGraphTool()
        assert tool.name == "KnowledgeGraph"

    def test_default_description(self) -> None:
        tool = KnowledgeGraphTool()
        assert "knowledge graph" in tool.description.lower()

    def test_default_get_graph_is_true(self) -> None:
        tool = KnowledgeGraphTool()
        assert tool.get_graph is True

    def test_default_update_graph_is_true(self) -> None:
        tool = KnowledgeGraphTool()
        assert tool.update_graph is True

    def test_default_search_is_true(self) -> None:
        tool = KnowledgeGraphTool()
        assert tool.search is True

    def test_default_read_only_is_false(self) -> None:
        tool = KnowledgeGraphTool()
        assert tool.read_only is False

    def test_default_embedding_model(self) -> None:
        tool = KnowledgeGraphTool()
        assert tool.embedding_model == "text-embedding-3-small"

    def test_default_embedding_provider(self) -> None:
        tool = KnowledgeGraphTool()
        assert tool.embedding_provider == "openai"


class TestKnowledgeGraphToolObserver:
    """observer() wiring — singleton actor creation (2.2)."""

    def test_observer_requires_orchestrator(self) -> None:
        tool = KnowledgeGraphTool()
        observer = MockActorToolObserver()
        observer._orchestrator = None  # type: ignore[assignment]
        with pytest.raises(ValueError, match="orchestrator"):
            tool.observer(observer)

    def test_observer_creates_actor_when_none(self) -> None:
        tool = KnowledgeGraphTool()
        observer = MockActorToolObserver()
        # get_team_member returns None → must createActor
        observer._orchestrator_proxy.get_team_member.return_value = None
        kg_addr = MockActorAddress(KG_ACTOR_NAME, KG_ACTOR_ROLE)
        observer._orchestrator_proxy.createActor.return_value = kg_addr

        tool.observer(observer)

        observer._orchestrator_proxy.get_team_member.assert_called_once_with(KG_ACTOR_NAME)
        observer._orchestrator_proxy.createActor.assert_called_once()

    def test_observer_reuses_existing_actor(self) -> None:
        tool = KnowledgeGraphTool()
        observer = MockActorToolObserver()
        existing_addr = MockActorAddress(KG_ACTOR_NAME, KG_ACTOR_ROLE)
        observer._orchestrator_proxy.get_team_member.return_value = existing_addr

        tool.observer(observer)

        observer._orchestrator_proxy.get_team_member.assert_called_once_with(KG_ACTOR_NAME)
        observer._orchestrator_proxy.createActor.assert_not_called()


class TestKnowledgeGraphToolReadOnly:
    """read_only=True disables update_graph (2.9)."""

    def test_read_only_disables_update_in_get_tools(self) -> None:
        tool = KnowledgeGraphTool(read_only=True)
        observer = MockActorToolObserver()
        observer.setup_kg_actor()
        tool.observer(observer)

        tools = tool.get_tools()
        tool_names = [t.__name__ for t in tools]
        assert "update_graph" not in tool_names

    def test_read_only_still_exposes_search(self) -> None:
        tool = KnowledgeGraphTool(read_only=True)
        observer = MockActorToolObserver()
        observer.setup_kg_actor()
        tool.observer(observer)

        tools = tool.get_tools()
        tool_names = [t.__name__ for t in tools]
        assert "search_graph" in tool_names


class TestKnowledgeGraphToolGetGraphFalse:
    """get_graph=False removes all get_graph exposure (2.10)."""

    def test_get_graph_false_no_system_prompts(self) -> None:
        tool = KnowledgeGraphTool(get_graph=False)
        observer = MockActorToolObserver()
        observer.setup_kg_actor()
        tool.observer(observer)

        prompts = tool.get_system_prompts()
        assert len(prompts) == 0

    def test_get_graph_false_no_commands_for_get_graph(self) -> None:
        tool = KnowledgeGraphTool(get_graph=False)
        observer = MockActorToolObserver()
        observer.setup_kg_actor()
        tool.observer(observer)

        commands = tool.get_commands()
        assert GetGraph not in commands


class TestGetSystemPrompts:
    """get_system_prompts() returns graph prompt callable (2.3)."""

    def test_returns_prompt_when_get_graph_enabled(self) -> None:
        tool = KnowledgeGraphTool()
        observer = MockActorToolObserver()
        observer.setup_kg_actor()
        tool.observer(observer)

        prompts = tool.get_system_prompts()
        assert len(prompts) == 1
        assert callable(prompts[0])

    def test_prompt_returns_empty_graph_message(self) -> None:
        tool = KnowledgeGraphTool()
        observer = MockActorToolObserver()
        observer.setup_kg_actor()
        tool.observer(observer)

        prompts = tool.get_system_prompts()
        result = prompts[0]()
        assert "empty" in result.lower()

    def test_prompt_contains_entities(self) -> None:
        tool = KnowledgeGraphTool()
        observer = MockActorToolObserver()
        actor = observer.setup_kg_actor()
        # Seed data
        actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(name="Alice", entity_type="Person", description="Engineer"),
                ]
            )
        )
        tool.observer(observer)

        prompts = tool.get_system_prompts()
        result = prompts[0]()
        assert "Alice" in result
        assert "Person" in result
        assert "Engineer" in result


class TestGetTools:
    """get_tools() returns correct factories based on channel config (2.4)."""

    def test_default_config_returns_update_and_search(self) -> None:
        tool = KnowledgeGraphTool()
        observer = MockActorToolObserver()
        observer.setup_kg_actor()
        tool.observer(observer)

        tools = tool.get_tools()
        tool_names = [t.__name__ for t in tools]
        # update_graph has TOOL_CALL, search has TOOL_CALL
        assert "update_graph" in tool_names
        assert "search_graph" in tool_names

    def test_default_config_no_get_graph_in_tools(self) -> None:
        """get_graph defaults to SYSTEM_PROMPT + COMMAND, NOT TOOL_CALL."""
        tool = KnowledgeGraphTool()
        observer = MockActorToolObserver()
        observer.setup_kg_actor()
        tool.observer(observer)

        tools = tool.get_tools()
        tool_names = [t.__name__ for t in tools]
        assert "get_graph" not in tool_names

    def test_get_graph_with_tool_call_channel(self) -> None:
        """When get_graph includes TOOL_CALL, it appears in get_tools()."""
        tool = KnowledgeGraphTool(get_graph=GetGraph(expose={TOOL_CALL, COMMAND}))
        observer = MockActorToolObserver()
        observer.setup_kg_actor()
        tool.observer(observer)

        tools = tool.get_tools()
        tool_names = [t.__name__ for t in tools]
        assert "get_graph" in tool_names


class TestGetCommands:
    """get_commands() returns correct command mappings (2.5)."""

    def test_default_config_has_get_graph_and_search_commands(self) -> None:
        tool = KnowledgeGraphTool()
        observer = MockActorToolObserver()
        observer.setup_kg_actor()
        tool.observer(observer)

        commands = tool.get_commands()
        assert GetGraph in commands
        assert SearchGraph in commands

    def test_no_update_graph_command_by_default(self) -> None:
        """update_graph defaults to TOOL_CALL only, not COMMAND."""
        tool = KnowledgeGraphTool()
        observer = MockActorToolObserver()
        observer.setup_kg_actor()
        tool.observer(observer)

        commands = tool.get_commands()
        assert UpdateGraph not in commands


class TestGetGraphFactory:
    """_get_graph_factory closure behavior (2.6)."""

    def test_get_graph_returns_formatted_string(self) -> None:
        tool = KnowledgeGraphTool(get_graph=GetGraph(expose={TOOL_CALL}))
        observer = MockActorToolObserver()
        actor = observer.setup_kg_actor()
        actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(name="Alice", entity_type="Person", description="Engineer"),
                    EntityCreate(name="Bob", entity_type="Person", description="Designer"),
                ],
                create_relations=[
                    RelationCreate(from_entity="Alice", to_entity="Bob", relation_type="KNOWS"),
                ],
            )
        )
        tool.observer(observer)

        tools = tool.get_tools()
        get_graph_fn = [t for t in tools if t.__name__ == "get_graph"][0]
        result = get_graph_fn()
        assert "Alice" in result
        assert "Bob" in result
        assert "KNOWS" in result

    def test_get_graph_emits_tool_call_event(self) -> None:
        tool = KnowledgeGraphTool(get_graph=GetGraph(expose={TOOL_CALL}))
        observer = MockActorToolObserver()
        observer.setup_kg_actor()
        tool.observer(observer)

        tools = tool.get_tools()
        get_graph_fn = [t for t in tools if t.__name__ == "get_graph"][0]
        get_graph_fn()

        assert len(observer.events) == 1
        assert observer.events[0].tool_name == "Get graph"


class TestUpdateGraphFactory:
    """_update_graph_factory closure behavior (2.7)."""

    def test_update_graph_calls_actor(self) -> None:
        tool = KnowledgeGraphTool()
        observer = MockActorToolObserver()
        actor = observer.setup_kg_actor()
        tool.observer(observer)

        tools = tool.get_tools()
        update_fn = [t for t in tools if t.__name__ == "update_graph"][0]
        result = update_fn(
            ManageGraph(
                create_entities=[
                    EntityCreate(name="Alice", entity_type="Person", description="Engineer"),
                ]
            )
        )
        assert result == "Done"
        assert len(actor.state.knowledge_graph.entities) == 1

    def test_update_graph_emits_tool_call_event(self) -> None:
        tool = KnowledgeGraphTool()
        observer = MockActorToolObserver()
        observer.setup_kg_actor()
        tool.observer(observer)

        tools = tool.get_tools()
        update_fn = [t for t in tools if t.__name__ == "update_graph"][0]
        update_fn(
            ManageGraph(
                create_entities=[
                    EntityCreate(name="X", entity_type="T", description="D"),
                ]
            )
        )

        assert len(observer.events) == 1
        assert observer.events[0].tool_name == "Update graph"


class TestSearchFactory:
    """_search_factory closure behavior (2.8)."""

    def test_search_returns_formatted_string(self) -> None:
        tool = KnowledgeGraphTool()
        observer = MockActorToolObserver()
        actor = observer.setup_kg_actor()
        actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(name="Alice", entity_type="Person", description="Engineer"),
                ]
            )
        )
        tool.observer(observer)

        tools = tool.get_tools()
        search_fn = [t for t in tools if t.__name__ == "search_graph"][0]
        result = search_fn(SearchQuery(query="Alice"))
        assert "Alice" in result

    def test_search_emits_tool_call_event(self) -> None:
        tool = KnowledgeGraphTool()
        observer = MockActorToolObserver()
        observer.setup_kg_actor()
        tool.observer(observer)

        tools = tool.get_tools()
        search_fn = [t for t in tools if t.__name__ == "search_graph"][0]
        search_fn(SearchQuery(query="nonexistent"))

        assert len(observer.events) == 1
        assert observer.events[0].tool_name == "Search graph"
