"""Tests for ``KnowledgeGraphSummaryState`` and its ``KnowledgeGraphTool`` provider (ADR-037 §5)."""

from __future__ import annotations

from unittest.mock import MagicMock

from akgentic.tool.core import COMMAND, LLM_CONTEXT, SYSTEM_PROMPT, Channels
from akgentic.tool.knowledge_graph import (
    GetGraph,
    KnowledgeGraphSummaryState,
    KnowledgeGraphTool,
    RootRow,
)
from akgentic.tool.knowledge_graph.kg_tool import _build_kg_summary_state
from akgentic.tool.knowledge_graph.models import Entity, GraphView, Relation

_FOOTER = "Use the get_graph tool to explore the full graph or subgraphs."


def _entity(
    name: str,
    entity_type: str = "Person",
    description: str = "d",
    is_root: bool = False,
) -> Entity:
    """Build an Entity with explicit fields for concise test setup."""
    return Entity(name=name, entity_type=entity_type, description=description, is_root=is_root)


def _relation(from_entity: str, to_entity: str, relation_type: str = "REL") -> Relation:
    """Build a Relation with explicit fields for concise test setup."""
    return Relation(from_entity=from_entity, to_entity=to_entity, relation_type=relation_type)


def _build_summary_view() -> GraphView:
    """Build a GraphView with varied types and root entities for summary tests."""
    entities = [
        _entity("Product", "Component", "Main product platform", is_root=True),
        _entity("AuthService", "Service", "Central authentication service", is_root=True),
        _entity("UserDB", "Database", "Primary user data store", is_root=True),
        _entity("Cache", "Component", "Redis cache layer"),
        _entity("Logger", "Service", "Logging service"),
    ]
    relations = [
        _relation("Product", "AuthService", "DEPENDS_ON"),
        _relation("AuthService", "UserDB", "STORES_IN"),
        _relation("Product", "Cache", "CONNECTS_TO"),
    ]
    return GraphView(entities=entities, relations=relations)


def make_state(
    view: GraphView,
    include_schema: bool = True,
    include_roots: bool = True,
) -> KnowledgeGraphSummaryState:
    """Build a KnowledgeGraphSummaryState the way the provider does."""
    return _build_kg_summary_state(view, include_schema, include_roots)


def make_tool(view: GraphView) -> KnowledgeGraphTool:
    """Build a KnowledgeGraphTool with a mocked KG actor proxy serving ``view``."""
    tool = KnowledgeGraphTool()
    mock_proxy = MagicMock()
    mock_proxy.get_graph.return_value = view
    tool._kg_proxy = mock_proxy
    return tool


# ── render_full: byte-identical port of the historical graph summary ─────────


def test_render_full_byte_identical_full_summary() -> None:
    """render_full reproduces the historical summary prompt byte for byte."""
    assert make_state(_build_summary_view()).render_full() == (
        "**Knowledge Graph Summary:**\n"
        "Entities: 5 | Relations: 3\n"
        "Entity types: Component, Database, Service\n"
        "Relation types: CONNECTS_TO, DEPENDS_ON, STORES_IN\n"
        "Root entities:\n"
        "- AuthService (Service): Central authentication service\n"
        "- Product (Component): Main product platform\n"
        "- UserDB (Database): Primary user data store\n"
        "\n"
        "Use the get_graph tool to explore the full graph or subgraphs."
    )


def test_render_full_both_schema_and_roots_enabled() -> None:
    """Full summary carries counts, schema, name-sorted roots, and the footer."""
    result = make_state(_build_summary_view()).render_full()
    assert "Knowledge Graph Summary:" in result
    assert "Entities: 5 | Relations: 3" in result
    assert "Entity types:" in result
    assert "Component" in result
    assert "Service" in result
    assert "Database" in result
    assert "Relation types:" in result
    assert "DEPENDS_ON" in result
    assert "Root entities:" in result
    assert "Product (Component): Main product platform" in result
    assert "AuthService (Service): Central authentication service" in result
    assert "UserDB (Database): Primary user data store" in result
    assert "Use the get_graph tool" in result


def test_render_full_schema_disabled() -> None:
    """include_schema=False suppresses the type lines; counts and roots stay."""
    result = make_state(_build_summary_view(), include_schema=False).render_full()
    assert "Entity types:" not in result
    assert "Relation types:" not in result
    assert "Entities: 5" in result
    assert "Root entities:" in result


def test_render_full_roots_disabled() -> None:
    """include_roots=False suppresses the roots block; counts and schema stay."""
    result = make_state(_build_summary_view(), include_roots=False).render_full()
    assert "Root entities:" not in result
    assert "Product (Component)" not in result
    assert "Entities: 5" in result
    assert "Entity types:" in result


def test_render_full_both_disabled_counts_and_footer_only() -> None:
    """Both flags off leaves counts and the footer only."""
    result = make_state(
        _build_summary_view(), include_schema=False, include_roots=False
    ).render_full()
    assert "Entities: 5 | Relations: 3" in result
    assert "Entity types:" not in result
    assert "Root entities:" not in result
    assert "Use the get_graph tool" in result


def test_render_full_empty_graph_sentinel() -> None:
    """An empty graph renders the sentinel line — not '' — with no footer."""
    result = make_state(GraphView()).render_full()
    assert result == "Knowledge graph is empty."
    assert _FOOTER not in result


def test_render_full_no_relation_types_line_when_no_relations() -> None:
    """A graph without relations omits the Relation types line entirely."""
    view = GraphView(entities=[_entity("Alice")])
    result = make_state(view).render_full()
    assert "Entity types: Person" in result
    assert "Relation types:" not in result


def test_render_full_scales_by_types_not_entities() -> None:
    """Summary length depends on distinct types + roots, not total count."""
    entities = [
        _entity(f"E{i}", "TypeA" if i % 2 == 0 else "TypeB", f"Entity {i}") for i in range(50)
    ]
    entities[0].is_root = True
    relations = [_relation(f"E{i}", f"E{i + 1}") for i in range(49)]
    result = make_state(GraphView(entities=entities, relations=relations)).render_full()
    lines = result.strip().split("\n")
    assert len(lines) < 15  # NOT 50+ lines


# ── render_delta: names only what moved ──────────────────────────────────────


def test_delta_added_entities_of_existing_type_reports_counts_only() -> None:
    """Adding entities of an already-present type reports counts, no type line."""
    previous = make_state(GraphView(entities=[_entity("Alice"), _entity("Bob")]))
    current = make_state(
        GraphView(entities=[_entity("Alice"), _entity("Bob"), _entity("Carol")])
    )

    delta = current.render_delta(previous)

    assert delta == "Entities: 3 (+1)."


def test_delta_new_entity_type_renders_once() -> None:
    """A new entity type renders exactly once, alongside the count movement."""
    previous = make_state(GraphView(entities=[_entity("Alice")]))
    current = make_state(GraphView(entities=[_entity("Alice"), _entity("R1", "Risk")]))

    delta = current.render_delta(previous)

    assert delta is not None
    assert "Entities: 2 (+1)." in delta
    assert delta.count("New entity type: Risk.") == 1


def test_delta_new_relation_type_renders_once() -> None:
    """A new relation type renders exactly once."""
    entities = [_entity("Alice"), _entity("Bob")]
    previous = make_state(GraphView(entities=entities))
    current = make_state(
        GraphView(entities=entities, relations=[_relation("Alice", "Bob", "KNOWS")])
    )

    delta = current.render_delta(previous)

    assert delta is not None
    assert "Relations: 1 (+1)." in delta
    assert delta.count("New relation type: KNOWS.") == 1


def test_delta_removed_entity_type_renders_once() -> None:
    """A disappeared entity type renders exactly once as removed."""
    previous = make_state(GraphView(entities=[_entity("Alice"), _entity("R1", "Risk")]))
    current = make_state(GraphView(entities=[_entity("Alice")]))

    delta = current.render_delta(previous)

    assert delta is not None
    assert delta.count("Removed entity type: Risk.") == 1


def test_delta_new_root_renders_once() -> None:
    """A new root renders exactly once, keyed by name."""
    previous = make_state(GraphView(entities=[_entity("Alice")]))
    current = make_state(
        GraphView(
            entities=[
                _entity("Alice"),
                _entity("MigrationPlan", "Project", "The plan", is_root=True),
            ]
        )
    )

    delta = current.render_delta(previous)

    assert delta is not None
    assert delta.count("New root: MigrationPlan (Project).") == 1


def test_delta_removed_root_renders_as_removed() -> None:
    """A root leaving the graph renders as removed."""
    previous = make_state(
        GraphView(entities=[_entity("Alice"), _entity("Plan", "Project", is_root=True)])
    )
    current = make_state(GraphView(entities=[_entity("Alice")]))

    delta = current.render_delta(previous)

    assert delta is not None
    assert "Removed root: Plan." in delta


def test_delta_changed_root_reported_once_as_change() -> None:
    """A root re-described on both sides is reported once as a change."""
    previous = make_state(
        GraphView(entities=[_entity("Plan", "Project", "Old description", is_root=True)])
    )
    current = make_state(
        GraphView(entities=[_entity("Plan", "Project", "New description", is_root=True)])
    )

    delta = current.render_delta(previous)

    assert delta == "Changed root: Plan (Project): New description."


def test_delta_pure_count_movement_reports_counts_only() -> None:
    """Count movement with unchanged types and roots yields the counts line only."""
    previous = make_state(
        GraphView(
            entities=[_entity("Alice"), _entity("Bob")],
            relations=[_relation("Alice", "Bob")],
        )
    )
    current = make_state(
        GraphView(
            entities=[_entity("Alice"), _entity("Bob"), _entity("Carol")],
            relations=[_relation("Alice", "Bob"), _relation("Bob", "Carol")],
        )
    )

    delta = current.render_delta(previous)

    assert delta == "Entities: 3 (+1) | Relations: 2 (+1)."


def test_delta_include_schema_false_suppresses_type_reporting() -> None:
    """A suppressed schema block never surfaces in the delta."""
    previous = make_state(GraphView(entities=[_entity("Alice")]), include_schema=False)
    current = make_state(
        GraphView(entities=[_entity("Alice"), _entity("R1", "Risk")]), include_schema=False
    )

    delta = current.render_delta(previous)

    assert delta == "Entities: 2 (+1)."


def test_delta_include_roots_false_suppresses_root_reporting() -> None:
    """A suppressed roots block never surfaces in the delta."""
    previous = make_state(GraphView(entities=[_entity("Alice")]), include_roots=False)
    current = make_state(
        GraphView(entities=[_entity("Alice"), _entity("Plan", is_root=True)]),
        include_roots=False,
    )

    delta = current.render_delta(previous)

    assert delta == "Entities: 2 (+1)."
    assert "root" not in delta.lower()


def test_delta_identical_states_return_none() -> None:
    """An unchanged graph diffs to None."""
    view = _build_summary_view()
    assert make_state(view).render_delta(make_state(view)) is None


def test_delta_two_empty_states_return_none() -> None:
    """Two empty graphs diff to None."""
    assert make_state(GraphView()).render_delta(make_state(GraphView())) is None


def test_delta_length_bounded_on_large_graph() -> None:
    """A 1000-entity / 3-type graph diffs to a delta of bounded length."""
    types = ["TypeA", "TypeB", "TypeC"]
    small = GraphView(entities=[_entity(f"E{i}", types[i % 3]) for i in range(10)])
    large = GraphView(
        entities=[_entity(f"E{i}", types[i % 3]) for i in range(1000)],
        relations=[_relation(f"E{i}", f"E{i + 1}") for i in range(999)],
    )

    delta = make_state(large).render_delta(make_state(small))

    assert delta is not None
    assert len(delta) < 200  # O(types + roots), never O(entities)


# ── serialization round-trip ─────────────────────────────────────────────────


def test_summary_state_round_trip() -> None:
    """KnowledgeGraphSummaryState round-trips through model_dump / model_validate."""
    state = make_state(_build_summary_view(), include_schema=True, include_roots=False)
    restored = KnowledgeGraphSummaryState.model_validate(state.model_dump())

    assert restored == state
    assert restored.render_full() == state.render_full()


def test_root_row_round_trip() -> None:
    """RootRow round-trips through model_dump / model_validate."""
    row = RootRow(name="Plan", entity_type="Project", description="The plan")

    assert RootRow.model_validate(row.model_dump()) == row


# ── provider gating on KnowledgeGraphTool.get_context_states() ───────────────


def test_default_tool_yields_one_named_provider() -> None:
    """The default KnowledgeGraphTool exposes exactly one provider, by stable name."""
    tool = make_tool(GraphView())

    providers = tool.get_context_states()

    assert [p.__name__ for p in providers] == ["knowledge_graph_summary_state"]


def test_disabled_get_graph_yields_no_provider() -> None:
    """get_graph=False drops the provider."""
    tool = make_tool(GraphView())
    tool.get_graph = False

    assert tool.get_context_states() == []


def test_expose_without_llm_context_yields_no_provider() -> None:
    """A get_graph narrowed to COMMAND only must not surface a provider
    (silent-drop trap)."""
    tool = make_tool(GraphView())
    tool.get_graph = GetGraph(expose={Channels.COMMAND})

    assert tool.get_context_states() == []


# ── provider behavior ────────────────────────────────────────────────────────


def test_provider_bakes_prompt_flags_into_state() -> None:
    """The provider stamps the resolved prompt flags onto every state."""
    tool = make_tool(_build_summary_view())
    tool.get_graph = GetGraph(prompt_include_schema=False, prompt_include_roots=False)
    provider = tool.get_context_states()[0]

    state = provider()

    assert isinstance(state, KnowledgeGraphSummaryState)
    assert state.include_schema is False
    assert state.include_roots is False
    assert "Entity types:" not in state.render_full()
    assert "Root entities:" not in state.render_full()


def test_provider_empty_graph_is_a_state_not_none() -> None:
    """An empty graph is a real state whose render_full is the sentinel line."""
    tool = make_tool(GraphView())
    provider = tool.get_context_states()[0]

    state = provider()

    assert isinstance(state, KnowledgeGraphSummaryState)
    assert state.render_full() == "Knowledge graph is empty."


def test_provider_returns_none_on_proxy_failure() -> None:
    """A raising KG proxy makes the provider return None — never raise."""
    tool = make_tool(GraphView())
    tool._kg_proxy.get_graph.side_effect = RuntimeError("actor gone")
    provider = tool.get_context_states()[0]

    assert provider() is None


# ── persisted-payload normalizer adoption (ADR-037 §4) ───────────────────────


def test_persisted_system_prompt_expose_revalidates_to_llm_context() -> None:
    """A payload with expose ['system_prompt', 'command'] resolves to LLM_CONTEXT
    and still yields the provider."""
    payload = KnowledgeGraphTool().model_dump()
    payload["get_graph"] = {"expose": ["system_prompt", "command"]}

    restored = KnowledgeGraphTool.model_validate(payload)

    assert isinstance(restored.get_graph, GetGraph)
    assert restored.get_graph.expose == {LLM_CONTEXT, COMMAND}

    restored._kg_proxy = MagicMock()
    providers = restored.get_context_states()
    assert [p.__name__ for p in providers] == ["knowledge_graph_summary_state"]


def test_other_kg_params_keep_system_prompt_untouched() -> None:
    """UpdateGraph / SearchGraph payloads keep system_prompt in expose —
    the normalizer is per-param, never a global rewrite."""
    payload = KnowledgeGraphTool().model_dump()
    payload["update_graph"] = {"expose": ["system_prompt", "tool_call"]}
    payload["search"] = {"expose": ["system_prompt", "command"]}

    restored = KnowledgeGraphTool.model_validate(payload)

    assert not isinstance(restored.update_graph, bool)
    assert SYSTEM_PROMPT in restored.update_graph.expose
    assert not isinstance(restored.search, bool)
    assert SYSTEM_PROMPT in restored.search.expose
