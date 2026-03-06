"""Tests for Knowledge Graph data models.

Covers: UUID auto-generation, defaults, Pydantic validation,
serialization round-trip for all models (Task 1.7).
"""

from __future__ import annotations

import uuid

import pytest

from akgentic.tool.knowledge_graph.models import (
    Entity,
    EntityCreate,
    EntityUpdate,
    KnowledgeGraph,
    KnowledgeGraphState,
    ManageGraph,
    Relation,
    RelationCreate,
    RelationDelete,
)

# ---------------------------------------------------------------------------
# Entity model (AC-2, Task 1.1)
# ---------------------------------------------------------------------------


class TestEntity:
    """Tests for Entity model."""

    def test_auto_generated_uuid(self) -> None:
        """Entity.id is a UUID auto-generated on creation."""
        e = Entity(name="Alice", entity_type="Person", description="A person")
        assert isinstance(e.id, uuid.UUID)

    def test_two_entities_have_different_uuids(self) -> None:
        e1 = Entity(name="Alice", entity_type="Person", description="A person")
        e2 = Entity(name="Bob", entity_type="Person", description="Another person")
        assert e1.id != e2.id

    def test_observations_default_empty(self) -> None:
        e = Entity(name="Alice", entity_type="Person", description="A person")
        assert e.observations == []

    def test_observations_set_explicitly(self) -> None:
        e = Entity(
            name="Alice",
            entity_type="Person",
            description="A person",
            observations=["friendly", "tall"],
        )
        assert e.observations == ["friendly", "tall"]

    def test_required_fields_enforced(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            Entity()  # type: ignore[call-arg]

    def test_all_fields_type_checked(self) -> None:
        e = Entity(name="Alice", entity_type="Person", description="A person")
        assert isinstance(e.name, str)
        assert isinstance(e.entity_type, str)
        assert isinstance(e.description, str)
        assert isinstance(e.observations, list)

    def test_serialization_round_trip(self) -> None:
        e = Entity(
            name="Alice",
            entity_type="Person",
            description="A person",
            observations=["friendly"],
        )
        data = e.model_dump()
        restored = Entity.model_validate(data)
        assert restored.name == e.name
        assert restored.entity_type == e.entity_type
        assert restored.description == e.description
        assert restored.observations == e.observations
        # UUID survives round-trip (as string in dump, parsed back)
        assert str(restored.id) == str(e.id)

    def test_is_root_defaults_false(self) -> None:
        e = Entity(name="Alice", entity_type="Person", description="A person")
        assert e.is_root is False

    def test_is_root_set_true(self) -> None:
        e = Entity(name="Alice", entity_type="Person", description="A person", is_root=True)
        assert e.is_root is True

    def test_is_root_survives_serialization(self) -> None:
        e = Entity(name="R", entity_type="T", description="d", is_root=True)
        restored = Entity.model_validate(e.model_dump())
        assert restored.is_root is True


# ---------------------------------------------------------------------------
# Relation model (AC-3, Task 1.2)
# ---------------------------------------------------------------------------


class TestRelation:
    """Tests for Relation model."""

    def test_auto_generated_uuid(self) -> None:
        r = Relation(from_entity="A", to_entity="B", relation_type="knows")
        assert isinstance(r.id, uuid.UUID)

    def test_description_defaults_empty(self) -> None:
        r = Relation(from_entity="A", to_entity="B", relation_type="knows")
        assert r.description == ""

    def test_description_set_explicitly(self) -> None:
        r = Relation(
            from_entity="A", to_entity="B", relation_type="knows", description="since 2020"
        )
        assert r.description == "since 2020"

    def test_required_fields_enforced(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            Relation()  # type: ignore[call-arg]

    def test_serialization_round_trip(self) -> None:
        r = Relation(from_entity="A", to_entity="B", relation_type="knows", description="well")
        data = r.model_dump()
        restored = Relation.model_validate(data)
        assert restored.from_entity == r.from_entity
        assert restored.to_entity == r.to_entity
        assert restored.relation_type == r.relation_type
        assert restored.description == r.description
        assert str(restored.id) == str(r.id)


# ---------------------------------------------------------------------------
# CRUD request models (Task 1.3)
# ---------------------------------------------------------------------------


class TestEntityCreate:
    def test_defaults(self) -> None:
        ec = EntityCreate(name="X", entity_type="Concept", description="desc")
        assert ec.observations == []

    def test_with_observations(self) -> None:
        ec = EntityCreate(name="X", entity_type="Concept", description="desc", observations=["a"])
        assert ec.observations == ["a"]

    def test_is_root_defaults_false(self) -> None:
        ec = EntityCreate(name="X", entity_type="Concept", description="desc")
        assert ec.is_root is False

    def test_is_root_set_true(self) -> None:
        ec = EntityCreate(name="X", entity_type="Concept", description="desc", is_root=True)
        assert ec.is_root is True


class TestEntityUpdate:
    def test_all_optional_fields_default_none(self) -> None:
        eu = EntityUpdate(name="X")
        assert eu.description is None
        assert eu.entity_type is None
        assert eu.add_observations is None
        assert eu.remove_observations is None
        assert eu.is_root is None

    def test_partial_fields(self) -> None:
        eu = EntityUpdate(name="X", description="new desc")
        assert eu.description == "new desc"
        assert eu.entity_type is None

    def test_is_root_none_is_noop(self) -> None:
        eu = EntityUpdate(name="X")
        assert eu.is_root is None

    def test_is_root_set_true(self) -> None:
        eu = EntityUpdate(name="X", is_root=True)
        assert eu.is_root is True

    def test_is_root_set_false(self) -> None:
        eu = EntityUpdate(name="X", is_root=False)
        assert eu.is_root is False


class TestRelationCreate:
    def test_defaults(self) -> None:
        rc = RelationCreate(from_entity="A", to_entity="B", relation_type="knows")
        assert rc.description == ""

    def test_with_description(self) -> None:
        rc = RelationCreate(
            from_entity="A", to_entity="B", relation_type="knows", description="well"
        )
        assert rc.description == "well"


class TestRelationDelete:
    def test_fields(self) -> None:
        rd = RelationDelete(from_entity="A", to_entity="B", relation_type="knows")
        assert rd.from_entity == "A"
        assert rd.to_entity == "B"
        assert rd.relation_type == "knows"


# ---------------------------------------------------------------------------
# ManageGraph batch model (Task 1.4)
# ---------------------------------------------------------------------------


class TestManageGraph:
    def test_all_defaults_empty(self) -> None:
        mg = ManageGraph()
        assert mg.create_entities == []
        assert mg.update_entities == []
        assert mg.delete_entities == []
        assert mg.create_relations == []
        assert mg.delete_relations == []

    def test_with_data(self) -> None:
        mg = ManageGraph(
            create_entities=[EntityCreate(name="A", entity_type="T", description="d")],
            delete_entities=["B"],
        )
        assert len(mg.create_entities) == 1
        assert mg.delete_entities == ["B"]


# ---------------------------------------------------------------------------
# KnowledgeGraph container (Task 1.5)
# ---------------------------------------------------------------------------


class TestKnowledgeGraph:
    def test_defaults_empty(self) -> None:
        kg = KnowledgeGraph()
        assert kg.entities == []
        assert kg.relations == []

    def test_with_data(self) -> None:
        e = Entity(name="A", entity_type="T", description="d")
        r = Relation(from_entity="A", to_entity="B", relation_type="knows")
        kg = KnowledgeGraph(entities=[e], relations=[r])
        assert len(kg.entities) == 1
        assert len(kg.relations) == 1


# ---------------------------------------------------------------------------
# KnowledgeGraphState (Task 1.6)
# ---------------------------------------------------------------------------


class TestKnowledgeGraphState:
    def test_default_knowledge_graph(self) -> None:
        state = KnowledgeGraphState()
        assert isinstance(state.knowledge_graph, KnowledgeGraph)
        assert state.knowledge_graph.entities == []

    def test_inherits_base_state(self) -> None:
        from akgentic.core.agent_state import BaseState

        state = KnowledgeGraphState()
        assert isinstance(state, BaseState)

    def test_notify_state_change_without_observer(self) -> None:
        """notify_state_change should be a no-op without observer."""
        state = KnowledgeGraphState()
        state.notify_state_change()  # Should not raise
