"""Tests for KnowledgeGraphActor CRUD operations.

Covers: all CRUD ops, cascade deletion, error collection, partial updates,
state change notification, deduplication logic, singleton constants (Task 2.9).
Also covers embedding wiring: entity/relation embedding on create, re-embedding
on description update, removal on delete, graceful degradation on API failure
(Task 5.11, Story 2.1 ACs 3-7).

Pattern: Instantiate KnowledgeGraphActor() directly, call on_start(),
test methods. Same pattern as test_planning_actor.py.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from akgentic.core.actor_address import ActorAddress

from akgentic.tool.errors import RetriableError
from akgentic.tool.knowledge_graph.kg_actor import (
    KG_ACTOR_NAME,
    KG_ACTOR_ROLE,
    KnowledgeGraphActor,
)
from akgentic.tool.knowledge_graph.models import (
    EntityCreate,
    EntityUpdate,
    GraphView,
    ManageGraph,
    RelationCreate,
    RelationDelete,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class MockActorAddress(ActorAddress):
    """Mock ActorAddress for testing (mirrors test_planning_actor.py)."""

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

    def send(self, recipient, message):  # type: ignore[no-untyped-def]
        pass

    def is_alive(self) -> bool:
        return True

    def stop(self) -> None:
        pass

    def handle_user_message(self) -> bool:
        return False

    def serialize(self):  # type: ignore[no-untyped-def]
        return {"name": self._name, "role": self._role, "agent_id": str(self._agent_id)}

    def __repr__(self) -> str:
        return f"MockActorAddress(name={self._name})"


def _actor() -> KnowledgeGraphActor:
    """Create and initialize a KnowledgeGraphActor for testing."""
    actor = KnowledgeGraphActor()
    actor.on_start()
    return actor


def _seed_entities(actor: KnowledgeGraphActor) -> None:
    """Seed actor with two entities: Alice (Person) and Bob (Person)."""
    actor.update_graph(
        ManageGraph(
            create_entities=[
                EntityCreate(name="Alice", entity_type="Person", description="Engineer"),
                EntityCreate(name="Bob", entity_type="Person", description="Designer"),
            ]
        )
    )


def _seed_with_relation(actor: KnowledgeGraphActor) -> None:
    """Seed actor with two entities and a relation between them."""
    actor.update_graph(
        ManageGraph(
            create_entities=[
                EntityCreate(name="Alice", entity_type="Person", description="Engineer"),
                EntityCreate(name="Bob", entity_type="Person", description="Designer"),
            ],
            create_relations=[
                RelationCreate(from_entity="Alice", to_entity="Bob", relation_type="knows"),
            ],
        )
    )


# ---------------------------------------------------------------------------
# Constants (AC-4)
# ---------------------------------------------------------------------------


class TestConstants:
    def test_actor_name(self) -> None:
        assert KG_ACTOR_NAME == "#KnowledgeGraphTool"

    def test_actor_role(self) -> None:
        assert KG_ACTOR_ROLE == "ToolActor"


# ---------------------------------------------------------------------------
# on_start / get_graph (Task 2.1, 2.8)
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_on_start_initializes_empty_graph(self) -> None:
        actor = _actor()
        gv = actor.get_graph()
        entities, relations = gv.entities, gv.relations
        assert entities == []
        assert relations == []

    def test_get_graph_returns_graph_view(self) -> None:
        actor = _actor()
        result = actor.get_graph()
        assert isinstance(result, GraphView)


# ---------------------------------------------------------------------------
# Create entities (AC-5, Task 2.2)
# ---------------------------------------------------------------------------


class TestCreateEntities:
    def test_create_single_entity(self) -> None:
        actor = _actor()
        result = actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(name="Alice", entity_type="Person", description="Engineer")
                ]
            )
        )
        assert result == "Done"
        entities = actor.get_graph().entities
        assert len(entities) == 1
        assert entities[0].name == "Alice"
        assert isinstance(entities[0].id, uuid.UUID)

    def test_create_multiple_entities(self) -> None:
        actor = _actor()
        _seed_entities(actor)
        entities = actor.get_graph().entities
        assert len(entities) == 2

    def test_duplicate_entity_name_raises_error(self) -> None:
        actor = _actor()
        _seed_entities(actor)
        with pytest.raises(RetriableError, match="already exists"):
            actor.update_graph(
                ManageGraph(
                    create_entities=[
                        EntityCreate(name="Alice", entity_type="Person", description="dup")
                    ]
                )
            )

    def test_entity_gets_uuid(self) -> None:
        actor = _actor()
        actor.update_graph(
            ManageGraph(create_entities=[EntityCreate(name="X", entity_type="T", description="d")])
        )
        entities = actor.get_graph().entities
        assert isinstance(entities[0].id, uuid.UUID)


# ---------------------------------------------------------------------------
# Create relations (AC-5, Task 2.3)
# ---------------------------------------------------------------------------


class TestCreateRelations:
    def test_create_relation_success(self) -> None:
        actor = _actor()
        _seed_with_relation(actor)
        relations = actor.get_graph().relations
        assert len(relations) == 1
        assert relations[0].from_entity == "Alice"
        assert relations[0].to_entity == "Bob"

    def test_relation_missing_from_entity(self) -> None:
        actor = _actor()
        _seed_entities(actor)
        with pytest.raises(RetriableError, match="non-existent from_entity"):
            actor.update_graph(
                ManageGraph(
                    create_relations=[
                        RelationCreate(from_entity="Nobody", to_entity="Bob", relation_type="knows")
                    ]
                )
            )

    def test_relation_missing_to_entity(self) -> None:
        actor = _actor()
        _seed_entities(actor)
        with pytest.raises(RetriableError, match="non-existent to_entity"):
            actor.update_graph(
                ManageGraph(
                    create_relations=[
                        RelationCreate(
                            from_entity="Alice", to_entity="Nobody", relation_type="knows"
                        )
                    ]
                )
            )

    def test_duplicate_relation_silently_skipped(self) -> None:
        actor = _actor()
        _seed_with_relation(actor)
        # Add same relation again — no error, just skipped
        result = actor.update_graph(
            ManageGraph(
                create_relations=[
                    RelationCreate(from_entity="Alice", to_entity="Bob", relation_type="knows"),
                ]
            )
        )
        assert result == "Done"
        relations = actor.get_graph().relations
        assert len(relations) == 1


# ---------------------------------------------------------------------------
# Update entities (AC-6, Task 2.4)
# ---------------------------------------------------------------------------


class TestUpdateEntities:
    def test_partial_update_description(self) -> None:
        actor = _actor()
        _seed_entities(actor)
        actor.update_graph(
            ManageGraph(update_entities=[EntityUpdate(name="Alice", description="Senior Engineer")])
        )
        entities = actor.get_graph().entities
        alice = next(e for e in entities if e.name == "Alice")
        assert alice.description == "Senior Engineer"
        assert alice.entity_type == "Person"  # unchanged

    def test_partial_update_entity_type(self) -> None:
        actor = _actor()
        _seed_entities(actor)
        actor.update_graph(
            ManageGraph(update_entities=[EntityUpdate(name="Alice", entity_type="Engineer")])
        )
        entities = actor.get_graph().entities
        alice = next(e for e in entities if e.name == "Alice")
        assert alice.entity_type == "Engineer"
        assert alice.description == "Engineer"  # unchanged (original value)

    def test_add_observations(self) -> None:
        actor = _actor()
        _seed_entities(actor)
        actor.update_graph(
            ManageGraph(
                update_entities=[EntityUpdate(name="Alice", add_observations=["friendly", "smart"])]
            )
        )
        entities = actor.get_graph().entities
        alice = next(e for e in entities if e.name == "Alice")
        assert alice.observations == ["friendly", "smart"]

    def test_remove_observations(self) -> None:
        actor = _actor()
        actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(
                        name="Alice",
                        entity_type="Person",
                        description="d",
                        observations=["a", "b", "c"],
                    )
                ]
            )
        )
        actor.update_graph(
            ManageGraph(update_entities=[EntityUpdate(name="Alice", remove_observations=["b"])])
        )
        entities = actor.get_graph().entities
        alice = next(e for e in entities if e.name == "Alice")
        assert alice.observations == ["a", "c"]

    def test_update_entity_not_found(self) -> None:
        actor = _actor()
        with pytest.raises(RetriableError, match="not found for update"):
            actor.update_graph(
                ManageGraph(update_entities=[EntityUpdate(name="Nobody", description="x")])
            )


# ---------------------------------------------------------------------------
# Delete entities with cascade (AC-7, Task 2.5)
# ---------------------------------------------------------------------------


class TestDeleteEntities:
    def test_delete_entity(self) -> None:
        actor = _actor()
        _seed_entities(actor)
        actor.update_graph(ManageGraph(delete_entities=["Alice"]))
        entities = actor.get_graph().entities
        assert len(entities) == 1
        assert entities[0].name == "Bob"

    def test_cascade_delete_relations(self) -> None:
        actor = _actor()
        _seed_with_relation(actor)
        actor.update_graph(ManageGraph(delete_entities=["Alice"]))
        gv = actor.get_graph()
        entities, relations = gv.entities, gv.relations
        assert len(entities) == 1
        assert len(relations) == 0  # relation cascade-deleted

    def test_delete_entity_not_found(self) -> None:
        actor = _actor()
        with pytest.raises(RetriableError, match="not found for deletion"):
            actor.update_graph(ManageGraph(delete_entities=["Nobody"]))


# ---------------------------------------------------------------------------
# Delete relations (Task 2.6)
# ---------------------------------------------------------------------------


class TestDeleteRelations:
    def test_delete_relation_success(self) -> None:
        actor = _actor()
        _seed_with_relation(actor)
        actor.update_graph(
            ManageGraph(
                delete_relations=[
                    RelationDelete(from_entity="Alice", to_entity="Bob", relation_type="knows")
                ]
            )
        )
        relations = actor.get_graph().relations
        assert len(relations) == 0

    def test_delete_relation_not_found(self) -> None:
        actor = _actor()
        _seed_entities(actor)
        with pytest.raises(RetriableError, match="not found"):
            actor.update_graph(
                ManageGraph(
                    delete_relations=[
                        RelationDelete(from_entity="Alice", to_entity="Bob", relation_type="knows")
                    ]
                )
            )


# ---------------------------------------------------------------------------
# Error collection across operations (AC-8, Task 2.7)
# ---------------------------------------------------------------------------


class TestErrorCollection:
    def test_multiple_errors_collected_single_raise(self) -> None:
        """All errors across operations collected into one RetriableError."""
        actor = _actor()
        with pytest.raises(RetriableError) as exc_info:
            actor.update_graph(
                ManageGraph(
                    update_entities=[EntityUpdate(name="X", description="d")],
                    delete_entities=["Y"],
                )
            )
        msg = str(exc_info.value)
        assert "X" in msg
        assert "Y" in msg

    def test_errors_from_mixed_operations(self) -> None:
        """Mix of successful and failing operations — errors collected, successes applied."""
        actor = _actor()
        _seed_entities(actor)

        with pytest.raises(RetriableError) as exc_info:
            actor.update_graph(
                ManageGraph(
                    create_entities=[
                        EntityCreate(name="Charlie", entity_type="T", description="d"),
                    ],
                    create_relations=[
                        RelationCreate(from_entity="Ghost", to_entity="Alice", relation_type="x"),
                    ],
                )
            )
        # Charlie was created successfully before the relation error
        entities = actor.get_graph().entities
        names = {e.name for e in entities}
        assert "Charlie" in names
        assert "Ghost" in str(exc_info.value)


# ---------------------------------------------------------------------------
# State change notification (AC-9, Task 2.7)
# ---------------------------------------------------------------------------


class TestStateNotification:
    def test_notify_called_on_update(self) -> None:
        """state.notify_state_change() is called after mutations."""
        actor = _actor()
        # Attach a mock observer to track calls
        call_count = 0

        class Observer:
            def notify_state_change(self, state: object) -> None:
                nonlocal call_count
                call_count += 1

        actor.state.observer(Observer())  # type: ignore[arg-type]
        # observer() itself calls notify once
        initial_count = call_count

        actor.update_graph(
            ManageGraph(create_entities=[EntityCreate(name="A", entity_type="T", description="d")])
        )
        assert call_count > initial_count

    def test_notify_called_even_on_error(self) -> None:
        """notify_state_change() fires even when errors are raised."""
        actor = _actor()
        call_count = 0

        class Observer:
            def notify_state_change(self, state: object) -> None:
                nonlocal call_count
                call_count += 1

        actor.state.observer(Observer())  # type: ignore[arg-type]
        initial_count = call_count

        with pytest.raises(RetriableError):
            actor.update_graph(ManageGraph(delete_entities=["Ghost"]))

        assert call_count > initial_count


# ---------------------------------------------------------------------------
# Operation order (Task 2.7 design decision)
# ---------------------------------------------------------------------------


class TestOperationOrder:
    def test_create_before_relations(self) -> None:
        """Entities created in same batch can be referenced by relations."""
        actor = _actor()
        result = actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(name="A", entity_type="T", description="d"),
                    EntityCreate(name="B", entity_type="T", description="d"),
                ],
                create_relations=[
                    RelationCreate(from_entity="A", to_entity="B", relation_type="links"),
                ],
            )
        )
        assert result == "Done"
        gv = actor.get_graph()
        entities, relations = gv.entities, gv.relations
        assert len(entities) == 2
        assert len(relations) == 1

    def test_delete_last(self) -> None:
        """Deletes happen after creates/updates so cascade works on full graph."""
        actor = _actor()
        _seed_with_relation(actor)

        # Delete Alice (cascade removes relation) in same batch as relation create
        # The new relation should be created before the delete cascade
        actor.update_graph(
            ManageGraph(
                create_relations=[
                    RelationCreate(from_entity="Bob", to_entity="Alice", relation_type="respects"),
                ],
                delete_entities=["Alice"],
            )
        )
        gv = actor.get_graph()
        entities, relations = gv.entities, gv.relations
        assert len(entities) == 1
        assert entities[0].name == "Bob"
        # Both relations (original + new) should be cascade-deleted
        assert len(relations) == 0


# ---------------------------------------------------------------------------
# is_root wiring (Story 1.4, AC-2, AC-3)
# ---------------------------------------------------------------------------


class TestIsRootWiring:
    """Verify is_root flows through create and update operations."""

    def test_create_entity_with_is_root_true(self) -> None:
        actor = _actor()
        actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(
                        name="Root", entity_type="Concept", description="entry point", is_root=True
                    )
                ]
            )
        )
        entities = actor.get_graph().entities
        assert len(entities) == 1
        assert entities[0].is_root is True

    def test_create_entity_without_is_root_defaults_false(self) -> None:
        actor = _actor()
        actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(name="Leaf", entity_type="Concept", description="not root")
                ]
            )
        )
        entities = actor.get_graph().entities
        assert entities[0].is_root is False

    def test_update_entity_toggle_is_root_true(self) -> None:
        actor = _actor()
        _seed_entities(actor)
        actor.update_graph(ManageGraph(update_entities=[EntityUpdate(name="Alice", is_root=True)]))
        alice = next(e for e in actor.get_graph().entities if e.name == "Alice")
        assert alice.is_root is True

    def test_update_entity_toggle_is_root_false(self) -> None:
        actor = _actor()
        actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(name="Root", entity_type="T", description="d", is_root=True)
                ]
            )
        )
        actor.update_graph(ManageGraph(update_entities=[EntityUpdate(name="Root", is_root=False)]))
        root = actor.get_graph().entities[0]
        assert root.is_root is False

    def test_is_root_survives_other_partial_updates(self) -> None:
        actor = _actor()
        actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(name="Root", entity_type="T", description="d", is_root=True)
                ]
            )
        )
        # Update description only — is_root should remain True
        actor.update_graph(
            ManageGraph(update_entities=[EntityUpdate(name="Root", description="updated")])
        )
        root = actor.get_graph().entities[0]
        assert root.is_root is True
        assert root.description == "updated"

    def test_update_is_root_none_is_noop(self) -> None:
        actor = _actor()
        actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(name="Root", entity_type="T", description="d", is_root=True)
                ]
            )
        )
        # is_root=None (default) means no change
        actor.update_graph(
            ManageGraph(update_entities=[EntityUpdate(name="Root", description="x")])
        )
        root = actor.get_graph().entities[0]
        assert root.is_root is True


# ---------------------------------------------------------------------------
# Embedding wiring (Task 5.11, Story 2.1 ACs 3-7)
# ---------------------------------------------------------------------------

_FAKE_VECTOR = [0.1, 0.2, 0.3]
_EMBED_MODULE = "akgentic.tool.knowledge_graph.kg_actor.EmbeddingService"


def _actor_with_mock_embed() -> tuple[KnowledgeGraphActor, MagicMock]:
    """Return (actor, mock_embed_method) with a pre-wired mock EmbeddingService."""
    actor = _actor()
    mock_svc = MagicMock()
    mock_svc.embed.return_value = [_FAKE_VECTOR]
    actor._embedding_svc = mock_svc  # inject mock directly, bypassing lazy init
    return actor, mock_svc


class TestEmbeddingOnCreate:
    """AC-3, AC-4: embedding called when entities/relations are created."""

    def test_embedding_called_on_entity_create(self) -> None:
        actor, mock_svc = _actor_with_mock_embed()
        actor.update_graph(
            ManageGraph(
                create_entities=[EntityCreate(name="Alice", entity_type="Person", description="Eng")]
            )
        )
        mock_svc.embed.assert_called_once()
        call_args = mock_svc.embed.call_args[0][0]
        assert "Alice" in call_args[0]
        assert "Eng" in call_args[0]

    def test_embedding_called_on_relation_create_with_description(self) -> None:
        actor, mock_svc = _actor_with_mock_embed()
        _seed_entities(actor)
        mock_svc.reset_mock()
        actor.update_graph(
            ManageGraph(
                create_relations=[
                    RelationCreate(
                        from_entity="Alice",
                        to_entity="Bob",
                        relation_type="knows",
                        description="long colleagues",
                    )
                ]
            )
        )
        mock_svc.embed.assert_called_once()
        call_args = mock_svc.embed.call_args[0][0]
        assert "long colleagues" in call_args[0]

    def test_embedding_skipped_on_relation_create_empty_description(self) -> None:
        actor, mock_svc = _actor_with_mock_embed()
        _seed_entities(actor)
        mock_svc.reset_mock()
        actor.update_graph(
            ManageGraph(
                create_relations=[
                    RelationCreate(from_entity="Alice", to_entity="Bob", relation_type="knows")
                ]
            )
        )
        mock_svc.embed.assert_not_called()


class TestEmbeddingOnUpdate:
    """AC-5: re-embedding on description change."""

    def test_embedding_updated_on_description_change(self) -> None:
        actor, mock_svc = _actor_with_mock_embed()
        actor.update_graph(
            ManageGraph(
                create_entities=[EntityCreate(name="Alice", entity_type="Person", description="Eng")]
            )
        )
        embed_count_before = mock_svc.embed.call_count
        actor.update_graph(
            ManageGraph(update_entities=[EntityUpdate(name="Alice", description="Senior Eng")])
        )
        assert mock_svc.embed.call_count == embed_count_before + 1
        # Verify the new description is embedded, not the old one
        last_call = mock_svc.embed.call_args[0][0]
        assert "Senior Eng" in last_call[0]

    def test_no_re_embedding_when_description_unchanged(self) -> None:
        actor, mock_svc = _actor_with_mock_embed()
        actor.update_graph(
            ManageGraph(
                create_entities=[EntityCreate(name="Alice", entity_type="Person", description="Eng")]
            )
        )
        embed_count_before = mock_svc.embed.call_count
        # Update entity_type only — no description change
        actor.update_graph(
            ManageGraph(update_entities=[EntityUpdate(name="Alice", entity_type="Engineer")])
        )
        assert mock_svc.embed.call_count == embed_count_before


class TestEmbeddingOnDelete:
    """AC-6: embedding removed when entities/relations are deleted."""

    def test_entity_embedding_removed_on_delete(self) -> None:
        actor, mock_svc = _actor_with_mock_embed()
        actor.update_graph(
            ManageGraph(
                create_entities=[EntityCreate(name="Alice", entity_type="Person", description="Eng")]
            )
        )
        # Check an embedding was stored
        assert len(actor._vector_index._entries) == 1

        actor.update_graph(ManageGraph(delete_entities=["Alice"]))
        assert len(actor._vector_index._entries) == 0

    def test_relation_embedding_removed_on_delete(self) -> None:
        actor, mock_svc = _actor_with_mock_embed()
        _seed_entities(actor)
        actor.update_graph(
            ManageGraph(
                create_relations=[
                    RelationCreate(
                        from_entity="Alice",
                        to_entity="Bob",
                        relation_type="knows",
                        description="old friends",
                    )
                ]
            )
        )
        # 2 entity embeddings + 1 relation embedding
        assert len(actor._vector_index._entries) == 3

        actor.update_graph(
            ManageGraph(
                delete_relations=[
                    RelationDelete(from_entity="Alice", to_entity="Bob", relation_type="knows")
                ]
            )
        )
        assert len(actor._vector_index._entries) == 2


class TestEmbeddingGracefulDegradation:
    """AC-7: embedding failures do not block CRUD operations."""

    def test_embedding_failure_does_not_block_entity_create(self) -> None:
        actor, mock_svc = _actor_with_mock_embed()
        mock_svc.embed.side_effect = RuntimeError("API key invalid")
        result = actor.update_graph(
            ManageGraph(
                create_entities=[EntityCreate(name="Alice", entity_type="Person", description="Eng")]
            )
        )
        assert result == "Done"
        assert len(actor.get_graph().entities) == 1

    def test_embedding_failure_does_not_block_relation_create(self) -> None:
        actor, mock_svc = _actor_with_mock_embed()
        _seed_entities(actor)
        mock_svc.embed.side_effect = RuntimeError("API timeout")
        result = actor.update_graph(
            ManageGraph(
                create_relations=[
                    RelationCreate(
                        from_entity="Alice",
                        to_entity="Bob",
                        relation_type="knows",
                        description="desc",
                    )
                ]
            )
        )
        assert result == "Done"
        assert len(actor.get_graph().relations) == 1
