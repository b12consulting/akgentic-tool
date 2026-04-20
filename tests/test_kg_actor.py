"""Tests for KnowledgeGraphActor CRUD operations.

Covers: all CRUD ops, cascade deletion, error collection, partial updates,
state change notification, deduplication logic, singleton constants (Task 2.9).
Also covers embedding wiring via VectorStoreActor proxy: entity/relation
embedding on create, re-embedding on description update, removal on delete,
graceful degradation on API failure.
Also covers vector search and hybrid search.

Pattern: Instantiate KnowledgeGraphActor() directly, call on_start(),
test methods.  The VectorStoreActor proxy is mocked.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from akgentic.core.actor_address import ActorAddress

from akgentic.tool.errors import RetriableError
from akgentic.tool.event import ToolStateEvent
from akgentic.tool.knowledge_graph.kg_actor import (
    KG_ACTOR_NAME,
    KG_ACTOR_ROLE,
    KG_COLLECTION,
    KnowledgeGraphActor,
    KnowledgeGraphConfig,
)
from akgentic.tool.knowledge_graph.models import (
    EntityCreate,
    EntityUpdate,
    GraphView,
    KnowledgeGraphStateEvent,
    ManageGraph,
    RelationCreate,
    RelationDelete,
    SearchQuery,
)
from akgentic.tool.vector_store.protocol import CollectionStatus
from akgentic.tool.vector_store.protocol import SearchHit as VsSearchHit
from akgentic.tool.vector_store.protocol import SearchResult as VsSearchResult

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
    """Create and initialize a KnowledgeGraphActor for testing.

    The VectorStoreActor proxy is not available (degraded mode) -- suitable
    for pure CRUD tests that do not need embedding.
    """
    from akgentic.tool.knowledge_graph.models import KnowledgeGraphState

    actor = KnowledgeGraphActor()
    # Bypass _acquire_vs_proxy (no orchestrator in unit tests)
    actor.state = KnowledgeGraphState()
    actor.state.observer(actor)
    actor._vs_proxy = None
    actor._state_event_seq = 0
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
# KnowledgeGraphConfig (AC-1)
# ---------------------------------------------------------------------------


class TestKnowledgeGraphConfig:
    """AC-8: KnowledgeGraphConfig is an empty BaseConfig subclass."""

    def test_config_is_base_config_subclass(self) -> None:
        from akgentic.core.agent_config import BaseConfig

        assert issubclass(KnowledgeGraphConfig, BaseConfig)

    def test_config_creates_with_name_and_role(self) -> None:
        cfg = KnowledgeGraphConfig(name="test", role="ToolActor")
        assert cfg.name == "test"
        assert cfg.role == "ToolActor"

    def test_config_has_no_embedding_fields(self) -> None:
        cfg = KnowledgeGraphConfig(name="test", role="ToolActor")
        assert not hasattr(cfg, "embedding_model")
        assert not hasattr(cfg, "embedding_provider")


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


def _make_mock_vs_proxy() -> MagicMock:
    """Create a mock VectorStoreActor proxy with default return values."""
    proxy = MagicMock()
    proxy.embed.return_value = [_FAKE_VECTOR]
    proxy.add.return_value = None
    proxy.remove.return_value = None
    proxy.create_collection.return_value = None
    proxy.search.return_value = VsSearchResult(
        hits=[], status=CollectionStatus.READY, indexing_pending=0
    )
    return proxy


def _actor_with_mock_embed() -> tuple[KnowledgeGraphActor, MagicMock]:
    """Return (actor, mock_vs_proxy) with a pre-wired mock VectorStoreActor proxy."""
    actor = _actor()
    mock_proxy = _make_mock_vs_proxy()
    actor._vs_proxy = mock_proxy
    return actor, mock_proxy


class TestEmbeddingOnCreate:
    """AC-4, AC-6: embedding called when entities/relations are created via vs_proxy."""

    def test_embedding_called_on_entity_create(self) -> None:
        actor, mock_proxy = _actor_with_mock_embed()
        actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(name="Alice", entity_type="Person", description="Eng")
                ]
            )
        )
        mock_proxy.embed.assert_called_once()
        call_args = mock_proxy.embed.call_args[0][0]
        assert "Alice" in call_args[0]
        assert "Eng" in call_args[0]

    def test_add_called_on_entity_create(self) -> None:
        actor, mock_proxy = _actor_with_mock_embed()
        actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(name="Alice", entity_type="Person", description="Eng")
                ]
            )
        )
        mock_proxy.add.assert_called_once()
        call_args = mock_proxy.add.call_args
        assert call_args[0][0] == KG_COLLECTION

    def test_embedding_called_on_relation_create_with_description(self) -> None:
        actor, mock_proxy = _actor_with_mock_embed()
        _seed_entities(actor)
        mock_proxy.reset_mock()
        mock_proxy.embed.return_value = [_FAKE_VECTOR]
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
        mock_proxy.embed.assert_called_once()
        call_args = mock_proxy.embed.call_args[0][0]
        assert "long colleagues" in call_args[0]

    def test_embedding_skipped_on_relation_create_empty_description(self) -> None:
        actor, mock_proxy = _actor_with_mock_embed()
        _seed_entities(actor)
        mock_proxy.reset_mock()
        actor.update_graph(
            ManageGraph(
                create_relations=[
                    RelationCreate(from_entity="Alice", to_entity="Bob", relation_type="knows")
                ]
            )
        )
        mock_proxy.embed.assert_not_called()


class TestEmbeddingOnUpdate:
    """AC-5: re-embedding on description change via vs_proxy."""

    def test_embedding_updated_on_description_change(self) -> None:
        actor, mock_proxy = _actor_with_mock_embed()
        actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(name="Alice", entity_type="Person", description="Eng")
                ]
            )
        )
        embed_count_before = mock_proxy.embed.call_count
        actor.update_graph(
            ManageGraph(update_entities=[EntityUpdate(name="Alice", description="Senior Eng")])
        )
        assert mock_proxy.embed.call_count == embed_count_before + 1
        # Verify the new description is embedded, not the old one
        last_call = mock_proxy.embed.call_args[0][0]
        assert "Senior Eng" in last_call[0]

    def test_remove_called_before_re_embedding(self) -> None:
        actor, mock_proxy = _actor_with_mock_embed()
        actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(name="Alice", entity_type="Person", description="Eng")
                ]
            )
        )
        mock_proxy.remove.reset_mock()
        actor.update_graph(
            ManageGraph(update_entities=[EntityUpdate(name="Alice", description="Senior Eng")])
        )
        mock_proxy.remove.assert_called_once()
        call_args = mock_proxy.remove.call_args
        assert call_args[0][0] == KG_COLLECTION

    def test_no_re_embedding_when_description_unchanged(self) -> None:
        actor, mock_proxy = _actor_with_mock_embed()
        actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(name="Alice", entity_type="Person", description="Eng")
                ]
            )
        )
        embed_count_before = mock_proxy.embed.call_count
        # Update entity_type only — no description change
        actor.update_graph(
            ManageGraph(update_entities=[EntityUpdate(name="Alice", entity_type="Engineer")])
        )
        assert mock_proxy.embed.call_count == embed_count_before


class TestEmbeddingOnDelete:
    """AC-7: embedding removed when entities/relations are deleted via vs_proxy."""

    def test_entity_embedding_removed_on_delete(self) -> None:
        actor, mock_proxy = _actor_with_mock_embed()
        actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(name="Alice", entity_type="Person", description="Eng")
                ]
            )
        )
        mock_proxy.remove.reset_mock()
        actor.update_graph(ManageGraph(delete_entities=["Alice"]))
        mock_proxy.remove.assert_called_once()
        call_args = mock_proxy.remove.call_args
        assert call_args[0][0] == KG_COLLECTION
        assert len(call_args[0][1]) == 1  # 1 entity ref_id

    def test_relation_embedding_removed_on_delete(self) -> None:
        actor, mock_proxy = _actor_with_mock_embed()
        _seed_entities(actor)
        mock_proxy.remove.reset_mock()
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
        mock_proxy.remove.reset_mock()
        actor.update_graph(
            ManageGraph(
                delete_relations=[
                    RelationDelete(from_entity="Alice", to_entity="Bob", relation_type="knows")
                ]
            )
        )
        mock_proxy.remove.assert_called_once()
        call_args = mock_proxy.remove.call_args
        assert call_args[0][0] == KG_COLLECTION
        assert len(call_args[0][1]) == 1  # 1 relation ref_id


class TestEmbeddingGracefulDegradation:
    """Embedding failures do not block CRUD operations."""

    def test_embedding_failure_does_not_block_entity_create(self) -> None:
        actor, mock_proxy = _actor_with_mock_embed()
        mock_proxy.embed.side_effect = RuntimeError("API key invalid")
        result = actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(name="Alice", entity_type="Person", description="Eng")
                ]
            )
        )
        assert result == "Done"
        assert len(actor.get_graph().entities) == 1

    def test_embedding_failure_does_not_block_relation_create(self) -> None:
        actor, mock_proxy = _actor_with_mock_embed()
        _seed_entities(actor)
        mock_proxy.embed.side_effect = RuntimeError("API timeout")
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

    def test_degraded_mode_when_no_proxy(self) -> None:
        """When _vs_proxy is None, CRUD works but no embedding calls are made."""
        actor = _actor()  # _vs_proxy is None
        result = actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(name="Alice", entity_type="Person", description="Eng")
                ]
            )
        )
        assert result == "Done"
        assert len(actor.get_graph().entities) == 1


# ---------------------------------------------------------------------------
# Vector search (Story 2.2 Task 1, ACs 1, 3, 5)
# ---------------------------------------------------------------------------


class TestVectorSearch:
    """AC-5: vector search returns ranked results or empty on no embeddings."""

    def test_vector_search_returns_top_k_ranked_by_score(self) -> None:
        actor, mock_proxy = _actor_with_mock_embed()
        actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(name="Alice", entity_type="Person", description="Engineer"),
                    EntityCreate(name="Bob", entity_type="Person", description="Designer"),
                ]
            )
        )
        entity_ids = [str(e.id) for e in actor.get_graph().entities]
        mock_proxy.embed.return_value = [_FAKE_VECTOR]
        mock_proxy.search.return_value = VsSearchResult(
            hits=[
                VsSearchHit(ref_type="entity", ref_id=entity_ids[0], text="", score=0.9),
                VsSearchHit(ref_type="entity", ref_id=entity_ids[1], text="", score=0.5),
            ],
            status=CollectionStatus.READY,
            indexing_pending=0,
        )
        result = actor.search(SearchQuery(query="engineer", top_k=2, mode="vector"))
        assert len(result.hits) == 2
        assert result.hits[0].score == 0.9
        assert result.hits[1].score == 0.5
        assert result.hits[0].entity is not None

    def test_vector_search_returns_empty_when_no_proxy(self) -> None:
        actor = _actor()  # _vs_proxy is None
        result = actor.search(SearchQuery(query="anything", top_k=5, mode="vector"))
        assert result.hits == []

    def test_vector_search_returns_empty_when_embed_call_fails(self) -> None:
        """Verify graceful empty result when embed() raises during vector search."""
        actor, mock_proxy = _actor_with_mock_embed()
        mock_proxy.embed.side_effect = RuntimeError("API quota exceeded")
        result = actor.search(SearchQuery(query="Alice", top_k=5, mode="vector"))
        assert result.hits == []

    def test_vector_search_returns_empty_when_embed_returns_empty(self) -> None:
        """VectorStoreActor returns [] on embed failure -- should result in empty search."""
        actor, mock_proxy = _actor_with_mock_embed()
        mock_proxy.embed.return_value = []
        result = actor.search(SearchQuery(query="Alice", top_k=5, mode="vector"))
        assert result.hits == []

    def test_vector_search_returns_empty_when_search_call_fails(self) -> None:
        """Graceful empty result when vs_proxy.search() raises during vector search."""
        actor, mock_proxy = _actor_with_mock_embed()
        mock_proxy.embed.return_value = [_FAKE_VECTOR]
        mock_proxy.search.side_effect = RuntimeError("Connection refused")
        result = actor.search(SearchQuery(query="Alice", top_k=5, mode="vector"))
        assert result.hits == []


# ---------------------------------------------------------------------------
# Hybrid search (Story 2.2 Task 2, ACs 2, 4)
# ---------------------------------------------------------------------------


class TestHybridSearch:
    """Hybrid merges keyword+vector, falls back to keyword when no embeddings."""

    def _setup_actor_with_known_vectors(
        self,
    ) -> tuple[KnowledgeGraphActor, MagicMock]:
        """Actor with Alice + Bob; mock vs_proxy."""
        actor, mock_proxy = _actor_with_mock_embed()
        actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(
                        name="Alice", entity_type="Person", description="Engineer login"
                    ),
                    EntityCreate(
                        name="Bob", entity_type="Person", description="Designer security"
                    ),
                ]
            )
        )
        return actor, mock_proxy

    def test_hybrid_falls_back_to_keyword_when_no_proxy(self) -> None:
        actor = _actor()  # _vs_proxy is None
        actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(name="Alice", entity_type="Person", description="Engineer")
                ]
            )
        )
        result = actor.search(SearchQuery(query="Alice", top_k=5, mode="hybrid"))
        assert len(result.hits) == 1
        assert result.hits[0].entity is not None
        assert result.hits[0].entity.name == "Alice"

    def test_hybrid_deduplicates_by_ref_id(self) -> None:
        actor, mock_proxy = self._setup_actor_with_known_vectors()
        alice_id = str(
            next(e for e in actor.get_graph().entities if e.name == "Alice").id
        )
        mock_proxy.embed.return_value = [_FAKE_VECTOR]
        mock_proxy.search.return_value = VsSearchResult(
            hits=[VsSearchHit(ref_type="entity", ref_id=alice_id, text="", score=0.8)],
            status=CollectionStatus.READY,
            indexing_pending=0,
        )
        result = actor.search(SearchQuery(query="Alice", top_k=10, mode="hybrid"))
        alice_hits = [h for h in result.hits if h.ref_id == alice_id]
        assert len(alice_hits) == 1, "Alice should appear exactly once (deduplication)"

    def test_hybrid_combined_hits_rank_higher_than_vector_only(self) -> None:
        actor, mock_proxy = self._setup_actor_with_known_vectors()
        entities = actor.get_graph().entities
        alice_id = str(next(e for e in entities if e.name == "Alice").id)
        bob_id = str(next(e for e in entities if e.name == "Bob").id)
        mock_proxy.embed.return_value = [_FAKE_VECTOR]
        mock_proxy.search.return_value = VsSearchResult(
            hits=[
                VsSearchHit(ref_type="entity", ref_id=alice_id, text="", score=0.8),
                VsSearchHit(ref_type="entity", ref_id=bob_id, text="", score=0.9),
            ],
            status=CollectionStatus.READY,
            indexing_pending=0,
        )
        result = actor.search(SearchQuery(query="login", top_k=5, mode="hybrid"))
        ref_ids = [h.ref_id for h in result.hits]
        assert ref_ids[0] == alice_id, "Alice (vector+keyword) should rank above Bob (vector-only)"

    def test_hybrid_top_k_limits_results(self) -> None:
        actor, mock_proxy = _actor_with_mock_embed()
        actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(name=f"Entity{i}", entity_type="T", description=f"desc {i}")
                    for i in range(10)
                ]
            )
        )
        entity_ids = [str(e.id) for e in actor.get_graph().entities]
        mock_proxy.embed.return_value = [_FAKE_VECTOR]
        mock_proxy.search.return_value = VsSearchResult(
            hits=[
                VsSearchHit(ref_type="entity", ref_id=eid, text="", score=0.5)
                for eid in entity_ids
            ],
            status=CollectionStatus.READY,
            indexing_pending=0,
        )
        result = actor.search(SearchQuery(query="desc", top_k=3, mode="hybrid"))
        assert len(result.hits) <= 3


# ---------------------------------------------------------------------------
# Search expansion: include_neighbors, include_edges, find_paths (Story 3.1, ACs 2-5)
# ---------------------------------------------------------------------------


def _seed_expansion_graph(actor: KnowledgeGraphActor) -> None:
    """Populate a test graph for search expansion tests.

    Graph topology:
        Alice --KNOWS--> Bob
        Alice --WORKS_AT--> Corp
        Bob --WORKS_AT--> Corp
        Corp --LOCATED_IN--> City
        Dave --KNOWS--> Eve   (disconnected island)
    """
    actor.update_graph(
        ManageGraph(
            create_entities=[
                EntityCreate(name="Alice", entity_type="Person", description="alice engineer"),
                EntityCreate(name="Bob", entity_type="Person", description="bob designer"),
                EntityCreate(name="Corp", entity_type="Organization", description="tech corporation"),
                EntityCreate(name="City", entity_type="Location", description="metro city"),
                EntityCreate(name="Dave", entity_type="Person", description="dave manager"),
                EntityCreate(name="Eve", entity_type="Person", description="eve analyst"),
            ],
            create_relations=[
                RelationCreate(from_entity="Alice", to_entity="Bob", relation_type="KNOWS"),
                RelationCreate(from_entity="Alice", to_entity="Corp", relation_type="WORKS_AT"),
                RelationCreate(from_entity="Bob", to_entity="Corp", relation_type="WORKS_AT"),
                RelationCreate(from_entity="Corp", to_entity="City", relation_type="LOCATED_IN"),
                RelationCreate(from_entity="Dave", to_entity="Eve", relation_type="KNOWS"),
            ],
        )
    )


class TestSearchIncludeNeighbors:
    """AC-2: include_neighbors=True returns 1-hop neighbors of entity hits."""

    def test_include_neighbors_returns_1hop_neighbors(self) -> None:
        actor = _actor()
        _seed_expansion_graph(actor)
        result = actor.search(SearchQuery(query="alice engineer", mode="keyword", include_neighbors=True))
        entity_hit_names = {h.entity.name for h in result.hits if h.entity}
        assert "Alice" in entity_hit_names
        # Alice's neighbors: Bob (KNOWS) and Corp (WORKS_AT)
        neighbor_names = {e.name for e in result.neighbors}
        assert "Bob" in neighbor_names
        assert "Corp" in neighbor_names
        # Neighbors should not include Alice herself
        assert "Alice" not in neighbor_names

    def test_include_neighbors_excludes_entities_in_hits(self) -> None:
        """Neighbors should not duplicate entities already in hits."""
        actor = _actor()
        _seed_expansion_graph(actor)
        # Search "alice" matches Alice (entity hit)
        result = actor.search(SearchQuery(query="alice", mode="keyword", include_neighbors=True))
        hit_names = {h.entity.name for h in result.hits if h.entity}
        neighbor_names = {e.name for e in result.neighbors}
        # No overlap between hits and neighbors
        assert not (hit_names & neighbor_names)

    def test_include_neighbors_no_entity_hits_returns_empty_neighbors(self) -> None:
        actor = _actor()
        _seed_expansion_graph(actor)
        # Search for something that only matches relations (by relation_type)
        result = actor.search(SearchQuery(query="zzz_no_match", mode="keyword", include_neighbors=True))
        assert result.neighbors == []

    def test_include_neighbors_false_returns_empty(self) -> None:
        """Default behavior: no neighbors expansion."""
        actor = _actor()
        _seed_expansion_graph(actor)
        result = actor.search(SearchQuery(query="alice", mode="keyword"))
        assert result.neighbors == []


class TestSearchIncludeEdges:
    """AC-3: include_edges=True returns all relations connected to entity hits."""

    def test_include_edges_returns_connected_relations(self) -> None:
        actor = _actor()
        _seed_expansion_graph(actor)
        result = actor.search(SearchQuery(query="alice engineer", mode="keyword", include_edges=True))
        entity_hit_names = {h.entity.name for h in result.hits if h.entity}
        assert "Alice" in entity_hit_names
        # Alice's relations: Alice→Bob (KNOWS), Alice→Corp (WORKS_AT)
        rel_types = {r.relation_type for r in result.connected_relations}
        assert "KNOWS" in rel_types
        assert "WORKS_AT" in rel_types

    def test_include_edges_deduplicates_relations(self) -> None:
        """When multiple entity hits share a relation, it appears only once."""
        actor = _actor()
        _seed_expansion_graph(actor)
        # Both Alice and Bob are connected via Corp's WORKS_AT relation
        result = actor.search(
            SearchQuery(query="designer", mode="keyword", include_edges=True)
        )
        # Find all relation IDs — should be unique
        rel_ids = [str(r.id) for r in result.connected_relations]
        assert len(rel_ids) == len(set(rel_ids))

    def test_include_edges_false_returns_empty(self) -> None:
        actor = _actor()
        _seed_expansion_graph(actor)
        result = actor.search(SearchQuery(query="alice", mode="keyword"))
        assert result.connected_relations == []


class TestSearchFindPaths:
    """AC-4, AC-5: find_paths computes BFS paths between top 5 entity hits."""

    def test_find_paths_with_5_entity_hits(self) -> None:
        """find_paths=True with 5+ entity hits finds paths between top 5."""
        actor = _actor()
        # Create 5 connected entities
        actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(name=f"Node{i}", entity_type="T", description="node target")
                    for i in range(5)
                ],
                create_relations=[
                    RelationCreate(from_entity=f"Node{i}", to_entity=f"Node{i+1}", relation_type="LINKS")
                    for i in range(4)
                ],
            )
        )
        result = actor.search(SearchQuery(query="node target", mode="keyword", find_paths=True))
        entity_hits = [h for h in result.hits if h.entity]
        assert len(entity_hits) == 5
        # Should find paths between connected nodes
        assert len(result.paths) > 0

    def test_find_paths_fewer_than_2_hits_returns_empty(self) -> None:
        """AC-5: fewer than 2 entity hits → empty paths."""
        actor = _actor()
        actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(name="OnlyOne", entity_type="T", description="unique item xyz123")
                ]
            )
        )
        result = actor.search(SearchQuery(query="unique item xyz123", mode="keyword", find_paths=True))
        entity_hits = [h for h in result.hits if h.entity]
        assert len(entity_hits) == 1
        assert result.paths == []

    def test_find_paths_max_10_pairs(self) -> None:
        """find_paths uses top 5 entities → max C(5,2) = 10 pairs."""
        actor = _actor()
        # Create 6 entities all matching but only top 5 used
        actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(name=f"N{i}", entity_type="T", description="path target")
                    for i in range(6)
                ],
                create_relations=[
                    RelationCreate(from_entity=f"N{i}", to_entity=f"N{i+1}", relation_type="LINKS")
                    for i in range(5)
                ],
            )
        )
        result = actor.search(SearchQuery(query="path target", mode="keyword", find_paths=True))
        # Max 10 paths (C(5,2) = 10 pairs)
        assert len(result.paths) <= 10

    def test_find_paths_no_path_between_disconnected_entities(self) -> None:
        """Disconnected entities: path not found → silently skipped."""
        actor = _actor()
        _seed_expansion_graph(actor)
        # Alice-Corp-Bob are connected; Dave-Eve are disconnected from them
        # Search for "person" matches Alice, Bob, Dave, Eve → paths between disconnected pairs skipped
        result = actor.search(SearchQuery(query="person", mode="keyword", find_paths=True))
        entity_hit_names = {h.entity.name for h in result.hits if h.entity}
        # We should have multiple entity hits but paths only between reachable pairs
        assert isinstance(result.paths, list)
        # Paths are returned (may be empty if none reachable) but no exception

    def test_find_paths_false_returns_empty(self) -> None:
        actor = _actor()
        _seed_expansion_graph(actor)
        result = actor.search(SearchQuery(query="alice", mode="keyword"))
        assert result.paths == []

    def test_find_paths_path_is_alternating_entity_relation_sequence(self) -> None:
        """Paths contain alternating [Entity, Relation, Entity, ...] sequences."""
        from akgentic.tool.knowledge_graph.models import Entity, Relation
        actor = _actor()
        actor.update_graph(
            ManageGraph(
                create_entities=[
                    EntityCreate(name="Start", entity_type="T", description="route node"),
                    EntityCreate(name="End", entity_type="T", description="route node"),
                ],
                create_relations=[
                    RelationCreate(from_entity="Start", to_entity="End", relation_type="CONNECTS"),
                ],
            )
        )
        result = actor.search(SearchQuery(query="route node", mode="keyword", find_paths=True))
        assert len(result.paths) == 1
        path = result.paths[0]
        # [Entity, Relation, Entity] for direct connection
        assert len(path) == 3
        assert isinstance(path[0], Entity)
        assert isinstance(path[1], Relation)
        assert isinstance(path[2], Entity)


# ---------------------------------------------------------------------------
# ToolStateEvent emission (AC-3)
# ---------------------------------------------------------------------------


class TestToolStateEventEmission:
    """Verify update_graph emits ToolStateEvent with KnowledgeGraphStateEvent payload."""

    def test_notify_event_called_on_non_empty_mutation(self) -> None:
        actor = _actor()
        with patch.object(actor, "notify_event") as mock_notify:
            actor.update_graph(
                ManageGraph(
                    create_entities=[
                        EntityCreate(name="Eve", entity_type="Person", description="Tester")
                    ]
                )
            )
            mock_notify.assert_called_once()
            event = mock_notify.call_args[0][0]
            assert isinstance(event, ToolStateEvent)
            assert isinstance(event.payload, KnowledgeGraphStateEvent)
            assert event.seq == 1
            assert len(event.payload.entities_added) == 1
            assert event.payload.entities_added[0].name == "Eve"

    def test_notify_event_not_called_on_empty_mutation(self) -> None:
        actor = _actor()
        with patch.object(actor, "notify_event") as mock_notify:
            actor.update_graph(ManageGraph())
            mock_notify.assert_not_called()

    def test_state_event_seq_increments(self) -> None:
        actor = _actor()
        with patch.object(actor, "notify_event"):
            actor.update_graph(
                ManageGraph(
                    create_entities=[
                        EntityCreate(name="A", entity_type="T", description="d1")
                    ]
                )
            )
            assert actor._state_event_seq == 1
            actor.update_graph(
                ManageGraph(
                    create_entities=[
                        EntityCreate(name="B", entity_type="T", description="d2")
                    ]
                )
            )
            assert actor._state_event_seq == 2
