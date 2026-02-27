"""Tests for PlanActor error handling and update operations."""

from __future__ import annotations

import uuid

from akgentic.core.actor_address import ActorAddress

from akgentic.tool.planning.planning_actor import (
    PlanActor,
    PlanItemCreate,
    PlanItemUpdate,
    UpdatePlan,
)


class MockActorAddress(ActorAddress):
    """Mock ActorAddress for testing."""

    def __init__(self, name: str = "test-agent", role: str = "test-role"):
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

    def send(self, recipient, message):
        pass

    def is_alive(self) -> bool:
        return True

    def handle_user_message(self) -> bool:
        return False

    def serialize(self):
        return {"name": self._name, "role": self._role, "agent_id": str(self._agent_id)}

    def __repr__(self) -> str:
        return f"MockActorAddress(name={self._name}, role={self._role}, agent_id={self._agent_id})"


def test_update_item_success() -> None:
    """Test successful item update returns None."""
    actor = PlanActor()
    actor.on_start()

    # Create an item
    create_req = PlanItemCreate(
        id=1, status="pending", description="Task 1", owner="Alice", dependencies=[]
    )
    actor_addr = MockActorAddress("test-agent")
    actor._create_item(create_req, actor_addr)

    # Update the item
    update_req = PlanItemUpdate(id=1, status="started", description="Updated Task 1")
    result = actor._update_item(update_req)

    # Should return None on success
    assert result is None

    # Verify the update was applied
    items = actor.get_planning()
    assert len(items) == 1
    assert items[0].status == "started"
    assert items[0].description == "Updated Task 1"


def test_update_item_not_found() -> None:
    """Test updating non-existent item returns error message."""
    actor = PlanActor()
    actor.on_start()

    # Try to update non-existent item
    update_req = PlanItemUpdate(id=999, status="started")
    result = actor._update_item(update_req)

    # Should return error message
    assert result is not None
    assert "Update error" in result
    assert "999" in result
    assert "found" in result.lower()


def test_update_planning_with_update_errors() -> None:
    """Test update_planning collects errors from failed updates."""
    actor = PlanActor()
    actor.on_start()
    actor_addr = MockActorAddress("test-agent")

    # Create one item
    create_req = PlanItemCreate(
        id=1, status="pending", description="Task 1", owner="Alice", dependencies=[]
    )
    actor._create_item(create_req, actor_addr)

    # Try to update both existing and non-existent items
    update_plan = UpdatePlan(
        create_items=[],
        update_items=[
            PlanItemUpdate(id=1, status="started"),  # Should succeed
            PlanItemUpdate(id=999, status="completed"),  # Should fail
        ],
        delete_items=[],
    )

    result = actor.update_planning(update_plan, actor_addr)

    # Should report errors
    assert "Done with errors" in result
    assert "Update error" in result
    assert "999" in result


def test_update_planning_delete_not_found() -> None:
    """Test deleting non-existent item returns error."""
    actor = PlanActor()
    actor.on_start()
    actor_addr = MockActorAddress("test-agent")

    # Try to delete non-existent item
    update_plan = UpdatePlan(
        create_items=[],
        update_items=[],
        delete_items=[999],
    )

    result = actor.update_planning(update_plan, actor_addr)

    # Should report error
    assert "Done with errors" in result
    assert "Delete error" in result
    assert "999" in result
    assert "found" in result.lower()


def test_update_planning_delete_success() -> None:
    """Test successful deletion."""
    actor = PlanActor()
    actor.on_start()
    actor_addr = MockActorAddress("test-agent")

    # Create an item
    create_req = PlanItemCreate(
        id=1, status="pending", description="Task 1", owner="Alice", dependencies=[]
    )
    actor._create_item(create_req, actor_addr)

    # Delete the item
    update_plan = UpdatePlan(
        create_items=[],
        update_items=[],
        delete_items=[1],
    )

    result = actor.update_planning(update_plan, actor_addr)

    # Should succeed without errors
    assert result == "Done"
    assert len(actor.get_planning()) == 0


def test_update_planning_mixed_operations_with_errors() -> None:
    """Test mixed create/update/delete operations with some errors."""
    actor = PlanActor()
    actor.on_start()
    actor_addr = MockActorAddress("test-agent")

    # Create initial item
    create_req = PlanItemCreate(
        id=1, status="pending", description="Task 1", owner="Alice", dependencies=[]
    )
    actor._create_item(create_req, actor_addr)

    # Mixed operations with some failures
    update_plan = UpdatePlan(
        create_items=[
            PlanItemCreate(
                id=2, status="pending", description="Task 2", owner="Bob", dependencies=[]
            )
        ],
        update_items=[
            PlanItemUpdate(id=1, status="completed"),  # Should succeed
            PlanItemUpdate(id=999, status="started"),  # Should fail
        ],
        delete_items=[1, 888],  # First succeeds, second fails
    )

    result = actor.update_planning(update_plan, actor_addr)

    # Should report multiple errors
    assert "Done with errors" in result
    assert "Update error" in result or "Delete error" in result

    # Verify operations that should succeed
    items = actor.get_planning()
    assert len(items) == 1  # Item 2 created, item 1 deleted
    assert items[0].id == 2
    assert items[0].description == "Task 2"


def test_update_planning_all_success() -> None:
    """Test update_planning returns 'Done' when no errors."""
    actor = PlanActor()
    actor.on_start()
    actor_addr = MockActorAddress("test-agent")

    # Create initial item
    create_req = PlanItemCreate(
        id=1, status="pending", description="Task 1", owner="Alice", dependencies=[]
    )
    actor._create_item(create_req, actor_addr)

    # All operations should succeed
    update_plan = UpdatePlan(
        create_items=[
            PlanItemCreate(
                id=2, status="pending", description="Task 2", owner="Bob", dependencies=[]
            )
        ],
        update_items=[PlanItemUpdate(id=1, status="completed")],
        delete_items=[],
    )

    result = actor.update_planning(update_plan, actor_addr)

    # Should return clean "Done"
    assert result == "Done"

    # Verify final state
    items = actor.get_planning()
    assert len(items) == 2
    assert items[0].status == "completed"
    assert items[1].id == 2
