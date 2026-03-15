"""Tests for PlanActor error handling and update operations."""

from __future__ import annotations

import pytest

from akgentic.tool.errors import RetriableError
from akgentic.tool.planning.planning_actor import (
    PlanActor,
    TaskCreate,
    TaskUpdate,
    UpdatePlan,
)
from tests.conftest import MockActorAddress


def test_update_item_success() -> None:
    """Test successful item update returns None."""
    actor = PlanActor()
    actor.on_start()

    # Create an item
    create_req = TaskCreate(
        id=1, status="pending", description="Task 1", owner="Alice", dependencies=[]
    )
    actor_addr = MockActorAddress("test-agent")
    actor._create_task(create_req, actor_addr)

    # Update the item
    update_req = TaskUpdate(id=1, status="started", description="Updated Task 1")
    result = actor._update_task(update_req)

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
    update_req = TaskUpdate(id=999, status="started")
    result = actor._update_task(update_req)

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
    create_req = TaskCreate(
        id=1, status="pending", description="Task 1", owner="Alice", dependencies=[]
    )
    actor._create_task(create_req, actor_addr)

    # Try to update both existing and non-existent items
    update_plan = UpdatePlan(
        create_tasks=[],
        update_tasks=[
            TaskUpdate(id=1, status="started"),  # Should succeed
            TaskUpdate(id=999, status="completed"),  # Should fail
        ],
        delete_tasks=[],
    )

    with pytest.raises(RetriableError, match="Update error.*999"):
        actor.update_planning(update_plan, actor_addr)


def test_update_planning_delete_not_found() -> None:
    """Test deleting non-existent item raises RetriableError."""
    actor = PlanActor()
    actor.on_start()
    actor_addr = MockActorAddress("test-agent")

    # Try to delete non-existent item
    update_plan = UpdatePlan(
        create_tasks=[],
        update_tasks=[],
        delete_tasks=[999],
    )

    with pytest.raises(RetriableError, match="Delete error.*999"):
        actor.update_planning(update_plan, actor_addr)


def test_update_planning_delete_success() -> None:
    """Test successful deletion."""
    actor = PlanActor()
    actor.on_start()
    actor_addr = MockActorAddress("test-agent")

    # Create an item
    create_req = TaskCreate(
        id=1, status="pending", description="Task 1", owner="Alice", dependencies=[]
    )
    actor._create_task(create_req, actor_addr)

    # Delete the item
    update_plan = UpdatePlan(
        create_tasks=[],
        update_tasks=[],
        delete_tasks=[1],
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
    create_req = TaskCreate(
        id=1, status="pending", description="Task 1", owner="Alice", dependencies=[]
    )
    actor._create_task(create_req, actor_addr)

    # Mixed operations with some failures
    update_plan = UpdatePlan(
        create_tasks=[
            TaskCreate(id=2, status="pending", description="Task 2", owner="Bob", dependencies=[])
        ],
        update_tasks=[
            TaskUpdate(id=1, status="completed"),  # Should succeed
            TaskUpdate(id=999, status="started"),  # Should fail
        ],
        delete_tasks=[1, 888],  # First succeeds, second fails
    )

    with pytest.raises(RetriableError) as exc_info:
        actor.update_planning(update_plan, actor_addr)

    # Should report multiple errors
    error_msg = str(exc_info.value)
    assert "Update error" in error_msg or "Delete error" in error_msg

    # Verify operations that should succeed (committed before the raise)
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
    create_req = TaskCreate(
        id=1, status="pending", description="Task 1", owner="Alice", dependencies=[]
    )
    actor._create_task(create_req, actor_addr)

    # All operations should succeed
    update_plan = UpdatePlan(
        create_tasks=[
            TaskCreate(id=2, status="pending", description="Task 2", owner="Bob", dependencies=[])
        ],
        update_tasks=[TaskUpdate(id=1, status="completed")],
        delete_tasks=[],
    )

    result = actor.update_planning(update_plan, actor_addr)

    # Should return clean "Done"
    assert result == "Done"

    # Verify final state
    items = actor.get_planning()
    assert len(items) == 2
    assert items[0].status == "completed"
    assert items[1].id == 2
