"""What this slice declares: ``WorkspaceHost`` and ``WorkspaceAttached`` (story 51-2).

Two declarations and nothing behind them, which is exactly why each is pinned by
its shape. A host class that grew a method would be a second place for the
get-or-create to drift from core's; a payload that moved module, grew a field or
stopped being frozen would break every persisted stream and the frontend's fold
without a single behavioural spec in this package going red.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import uuid
from collections.abc import Generator
from pathlib import Path
from types import FunctionType

import pydantic
import pytest
from akgentic.core import ActorRegistry
from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.agent_config import BaseConfig
from akgentic.core.messages.message import Message
from akgentic.core.resource_host import ResourceHost

from akgentic.tool.workspace import event as event_module
from akgentic.tool.workspace.event import WorkspaceAttached
from akgentic.tool.workspace.host import WorkspaceHost


@pytest.fixture
def system() -> Generator[ActorSystem, None, None]:
    """A real actor system, with every actor stopped afterwards — both hosts included."""
    actor_system = ActorSystem()
    try:
        yield actor_system
    finally:
        actor_system.shutdown(timeout=10)
        ActorRegistry.stop_all()


class TestWorkspaceHostIsItsOwnConcreteClass:
    """One host per concrete class per process, found by its own class and no other."""

    def test_each_host_is_found_by_its_own_class_alone(self, system: ActorSystem) -> None:
        """The base host running beside it is what makes the exact lookup observable.

        With a workspace host alone, a subclass-inclusive lookup and an exact one
        give the same answer, and the spec could not tell them apart.
        """
        assert issubclass(WorkspaceHost, ResourceHost)
        assert WorkspaceHost is not ResourceHost

        base = system.createActor(
            ResourceHost, config=BaseConfig(name="#ResourceHost", role="ResourceHost")
        )
        workspace = system.createActor(
            WorkspaceHost, config=BaseConfig(name="#WorkspaceHost", role="ResourceHost")
        )

        assert [a.agent_id for a in ActorSystem.find_by_class(WorkspaceHost)] == [
            workspace.agent_id
        ]
        assert [a.agent_id for a in ActorSystem.find_by_class(ResourceHost)] == [base.agent_id]

    def test_it_declares_no_method_and_no_attribute_of_its_own(self) -> None:
        """Everything it does is core's; it carries its identity and a docstring.

        **Not** "no non-dunder key": pykka's ``Actor`` is an ``abc.ABC``, so
        ``ABCMeta`` puts ``_abc_impl`` in every subclass's own dict. A public
        name or any function is what would be a second behaviour.
        """
        own = vars(WorkspaceHost)
        assert WorkspaceHost.__doc__
        assert [name for name, value in own.items() if isinstance(value, FunctionType)] == []
        assert [name for name in own if not name.startswith("_")] == []


class TestWorkspaceAttachedIsAFrozenTopLevelPayload:
    """The wire contract, pinned by shape: module, name, fields, immutability."""

    def test_it_is_a_frozen_dataclass_with_exactly_two_fields(self) -> None:
        assert dataclasses.is_dataclass(WorkspaceAttached)
        assert [field.name for field in dataclasses.fields(WorkspaceAttached)] == [
            "agent_id",
            "workspace_path",
        ]
        event = WorkspaceAttached(agent_id=uuid.uuid4(), workspace_path="u-alice/notes")
        with pytest.raises(dataclasses.FrozenInstanceError):
            event.workspace_path = "u-bob/notes"  # type: ignore[misc]

    def test_it_lives_at_the_top_level_of_the_module_the_wire_names(self) -> None:
        """The serializer persists ``module.ClassName``; replay resolves it with no alias."""
        assert WorkspaceAttached.__module__ == "akgentic.tool.workspace.event"
        assert WorkspaceAttached.__qualname__ == "WorkspaceAttached"

    def test_it_is_a_payload_and_not_an_envelope(self) -> None:
        """It rides inside core's ``EventMessage``; it is neither a message nor a model."""
        assert not issubclass(WorkspaceAttached, Message)
        assert not issubclass(WorkspaceAttached, pydantic.BaseModel)

    def test_its_module_imports_the_standard_library_and_nothing_else(self) -> None:
        """A leaf: resolving the payload class never drags in the card or the actor.

        Read from the module's own import statements rather than from
        ``sys.modules``: importing ``akgentic.tool.workspace.event`` runs the
        package's ``__init__`` first, which imports everything, so a runtime
        check could never be red. Whole-set equality, so a new import of any
        kind — the card, the actor, ``typing`` under a guard — fails it.
        """
        tree = ast.parse(Path(inspect.getfile(event_module)).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, "a relative import in the payload module"
                assert node.module is not None
                imported.add(node.module)
        assert imported == {"uuid", "dataclasses"}
