"""What this slice declares: ``WorkspaceAttached``, and the host that no longer exists.

**Renamed from ``test_workspace_host.py`` by story 52-6, and half of it deleted.**
``TestWorkspaceHostIsItsOwnConcreteClass`` pinned the shape of a ``WorkspaceHost``
that had no methods of its own — a host class that grew one would have been a
second place for core's get-or-create to drift from. The class is deleted, so
those two specs guard nothing; what replaces them is
:class:`TestTheHostIsGoneFromTheModuleTree`, which asserts the **absence** as an
executable fact about the package's surface rather than leaving it to prose.

``WorkspaceAttached`` is untouched by any of that and was never about hosting:
it is the domain event a team's stream carries for every bind, emitted by the
card since 52-5. It is pinned by shape because a payload that moved module, grew
a field or stopped being frozen would break every persisted stream and the
frontend's fold without a single behavioural spec in this package going red.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import uuid
from pathlib import Path

import pydantic
import pytest
from akgentic.core.messages.message import Message

import akgentic.tool.workspace as ws
from akgentic.tool.workspace import event as event_module
from akgentic.tool.workspace.event import WorkspaceAttached


class TestTheHostIsGoneFromTheModuleTree:
    """AC 1. Neither the module nor either name resolves, in this process or any other.

    An import assertion rather than a source grep: ``__init__`` re-exports by
    name, so a module left on disk and dropped from ``__all__`` would still be
    importable and still be found by the deserializer's ``import_module`` +
    ``getattr``, which is exactly how a retired class comes back to life.
    """

    def test_the_host_module_does_not_import(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("akgentic.tool.workspace.host")

    @pytest.mark.parametrize("name", ["WorkspaceHost", "workspace_host_address"])
    def test_neither_name_is_an_attribute_of_the_package(self, name: str) -> None:
        assert not hasattr(ws, name)
        assert name not in ws.__all__

    def test_the_state_class_and_the_tick_went_with_it(self) -> None:
        """``WorkspaceState`` and ``SweepTick`` existed only for the hosted lifetime.

        Named here rather than in a file of their own because they are the same
        deletion: the state was the thing a host's store carried, and the tick
        was the thing that reaped a tree no team owned.
        """
        for name in ("WorkspaceState", "SweepTick"):
            assert not hasattr(ws, name)
            assert name not in ws.__all__
        from akgentic.tool.workspace import models

        assert not hasattr(models, "WorkspaceState")
        assert not hasattr(models, "SweepTick")


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
