"""The ``akgentic.tool.event`` compatibility façade, and the untouched package root.

``akgentic-tool`` ships on PyPI and a sibling package imports ``CommandsAnnouncedEvent``
from ``akgentic.tool.event``. The façade is what lets the split land without editing
that consumer, so these tests pin its three load-bearing properties: it warns on
access and not on import, it names the destination, and it resolves to the very same
object as the new module.

One symbol took a different exit. ``ToolStatePayload`` was not moved but **removed**, so
a class below pins the opposite behaviour: no warning, no resolution, on every path
the name ever had.

The three observer entries followed it out of the shim (2026-08-17): their Stable-tier
surface is the ``akgentic.tool`` package root, which never changed, and their
``event.py`` residence was an accident of the pre-split layout. The façade now carries
exactly the four event symbols — the ones that are load-bearing for persisted
``__model__`` markers and for the sibling ``CommandsAnnouncedEvent`` import.
"""

from __future__ import annotations

import importlib
import re
import warnings

import pytest

import akgentic.tool
import akgentic.tool.event as event_shim
from akgentic.tool.event import _MOVED

# Frozen before the split: the package-root API may not change, so a consumer importing
# from ``akgentic.tool`` needs no migration at all. The single exception is a symbol whose
# tier places it outside that promise and whose removal is authorised on that basis —
# ``ToolStatePayload`` left this list for that reason and is the only name that has. A
# move never justifies an edit here; only a removal decision does, and the list keeps
# guarding every remaining name.
# A brand-new export breaks no consumer, so a new tool card joins the list with its
# story (``NotificationTool``, epic 28). What the list still forbids is a name leaving
# or changing.
_EXPECTED_ROOT_ALL: list[str] = [
    "BaseToolParam",
    "ContextState",  # epic 31: the context-state contract joins the stable surface
    "ContextUpdater",  # epic 35: the context-update engine joins the stable surface
    "ToolCard",
    "ToolFactory",
    "ToolState",  # epic 35: the persistent tool-state slot joins the stable surface
    "CommandRegistry",
    "normalize_system_prompt_to_llm_context",  # epic 31
    "COMMAND",
    "LLM_CONTEXT",  # epic 31: the fourth expose channel
    "SYSTEM_PROMPT",
    "TOOL_CALL",
    "Channels",
    "RetriableError",
    "CommandNotRecognized",
    "ToolObserverGone",
    "ToolObserver",
    "ActorToolObserver",
    "ToolStateCarrier",  # epic 35: how tools reach the slot — beside the observers
    "TeamManagementToolObserver",
    "ToolStateEvent",
    "KnowledgeGraphStateEvent",
    "CommandArg",
    "CommandDescriptor",
    "CommandsAnnouncedEvent",
    "mcp",
    "planning",
    "sandbox",
    "search",
    "team",
    "workspace",
    "BwrapSandboxActor",
    "ExecTool",
    "MailboxTool",  # epic 34: the mailbox card joins the stable surface
    "MetadataTool",
    "NotificationTool",
    "SeatbeltSandboxActor",
    "SkillTool",
    "WorkspaceTool",
]

_VECTOR_SEARCH_EXTRAS = ["VectorEntry", "EmbeddingService", "VectorIndex"]


def _from_import(module: str, name: str) -> object:
    """Run ``from <module> import <name>`` exactly as a consumer would write it.

    Built at runtime rather than written literally because the statement form is the
    thing under test, not the binding it produces: a literal import of a name that is
    expected to be missing cannot be written at all, and one that resolves would be an
    unused import. ``importlib`` is not a substitute — it raises ``AttributeError``
    where the ``from`` form raises ``ImportError``, which is the distinction being pinned.
    """
    namespace: dict[str, object] = {}
    exec(f"from {module} import {name}", namespace)
    return namespace[name]


class TestShimIsSilentOnImport:
    """A shim that warns at import time punishes consumers of untouched symbols."""

    def test_importing_the_shim_emits_no_warning(self) -> None:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            importlib.reload(event_shim)
        assert [str(record.message) for record in records] == []

    def test_importing_the_package_root_emits_no_deprecation_warning(self) -> None:
        """The root must source from the new modules, never through the façade."""
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            importlib.reload(akgentic.tool)
        deprecations = [
            str(record.message)
            for record in records
            if issubclass(record.category, DeprecationWarning)
        ]
        assert deprecations == []


class TestShimWarnsOnAccess:
    def test_sibling_package_import_still_resolves_and_warns_once(self) -> None:
        """The verbatim statement a sibling package uses, pinned end to end."""
        from akgentic.tool.core.event import CommandsAnnouncedEvent as MovedHome

        with pytest.warns(DeprecationWarning) as records:
            from akgentic.tool.event import CommandsAnnouncedEvent

        assert CommandsAnnouncedEvent is MovedHome
        assert len(records) == 1
        message = str(records[0].message)
        assert "CommandsAnnouncedEvent" in message
        assert "akgentic.tool.core.event" in message

    @pytest.mark.parametrize(("name", "module"), sorted(_MOVED.items()))
    def test_every_moved_name_resolves_to_its_new_home(self, name: str, module: str) -> None:
        expected = getattr(importlib.import_module(module), name)
        with pytest.warns(DeprecationWarning):
            resolved = getattr(event_shim, name)
        assert resolved is expected

    def test_moved_map_covers_exactly_the_four_event_symbols(self) -> None:
        """Guard the guard: a drifted ``_MOVED`` would make the sweeps above vacuous."""
        assert sorted(_MOVED) == [
            "CommandArg",
            "CommandDescriptor",
            "CommandsAnnouncedEvent",
            "ToolStateEvent",
        ]
        assert set(_MOVED.values()) == {"akgentic.tool.core.event"}

    def test_repeated_access_keeps_warning(self) -> None:
        """Caching into ``globals()`` would silence every access after the first."""
        for _ in range(2):
            with pytest.warns(DeprecationWarning):
                _ = event_shim.CommandArg

    def test_warning_names_no_removal_release(self) -> None:
        """No version number and no date — the schedule is deliberately open."""
        with pytest.warns(DeprecationWarning) as records:
            _ = event_shim.ToolStateEvent
        message = str(records[0].message)
        assert "no removal release is scheduled" in message
        assert not re.search(r"\d+\.\d+", message), message

    def test_unknown_attribute_raises_without_warning(self) -> None:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            with pytest.raises(AttributeError):
                _ = event_shim.does_not_exist
        assert [str(record.message) for record in records] == []

    def test_dir_lists_the_moved_names(self) -> None:
        assert sorted(_MOVED) == [name for name in dir(event_shim) if name in _MOVED]


class TestPackageRootApiUnchanged:
    def test_all_matches_the_pre_split_list(self) -> None:
        assert set(akgentic.tool.__all__) - set(_VECTOR_SEARCH_EXTRAS) == set(_EXPECTED_ROOT_ALL)

    def test_every_exported_name_resolves(self) -> None:
        unresolved = [name for name in akgentic.tool.__all__ if not hasattr(akgentic.tool, name)]
        assert not unresolved, f"exported but unresolvable: {unresolved}"

    def test_root_unknown_attribute_still_raises(self) -> None:
        with pytest.raises(AttributeError):
            _ = akgentic.tool.does_not_exist


class TestToolStatePayloadIsGoneNotMoved:
    """The alias was removed outright — the first symbol in the split to get no shim.

    A moved symbol resolves from the façade and warns. A removed one does not resolve at
    all and must not warn: the ``_MOVED`` lookup misses, so access takes the unknown-name
    branch. Both statement forms are pinned on every path the name ever had, because
    ``from X import Y`` and ``X.Y`` raise different exception types — CPython converts a
    module ``__getattr__``'s ``AttributeError`` into ``ImportError`` for the ``from``
    form — and only the pair proves the name is unreachable rather than merely awkward.
    """

    @pytest.mark.parametrize(
        "module",
        [
            "akgentic.tool.event",
            "akgentic.tool",
            "akgentic.tool.knowledge_graph.event",
        ],
    )
    def test_from_import_raises_import_error(self, module: str) -> None:
        with pytest.raises(ImportError):
            _from_import(module, "ToolStatePayload")

    def test_facade_attribute_access_raises_and_does_not_warn(self) -> None:
        """A shimmed name warns here; a removed one must reach the unknown-name branch."""
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            with pytest.raises(AttributeError):
                _ = event_shim.ToolStatePayload
        assert [str(record.message) for record in records] == []

    def test_package_root_attribute_access_raises(self) -> None:
        with pytest.raises(AttributeError):
            _ = akgentic.tool.ToolStatePayload

    def test_the_name_survives_in_no_export_list(self) -> None:
        assert "ToolStatePayload" not in akgentic.tool.__all__
        assert "ToolStatePayload" not in _MOVED
        assert "ToolStatePayload" not in dir(event_shim)


class TestObserversAreGoneFromTheShim:
    """The three observer entries were withdrawn from the façade, not moved again.

    Unlike ``ToolStatePayload`` they still exist — at the package root, which is their
    Stable-tier surface, and in their post-split homes. What is pinned here is that the
    ``akgentic.tool.event`` path no longer serves them on either statement form, without
    warning, while the root keeps resolving them silently.
    """

    _OBSERVERS = ["ToolObserver", "ActorToolObserver", "TeamManagementToolObserver"]

    @pytest.mark.parametrize("name", _OBSERVERS)
    def test_from_import_raises_import_error(self, name: str) -> None:
        with pytest.raises(ImportError):
            _from_import("akgentic.tool.event", name)

    @pytest.mark.parametrize("name", _OBSERVERS)
    def test_facade_attribute_access_raises_and_does_not_warn(self, name: str) -> None:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            with pytest.raises(AttributeError):
                _ = getattr(event_shim, name)
        assert [str(record.message) for record in records] == []

    @pytest.mark.parametrize("name", _OBSERVERS)
    def test_package_root_still_resolves_without_warning(self, name: str) -> None:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            resolved = getattr(akgentic.tool, name)
        assert resolved is not None
        assert [str(record.message) for record in records] == []

    @pytest.mark.parametrize("name", _OBSERVERS)
    def test_the_name_survives_in_no_shim_export_list(self, name: str) -> None:
        assert name not in _MOVED
        assert name not in dir(event_shim)


class TestKnowledgeGraphStateEventIsUntouched:
    """Nothing that carries data changed: only the alias naming the class went away."""

    def test_every_path_resolves_to_the_one_class(self) -> None:
        from akgentic.tool import knowledge_graph
        from akgentic.tool.knowledge_graph import event, models

        assert event.KnowledgeGraphStateEvent is models.KnowledgeGraphStateEvent
        assert knowledge_graph.KnowledgeGraphStateEvent is models.KnowledgeGraphStateEvent
        assert akgentic.tool.KnowledgeGraphStateEvent is models.KnowledgeGraphStateEvent
