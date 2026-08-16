"""The ``akgentic.tool.event`` compatibility façade, and the untouched package root.

``akgentic-tool`` ships on PyPI and a sibling package imports ``CommandsAnnouncedEvent``
from ``akgentic.tool.event``. The façade is what lets the split land without editing
that consumer, so these tests pin its three load-bearing properties: it warns on
access and not on import, it names the destination, and it resolves to the very same
object as the new module.
"""

from __future__ import annotations

import importlib
import re
import warnings

import pytest

import akgentic.tool
import akgentic.tool.event as event_shim
from akgentic.tool.event import _MOVED

# Frozen before the split: the package-root API may not change, so a consumer
# importing from ``akgentic.tool`` needs no migration at all.
_EXPECTED_ROOT_ALL: list[str] = [
    "BaseToolParam",
    "ToolCard",
    "ToolFactory",
    "CommandRegistry",
    "COMMAND",
    "SYSTEM_PROMPT",
    "TOOL_CALL",
    "Channels",
    "RetriableError",
    "CommandNotRecognized",
    "ToolObserverGone",
    "ToolObserver",
    "ActorToolObserver",
    "TeamManagementToolObserver",
    "ToolStateEvent",
    "ToolStatePayload",
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
    "SeatbeltSandboxActor",
    "WorkspaceTool",
]

_VECTOR_SEARCH_EXTRAS = ["VectorEntry", "EmbeddingService", "VectorIndex"]


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

    def test_tool_state_payload_alias_resolves_to_the_kg_event(self) -> None:
        """The alias survives the split so existing annotations keep resolving."""
        from akgentic.tool.knowledge_graph.models import KnowledgeGraphStateEvent

        with pytest.warns(DeprecationWarning):
            assert event_shim.ToolStatePayload is KnowledgeGraphStateEvent

    def test_repeated_access_keeps_warning(self) -> None:
        """Caching into ``globals()`` would silence every access after the first."""
        for _ in range(2):
            with pytest.warns(DeprecationWarning):
                _ = event_shim.ToolObserver

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

    def test_root_tool_state_payload_resolves_without_warning(self) -> None:
        """The root path is not deprecated — only ``akgentic.tool.event`` warns."""
        from akgentic.tool.knowledge_graph.models import KnowledgeGraphStateEvent

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            payload = akgentic.tool.ToolStatePayload
        assert payload is KnowledgeGraphStateEvent
        assert [str(record.message) for record in records] == []

    def test_root_unknown_attribute_still_raises(self) -> None:
        with pytest.raises(AttributeError):
            _ = akgentic.tool.does_not_exist
