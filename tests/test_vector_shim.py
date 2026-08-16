"""The ``akgentic.tool.vector`` compatibility façade, and the untouched package root.

``akgentic-tool`` ships on PyPI, so an import of ``akgentic.tool.vector`` may live in code
this repo cannot edit. The façade is what lets the move land without touching it, so these
tests pin its three load-bearing properties: it warns on access and not on import, it names
the destination, and it resolves to the very same object as the new module.

Mirrors ``test_event_shim.py``, with one addition it has no need for:
``_check_vector_search_dependencies`` is private, and PEP 562 ``__getattr__`` serves
underscore names exactly like any other. A parametrised loop written to skip underscores
would leave this story's only private moved symbol untested, so it is also covered by name.
"""

from __future__ import annotations

import importlib
import re
import warnings

import pytest

import akgentic.tool
import akgentic.tool.vector as vector_shim
from akgentic.tool.vector import _MOVED

_NEW_HOME = "akgentic.tool.vector_store.vector"

_MOVED_NAMES = [
    "VectorEntry",
    "EmbeddingService",
    "VectorIndex",
    "_check_vector_search_dependencies",
]


class TestShimIsSilentOnImport:
    """A shim that warns at import time punishes consumers of untouched symbols."""

    def test_importing_the_shim_emits_no_warning(self) -> None:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            importlib.reload(vector_shim)
        assert [str(record.message) for record in records] == []

    def test_importing_the_package_root_emits_no_deprecation_warning(self) -> None:
        """The root must source from the new module, never through the façade."""
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
    def test_moved_map_covers_exactly_the_four_symbols(self) -> None:
        """Guard the guard: a shrunken ``_MOVED`` would make the sweep below vacuous."""
        assert sorted(_MOVED) == sorted(_MOVED_NAMES)
        assert set(_MOVED.values()) == {_NEW_HOME}

    @pytest.mark.parametrize(("name", "module"), sorted(_MOVED.items()))
    def test_every_moved_name_resolves_to_its_new_home(self, name: str, module: str) -> None:
        expected = getattr(importlib.import_module(module), name)
        with pytest.warns(DeprecationWarning) as records:
            resolved = getattr(vector_shim, name)
        assert resolved is expected
        assert len(records) == 1
        message = str(records[0].message)
        assert name in message
        assert _NEW_HOME in message

    def test_the_private_dependency_check_resolves_through_the_facade(self) -> None:
        """The one underscore name, covered by name rather than only by the loop."""
        from akgentic.tool.vector_store.vector import _check_vector_search_dependencies

        with pytest.warns(DeprecationWarning) as records:
            resolved = vector_shim._check_vector_search_dependencies

        assert resolved is _check_vector_search_dependencies
        message = str(records[0].message)
        assert "_check_vector_search_dependencies" in message
        assert _NEW_HOME in message

    def test_repeated_access_keeps_warning(self) -> None:
        """Caching into ``globals()`` would silence every access after the first."""
        for _ in range(2):
            with pytest.warns(DeprecationWarning):
                _ = vector_shim.VectorEntry

    def test_warning_names_no_removal_release(self) -> None:
        """No version number and no date — the schedule is deliberately open."""
        with pytest.warns(DeprecationWarning) as records:
            _ = vector_shim.VectorIndex
        message = str(records[0].message)
        assert "no removal release is scheduled" in message
        assert not re.search(r"\d+\.\d+", message), message

    def test_unknown_attribute_raises_without_warning(self) -> None:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            with pytest.raises(AttributeError):
                _ = vector_shim.does_not_exist
        assert [str(record.message) for record in records] == []

    def test_dir_lists_the_moved_names(self) -> None:
        assert sorted(_MOVED) == [name for name in dir(vector_shim) if name in _MOVED]


class TestPackageRootStaysSilent:
    """The root API does not change, so nothing reached through it may deprecate."""

    @pytest.mark.parametrize("name", ["VectorEntry", "EmbeddingService", "VectorIndex"])
    def test_root_access_resolves_without_warning(self, name: str) -> None:
        expected = getattr(importlib.import_module(_NEW_HOME), name)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            resolved = getattr(akgentic.tool, name)
        assert resolved is expected
        assert [str(record.message) for record in records] == []
