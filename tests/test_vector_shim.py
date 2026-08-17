"""The removed ``akgentic.tool.vector`` module, and the untouched package root.

The compatibility façade that lived here after the ``vector_store`` move was withdrawn:
all four of its entries were Internal-tier (a courtesy, not a guarantee), and the module
found no consumer. These tests pin the *removal* the way ``ToolStatePayload``'s removal is
pinned in ``test_event_shim.py``: the old path does not resolve at all and does not warn,
while the package root — the Stable-tier surface — keeps serving every public symbol
silently from the new home.
"""

from __future__ import annotations

import importlib
import importlib.resources
import warnings

import pytest

import akgentic.tool

_NEW_HOME = "akgentic.tool.vector_store.vector"

_REMOVED_NAMES = [
    "VectorEntry",
    "EmbeddingService",
    "VectorIndex",
    "_check_vector_search_dependencies",
]


class TestVectorModuleIsGoneNotMoved:
    """A moved module resolves and warns; a removed one must not exist at all."""

    def test_importing_the_module_raises(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("akgentic.tool.vector")

    @pytest.mark.parametrize("name", _REMOVED_NAMES)
    def test_from_import_raises_import_error(self, name: str) -> None:
        """The verbatim statement a consumer would have written, per symbol."""
        namespace: dict[str, object] = {}
        with pytest.raises(ImportError):
            exec(f"from akgentic.tool.vector import {name}", namespace)

    def test_no_file_survives_on_disk(self) -> None:
        """An orphaned ``vector.py`` would resurrect the path on the next install."""
        package_dir = importlib.resources.files("akgentic.tool")
        assert not (package_dir / "vector.py").is_file()


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
