"""The ``akgentic.tool.vector`` façade, and the untouched package root.

The module moved to ``vector_store`` and was then withdrawn on the finding that it had
no consumer. A source sweep produced that finding, and it missed the consumer that
matters: ``VectorEntry`` is a ``SerializableBaseModel``, so rows persisted before the
move record ``akgentic.tool.vector.VectorEntry`` in ``__model__`` and resolve it at read
time through ``import_module`` plus ``getattr``. The path is therefore load-bearing for
stored data, and these tests pin it the way ``test_event_shim.py`` pins ``event.py``.

The other three entries stay withdrawn: ``__model__`` is written for models and
dataclasses alone, so a plain class and a function cannot occur in a payload. The
package root — the Stable-tier surface — keeps serving every public symbol silently
from the new home.
"""

from __future__ import annotations

import importlib
import warnings

import pytest
from akgentic.core.utils.deserializer import import_class

import akgentic.tool

_NEW_HOME = "akgentic.tool.vector_store.vector"
_OLD_HOME = "akgentic.tool.vector"

# The marker a pre-move row carries. Read back verbatim, never rewritten.
_LEGACY_ENTRY_MODEL = "akgentic.tool.vector.VectorEntry"

# Withdrawn by story 27-10 and staying withdrawn: none is a model or a dataclass, so
# none can appear in a ``__model__`` marker.
_STILL_REMOVED_NAMES = [
    "EmbeddingService",
    "VectorIndex",
    "_check_vector_search_dependencies",
]


class TestPersistedRowsKeepResolving:
    """The reason the module is on disk at all: stored payloads name this path."""

    def test_the_legacy_marker_deserializes_to_the_moved_class(self) -> None:
        """A row written before the move still resolves to the class in its new home."""
        expected = importlib.import_module(_NEW_HOME).VectorEntry
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert import_class(_LEGACY_ENTRY_MODEL) is expected

    def test_a_stored_payload_round_trips_through_the_old_path(self) -> None:
        """End to end: the envelope a pre-move writer produced still loads today.

        ``import_class`` alone proves the path resolves; this proves the object it
        yields still reconstructs the row, which is what a reader actually needs.
        """
        entry_cls = importlib.import_module(_NEW_HOME).VectorEntry
        payload = entry_cls(
            ref_type="entity", ref_id="e1", text="hello", vector=[0.1, 0.2]
        ).model_dump()
        payload["__model__"] = _LEGACY_ENTRY_MODEL

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            restored = import_class(payload.pop("__model__"))(**payload)

        assert isinstance(restored, entry_cls)
        assert restored.ref_type == "entity"
        assert restored.ref_id == "e1"
        assert restored.text == "hello"
        assert restored.vector == [0.1, 0.2]

    def test_resolution_warns_so_legacy_rows_are_findable(self) -> None:
        """Access through the old path is answered, and announced."""
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            importlib.import_module(_OLD_HOME).VectorEntry
        assert [r.category for r in records] == [DeprecationWarning]
        assert _NEW_HOME in str(records[0].message)

    def test_importing_the_module_does_not_warn(self) -> None:
        """PEP 562 fires on attribute access, so a bare import stays silent.

        An import-time warning would reach every consumer of the package, including
        the ones that touch none of the moved symbols.
        """
        importlib.invalidate_caches()
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            importlib.reload(importlib.import_module(_OLD_HOME))
        assert [str(r.message) for r in records] == []


class TestTheWithdrawnEntriesStayWithdrawn:
    """Only what can appear in a ``__model__`` marker comes back."""

    @pytest.mark.parametrize("name", _STILL_REMOVED_NAMES)
    def test_from_import_raises_import_error(self, name: str) -> None:
        """The verbatim statement a consumer would have written, per symbol."""
        namespace: dict[str, object] = {}
        with pytest.raises(ImportError):
            exec(f"from {_OLD_HOME} import {name}", namespace)

    @pytest.mark.parametrize("name", _STILL_REMOVED_NAMES)
    def test_attribute_access_raises_without_warning(self, name: str) -> None:
        """A withdrawn name is absent, not deprecated — no warning detour."""
        module = importlib.import_module(_OLD_HOME)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            with pytest.raises(AttributeError):
                getattr(module, name)
        assert [str(r.message) for r in records] == []

    def test_the_facade_advertises_only_what_it_serves(self) -> None:
        """``dir()`` is the discoverable surface, so it must not overstate."""
        assert dir(importlib.import_module(_OLD_HOME)) == ["VectorEntry"]


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
