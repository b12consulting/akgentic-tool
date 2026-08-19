"""Compatibility façade for the pre-move ``akgentic.tool.vector`` module.

The module moved to ``akgentic.tool.vector_store.vector`` and was withdrawn outright,
on the finding that no consumer imported it. That finding was drawn from a source
sweep, and one class here has a consumer a source sweep cannot see: ``VectorEntry`` is
a ``SerializableBaseModel``, so every row persisted before the move carries
``akgentic.tool.vector.VectorEntry`` in its ``__model__`` marker. Reading one back is
``import_module`` plus ``getattr`` on that recorded path, so the path has to keep
resolving for as long as such rows exist — which is not a property of this repository.

Only ``VectorEntry`` is served here, and the boundary is the serializer's: ``__model__``
is written for Pydantic models and dataclasses alone. ``EmbeddingService`` and
``VectorIndex`` are plain classes and ``_check_vector_search_dependencies`` is a
function, so none of the three can appear in a stored payload and none is restored —
their withdrawal stands, and importing one from here still raises.

Resolution goes through PEP 562, so the warning fires on **attribute access** and never
at import time. An import-time warning would hit every consumer of the package,
including code that touches none of the moved symbols.
"""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Redundant ``X as X`` alias: mypy strict implies no_implicit_reexport, so a
    # façade has to re-export deliberately (same form as ``event.py``).
    from akgentic.tool.vector_store.vector import VectorEntry as VectorEntry

# No removal release is named here, and that is deliberate rather than an oversight.
# This shim outlives the code that reads it: it can only be withdrawn once no persisted
# row still carries the pre-move marker, which is a fact about deployed databases and
# not about a version number. The schedule stays open until someone can establish that.
_SHIM_REMOVAL_NOTICE = "no removal release is scheduled"

_MOVED: dict[str, str] = {
    "VectorEntry": "akgentic.tool.vector_store.vector",
}


def __getattr__(name: str) -> Any:
    """Resolve a moved symbol from its new home, warning once per access.

    The resolved object is deliberately **not** cached into ``globals()``: caching
    silences every access after the first, which would make the warning depend on
    module-import order across a session rather than on what the caller wrote.
    """
    module = _MOVED.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"akgentic.tool.vector.{name} has moved to {module}.{name}; "
        f"import it from there ({_SHIM_REMOVAL_NOTICE}).",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    """Expose the moved name so ``dir()`` and tab-completion still find it."""
    return sorted(_MOVED)
