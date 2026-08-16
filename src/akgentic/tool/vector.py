"""Compatibility façade for the pre-move ``akgentic.tool.vector`` module.

The vector primitives that lived here moved into ``akgentic.tool.vector_store.vector``,
next to the backends, actors and protocols built on them. This module stays on disk so no
consumer has to change an import line — sibling packages and anyone on PyPI keep working,
and ``VectorEntry`` payloads persisted with a pre-move ``__model__`` marker keep resolving,
since that lookup is ``import_module`` plus ``getattr``.

Resolution goes through PEP 562, so the warning fires on **attribute access** and
never at import time. An import-time warning would hit every consumer of the
package, including code that touches none of the moved symbols.
"""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Redundant ``X as X`` aliases: mypy strict implies no_implicit_reexport, so a
    # façade has to re-export deliberately (same form as core/__init__.py).
    from akgentic.tool.vector_store.vector import EmbeddingService as EmbeddingService
    from akgentic.tool.vector_store.vector import VectorEntry as VectorEntry
    from akgentic.tool.vector_store.vector import VectorIndex as VectorIndex
    from akgentic.tool.vector_store.vector import (
        _check_vector_search_dependencies as _check_vector_search_dependencies,
    )

# No removal release is named here, and that is deliberate rather than an oversight.
# Naming one commits this shim's fate to a version number chosen before anyone knows
# what that release will carry — akgentic-llm learned this when its own shim was
# announced for 2.0.0 and 2.0.0 turned out to be a dependency-forced major that the
# shim shipped straight through. The schedule stays open until someone actually
# schedules it.
_SHIM_REMOVAL_NOTICE = "no removal release is scheduled"

_MOVED: dict[str, str] = {
    "VectorEntry": "akgentic.tool.vector_store.vector",
    "EmbeddingService": "akgentic.tool.vector_store.vector",
    "VectorIndex": "akgentic.tool.vector_store.vector",
    "_check_vector_search_dependencies": "akgentic.tool.vector_store.vector",
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
    """Expose the moved names so ``dir()`` and tab-completion still find them."""
    return sorted(_MOVED)
