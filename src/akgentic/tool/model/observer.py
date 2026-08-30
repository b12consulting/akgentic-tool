"""Observer protocol for the model-switch tool.

``ModelSwitchToolObserver`` is one tool's contract, not the package's: ``ModelTool``
is its only consumer. It lives beside that tool rather than in ``core/`` so the global
surface stays limited to what more than one domain actually needs.

It is a **sibling** of ``ActorToolObserver``, not a widening of it: every observer
that offers no model switch keeps satisfying the base protocol unchanged.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from akgentic.tool.core.observer import ActorToolObserver
from akgentic.tool.model.state import ModelRow


@runtime_checkable
class ModelSwitchToolObserver(ActorToolObserver, Protocol):
    """Observer protocol for runtime model switching.

    Extends ``ActorToolObserver`` with the two operations ``ModelTool`` needs:
    reading the roster as serializable rows, and making one entry the model in
    force. The implementation lives in ``akgentic-agent``, which may import both
    this package and ``akgentic-llm`` and so can project the roster's own
    configuration model onto ``ModelRow``.
    """

    def list_model_rows(self) -> list[ModelRow]:
        """Project the roster, one row per entry.

        Rebuilt from the roster on every call — the rows are never stored.

        Returns:
            One ``ModelRow`` per roster entry, with ``active`` set on the entry
            currently in force.
        """
        ...

    def switch_model(self, key: str) -> str:
        """Make the roster entry named by *key* the model in force.

        Args:
            key: Roster key of the target entry, ``f"{provider}:{model}"``.

        Returns:
            A human-readable confirmation of the outcome.
        """
        ...
