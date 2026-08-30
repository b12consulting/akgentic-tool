"""The model-switch domain's serializable vocabulary (akgentic-llm ADR-018 §6).

``ModelRow`` is owned here because ``akgentic-tool`` may import ``akgentic-core``
only: the roster's own configuration model lives in ``akgentic-llm``, which this
package must never name. ``akgentic-agent`` maps one onto the other — it may
import both packages, so the mapping has a legal home and neither package gains
an import edge.

Two models, one subject and two roles: ``ModelRow`` is a projection the observer
rebuilds per call and nobody stores, while ``ActiveModelState`` is the diffable
``LLM_CONTEXT`` state ``ModelTool`` publishes and the context updater persists as
a baseline.
"""

from typing import Self

from akgentic.core.utils import SerializableBaseModel
from akgentic.tool.core import ContextState


class ModelRow(SerializableBaseModel):
    """One roster entry as the model-switch contract carries it.

    A **projection**, not a second source of truth: a row is rebuilt from the
    roster on every call and never stored. The only thing this feature persists
    is ``ToolState.active_model``, a single key.

    Deliberately not a ``ContextState`` — a row is not diffable state, and the
    ``LLM_CONTEXT`` state describing the model in force belongs to the card.

    Attributes:
        key: The roster key, ``f"{provider}:{model}"`` — what ``switch_model``
            takes and what ``ToolState.active_model`` stores.
        provider: The provider the entry resolves through.
        model: The provider-side model name.
        active: Whether this entry is the one currently in force.
        context_length: The entry's declared context window, or ``None`` when it
            declares none.
    """

    key: str
    provider: str
    model: str
    active: bool
    context_length: int | None


class ActiveModelState(ContextState):
    """The model in force at one point in time, as the ``LLM_CONTEXT`` channel carries it.

    **The key, and nothing else.** This state is persisted as a baseline inside
    ``ToolState.context_baselines``, so every column it carries is paid for on
    every checkpoint of every agent; one string keeps that O(1) (NFR3). Nothing is
    lost by the narrowing: provider, model name and context window are all
    derivable from the key by whoever holds the roster, and the roster is
    ``akgentic-llm``'s — a package this one may not import anyway.

    Attributes:
        key: The roster key of the model in force, ``f"{provider}:{model}"``.
    """

    key: str

    def render_full(self) -> str:
        """The model in force, on one line."""
        return f"**Active model:** {self.key}"

    def render_delta(self, previous: Self) -> str | None:
        """The move from ``previous`` to here, or ``None`` when nothing moved."""
        if previous.key == self.key:
            return None
        return f"**Active model changed:** {previous.key} → {self.key}"
