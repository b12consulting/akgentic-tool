"""The model-switch domain's serializable vocabulary (akgentic-llm ADR-018 §6).

``ModelRow`` is owned here because ``akgentic-tool`` may import ``akgentic-core``
only: the roster's own configuration model lives in ``akgentic-llm``, which this
package must never name. ``akgentic-agent`` maps one onto the other — it may
import both packages, so the mapping has a legal home and neither package gains
an import edge.
"""

from akgentic.core.utils import SerializableBaseModel


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
