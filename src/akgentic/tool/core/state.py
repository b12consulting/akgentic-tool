"""The ``ToolState`` slot: per-agent durable state owned by the tool layer (ADR-041 §1).

Context-update baselines are persisted **as a cache** on the agent's state, so a
restored agent resumes delta delivery instead of re-sending a full snapshot on
its first turn. The slot travels with the agent's state object, which
``init_state()`` replaces wholesale on restore — so no consumer may hold a
``ToolState`` reference. Read it live through the observer instead:
``observer.state.tool_state``, on every call.
"""

from pydantic import Field

from akgentic.core.utils import SerializableBaseModel

from .context_state import ContextState


class ToolState(SerializableBaseModel):
    """Per-agent persistent state owned by the tool layer, carried by the agent's state.

    Keys of ``context_baselines`` are provider ``__name__``s — the name of the
    zero-arg provider callable that ``ToolCard.get_context_states`` yields.
    Downstream delivery wiring relies on that convention to pair each provider
    with its persisted baseline.

    Never store a ``ToolState`` reference: the agent's state object is replaced
    wholesale on restore, so a cached reference goes silently stale. Reach the
    live slot through ``observer.state.tool_state`` at the moment of use.
    """

    context_baselines: dict[str, ContextState] = Field(default_factory=dict)
    context_update_seq: int = 0
