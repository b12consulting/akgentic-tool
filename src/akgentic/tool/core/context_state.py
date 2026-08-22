"""The ``ContextState`` contract: a card's prompt-relevant state, structured for diffing.

Implements ADR-037 §3. A stateful tool card exposes its volatile prompt content as
``ContextState`` snapshots delivered on the ``LLM_CONTEXT`` channel — pushed into the
context tail per turn as deltas — instead of re-rendering it into the system prompt,
which would invalidate the cached prefix on every change.
"""

from abc import ABC, abstractmethod
from typing import Self

from akgentic.core.utils import SerializableBaseModel


class ContextState(SerializableBaseModel, ABC):
    """A card's prompt-relevant state at one point in time, structured for diffing.

    Contract rules for the agent-side caller:

    - ``render_delta`` may assume ``previous`` is the **same concrete type** as
      ``self``; the caller must never diff across types (a card reconfigured
      between turns).
    - A first-seen state is never a delta: the caller renders ``render_full()``;
      ``render_delta`` is never asked to diff against nothing.
    - A **provider** — the zero-arg callable returning ``ContextState | None``
      that :meth:`ToolCard.get_context_states` yields — never raises. It returns
      ``None`` when its state is unavailable (collected observer, stopped actor).
    """

    @abstractmethod
    def render_full(self) -> str:
        """The whole state, as the model should first see it.

        Returns ``""`` when there is nothing to say.
        """

    @abstractmethod
    def render_delta(self, previous: Self) -> str | None:
        """What changed since ``previous``. ``None`` when nothing did."""
