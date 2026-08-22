"""The ``ContextUpdater``: at most one **Context update** block per turn.

Implements ADR-041 §4 (engine, weak observer) and §5 (trust rules), carrying the
delivery behaviour of ADR-037 §6 (marker grammar, one block per turn) and §7
(baseline-as-cache) in from the agent side. The engine reads every context-state
provider in factory order, diffs against the baselines persisted in
``observer.state.tool_state``, and returns the composed block — the append stays
with the agent, so a restored agent resumes delta delivery from its persisted
baselines instead of re-sending a full snapshot.
"""

from __future__ import annotations

import logging
import re
import weakref
from collections.abc import Callable, Iterator, Sequence

from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

from .context_state import ContextState
from .observer import ActorToolObserver
from .state import ToolState

logger = logging.getLogger(__name__)

# The Context update marker line (ADR-037 §6). The numbered pattern recovers the
# block counter from a restored history and detects markers the persisted
# counter has never seen.
_CONTEXT_UPDATE_MARKER = re.compile(r"\*\*Context update (\d+)\*\*")


def _iter_user_prompt_texts(messages: Sequence[ModelMessage]) -> Iterator[str]:
    """Yield the user-prompt text of *messages*, one lazy pass.

    Scan scope is deliberately narrow: only ``UserPromptPart`` content on
    ``ModelRequest`` messages — a ``str`` content directly, the ``str`` items of
    a multimodal ``list`` content otherwise. Tool returns, retry prompts, system
    prompts and ``ModelResponse`` messages are never inspected (a model echoing
    the marker must not count), and nothing is concatenated. Never construct a
    ``ModelRequest`` here or anywhere in ``src/`` — appending one is the retired
    defect this module's imports must not resurrect; they exist for
    ``isinstance`` checks only.
    """
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if not isinstance(part, UserPromptPart):
                continue
            if isinstance(part.content, str):
                yield part.content
            elif isinstance(part.content, list):
                yield from (item for item in part.content if isinstance(item, str))


class ContextUpdater:
    """Composes at most one Context update block per turn against persisted baselines.

    Holds the observer via ``weakref.ref`` and dereferences
    ``observer.state.tool_state`` on every call — the agent replaces its state
    object wholesale on restore, so neither the carrier nor the ``ToolState``
    may be retained across calls (see :class:`~akgentic.tool.core.state.ToolState`).
    A collected observer degrades: :meth:`compose_update` returns ``None`` and
    :meth:`reset` is a no-op.
    """

    def __init__(
        self,
        observer: ActorToolObserver,
        providers: Sequence[Callable[[], ContextState | None]],
    ) -> None:
        """Build the engine over *observer*'s persisted state and *providers*.

        Args:
            observer: The actor-aware observer whose ``state.tool_state`` slot
                carries the baselines and the block counter. Held weakly.
            providers: Zero-arg context-state providers in factory order, as
                :meth:`ToolFactory.get_context_states` returns them.
        """
        self._observer_ref: weakref.ref[ActorToolObserver] = weakref.ref(observer)
        self._providers = providers

    def compose_update(self, messages: Sequence[ModelMessage]) -> str | None:
        """Compose this turn's Context update block, or ``None`` for nothing to say.

        Reconciles the persisted counter and baselines against the marker
        numbers visible in *messages*, then reads every provider in order and
        diffs against its baseline. Never appends anything and never raises —
        provider and renderer failures degrade per section. ``full_snapshot``
        is captured immediately after reconciliation, before the provider loop:
        a no-change delta advances its baseline mid-loop, so the wording flag
        must be taken first.

        Baselines of contributing providers and the counter advance in place on
        the live ``ToolState`` only when a block is produced (the no-change
        baseline advance excepted); a turn with nothing to say moves nothing,
        so an unchanged turn stays byte-identical for the prompt cache.
        """
        observer = self._observer_ref()
        if observer is None:
            return None
        tool_state = observer.state.tool_state
        self._reconcile_baselines(tool_state, messages)
        full_snapshot = not tool_state.context_baselines
        sections: list[str] = []
        advanced: dict[str, ContextState] = {}
        for provider in self._providers:
            contribution = self._render_section(provider, tool_state)
            if contribution is None:
                continue
            rendering, state = contribution
            sections.append(rendering)
            advanced[provider.__name__] = state
        if not sections:
            return None
        block = self._compose_block(tool_state.context_update_seq + 1, sections, full_snapshot)
        tool_state.context_baselines.update(advanced)
        tool_state.context_update_seq += 1
        return block

    def reset(self) -> None:
        """Zero both persisted fields in place — the ``/clear`` path.

        The one legitimate zeroing of the counter: it corresponds to a history
        whose markers were wiped with it. A collected observer makes this a
        no-op.
        """
        observer = self._observer_ref()
        if observer is None:
            return
        tool_state = observer.state.tool_state
        tool_state.context_baselines.clear()
        tool_state.context_update_seq = 0

    @staticmethod
    def _reconcile_baselines(tool_state: ToolState, messages: Sequence[ModelMessage]) -> None:
        """Reconcile the persisted state against the visible history (ADR-041 §5).

        ``highest`` is the largest marker number in the user-role prompt texts.
        ``seq == highest`` trusts the baselines (including the fresh ``0 == 0``
        case). ``seq < highest`` — a stale save — raises the counter to
        ``highest`` and keeps the baselines: the next delta re-states what the
        missed blocks said, a repeat, never an omission. ``seq > highest`` —
        the last delivered marker is no longer visible (compaction, trimming,
        an out-of-band wipe) — drops every baseline so the next block is a full
        snapshot; the counter stays where it is, because a partially trimmed
        history may still show older numbers. Reconciliation only ever moves
        the counter upward; the one legitimate zeroing is :meth:`reset`.
        """
        highest = 0
        for text in _iter_user_prompt_texts(messages):
            for match in _CONTEXT_UPDATE_MARKER.finditer(text):
                highest = max(highest, int(match.group(1)))
        seq = tool_state.context_update_seq
        if seq == highest:
            return
        if seq < highest:
            tool_state.context_update_seq = highest
            return
        tool_state.context_baselines.clear()

    @staticmethod
    def _render_section(
        provider: Callable[[], ContextState | None], tool_state: ToolState
    ) -> tuple[str, ContextState] | None:
        """Compute one provider's section for this turn, or ``None`` for no section.

        First-seen states — and states whose concrete type differs from their
        baseline's (a card reconfigured mid-life) — render ``render_full()``;
        otherwise ``render_delta(baseline)``. The rendering is used verbatim.

        Degradation, never failure: a provider or renderer that raises is
        logged and skipped without advancing its baseline; a ``None`` state or
        an empty full rendering contributes nothing. A ``None`` delta means no
        change, so the baseline may advance to the current (equal) state even
        though no section is produced.

        Returns:
            The ``(rendering, state)`` pair to contribute, or ``None``.
        """
        name = provider.__name__
        try:
            state = provider()
        except Exception:
            logger.exception("context-state provider '%s' raised; skipped", name)
            return None
        if state is None:
            return None
        baseline = tool_state.context_baselines.get(name)
        try:
            if baseline is None or type(state) is not type(baseline):
                rendering: str | None = state.render_full()
            else:
                rendering = state.render_delta(baseline)
        except Exception:
            logger.exception("context-state renderer '%s' raised; skipped", name)
            return None
        if rendering is None:
            tool_state.context_baselines[name] = state
            return None
        if not rendering:
            return None
        return rendering, state

    @staticmethod
    def _compose_block(number: int, sections: list[str], full_snapshot: bool) -> str:
        """Compose one Context update block: marker line + verbatim sections.

        The marker is ``**Context update N**`` with a fixed suffix — worded as
        current state when the baselines were empty as delivery began (every
        section renders full then), as change only when the block was diffed
        against surviving baselines. Both suffixes are fixed strings (ADR-037
        §6) — nothing turn-varying beyond ``N`` may appear: a timestamp would
        defeat the cache property the delta channel exists for. Sections are
        joined to the marker and to each other with blank lines, never
        re-wrapped — the renderers own their internal join style.
        """
        suffix = "current state." if full_snapshot else "state has changed since the last update."
        marker = f"**Context update {number}** — {suffix}"
        return "\n\n".join([marker, *sections])
