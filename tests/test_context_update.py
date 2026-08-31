"""Tests for the ``ContextUpdater`` engine and ``ToolFactory.get_context_updater()`` (Story 35-2).

Covers the delivery behaviour moved in from the agent side: full-snapshot and
delta composition, per-provider degradation, the weak observer, the trust-rule
reconciliation table, ``reset()``, and the factory build path.
"""

from __future__ import annotations

import gc
import logging
import uuid
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, Self

import pytest
from akgentic.core.actor_address import ActorAddress
from akgentic.core.agent import AkgentType
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from akgentic.tool import ContextUpdater as RootContextUpdater
from akgentic.tool.core import ContextUpdater as CoreContextUpdater
from akgentic.tool.core import ToolCard, ToolFactory
from akgentic.tool.core.context_state import ContextState
from akgentic.tool.core.context_update import ContextUpdater
from akgentic.tool.core.state import ToolState

# ---------------------------------------------------------------------------
# Local context states with distinguishable full/delta renderings
# ---------------------------------------------------------------------------


class _CounterState(ContextState):
    value: int = 0

    def render_full(self) -> str:
        return f"counter full: {self.value}"

    def render_delta(self, previous: Self) -> str | None:
        if self.value == previous.value:
            return None
        return f"counter delta: {previous.value} -> {self.value}"


class _LabelState(ContextState):
    label: str = "start"

    def render_full(self) -> str:
        return f"label full: {self.label}"

    def render_delta(self, previous: Self) -> str | None:
        if self.label == previous.label:
            return None
        return f"label delta: {self.label}"


class _ExplodingState(ContextState):
    def render_full(self) -> str:
        raise RuntimeError("render boom")

    def render_delta(self, previous: Self) -> str | None:
        raise RuntimeError("render boom")


# ---------------------------------------------------------------------------
# Fake observer (35-1's shape) and message helpers
# ---------------------------------------------------------------------------


class _Carrier:
    """Minimal ``ToolStateCarrier``: an object exposing ``tool_state``."""

    def __init__(self) -> None:
        self.tool_state = ToolState()


class _Observer:
    """Minimal object satisfying every ``ActorToolObserver`` member."""

    def __init__(self) -> None:
        self.myAddress = SimpleNamespace()  # noqa: N815 — protocol member name
        self.orchestrator = None
        self.team_id = uuid.uuid4()
        self.state = _Carrier()

    def notify_event(self, event: object) -> None:
        pass

    def proxy_ask(
        self,
        actor: ActorAddress,
        actor_type: type[AkgentType] | None = None,
        timeout: int | None = None,
    ) -> Any:
        return None

    def proxy_tell(
        self,
        actor: ActorAddress,
        actor_type: type[AkgentType] | None = None,
    ) -> Any:
        return None


def _user(content: str | list[str]) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=content)])


def _response(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def _make_env() -> tuple[_Observer, ContextUpdater, dict[str, ContextState | None]]:
    """One observer, one updater over two providers reading mutable holders."""
    observer = _Observer()
    states: dict[str, ContextState | None] = {
        "counter_provider": _CounterState(value=1),
        "label_provider": _LabelState(label="start"),
    }

    def counter_provider() -> ContextState | None:
        return states["counter_provider"]

    def label_provider() -> ContextState | None:
        return states["label_provider"]

    updater = ContextUpdater(observer, [counter_provider, label_provider])
    return observer, updater, states


# ---------------------------------------------------------------------------
# AC 11.1 — first compose is a full snapshot
# ---------------------------------------------------------------------------


def test_first_compose_renders_full_snapshot_numbered_one() -> None:
    observer, updater, _states = _make_env()
    block = updater.compose_update([])
    assert block == (
        "**Context update 1** — current state.\n\ncounter full: 1\n\nlabel full: start"
    )
    tool_state = observer.state.tool_state
    assert set(tool_state.context_baselines) == {"counter_provider", "label_provider"}
    assert tool_state.context_update_seq == 1


# ---------------------------------------------------------------------------
# AC 11.2 — an unchanged turn says nothing and moves nothing
# ---------------------------------------------------------------------------


def test_unchanged_second_compose_returns_none_and_keeps_counter() -> None:
    observer, updater, states = _make_env()
    first = updater.compose_update([])
    assert first is not None
    # An equal-but-distinct state: the no-change delta still advances its baseline.
    states["counter_provider"] = _CounterState(value=1)
    assert updater.compose_update([_user(first)]) is None
    tool_state = observer.state.tool_state
    assert tool_state.context_update_seq == 1
    assert tool_state.context_baselines["counter_provider"] is states["counter_provider"]


# ---------------------------------------------------------------------------
# AC 11.3 — a delta names only what moved
# ---------------------------------------------------------------------------


def test_delta_compose_renders_delta_with_changed_suffix() -> None:
    observer, updater, states = _make_env()
    first = updater.compose_update([])
    assert first is not None
    states["counter_provider"] = _CounterState(value=2)
    block = updater.compose_update([_user(first)])
    assert block == (
        "**Context update 2** — state has changed since the last update.\n\ncounter delta: 1 -> 2"
    )
    assert observer.state.tool_state.context_update_seq == 2


# ---------------------------------------------------------------------------
# AC 11.4 / 11.5 — degradation, never failure
# ---------------------------------------------------------------------------


def test_raising_provider_is_skipped_without_advancing_its_baseline(
    caplog: pytest.LogCaptureFixture,
) -> None:
    observer = _Observer()

    def broken_provider() -> ContextState | None:
        raise RuntimeError("boom")

    def good_provider() -> ContextState | None:
        return _CounterState(value=7)

    updater = ContextUpdater(observer, [broken_provider, good_provider])
    with caplog.at_level(logging.ERROR):
        block = updater.compose_update([])
    assert block == "**Context update 1** — current state.\n\ncounter full: 7"
    tool_state = observer.state.tool_state
    assert "broken_provider" not in tool_state.context_baselines
    assert "good_provider" in tool_state.context_baselines
    assert "broken_provider" in caplog.text


def test_raising_renderer_is_skipped_without_advancing_its_baseline(
    caplog: pytest.LogCaptureFixture,
) -> None:
    observer = _Observer()

    def exploding_provider() -> ContextState | None:
        return _ExplodingState()

    def good_provider() -> ContextState | None:
        return _CounterState(value=7)

    updater = ContextUpdater(observer, [exploding_provider, good_provider])
    with caplog.at_level(logging.ERROR):
        block = updater.compose_update([])
    assert block == "**Context update 1** — current state.\n\ncounter full: 7"
    tool_state = observer.state.tool_state
    assert "exploding_provider" not in tool_state.context_baselines
    assert "exploding_provider" in caplog.text


def test_none_state_contributes_nothing() -> None:
    observer, updater, states = _make_env()
    states["counter_provider"] = None
    block = updater.compose_update([])
    assert block == "**Context update 1** — current state.\n\nlabel full: start"
    assert "counter_provider" not in observer.state.tool_state.context_baselines


def test_empty_full_rendering_contributes_and_advances_nothing() -> None:
    observer = _Observer()

    class _EmptyFullState(_LabelState):
        def render_full(self) -> str:
            return ""

    def blank_provider() -> ContextState | None:
        return _EmptyFullState()

    updater = ContextUpdater(observer, [blank_provider])
    assert updater.compose_update([]) is None
    tool_state = observer.state.tool_state
    assert tool_state.context_baselines == {}
    assert tool_state.context_update_seq == 0


def test_empty_delta_rendering_contributes_and_advances_nothing() -> None:
    observer = _Observer()

    class _EmptyDeltaState(_LabelState):
        def render_delta(self, previous: Self) -> str | None:
            return ""

    holder: dict[str, ContextState] = {"state": _EmptyDeltaState(label="start")}

    def blank_delta_provider() -> ContextState | None:
        return holder["state"]

    updater = ContextUpdater(observer, [blank_delta_provider])
    first = updater.compose_update([])
    assert first is not None
    baseline = observer.state.tool_state.context_baselines["blank_delta_provider"]
    holder["state"] = _EmptyDeltaState(label="moved")
    assert updater.compose_update([_user(first)]) is None
    tool_state = observer.state.tool_state
    # Unlike a None delta, an empty delta does not advance its baseline.
    assert tool_state.context_baselines["blank_delta_provider"] is baseline
    assert tool_state.context_update_seq == 1


def test_type_changed_state_renders_full_against_existing_baseline() -> None:
    observer, updater, states = _make_env()
    first = updater.compose_update([])
    assert first is not None
    # A card reconfigured mid-life: a different concrete type renders full, never a delta.
    states["counter_provider"] = _LabelState(label="swapped")
    block = updater.compose_update([_user(first)])
    assert block == (
        "**Context update 2** — state has changed since the last update.\n\nlabel full: swapped"
    )
    baselines = observer.state.tool_state.context_baselines
    assert isinstance(baselines["counter_provider"], _LabelState)


# ---------------------------------------------------------------------------
# AC 11.6 — a collected observer degrades, never raises
# ---------------------------------------------------------------------------


def test_collected_observer_degrades_to_none_and_noop() -> None:
    observer = _Observer()
    provider_calls: list[str] = []

    def counting_provider() -> ContextState | None:
        provider_calls.append("called")
        return _CounterState(value=1)

    updater = ContextUpdater(observer, [counting_provider])
    del observer
    gc.collect()
    assert updater.compose_update([]) is None
    updater.reset()
    assert provider_calls == []


# ---------------------------------------------------------------------------
# AC 11.7 — seq > highest: baselines dropped, full snapshot numbered seq + 1
# ---------------------------------------------------------------------------


def test_marker_absent_history_with_persisted_seq_drops_baselines() -> None:
    observer, updater, _states = _make_env()
    tool_state = observer.state.tool_state
    tool_state.context_update_seq = 3
    tool_state.context_baselines["counter_provider"] = _CounterState(value=1)
    tool_state.context_baselines["label_provider"] = _LabelState(label="start")
    block = updater.compose_update([_user("no markers anywhere")])
    # Full renderings despite equal states: only a baseline drop explains them.
    assert block == (
        "**Context update 4** — current state.\n\ncounter full: 1\n\nlabel full: start"
    )
    assert tool_state.context_update_seq == 4


def test_marker_in_model_response_is_never_counted() -> None:
    """The echo rule: a model repeating the marker does not make it visible."""
    observer, updater, _states = _make_env()
    tool_state = observer.state.tool_state
    tool_state.context_update_seq = 2
    tool_state.context_baselines["counter_provider"] = _CounterState(value=1)
    tool_state.context_baselines["label_provider"] = _LabelState(label="start")
    block = updater.compose_update([_response("**Context update 2** — current state.")])
    assert block == (
        "**Context update 3** — current state.\n\ncounter full: 1\n\nlabel full: start"
    )


# ---------------------------------------------------------------------------
# AC 11.8 — seq < highest: counter raised, baselines kept
# ---------------------------------------------------------------------------


def test_seq_below_highest_raises_counter_and_keeps_baselines() -> None:
    observer, updater, states = _make_env()
    tool_state = observer.state.tool_state
    tool_state.context_update_seq = 1
    tool_state.context_baselines["counter_provider"] = _CounterState(value=1)
    tool_state.context_baselines["label_provider"] = _LabelState(label="start")
    states["counter_provider"] = _CounterState(value=9)
    history = [_user("**Context update 5** — state has changed since the last update.")]
    block = updater.compose_update(history)
    # A delta against the kept baseline, numbered highest + 1.
    assert block == (
        "**Context update 6** — state has changed since the last update.\n\ncounter delta: 1 -> 9"
    )
    assert tool_state.context_update_seq == 6


def test_marker_in_multimodal_list_content_is_counted() -> None:
    observer, updater, states = _make_env()
    tool_state = observer.state.tool_state
    tool_state.context_update_seq = 2
    tool_state.context_baselines["counter_provider"] = _CounterState(value=1)
    tool_state.context_baselines["label_provider"] = _LabelState(label="start")
    states["counter_provider"] = _CounterState(value=3)
    message = _user(["a caption", "**Context update 2** — current state."])
    block = updater.compose_update([message])
    # seq == highest through the list item: baselines trusted, so a delta.
    assert block == (
        "**Context update 3** — state has changed since the last update.\n\ncounter delta: 1 -> 3"
    )


# ---------------------------------------------------------------------------
# AC 11.9 — reset() zeroes both fields
# ---------------------------------------------------------------------------


def test_reset_zeroes_baselines_and_counter() -> None:
    observer, updater, _states = _make_env()
    assert updater.compose_update([]) is not None
    tool_state = observer.state.tool_state
    assert tool_state.context_baselines
    assert tool_state.context_update_seq == 1
    updater.reset()
    assert tool_state.context_baselines == {}
    assert tool_state.context_update_seq == 0


# ---------------------------------------------------------------------------
# AC 11.10 — factory build path
# ---------------------------------------------------------------------------


class _StateCard(ToolCard):
    name: str = "state-card"
    description: str = "card exposing one context-state provider"

    def get_tools(self) -> list[Callable[..., Any]]:
        return []

    def get_context_states(self) -> list[Callable[[], ContextState | None]]:
        def shared_provider() -> ContextState | None:
            return None

        return [shared_provider]


def test_get_context_updater_builds_engine_from_factory() -> None:
    observer = _Observer()
    factory = ToolFactory(tool_cards=[_StateCard()], observer=observer)
    updater = factory.get_context_updater()
    assert isinstance(updater, ContextUpdater)
    # The card's provider returns None, so there is nothing to say.
    assert updater.compose_update([]) is None


def test_duplicate_provider_names_raise_through_get_context_updater() -> None:
    observer = _Observer()
    factory = ToolFactory(tool_cards=[_StateCard(), _StateCard()], observer=observer)
    with pytest.raises(ValueError, match="shared_provider"):
        factory.get_context_updater()


def test_get_context_updater_requires_an_observer() -> None:
    factory = ToolFactory(tool_cards=[])
    with pytest.raises(ValueError, match="observer"):
        factory.get_context_updater()


def test_get_context_updater_rejects_a_non_actor_observer() -> None:
    class _EventOnlyObserver:
        def notify_event(self, event: object) -> None:
            pass

    factory = ToolFactory(tool_cards=[], observer=_EventOnlyObserver())
    with pytest.raises(ValueError, match="ActorToolObserver"):
        factory.get_context_updater()


# ---------------------------------------------------------------------------
# AC 10 — export symmetry
# ---------------------------------------------------------------------------


def test_context_updater_is_exported_from_core_and_root() -> None:
    assert CoreContextUpdater is ContextUpdater
    assert RootContextUpdater is ContextUpdater
