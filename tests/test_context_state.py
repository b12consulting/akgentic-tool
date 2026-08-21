"""Story 31-1: ``ContextState`` contract, ``LLM_CONTEXT`` channel, card hook,
factory aggregation, and the per-param ``expose`` normalizer."""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, Self

import pytest
from pydantic import field_validator

from akgentic.tool.core import (
    LLM_CONTEXT,
    BaseToolParam,
    Channels,
    ContextState,
    ToolCard,
    ToolFactory,
    normalize_system_prompt_to_llm_context,
)


class _RosterState(ContextState):
    """Concrete test state: a member roster diffed by additions."""

    members: list[str] = []

    def render_full(self) -> str:
        return ", ".join(self.members)

    def render_delta(self, previous: Self) -> str | None:
        added = [m for m in self.members if m not in previous.members]
        if not added:
            return None
        return "joined: " + ", ".join(added)


# ---------------------------------------------------------------------------
# AC 2 — LLM_CONTEXT channel
# ---------------------------------------------------------------------------


def test_llm_context_channel_member_and_alias() -> None:
    """Channels gains LLM_CONTEXT; the module-level alias matches; SYSTEM_PROMPT stays."""
    assert Channels.LLM_CONTEXT == "llm_context"
    assert LLM_CONTEXT is Channels.LLM_CONTEXT
    # No member removed or redefined.
    assert Channels.SYSTEM_PROMPT == "system_prompt"
    assert Channels.TOOL_CALL == "tool_call"
    assert Channels.COMMAND == "command"


# ---------------------------------------------------------------------------
# AC 1 — abstract contract enforcement
# ---------------------------------------------------------------------------


def test_subclass_missing_render_delta_raises_type_error() -> None:
    class _FullOnly(ContextState):
        def render_full(self) -> str:
            return "state"

    with pytest.raises(TypeError):
        _FullOnly()


def test_subclass_missing_render_full_raises_type_error() -> None:
    class _DeltaOnly(ContextState):
        def render_delta(self, previous: Self) -> str | None:
            return None

    with pytest.raises(TypeError):
        _DeltaOnly()


def test_concrete_subclass_render_semantics() -> None:
    """Empty state renders ""; an unchanged state diffs to None."""
    empty = _RosterState()
    assert empty.render_full() == ""

    a = _RosterState(members=["alice"])
    b = _RosterState(members=["alice", "bob"])
    assert a.render_delta(a) is None
    assert b.render_delta(a) == "joined: bob"


# ---------------------------------------------------------------------------
# AC 6 — serialization round-trip (NFR3)
# ---------------------------------------------------------------------------


def test_concrete_state_round_trips_through_model_dump() -> None:
    state = _RosterState(members=["alice", "bob"])
    restored = _RosterState.model_validate(state.model_dump())
    assert restored == state
    assert restored.render_full() == "alice, bob"


# ---------------------------------------------------------------------------
# AC 3 — card hook default
# ---------------------------------------------------------------------------


class _BareCard(ToolCard):
    def get_tools(self) -> list[Callable[..., object]]:
        return []


def test_tool_card_get_context_states_defaults_to_empty() -> None:
    assert _BareCard().get_context_states() == []


# ---------------------------------------------------------------------------
# AC 4 — factory aggregation in dependency order, duplicate-name raise
# ---------------------------------------------------------------------------


class _PrereqCard(ToolCard):
    def get_tools(self) -> list[Callable[..., object]]:
        return []

    def get_context_states(self) -> list[Callable[[], ContextState | None]]:
        def prereq_state() -> ContextState | None:
            return _RosterState(members=["alice"])

        return [prereq_state]


class _DependentCard(ToolCard):
    depends_on: ClassVar[list[str]] = ["_PrereqCard"]

    def get_tools(self) -> list[Callable[..., object]]:
        return []

    def get_context_states(self) -> list[Callable[[], ContextState | None]]:
        def dependent_state() -> ContextState | None:
            return None

        return [dependent_state]


def test_factory_aggregates_providers_in_dependency_order() -> None:
    """The dependent card's providers come after its prerequisite's, whatever the input order."""
    factory = ToolFactory(tool_cards=[_DependentCard(), _PrereqCard()])
    providers = factory.get_context_states()
    assert [p.__name__ for p in providers] == ["prereq_state", "dependent_state"]
    state = providers[0]()
    assert isinstance(state, _RosterState)
    assert providers[1]() is None


class _CollidingCardA(ToolCard):
    def get_tools(self) -> list[Callable[..., object]]:
        return []

    def get_context_states(self) -> list[Callable[[], ContextState | None]]:
        def shared_state() -> ContextState | None:
            return None

        return [shared_state]


class _CollidingCardB(ToolCard):
    def get_tools(self) -> list[Callable[..., object]]:
        return []

    def get_context_states(self) -> list[Callable[[], ContextState | None]]:
        def shared_state() -> ContextState | None:
            return None

        return [shared_state]


def test_factory_duplicate_provider_name_raises_naming_both_cards() -> None:
    factory = ToolFactory(tool_cards=[_CollidingCardA(), _CollidingCardB()])
    with pytest.raises(ValueError) as exc:
        factory.get_context_states()
    msg = str(exc.value)
    assert "shared_state" in msg
    assert "_CollidingCardA" in msg
    assert "_CollidingCardB" in msg


# ---------------------------------------------------------------------------
# AC 5 — per-param opt-in expose normalizer
# ---------------------------------------------------------------------------


class _MigratedParam(BaseToolParam):
    """Param class that opted in to the SYSTEM_PROMPT → LLM_CONTEXT normalization."""

    expose: set[Channels] = {Channels.LLM_CONTEXT, Channels.COMMAND}

    _normalize_expose = field_validator("expose", mode="after")(
        normalize_system_prompt_to_llm_context
    )


class _UntouchedParam(BaseToolParam):
    """Param class that did NOT opt in — SYSTEM_PROMPT must survive validation."""

    expose: set[Channels] = {Channels.SYSTEM_PROMPT, Channels.COMMAND}


def test_normalizer_maps_persisted_system_prompt_to_llm_context() -> None:
    param = _MigratedParam.model_validate({"expose": ["system_prompt", "command"]})
    assert param.expose == {Channels.LLM_CONTEXT, Channels.COMMAND}


def test_normalizer_is_a_no_op_without_system_prompt() -> None:
    param = _MigratedParam.model_validate({"expose": ["tool_call", "command"]})
    assert param.expose == {Channels.TOOL_CALL, Channels.COMMAND}


def test_param_not_using_normalizer_keeps_system_prompt() -> None:
    """The one test a global rewrite fails: normalization is opt-in per param class."""
    param = _UntouchedParam.model_validate({"expose": ["system_prompt", "command"]})
    assert param.expose == {Channels.SYSTEM_PROMPT, Channels.COMMAND}
