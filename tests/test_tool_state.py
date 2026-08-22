"""Tests for ``ToolState`` and the observer-side state contract (Story 35-1).

Covers the default slot, the polymorphic round-trip that makes the persisted
cache trustworthy, and the runtime-checkable structural contracts
(``ToolStateCarrier``, the widened ``ActorToolObserver``).
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

from akgentic.core.actor_address import ActorAddress
from akgentic.core.agent import AkgentType

from akgentic.tool import ToolState as RootToolState
from akgentic.tool import ToolStateCarrier as RootToolStateCarrier
from akgentic.tool.core import ToolState as CoreToolState
from akgentic.tool.core.observer import ActorToolObserver, ToolStateCarrier
from akgentic.tool.core.state import ToolState
from akgentic.tool.planning.state import PlanningState, TaskRow
from akgentic.tool.team.state import TeamMemberRow, TeamRosterState

# ---------------------------------------------------------------------------
# AC 2 — default state
# ---------------------------------------------------------------------------


def test_default_tool_state_is_empty() -> None:
    """A fresh slot carries no baselines and a zero sequence."""
    state = ToolState()
    assert state.context_baselines == {}
    assert state.context_update_seq == 0


def test_tool_state_is_exported_from_core_and_root() -> None:
    """AC 8: one class, importable from ``akgentic.tool.core`` and ``akgentic.tool``."""
    assert CoreToolState is ToolState
    assert RootToolState is ToolState
    assert RootToolStateCarrier is ToolStateCarrier


# ---------------------------------------------------------------------------
# AC 3 — polymorphic round-trip
# ---------------------------------------------------------------------------


def test_round_trip_preserves_concrete_context_state_types() -> None:
    """Two different concrete baselines survive dump/validate with types intact."""
    roster = TeamRosterState(
        members=[
            TeamMemberRow(name="alice", role="dev", is_self=True),
            TeamMemberRow(name="bob", role="pm", is_self=False),
        ]
    )
    planning = PlanningState(
        total=1,
        owner_counts={"alice": 1},
        tasks=[
            TaskRow(
                id=1, status="open", description="ship it", owner="alice", creator="bob", output=""
            )
        ],
        agent_name="alice",
        filter_by_agent=True,
    )
    state = ToolState(
        context_baselines={"roster_provider": roster, "planning_provider": planning},
        context_update_seq=7,
    )

    restored = ToolState.model_validate(state.model_dump())

    assert isinstance(restored.context_baselines["roster_provider"], TeamRosterState)
    assert isinstance(restored.context_baselines["planning_provider"], PlanningState)
    assert restored.context_baselines["roster_provider"] == roster
    assert restored.context_baselines["planning_provider"] == planning
    assert restored.context_update_seq == 7


# ---------------------------------------------------------------------------
# AC 6 — runtime conformance of the structural contracts
# ---------------------------------------------------------------------------


class _Carrier:
    """Minimal ``ToolStateCarrier``: an object exposing ``tool_state``."""

    def __init__(self) -> None:
        self.tool_state = ToolState()


class _FullObserver:
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


class _StatelessObserver(_FullObserver):
    """The same observer with the ``state`` member deleted — must fail the check."""

    def __init__(self) -> None:
        super().__init__()
        del self.state


def test_tool_state_carrier_runtime_check() -> None:
    """An object with ``tool_state`` satisfies the carrier; a bare object does not."""
    assert isinstance(_Carrier(), ToolStateCarrier)
    assert not isinstance(object(), ToolStateCarrier)


def test_observer_state_satisfies_carrier() -> None:
    """The value behind ``observer.state`` is itself a conforming carrier."""
    observer = _FullObserver()
    assert isinstance(observer.state, ToolStateCarrier)
    assert isinstance(observer.state.tool_state, ToolState)


def test_actor_tool_observer_requires_state() -> None:
    """``state`` is now part of the protocol: with it the check passes, without it it fails."""
    assert isinstance(_FullObserver(), ActorToolObserver)
    assert not isinstance(_StatelessObserver(), ActorToolObserver)
