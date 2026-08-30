"""Tests for the model domain's vocabulary and observer contract (Story 36-1).

Covers the ``ModelRow`` projection round-trip, the runtime conformance of
``ModelSwitchToolObserver`` in both directions, and the export identity of both
symbols through ``akgentic.tool.model`` and the package root.

The conforming fake supplies the six members ``ActorToolObserver`` declares on
this branch base — ``notify_event``, ``myAddress``, ``orchestrator``,
``team_id``, ``state``, ``proxy_ask`` — and nothing more, so the two negative
cases isolate the two new members rather than an incidental omission.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

from akgentic.core.actor_address import ActorAddress
from akgentic.core.agent import AkgentType

import akgentic.tool as tool_package
import akgentic.tool.model as model_package
from akgentic.tool import ModelRow as RootModelRow
from akgentic.tool import ModelSwitchToolObserver as RootModelSwitchToolObserver
from akgentic.tool.core.observer import ActorToolObserver
from akgentic.tool.core.state import ToolState
from akgentic.tool.model import ModelRow as PackageModelRow
from akgentic.tool.model import ModelSwitchToolObserver as PackageModelSwitchToolObserver
from akgentic.tool.model.observer import ModelSwitchToolObserver
from akgentic.tool.model.state import ModelRow

# ---------------------------------------------------------------------------
# AC 2 — ModelRow is a serializable projection
# ---------------------------------------------------------------------------


def test_model_row_round_trip_preserves_every_field() -> None:
    """A populated row survives dump/validate with every field equal."""
    row = ModelRow(
        key="openai:gpt-5.2",
        provider="openai",
        model="gpt-5.2",
        active=True,
        context_length=400_000,
    )

    restored = ModelRow.model_validate(row.model_dump())

    assert restored == row
    assert restored.key == "openai:gpt-5.2"
    assert restored.provider == "openai"
    assert restored.model == "gpt-5.2"
    assert restored.active is True
    assert restored.context_length == 400_000


def test_model_row_round_trips_an_undeclared_context_length() -> None:
    """``context_length`` is optional in value: ``None`` survives the round-trip."""
    row = ModelRow(
        key="anthropic:claude-opus-5",
        provider="anthropic",
        model="claude-opus-5",
        active=False,
        context_length=None,
    )

    restored = ModelRow.model_validate(row.model_dump())

    assert restored == row
    assert restored.context_length is None
    assert restored.active is False


def test_model_row_declares_its_fields_in_the_decided_order() -> None:
    """Field order is part of the contract, and it is what the dump emits."""
    assert list(ModelRow.model_fields) == [
        "key",
        "provider",
        "model",
        "active",
        "context_length",
    ]


# ---------------------------------------------------------------------------
# AC 4 — runtime conformance of ModelSwitchToolObserver, both ways
# ---------------------------------------------------------------------------


class _Carrier:
    """Minimal ``ToolStateCarrier``: an object exposing ``tool_state``."""

    def __init__(self) -> None:
        self.tool_state = ToolState()


class _ObserverBase:
    """Every member ``ActorToolObserver`` declares on this base, and nothing more."""

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


class _ModelSwitchObserver(_ObserverBase):
    """The base plus both new members — the conforming case."""

    def list_model_rows(self) -> list[ModelRow]:
        return []

    def switch_model(self, key: str) -> str:
        return f"switched to {key}"


class _WithoutListModelRows(_ObserverBase):
    """Otherwise identical, missing ``list_model_rows``."""

    def switch_model(self, key: str) -> str:
        return f"switched to {key}"


class _WithoutSwitchModel(_ObserverBase):
    """Otherwise identical, missing ``switch_model``."""

    def list_model_rows(self) -> list[ModelRow]:
        return []


def test_conforming_observer_satisfies_the_protocol() -> None:
    """Every base member plus both new methods passes the runtime check."""
    assert isinstance(_ModelSwitchObserver(), ModelSwitchToolObserver)


def test_observer_missing_list_model_rows_fails() -> None:
    """``list_model_rows`` is required — its absence alone fails the check."""
    assert not isinstance(_WithoutListModelRows(), ModelSwitchToolObserver)


def test_observer_missing_switch_model_fails() -> None:
    """``switch_model`` is required — its absence alone fails the check."""
    assert not isinstance(_WithoutSwitchModel(), ModelSwitchToolObserver)


def test_the_negative_cases_isolate_the_new_members() -> None:
    """Guard the guards: both negatives still satisfy the base protocol.

    Without this, a fake that failed for a missing ``team_id`` would look like
    proof that ``switch_model`` is required.
    """
    assert isinstance(_WithoutListModelRows(), ActorToolObserver)
    assert isinstance(_WithoutSwitchModel(), ActorToolObserver)


def test_the_base_protocol_is_not_widened() -> None:
    """A plain ``ActorToolObserver`` stand-in is unaffected by the sibling protocol."""
    base = _ObserverBase()
    assert isinstance(base, ActorToolObserver)
    assert not isinstance(base, ModelSwitchToolObserver)


# ---------------------------------------------------------------------------
# AC 8 — exports
# ---------------------------------------------------------------------------


def test_both_symbols_are_exported_by_identity() -> None:
    """One class each, reachable from the domain package and from the root."""
    assert PackageModelRow is ModelRow
    assert RootModelRow is ModelRow
    assert PackageModelSwitchToolObserver is ModelSwitchToolObserver
    assert RootModelSwitchToolObserver is ModelSwitchToolObserver


def test_both_symbols_are_named_in_both_all_lists() -> None:
    """The export surface declares them, not merely the import machinery."""
    assert "ModelRow" in model_package.__all__
    assert "ModelSwitchToolObserver" in model_package.__all__
    assert "ModelRow" in tool_package.__all__
    assert "ModelSwitchToolObserver" in tool_package.__all__
