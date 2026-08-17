"""The ``NotificationTool`` card: wiring, capabilities, ownership (AC1–4, 9–12)."""

from __future__ import annotations

import gc
from typing import Any

import akgentic.tool
import pytest
from akgentic.core.agent_config import BaseConfig
from akgentic.tool.core import ToolCard, ToolFactory
from akgentic.tool.errors import RetriableError
from akgentic.tool.notification.actor import (
    NOTIFICATION_ACTOR_NAME,
    NOTIFICATION_ACTOR_ROLE,
    NotificationActor,
)
from akgentic.tool.notification.models import DEFAULT_MESSAGE_CLASS, NotificationConfig
from akgentic.tool.notification.tool import (
    DEFAULT_MAX_DELAY_SECONDS,
    CancelNotification,
    ListPendingNotifications,
    NotificationTool,
)
from pydantic_ai.tools import Tool

from tests.notification.conftest import (
    FAKE_MESSAGE_PATH,
    FakeActorToolObserver,
    FakeOrchestratorProxy,
)


def _tool_named(card: NotificationTool, name: str) -> Any:
    """Return the card's tool callable called *name*."""
    return next(tool for tool in card.get_tools() if tool.__name__ == name)


class TestCardSurface:
    def test_declares_exactly_two_fields(self) -> None:
        assert set(NotificationTool.model_fields) == {"message_class", "max_delay_seconds"}

    def test_defaults(self) -> None:
        card = NotificationTool()
        assert card.message_class == DEFAULT_MESSAGE_CLASS
        assert card.max_delay_seconds == DEFAULT_MAX_DELAY_SECONDS

    def test_is_a_tool_card_exported_from_the_package_root(self) -> None:
        assert issubclass(NotificationTool, ToolCard)
        assert akgentic.tool.NotificationTool is NotificationTool
        assert "NotificationTool" in akgentic.tool.__all__

    def test_the_actor_proxy_is_a_private_attribute_not_a_field(self) -> None:
        assert "_notification_proxy" not in NotificationTool.model_fields
        assert "_notification_proxy" in NotificationTool.__private_attributes__

    def test_a_bound_card_still_round_trips(self, wired_card: NotificationTool) -> None:
        """Golden Rule #1b as the guarantee it exists for: a *bound* card serializes.

        Asserting on ``arbitrary_types_allowed`` would be vacuous — every model in
        this package inherits it from ``SerializableBaseModel``. What the rule
        protects is that a card holding a live actor proxy still round-trips,
        which holds only while that proxy is a private attribute.
        """
        assert wired_card._notification_proxy is not None  # genuinely bound

        restored = NotificationTool.model_validate(wired_card.model_dump())
        assert restored.message_class == FAKE_MESSAGE_PATH
        assert restored.max_delay_seconds == DEFAULT_MAX_DELAY_SECONDS
        assert restored._notification_proxy is None, "runtime state must not survive"

    def test_survives_a_json_round_trip(self) -> None:
        card = NotificationTool(message_class=FAKE_MESSAGE_PATH, max_delay_seconds=42)
        restored = NotificationTool.model_validate_json(card.model_dump_json())
        assert restored.message_class == FAKE_MESSAGE_PATH
        assert restored.max_delay_seconds == 42


class TestWiring:
    def test_binds_the_singleton_under_its_prefixed_name(
        self, orchestrator_proxy: FakeOrchestratorProxy, wired_card: NotificationTool
    ) -> None:
        assert len(orchestrator_proxy.create_calls) == 1
        actor_class, config = orchestrator_proxy.create_calls[0]
        assert actor_class is NotificationActor
        assert isinstance(config, NotificationConfig)
        assert config.name == NOTIFICATION_ACTOR_NAME == "#NotificationTool"
        assert config.role == NOTIFICATION_ACTOR_ROLE
        assert config.message_class == FAKE_MESSAGE_PATH

    @pytest.mark.parametrize(
        ("bad_path", "expected"),
        [
            ("tests.notification.no_such_module.Thing", "not importable"),
            ("tests.notification.conftest.NoSuchClass", "no attribute"),
            ("tests.notification.conftest.NotAMessage", "not a Message subclass"),
            ("akgentic.core.messages.message.Message", "does not declare"),
        ],
    )
    def test_a_bad_message_class_raises_at_bind_time(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        observer: FakeActorToolObserver,
        bad_path: str,
        expected: str,
    ) -> None:
        """And leaves no actor behind: validation runs before the singleton is bound."""
        card = NotificationTool(message_class=bad_path)
        with pytest.raises(ValueError, match=expected):
            card.observer(observer)
        assert orchestrator_proxy.create_calls == []

    def test_requires_an_orchestrator(self, observer: FakeActorToolObserver) -> None:
        observer._orchestrator = None
        card = NotificationTool(message_class=FAKE_MESSAGE_PATH)
        with pytest.raises(ValueError, match="orchestrator"):
            card.observer(observer)

    def test_observer_returns_self_for_chaining(self, observer: FakeActorToolObserver) -> None:
        card = NotificationTool(message_class=FAKE_MESSAGE_PATH)
        assert card.observer(observer) is card

    def test_using_an_unwired_card_is_a_programming_error(self) -> None:
        """Not a ``RetriableError``: no LLM input can fix a card that was never wired."""
        with pytest.raises(RuntimeError, match="observer"):
            NotificationTool().get_tools()

    def test_two_cards_share_one_actor(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        observer: FakeActorToolObserver,
        wired_card: NotificationTool,
    ) -> None:
        """AC9: one ``#NotificationTool``, and state crosses between the two cards."""
        second_card = NotificationTool(message_class=FAKE_MESSAGE_PATH)
        second_card.observer(observer)

        assert len(orchestrator_proxy.children) == 1
        assert len(orchestrator_proxy.create_calls) == 2  # both asked; one actor exists
        assert second_card._notification_proxy is wired_card._notification_proxy

        _tool_named(wired_card, "send_notification_message")("via the first card", 30)
        listed = _tool_named(second_card, "list_pending_notifications")()
        assert "via the first card" in listed


class TestCapabilitySurface:
    def test_get_tools_returns_the_three_capabilities(self, wired_card: NotificationTool) -> None:
        assert [tool.__name__ for tool in wired_card.get_tools()] == [
            "send_notification_message",
            "list_pending_notifications",
            "cancel_notification",
        ]

    def test_get_commands_returns_only_list_and_cancel(self, wired_card: NotificationTool) -> None:
        commands = wired_card.get_commands()
        assert set(commands) == {ListPendingNotifications, CancelNotification}
        assert {fn.__name__ for fn in commands.values()} == {
            "list_pending_notifications",
            "cancel_notification",
        }

    def test_the_send_docstring_states_the_cap(self, observer: FakeActorToolObserver) -> None:
        """AC2: the LLM sees the cap in the tool schema, which is the docstring."""
        card = NotificationTool(message_class=FAKE_MESSAGE_PATH, max_delay_seconds=45)
        card.observer(observer)
        doc = _tool_named(card, "send_notification_message").__doc__
        assert doc is not None
        assert "45 seconds" in doc

    def test_the_send_schema_the_model_sees_states_the_cap(
        self, observer: FakeActorToolObserver
    ) -> None:
        """AC2 as the model experiences it, not as a raw ``__doc__`` string.

        The cap is appended to the docstring at bind time, so the append must
        leave a docstring the schema builder can still parse. Asserting on
        ``__doc__`` alone cannot see that: a docstring whose sections stopped
        parsing still contains the substring. ``require_parameter_descriptions``
        is what makes a broken Args section fail here rather than silently ship
        an argument-less schema.
        """
        card = NotificationTool(message_class=FAKE_MESSAGE_PATH, max_delay_seconds=45)
        card.observer(observer)

        tool = Tool(
            _tool_named(card, "send_notification_message"),
            require_parameter_descriptions=True,
        )

        assert "45 seconds" in (tool.description or "")
        assert set(tool.function_schema.json_schema["properties"]) == {"content", "delay_seconds"}

    def test_every_capability_reaches_the_model_with_described_parameters(
        self, wired_card: NotificationTool
    ) -> None:
        """An undescribed argument is an argument the model has to guess at."""
        for capability in wired_card.get_tools():
            Tool(capability, require_parameter_descriptions=True)

    def test_the_commands_reach_the_registry_by_name(self, observer: FakeActorToolObserver) -> None:
        """Signatures must be derivable, or registry construction raises."""
        card = NotificationTool(message_class=FAKE_MESSAGE_PATH)
        registry = ToolFactory(tool_cards=[card], observer=observer).get_command_registry()

        assert registry.has("list_pending_notifications")
        assert registry.has("cancel_notification")
        assert not registry.has("send_notification_message")
        assert registry.dispatch("/list_pending_notifications") == "No pending notifications."

    def test_a_command_coerces_its_integer_argument(
        self, observer: FakeActorToolObserver, wired_card: NotificationTool
    ) -> None:
        """``/cancel_notification 1`` must reach the callable with an ``int``."""
        _tool_named(wired_card, "send_notification_message")("cancel me by command", 60)

        card = NotificationTool(message_class=FAKE_MESSAGE_PATH)
        registry = ToolFactory(tool_cards=[card], observer=observer).get_command_registry()

        assert registry.dispatch("/cancel_notification 1") == "Notification 1 cancelled."


class TestSchedule:
    def test_returns_an_id_and_stores_an_absolute_due_time(
        self, wired_card: NotificationTool, notification_actor: NotificationActor
    ) -> None:
        result = _tool_named(wired_card, "send_notification_message")("check CI", 30)

        assert "1" in result
        entry = notification_actor.state.pending[1]
        assert entry.notification_id == 1
        assert entry.content == "check CI"
        assert entry.owner is not None
        assert (entry.fire_at - entry.created_at).total_seconds() == pytest.approx(30, abs=1)

    def test_ids_are_monotonic(self, wired_card: NotificationTool) -> None:
        send = _tool_named(wired_card, "send_notification_message")
        first, second = send("a", 10), send("b", 10)
        assert "1" in first
        assert "2" in second

    @pytest.mark.parametrize("delay", [0, -1, DEFAULT_MAX_DELAY_SECONDS + 1])
    def test_a_delay_outside_the_range_is_retriable(
        self, wired_card: NotificationTool, notification_actor: NotificationActor, delay: int
    ) -> None:
        send = _tool_named(wired_card, "send_notification_message")
        with pytest.raises(RetriableError, match=f"between 1 and {DEFAULT_MAX_DELAY_SECONDS}"):
            send("nope", delay)
        assert notification_actor.state.pending == {}

    @pytest.mark.parametrize("delay", [1, DEFAULT_MAX_DELAY_SECONDS])
    def test_both_ends_of_the_range_are_accepted(
        self, wired_card: NotificationTool, notification_actor: NotificationActor, delay: int
    ) -> None:
        _tool_named(wired_card, "send_notification_message")("edge", delay)
        assert len(notification_actor.state.pending) == 1


class TestListAndCancelAreOwnershipScoped:
    def test_list_shows_only_the_callers_own_entries(
        self, orchestrator_proxy: FakeOrchestratorProxy, wired_card: NotificationTool
    ) -> None:
        other_observer = FakeActorToolObserver(orchestrator_proxy, name="bob")
        other_card = NotificationTool(message_class=FAKE_MESSAGE_PATH)
        other_card.observer(other_observer)

        _tool_named(wired_card, "send_notification_message")("alice's own", 60)
        _tool_named(other_card, "send_notification_message")("bob's own", 60)

        alice_view = _tool_named(wired_card, "list_pending_notifications")()
        bob_view = _tool_named(other_card, "list_pending_notifications")()

        assert "alice's own" in alice_view
        assert "bob's own" not in alice_view
        assert "bob's own" in bob_view
        assert "alice's own" not in bob_view

    def test_list_reports_the_remaining_time(self, wired_card: NotificationTool) -> None:
        _tool_named(wired_card, "send_notification_message")("soon", 60)
        listed = _tool_named(wired_card, "list_pending_notifications")()
        assert "id 1: soon" in listed
        assert "60 seconds" in listed

    def test_list_is_friendly_when_there_is_nothing(self, wired_card: NotificationTool) -> None:
        assert _tool_named(wired_card, "list_pending_notifications")() == (
            "No pending notifications."
        )

    def test_cancel_removes_the_callers_own_entry(
        self, wired_card: NotificationTool, notification_actor: NotificationActor
    ) -> None:
        _tool_named(wired_card, "send_notification_message")("drop me", 60)
        result = _tool_named(wired_card, "cancel_notification")(1)
        assert "cancelled" in result
        assert notification_actor.state.pending == {}

    def test_cancelling_an_unknown_id_is_retriable(self, wired_card: NotificationTool) -> None:
        with pytest.raises(RetriableError, match="No pending notification with id 99"):
            _tool_named(wired_card, "cancel_notification")(99)

    def test_cancelling_another_agents_entry_is_refused_and_keeps_it(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        wired_card: NotificationTool,
        notification_actor: NotificationActor,
    ) -> None:
        """The error must not be a disguised delete: bob's entry is still there."""
        other_observer = FakeActorToolObserver(orchestrator_proxy, name="bob")
        other_card = NotificationTool(message_class=FAKE_MESSAGE_PATH)
        other_card.observer(other_observer)
        _tool_named(other_card, "send_notification_message")("bob's own", 60)

        with pytest.raises(RetriableError, match="No pending notification with id 1"):
            _tool_named(wired_card, "cancel_notification")(1)

        assert notification_actor.state.pending[1].content == "bob's own"


class TestWeakObserver:
    def test_closures_do_not_pin_the_owning_agent(
        self, orchestrator_proxy: FakeOrchestratorProxy
    ) -> None:
        """ADR-030: a closure holds the *accessor*, so the agent is still collectable."""
        observer = FakeActorToolObserver(orchestrator_proxy)
        card = NotificationTool(message_class=FAKE_MESSAGE_PATH)
        card.observer(observer)
        send = _tool_named(card, "send_notification_message")
        cancel = _tool_named(card, "cancel_notification")

        del observer
        gc.collect()

        assert card._observer_or_none() is None
        with pytest.raises(RetriableError, match="shutting down"):
            send("too late", 10)
        with pytest.raises(RetriableError, match="shutting down"):
            cancel(1)


def test_the_actor_config_carries_a_plain_base_config_shape() -> None:
    """The singleton's config is ordinary serializable configuration."""
    config = NotificationConfig(name=NOTIFICATION_ACTOR_NAME, message_class=FAKE_MESSAGE_PATH)
    assert isinstance(config, BaseConfig)
    restored = NotificationConfig.model_validate(config.model_dump())
    assert restored.message_class == FAKE_MESSAGE_PATH
