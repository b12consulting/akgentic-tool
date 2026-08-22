"""Tests for ``MailboxTool``: three capabilities on three channels (ADR-040 §7)."""

from __future__ import annotations

import gc

import pytest
from akgentic.core.messages import Message

from akgentic.tool import ToolFactory
from akgentic.tool.core import COMMAND, LLM_CONTEXT
from akgentic.tool.errors import ToolObserverGone
from akgentic.tool.mailbox import (
    MailboxState,
    MailboxStatus,
    MailboxTool,
    MailboxToolObserver,
    ReadMailbox,
    Stop,
)
from tests.mailbox.conftest import _user_message


class _FakeObserver:
    """Structural MailboxToolObserver: real messages, no actor system."""

    def __init__(self, messages: list[Message] | None = None) -> None:
        self._messages = messages or []

    def notify_event(self, event: object) -> None:
        pass

    def get_mailbox(self) -> list[Message]:
        return list(self._messages)


def _wired_card(
    messages: list[Message] | None = None, **fields: object
) -> tuple[MailboxTool, _FakeObserver]:
    """A card with an attached fake observer; the observer is returned to keep it alive."""
    observer = _FakeObserver(messages)
    assert isinstance(observer, MailboxToolObserver)  # structural conformance
    card = MailboxTool.model_validate(fields)
    card.observer(observer)
    return card, observer


# ── the silent-drop assertion (backlog row 28): assert the lists, not behaviour ──


def test_default_card_yields_exactly_one_provider_one_tool_one_command() -> None:
    card, observer = _wired_card()

    providers = card.get_context_states()
    assert [provider.__name__ for provider in providers] == ["mailbox_state"]

    tools = card.get_tools()
    assert [tool.__name__ for tool in tools] == ["read_mailbox"]

    commands = card.get_commands()
    assert set(commands) == {Stop}
    assert commands[Stop].__name__ == "stop"


def test_factory_registers_exactly_one_command_under_canonical_name_stop() -> None:
    observer = _FakeObserver()
    factory = ToolFactory([MailboxTool()], observer=observer)

    registry = factory.get_command_registry()
    descriptors = registry.descriptors()
    assert [descriptor.name for descriptor in descriptors] == ["stop"]
    assert descriptors[0].args == []
    assert registry.has("stop")


def test_factory_aggregates_the_single_mailbox_state_provider() -> None:
    observer = _FakeObserver()
    factory = ToolFactory([MailboxTool()], observer=observer)

    providers = factory.get_context_states()
    assert [provider.__name__ for provider in providers] == ["mailbox_state"]


# ── read_mailbox (FR4) ───────────────────────────────────────────────────────


def test_read_mailbox_renders_sender_type_and_full_content() -> None:
    long_content = "first line\n" + "x" * 300
    card, observer = _wired_card(
        [_user_message("@Alice", long_content), _user_message("@Bob", "ping")]
    )
    read_mailbox = card.get_tools()[0]

    rendered = read_mailbox()
    assert "@Alice" in rendered
    assert "@Bob" in rendered
    assert "UserMessage" in rendered
    assert long_content in rendered  # full content, not the ~120-char preview
    assert "ping" in rendered


class _NoneContentMessage(Message):
    """A message whose content is explicitly ``None``."""

    content: str | None = None


def test_read_mailbox_renders_none_content_as_empty() -> None:
    # A content-less message still renders its sender and type, never "None".
    card, observer = _wired_card([_NoneContentMessage()])
    rendered = card.get_tools()[0]()

    assert "unknown" in rendered
    assert "_NoneContentMessage" in rendered
    assert "None" not in rendered.replace("_NoneContentMessage", "")


def test_read_mailbox_empty_mailbox_returns_a_sentence_not_empty_string() -> None:
    card, observer = _wired_card([])
    read_mailbox = card.get_tools()[0]

    rendered = read_mailbox()
    assert rendered != ""
    assert "mailbox is empty" in rendered


def test_read_mailbox_docstring_carries_the_redelivery_contract() -> None:
    # The docstring IS the model-facing tool description — functional surface.
    card, observer = _wired_card()
    doc = card.get_tools()[0].__doc__ or ""

    assert "not consume" in doc.lower()
    assert "delivered" in doc
    assert "as its own turn" in doc


def test_read_mailbox_twice_returns_the_same_messages() -> None:
    # The peek is non-consuming: a second read lists the same pending messages.
    card, observer = _wired_card([_user_message("@Alice", "hello")])
    read_mailbox = card.get_tools()[0]

    assert read_mailbox() == read_mailbox()
    assert "@Alice" in read_mailbox()


def test_read_mailbox_after_observer_collected_raises_tool_observer_gone() -> None:
    # In-life code: the raising form is the defined outcome (accessor captured, not observer).
    card, observer = _wired_card([])
    read_mailbox = card.get_tools()[0]

    del observer
    gc.collect()
    with pytest.raises(ToolObserverGone):
        read_mailbox()


# ── stop (FR5) ───────────────────────────────────────────────────────────────


def test_stop_invoked_directly_returns_none() -> None:
    card, observer = _wired_card()
    stop = card.get_commands()[Stop]

    assert stop() is None


def test_stop_dispatch_through_registry_returns_none() -> None:
    # The registry must propagate None, not string-render it as "None".
    observer = _FakeObserver()
    factory = ToolFactory([MailboxTool()], observer=observer)

    assert factory.get_command_registry().dispatch("/stop") is None


# ── ownership: the cancel vocabulary lives with the enforcement, not the card ──


def test_cancel_vocabulary_is_not_importable_from_this_package() -> None:
    # A stale re-export is exactly what story 34-4 removes; assert the absence.
    import akgentic.tool.mailbox as mailbox_module

    with pytest.raises(ImportError):
        from akgentic.tool.mailbox import is_cancel  # noqa: F401
    with pytest.raises(ImportError):
        from akgentic.tool.mailbox import render_arrival_notice  # noqa: F401

    assert "is_cancel" not in dir(mailbox_module)
    assert "render_arrival_notice" not in dir(mailbox_module)


def test_mailbox_tool_observer_still_imports_from_the_package_root() -> None:
    # The agent's capability module imports this symbol from exactly this path.
    from akgentic.tool.mailbox import MailboxToolObserver as ObserverProtocol

    assert isinstance(_FakeObserver(), ObserverProtocol)


# ── channel gating (FR3): disabling removes exactly one capability ───────────


def test_disabling_mailbox_status_removes_only_the_provider() -> None:
    card, observer = _wired_card(mailbox_status=False)

    assert card.get_context_states() == []
    assert [tool.__name__ for tool in card.get_tools()] == ["read_mailbox"]
    assert set(card.get_commands()) == {Stop}


def test_disabling_read_mailbox_removes_only_the_tool() -> None:
    card, observer = _wired_card(read_mailbox=False)

    assert [provider.__name__ for provider in card.get_context_states()] == ["mailbox_state"]
    assert card.get_tools() == []
    assert set(card.get_commands()) == {Stop}


def test_disabling_stop_removes_only_the_command() -> None:
    card, observer = _wired_card(stop=False)

    assert [provider.__name__ for provider in card.get_context_states()] == ["mailbox_state"]
    assert [tool.__name__ for tool in card.get_tools()] == ["read_mailbox"]
    assert card.get_commands() == {}


def test_capability_exposed_off_its_served_channel_is_dropped() -> None:
    # The gate is param resolved AND the served channel in its expose set.
    card, observer = _wired_card(
        mailbox_status=MailboxStatus(expose={COMMAND}),
        read_mailbox=ReadMailbox(expose={COMMAND}),
        stop=Stop(expose={LLM_CONTEXT}),
    )

    assert card.get_context_states() == []
    assert card.get_tools() == []
    assert card.get_commands() == {}


# ── serializability (FR7 / Golden Rule #1b) ──────────────────────────────────


def test_default_card_round_trips_through_pydantic() -> None:
    card = MailboxTool()
    restored = MailboxTool.model_validate(card.model_dump())
    assert restored == card


@pytest.mark.parametrize("field", ["mailbox_status", "read_mailbox", "stop"])
def test_card_with_disabled_param_round_trips(field: str) -> None:
    card = MailboxTool.model_validate({field: False})
    restored = MailboxTool.model_validate(card.model_dump())
    assert restored == card
    assert getattr(restored, field) is False


def test_card_with_explicit_params_round_trips() -> None:
    card = MailboxTool(
        mailbox_status=MailboxStatus(expose={LLM_CONTEXT, COMMAND}),
        read_mailbox=ReadMailbox(instructions="prefer wrapping up"),
        stop=Stop(),
    )
    restored = MailboxTool.model_validate(card.model_dump())
    assert restored == card
    assert isinstance(restored.mailbox_status, MailboxStatus)
    assert isinstance(restored.read_mailbox, ReadMailbox)
    assert restored.read_mailbox.instructions == "prefer wrapping up"
    assert isinstance(restored.stop, Stop)


# ── provider wiring (NFR2): the card's provider is 34-1's never-raise factory ──


def test_card_provider_returns_state_from_live_observer() -> None:
    card, observer = _wired_card([_user_message("@Alice", "hello")])
    provider = card.get_context_states()[0]

    state = provider()
    assert isinstance(state, MailboxState)
    assert state.rows[0].sender == "@Alice"


def test_card_provider_returns_none_when_observer_collected() -> None:
    card, observer = _wired_card([])
    provider = card.get_context_states()[0]

    del observer
    gc.collect()
    assert provider() is None
