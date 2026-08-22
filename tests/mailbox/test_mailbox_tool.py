"""Tests for ``MailboxTool``: two capabilities on two channels (ADR-040 §7, ADR-019 §4)."""

from __future__ import annotations

import gc
import uuid

import pytest
from akgentic.core.messages import CancelMessage, Message

from akgentic.tool import ToolFactory
from akgentic.tool.core import COMMAND, LLM_CONTEXT, TOOL_CALL
from akgentic.tool.errors import ToolObserverGone
from akgentic.tool.mailbox import (
    MailboxTool,
    MailboxToolObserver,
    ReadMailbox,
    Stop,
)
from tests.mailbox.conftest import _typed_message, _user_message


class _FakeObserver:
    """Structural MailboxToolObserver: real messages, no actor system.

    ``consume_mailbox`` really removes from the backing list and returns what it
    removed — a stub returning ``[]`` would make the consumption proof
    unfalsifiable.
    """

    def __init__(self, messages: list[Message] | None = None) -> None:
        self._messages = messages or []

    def notify_event(self, event: object) -> None:
        pass

    def get_mailbox(self) -> list[Message]:
        return list(self._messages)

    def consume_mailbox(self, message_ids: list[uuid.UUID]) -> list[Message]:
        wanted = set(message_ids)
        removed = [message for message in self._messages if message.id in wanted]
        self._messages = [message for message in self._messages if message.id not in wanted]
        return removed

    def deliver(self, message: Message) -> None:
        """Queue one more message, as an arrival between two reads would."""
        self._messages.append(message)


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


def test_default_card_yields_one_tool_one_command_and_no_provider() -> None:
    card, observer = _wired_card()

    assert card.get_context_states() == []

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


# ── the LLM_CONTEXT capability is gone (ADR-019 §4b) ─────────────────────────


def test_card_serves_no_context_state_provider() -> None:
    card, observer = _wired_card([_user_message("@Alice", "hello")])

    assert card.get_context_states() == []


def test_factory_aggregates_no_mailbox_provider() -> None:
    observer = _FakeObserver([_user_message("@Alice", "hello")])
    factory = ToolFactory([MailboxTool()], observer=observer)

    assert factory.get_context_states() == []


def test_card_exposes_no_mailbox_status_field() -> None:
    # Removed, not deprecated. ToolCard keeps Pydantic's default extra="ignore",
    # so the value is dropped silently — assert the absent field, never a raise.
    card = MailboxTool(mailbox_status=True)  # type: ignore[call-arg]

    assert "mailbox_status" not in card.model_dump()
    assert not hasattr(card, "mailbox_status")
    assert "mailbox_status" not in MailboxTool.model_fields


def test_state_vocabulary_is_not_importable_from_this_package() -> None:
    # A stale re-export is exactly what this story removes; assert the absence.
    import akgentic.tool.mailbox as mailbox_module

    with pytest.raises(ImportError):
        from akgentic.tool.mailbox import MailboxStatus  # noqa: F401
    with pytest.raises(ImportError):
        from akgentic.tool.mailbox import MailboxState  # noqa: F401
    with pytest.raises(ImportError):
        from akgentic.tool.mailbox import MailboxRow  # noqa: F401
    with pytest.raises(ImportError):
        from akgentic.tool.mailbox import make_mailbox_state_provider  # noqa: F401

    for symbol in ("MailboxStatus", "MailboxState", "MailboxRow", "make_mailbox_state_provider"):
        assert symbol not in dir(mailbox_module)


# ── read_mailbox renders (FR4) ───────────────────────────────────────────────


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
    assert long_content in rendered  # full content, not a preview
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


def test_read_mailbox_docstring_carries_the_absorption_contract() -> None:
    # The docstring IS the model-facing tool description — functional surface,
    # not documentation: it is what tells the model the read is one-shot.
    card, observer = _wired_card()
    doc = (card.get_tools()[0].__doc__ or "").lower()

    assert "absorbs" in doc
    assert "not be delivered to you again" in doc
    assert "as its own turn" in doc


def test_read_mailbox_after_observer_collected_raises_tool_observer_gone() -> None:
    # In-life code: the raising form is the defined outcome (accessor captured, not observer).
    card, observer = _wired_card([])
    read_mailbox = card.get_tools()[0]

    del observer
    gc.collect()
    with pytest.raises(ToolObserverGone):
        read_mailbox()


# ── read_mailbox consumes what it renders (AC 2, ADR-019 §4) ─────────────────


def test_read_mailbox_consumes_what_it_rendered_and_a_second_read_sees_only_new_mail() -> None:
    # The double-answer failure, removed by construction: what the first call
    # showed the model is gone from the mailbox, so the second cannot show it again.
    card, observer = _wired_card([_user_message("@Alice", "the first message")])
    read_mailbox = card.get_tools()[0]

    first = read_mailbox()
    assert "the first message" in first

    observer.deliver(_user_message("@Bob", "the second message"))

    second = read_mailbox()
    assert "the second message" in second
    assert "the first message" not in second

    assert observer.get_mailbox() == []


def test_read_mailbox_renders_only_what_consumption_returned() -> None:
    # consume_mailbox is the authority: a message the peek saw but consumption
    # declined (a reply_to envelope, an id dequeued in between) must not render.
    class _PartialObserver(_FakeObserver):
        def consume_mailbox(self, message_ids: list[uuid.UUID]) -> list[Message]:
            # Refuse the first id, as an envelope carrying a reply_to would be.
            return super().consume_mailbox(message_ids[1:])

    kept = _user_message("@Alice", "blocked on a reply")
    absorbed = _user_message("@Bob", "ordinary mail")
    observer = _PartialObserver([kept, absorbed])
    card = MailboxTool()
    card.observer(observer)

    rendered = card.get_tools()[0]()

    assert "ordinary mail" in rendered
    assert "blocked on a reply" not in rendered
    assert observer.get_mailbox() == [kept]


def test_read_mailbox_empty_after_consumption_returns_the_empty_sentence() -> None:
    card, observer = _wired_card([_user_message("@Alice", "hello")])
    read_mailbox = card.get_tools()[0]

    read_mailbox()
    assert "mailbox is empty" in read_mailbox()


# ── a cancel is never rendered and never consumed (AC 3, AC 4) ───────────────


def test_cancel_is_neither_rendered_nor_consumed_while_ordinary_mail_is() -> None:
    # Both halves matter: the rendering assertion alone passes with the cancel
    # silently consumed, which would let the model read its way out of a cancel.
    cancel = CancelMessage(reason="user pressed stop")
    slash_stop = _user_message("@Alice", "/stop")
    ordinary = _user_message("@Bob", "ordinary mail")
    card, observer = _wired_card([cancel, slash_stop, ordinary])

    rendered = card.get_tools()[0]()

    assert "ordinary mail" in rendered
    assert "CancelMessage" not in rendered
    assert "/stop" not in rendered

    surviving = observer.get_mailbox()
    assert surviving == [cancel, slash_stop]


def test_mailbox_of_cancels_only_reads_as_empty() -> None:
    # Correct: the read must not teach the model that it is being cancelled.
    cancel = CancelMessage(reason="user pressed stop")
    card, observer = _wired_card([cancel])

    assert "mailbox is empty" in card.get_tools()[0]()
    assert observer.get_mailbox() == [cancel]


@pytest.mark.parametrize("content", ["/stop", "/stop now", "  /stop now", "/stop\nplease"])
def test_first_token_exactly_stop_is_excluded(content: str) -> None:
    card, observer = _wired_card([_user_message("@Alice", content)])

    assert "mailbox is empty" in card.get_tools()[0]()
    assert len(observer.get_mailbox()) == 1


@pytest.mark.parametrize("content", ["/stopwatch", "please /stop", "/stopping now", "stop"])
def test_a_near_miss_is_ordinary_mail_and_is_consumed(content: str) -> None:
    card, observer = _wired_card([_user_message("@Alice", content)])

    rendered = card.get_tools()[0]()

    assert content.strip() in rendered
    assert observer.get_mailbox() == []


# ── the reply protocol per message (AC 5) ────────────────────────────────────


@pytest.mark.parametrize(
    ("message_type", "article", "protocol_fragment"),
    [
        ("request", "a", "A reply is expected: respond to @Alice with the result."),
        ("response", "a", "This is a reply to something you asked."),
        ("instruction", "an", "Carry it out; acknowledge to @Alice only if asked to."),
        ("notification", "a", "Informational message. No reply is expected."),
        ("acknowledgment", "an", "Receipt confirmed. No further action needed."),
    ],
)
def test_each_known_type_renders_its_reply_protocol(
    message_type: str, article: str, protocol_fragment: str
) -> None:
    card, observer = _wired_card([_typed_message("@Alice", message_type, "the body")])

    rendered = card.get_tools()[0]()

    assert f"You received {article} {message_type} from @Alice." in rendered
    assert protocol_fragment in rendered
    assert "the body" in rendered


def test_a_message_without_a_type_renders_without_a_protocol_line() -> None:
    card, observer = _wired_card([_user_message("@Alice", "just content")])

    rendered = card.get_tools()[0]()

    assert "@Alice" in rendered
    assert "UserMessage" in rendered
    assert "just content" in rendered
    assert "You received" not in rendered


def test_an_unknown_type_renders_without_a_protocol_line_and_does_not_raise() -> None:
    card, observer = _wired_card([_typed_message("@Alice", "telepathy", "the body")])

    rendered = card.get_tools()[0]()

    assert "@Alice" in rendered
    assert "the body" in rendered
    assert "You received" not in rendered


# ── stop (FR5, AC 7) ─────────────────────────────────────────────────────────


def test_stop_invoked_directly_reports_there_is_nothing_to_cancel() -> None:
    card, observer = _wired_card()
    stop = card.get_commands()[Stop]

    assert stop() == "There is no run to cancel."


def test_stop_dispatch_through_registry_returns_the_idle_string() -> None:
    # Dispatch reaches a command only while the agent is idle, and a mid-run
    # cancel never survives to be dequeued — so this is the only case there is.
    observer = _FakeObserver()
    factory = ToolFactory([MailboxTool()], observer=observer)

    assert factory.get_command_registry().dispatch("/stop") == "There is no run to cancel."


def test_stop_still_announces_a_description_to_the_command_palette() -> None:
    # The docstring is the wire surface: descriptors() feeds
    # CommandDescriptor.description from it and CommandsAnnouncedEvent carries it
    # to every frontend. Rewriting the docstring must not empty the palette entry.
    observer = _FakeObserver()
    factory = ToolFactory([MailboxTool()], observer=observer)

    descriptor = factory.get_command_registry().descriptors()[0]
    assert "cancel" in descriptor.description.lower()


# ── ownership: the cancel vocabulary lives with the enforcement, not the card ──


def test_cancel_vocabulary_is_not_importable_from_this_package() -> None:
    # The card's exclusion filter is private and must not become a second
    # public vocabulary; the agent holds the canonical predicate.
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


def test_disabling_read_mailbox_removes_only_the_tool() -> None:
    card, observer = _wired_card(read_mailbox=False)

    assert card.get_tools() == []
    assert set(card.get_commands()) == {Stop}


def test_disabling_stop_removes_only_the_command() -> None:
    card, observer = _wired_card(stop=False)

    assert [tool.__name__ for tool in card.get_tools()] == ["read_mailbox"]
    assert card.get_commands() == {}


def test_capability_exposed_off_its_served_channel_is_dropped() -> None:
    # The gate is param resolved AND the served channel in its expose set.
    card, observer = _wired_card(
        read_mailbox=ReadMailbox(expose={COMMAND}),
        stop=Stop(expose={LLM_CONTEXT}),
    )

    assert card.get_tools() == []
    assert card.get_commands() == {}
    assert card.get_context_states() == []


# ── serializability (FR7 / Golden Rule #1b) ──────────────────────────────────


def test_default_card_round_trips_through_pydantic() -> None:
    card = MailboxTool()
    restored = MailboxTool.model_validate(card.model_dump())
    assert restored == card


@pytest.mark.parametrize("field", ["read_mailbox", "stop"])
def test_card_with_disabled_param_round_trips(field: str) -> None:
    card = MailboxTool.model_validate({field: False})
    restored = MailboxTool.model_validate(card.model_dump())
    assert restored == card
    assert getattr(restored, field) is False


def test_card_with_explicit_params_round_trips() -> None:
    card = MailboxTool(
        read_mailbox=ReadMailbox(instructions="prefer wrapping up", expose={TOOL_CALL}),
        stop=Stop(),
    )
    restored = MailboxTool.model_validate(card.model_dump())
    assert restored == card
    assert isinstance(restored.read_mailbox, ReadMailbox)
    assert restored.read_mailbox.instructions == "prefer wrapping up"
    assert isinstance(restored.stop, Stop)
