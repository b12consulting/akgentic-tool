"""Tests for the mailbox context state and its provider building-block (ADR-040 §2)."""

from __future__ import annotations

import gc
import weakref

from akgentic.core.messages import Message, ResultMessage

from akgentic.tool.mailbox import (
    MailboxRow,
    MailboxState,
    MailboxToolObserver,
    make_mailbox_state_provider,
)
from tests.mailbox.conftest import _user_message


def _row(sender: str, message_type: str = "UserMessage", preview: str = "hi") -> MailboxRow:
    return MailboxRow(sender=sender, message_type=message_type, preview=preview)


class _FakeObserver:
    """Structural MailboxToolObserver: real messages, no actor system."""

    def __init__(self, messages: list[Message]) -> None:
        self._messages = messages

    def notify_event(self, event: object) -> None:
        pass

    def get_mailbox(self) -> list[Message]:
        return list(self._messages)


# ── render_full ──────────────────────────────────────────────────────────────


def test_render_full_empty_mailbox_says_nothing() -> None:
    # AC 3: empty rows -> "" (say nothing).
    assert MailboxState(rows=[]).render_full() == ""


def test_render_full_renders_count_and_senders() -> None:
    # AC 3: count + senders, worded from live state.
    state = MailboxState(rows=[_row("@Alice"), _row("@Bob"), _row("@Alice", preview="again")])
    assert state.render_full() == (
        "3 messages pending from @Alice, @Bob, consider wrapping up the current thread"
    )


def test_render_full_singular_message() -> None:
    state = MailboxState(rows=[_row("@Alice")])
    assert state.render_full() == (
        "1 message pending from @Alice, consider wrapping up the current thread"
    )


# ── render_delta: arrivals only, multiset semantics ──────────────────────────


def test_render_delta_names_only_new_rows() -> None:
    # AC 4: arrivals are narrated, pre-existing rows are not re-listed.
    previous = MailboxState(rows=[_row("@Alice")])
    current = MailboxState(rows=[_row("@Alice"), _row("@Bob", preview="ping")])
    delta = current.render_delta(previous)
    assert delta == "New message pending from @Bob (UserMessage): ping"


def test_render_delta_departures_only_is_none() -> None:
    # AC 4: departures became their own turns — never narrated.
    previous = MailboxState(rows=[_row("@Alice"), _row("@Bob")])
    current = MailboxState(rows=[_row("@Bob")])
    assert current.render_delta(previous) is None


def test_render_delta_no_change_is_none() -> None:
    previous = MailboxState(rows=[_row("@Alice")])
    current = MailboxState(rows=[_row("@Alice")])
    assert current.render_delta(previous) is None


def test_render_delta_duplicate_rows_use_multiset_counts() -> None:
    # AC 4: a second identical row IS an arrival — occurrence counts, not set membership.
    previous = MailboxState(rows=[_row("@Alice")])
    current = MailboxState(rows=[_row("@Alice"), _row("@Alice")])
    assert current.render_delta(previous) == "New message pending from @Alice (UserMessage): hi"


def test_render_delta_multiple_arrivals_join_with_newlines() -> None:
    previous = MailboxState(rows=[])
    current = MailboxState(rows=[_row("@Alice"), _row("@Bob", preview="ping")])
    assert current.render_delta(previous) == (
        "New message pending from @Alice (UserMessage): hi\n"
        "New message pending from @Bob (UserMessage): ping"
    )


def test_render_delta_arrival_without_preview_has_no_colon() -> None:
    previous = MailboxState(rows=[])
    current = MailboxState(rows=[_row("@Alice", preview="")])
    assert current.render_delta(previous) == "New message pending from @Alice (UserMessage)"


# ── from_messages mapping ────────────────────────────────────────────────────


def test_from_messages_maps_sender_type_and_first_line_preview() -> None:
    message = _user_message("@Alice", "first line\nsecond line")
    state = MailboxState.from_messages([message])
    assert state.rows == [
        MailboxRow(sender="@Alice", message_type="UserMessage", preview="first line")
    ]


def test_from_messages_truncates_preview_to_120_chars() -> None:
    message = _user_message("@Alice", "x" * 500)
    state = MailboxState.from_messages([message])
    assert state.rows[0].preview == "x" * 120


def test_from_messages_defends_missing_sender_and_content() -> None:
    # A bare Message has no content and no sender — the row still builds.
    state = MailboxState.from_messages([Message()])
    assert state.rows == [MailboxRow(sender="unknown", message_type="Message", preview="")]


def test_from_messages_carries_concrete_type_name() -> None:
    state = MailboxState.from_messages([ResultMessage(content="done")])
    assert state.rows[0].message_type == "ResultMessage"


# ── serializability (FR7) ────────────────────────────────────────────────────


def test_state_round_trips_through_model_dump_and_validate() -> None:
    # AC 8: MailboxRow and MailboxState survive dump -> validate unchanged.
    state = MailboxState(rows=[_row("@Alice"), _row("@Bob", preview="ping")])
    restored = MailboxState.model_validate(state.model_dump())
    assert restored == state
    assert restored.rows == state.rows


# ── provider building-block (NFR2) ───────────────────────────────────────────


def test_provider_returns_snapshot_from_live_observer() -> None:
    observer = _FakeObserver([_user_message("@Alice", "hello")])
    assert isinstance(observer, MailboxToolObserver)  # structural conformance
    provider = make_mailbox_state_provider(lambda: observer)
    state = provider()
    assert state is not None
    assert state.rows[0].sender == "@Alice"


def test_provider_returns_none_when_observer_collected() -> None:
    # AC 5: drive the weakref accessor to None — the provider returns None, never raises.
    observer = _FakeObserver([])
    ref: weakref.ref[_FakeObserver] = weakref.ref(observer)
    provider = make_mailbox_state_provider(lambda: ref())
    del observer
    gc.collect()
    assert provider() is None


def test_provider_swallows_observer_failure() -> None:
    # AC 5: no code path raises — a failing observer yields None.
    class _BrokenObserver(_FakeObserver):
        def get_mailbox(self) -> list[Message]:
            raise RuntimeError("actor stopped mid-call")

    provider = make_mailbox_state_provider(lambda: _BrokenObserver([]))
    assert provider() is None
