"""Tests for the run-cancellation vocabulary (ADR-040 §4, §5)."""

from __future__ import annotations

from akgentic.core.messages import CancelMessage, Message, UserMessage

from akgentic.tool.mailbox import is_cancel, render_arrival_notice
from tests.mailbox.conftest import _user_message

# ── is_cancel: both spellings of one intent ──────────────────────────────────


def test_cancel_message_instance_is_cancel() -> None:
    # AC 6: the typed spelling (programmatic senders).
    assert is_cancel(CancelMessage(reason="user pressed Esc")) is True


def test_stop_content_is_cancel() -> None:
    # AC 6: the string spelling (human / frontend Esc).
    assert is_cancel(UserMessage(content="/stop")) is True


def test_stop_with_leading_space_and_trailing_words_is_cancel() -> None:
    assert is_cancel(UserMessage(content="  /stop now")) is True


def test_ordinary_content_is_not_cancel() -> None:
    assert is_cancel(UserMessage(content="please summarize the thread")) is False


def test_stopwatch_is_not_cancel() -> None:
    # Exact-token rule: /stop followed by end or whitespace only.
    assert is_cancel(UserMessage(content="/stopwatch")) is False


def test_message_without_content_is_not_cancel() -> None:
    # A content-less message is simply False, never an error.
    assert is_cancel(Message()) is False


class _PayloadMessage(Message):
    """A message whose content is not a string."""

    content: int


def test_non_string_content_is_not_cancel() -> None:
    # The non-str content guard: simply False, never an error.
    assert is_cancel(_PayloadMessage(content=5)) is False


def test_empty_content_is_not_cancel() -> None:
    assert is_cancel(UserMessage(content="")) is False


def test_stop_mid_sentence_is_not_cancel() -> None:
    # The first token must be /stop — mentioning it later is not a cancel.
    assert is_cancel(UserMessage(content="please /stop")) is False


# ── render_arrival_notice: the mid-run doorbell wording ──────────────────────


def test_arrival_notice_empty_list_says_nothing() -> None:
    # AC 7: empty list -> "".
    assert render_arrival_notice([]) == ""


def test_arrival_notice_renders_count_senders_and_pointer() -> None:
    # AC 7: one line — count, senders, and the read_mailbox pointer.
    messages: list[Message] = [
        _user_message("@Alice", "hello"),
        _user_message("@Bob", "ping"),
    ]
    assert render_arrival_notice(messages) == (
        "2 new messages arrived (from @Alice, @Bob) — "
        "call `read_mailbox` to see them, or finish your current work first."
    )


def test_arrival_notice_singular_message() -> None:
    assert render_arrival_notice([_user_message("@Alice", "hello")]) == (
        "1 new message arrived (from @Alice) — "
        "call `read_mailbox` to see them, or finish your current work first."
    )


def test_arrival_notice_defends_senderless_message() -> None:
    # AC 7: a message without a usable sender/content still renders.
    assert render_arrival_notice([Message()]) == (
        "1 new message arrived (from unknown) — "
        "call `read_mailbox` to see them, or finish your current work first."
    )
