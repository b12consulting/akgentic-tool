"""Tests for ``MailboxCapability``'s injected wording (ADR-040 §5, ADR-044).

The subject is narrow: where the two injected strings come from and that an
override reaches both use sites. Cancellation, the offer filter, the withdrawal
hook and the silent no-op paths are exercised by ``akgentic-agent``'s wiring
suite and are not re-litigated here.

**The sentinel specs are the only real guard.** A default-constructed
capability's ``_absorbed_prefix`` *equals* ``ABSORBED_PREFIX``, so no
default-path assertion can tell the instance attribute apart from the constant —
which is exactly the edit a later "simplification" makes. Only a
sentinel-valued construction can see it, and those two specs are mutation
targets, not decoration.
"""

from __future__ import annotations

import inspect
import uuid
from typing import Any

import pytest
from akgentic.core.messages import Message
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition

from akgentic.tool.mailbox import (
    ABSORBED_PREFIX,
    ARRIVAL_CLOSING,
    MailboxCapability,
    MailboxMessage,
    MailboxTool,
)
from akgentic.tool.mailbox.capability import MailboxAccess

_SENTINEL_PREFIX = "SENTINEL-PREFIX-9f3c: only an injected prefix may carry this."
_SENTINEL_CLOSING = "SENTINEL-CLOSING-4a71: only an injected closing may carry this."


class _Errand(MailboxMessage):
    """A deliverable message class: it renders, and it previews.

    ``after_tool_execute`` returns early for anything that is not a
    ``MailboxMessage`` (the message is left queued to arrive as its own turn), so
    a prefix spec built on a plain ``Message`` would pass vacuously.
    """

    errand: str = "collect the report"

    def rendering(self) -> str:
        return f"You were asked to {self.errand}."

    def rendering_preview(self) -> str:
        return f"errand: {self.errand}"


class _FakeMailbox:
    """Structural ``MailboxAccess``: all three methods, backed by a real list.

    ``test_mailbox_tool.py``'s ``_FakeObserver`` satisfies ``MailboxToolObserver``
    but **not** this protocol — it has no ``current_message``. ``MailboxAccess``
    is ``@runtime_checkable``, which checks member presence and not signatures, so
    the missing method would surface as an ``AttributeError`` deep inside the
    offer filter rather than at the isinstance below.
    """

    def __init__(self, pending: list[Message], current: Message | None) -> None:
        self._pending = list(pending)
        self._current = current
        self.consumed: list[uuid.UUID] = []

    def get_mailbox(self) -> list[Message]:
        return list(self._pending)

    def consume_mailbox(self, message_ids: list[uuid.UUID]) -> list[Message]:
        wanted = set(message_ids)
        self.consumed.extend(message_ids)
        removed = [message for message in self._pending if message.id in wanted]
        self._pending = [message for message in self._pending if message.id not in wanted]
        return removed

    def current_message(self) -> Message | None:
        return self._current


class _FakeRunContext:
    """A ``RunContext`` stand-in exposing only what the two use sites touch.

    ``enqueue`` returns an id, as the real one does when the queue accepts the
    entry; ``after_node_run`` keys its withdrawal on that id, so returning
    ``None`` here would quietly disable a path this module does not test but must
    not misrepresent.
    """

    def __init__(self) -> None:
        self.enqueued: list[tuple[str, str]] = []
        self.pending_messages: list[Any] | None = None

    def enqueue(self, content: str, *, priority: str = "normal") -> str:
        self.enqueued.append((content, priority))
        return f"enqueue-{len(self.enqueued)}"


def _capability(
    pending: list[Message],
    current: Message | None,
    **wording: str,
) -> tuple[MailboxCapability, _FakeMailbox]:
    """A capability over a fake mailbox; the mailbox is returned to be asserted on."""
    mailbox = _FakeMailbox(pending, current)
    assert isinstance(mailbox, MailboxAccess)  # structural conformance
    capability = MailboxCapability(mailbox, MailboxTool(), **wording)
    return capability, mailbox


async def _notice_for(capability: MailboxCapability, ctx: _FakeRunContext) -> str:
    """Drive one ``before_model_request`` and return the notice it enqueued."""
    request_context = object()
    returned = await capability.before_model_request(ctx, request_context)  # type: ignore[arg-type]

    assert returned is request_context  # the hook returns it unchanged
    assert len(ctx.enqueued) == 1
    notice, priority = ctx.enqueued[0]
    assert priority == "asap"
    return notice


async def _absorb(capability: MailboxCapability, ctx: _FakeRunContext, target: Message) -> str:
    """Drive one completed ``read_mailbox`` call and return the injected content."""
    result = await capability.after_tool_execute(
        ctx,  # type: ignore[arg-type]
        call=ToolCallPart(tool_name="read_mailbox", args={"message_id": str(target.id)}),
        tool_def=ToolDefinition(name="read_mailbox"),
        args={"message_id": str(target.id)},
        result="acknowledged",
    )

    assert result == "acknowledged"  # the tool's own return passes through untouched
    assert len(ctx.enqueued) == 1
    injected, priority = ctx.enqueued[0]
    assert priority == "asap"
    return injected


# ── AC 2: the wording arrives as keyword-only constructor parameters ─────────


def test_the_two_wording_parameters_are_keyword_only_and_default_to_the_constants() -> None:
    # The kinds are the assertion, not just the names. `akgentic-agent` calls
    # `MailboxCapability(observer=self, card=mailbox_card)`, and dropping the `*`
    # would leave that call compiling while quietly making the wording positional
    # — a third positional argument then binds to a prefix nobody meant to pass.
    signature = inspect.signature(MailboxCapability.__init__)
    parameters = signature.parameters

    assert list(parameters) == ["self", "observer", "card", "absorbed_prefix", "arrival_closing"]

    for name in ("observer", "card"):
        assert parameters[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert parameters[name].default is inspect.Parameter.empty

    assert parameters["absorbed_prefix"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["absorbed_prefix"].default == ABSORBED_PREFIX
    assert parameters["arrival_closing"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["arrival_closing"].default == ARRIVAL_CLOSING


def test_two_argument_construction_still_works_positionally_and_by_keyword() -> None:
    # `akgentic-agent` builds this with exactly two arguments and must keep
    # compiling and behaving identically across this change.
    mailbox = _FakeMailbox([], None)
    card = MailboxTool()

    positional = MailboxCapability(mailbox, card)
    by_keyword = MailboxCapability(observer=mailbox, card=card)

    for capability in (positional, by_keyword):
        assert capability._absorbed_prefix == ABSORBED_PREFIX
        assert capability._arrival_closing == ARRIVAL_CLOSING


def test_the_card_still_decides_whether_the_doorbell_rings() -> None:
    # The half of the card seam that survives: the capability reads `read_mailbox`
    # off the card whole, and `read_mailbox=False` suppresses the notice.
    mailbox = _FakeMailbox([], None)

    assert MailboxCapability(mailbox, MailboxTool())._arrival_notice is True
    assert MailboxCapability(mailbox, MailboxTool(read_mailbox=False))._arrival_notice is False


# ── AC 3: plain assignment, so an explicit empty string is honoured ──────────


@pytest.mark.parametrize("wording", ["absorbed_prefix", "arrival_closing"])
def test_an_explicitly_empty_string_is_kept_and_not_replaced_by_the_default(wording: str) -> None:
    # The old `card.absorbed_prefix or ABSORBED_PREFIX` existed because an empty
    # string could arrive from a catalog entry, where it was a mistake rather than
    # a choice. With no catalog path there is only a caller, and a caller passing
    # "" demonstrably did not ask for the shipped prose.
    capability, _mailbox = _capability([], None, **{wording: ""})

    assert getattr(capability, f"_{wording}") == ""


# ── AC 4: both use sites read the instance attribute, not the constant ───────


@pytest.mark.asyncio
async def test_a_default_capability_injects_exactly_the_shipped_wording() -> None:
    errand = _Errand()
    capability, _mailbox = _capability([errand], _Errand())
    notice_ctx, absorb_ctx = _FakeRunContext(), _FakeRunContext()

    notice = await _notice_for(capability, notice_ctx)
    injected = await _absorb(capability, absorb_ctx, errand)

    assert notice.splitlines()[-1] == ARRIVAL_CLOSING
    assert injected == f"{ABSORBED_PREFIX}\n\n{errand.rendering()}"


@pytest.mark.asyncio
async def test_an_overridden_prefix_reaches_the_absorbed_message_injection() -> None:
    # THE mutation target. Inlining `ABSORBED_PREFIX` at the enqueue site — the
    # tempting "simplification", since the attribute usually holds exactly that —
    # silently deletes the override, and no default-path spec can see it.
    errand = _Errand()
    capability, mailbox = _capability(
        [errand], _Errand(), absorbed_prefix=_SENTINEL_PREFIX, arrival_closing=_SENTINEL_CLOSING
    )
    ctx = _FakeRunContext()

    injected = await _absorb(capability, ctx, errand)

    assert injected == f"{_SENTINEL_PREFIX}\n\n{errand.rendering()}"
    assert ABSORBED_PREFIX not in injected
    assert mailbox.consumed == [errand.id]  # the acknowledgement is made true here


@pytest.mark.asyncio
async def test_an_overridden_closing_reaches_the_arrival_notice() -> None:
    # The same mutation, on the other site: `render_arrival_notice`'s third
    # argument defaults to `ARRIVAL_CLOSING`, so passing the constant instead of
    # the attribute renders identically on every default-path spec.
    errand = _Errand()
    capability, _mailbox = _capability(
        [errand], _Errand(), absorbed_prefix=_SENTINEL_PREFIX, arrival_closing=_SENTINEL_CLOSING
    )
    ctx = _FakeRunContext()

    notice = await _notice_for(capability, ctx)

    assert notice.splitlines()[-1] == _SENTINEL_CLOSING
    assert ARRIVAL_CLOSING not in notice
    assert f"(id: {errand.id})" in notice  # the id is what makes the closing honest


# ── both wording defaults are public, because both are defaults of public surfaces ──


def test_both_wording_constants_are_on_the_package_surface() -> None:
    # `ARRIVAL_CLOSING` was `_CLOSING_WITH_IDS`, private, while `ABSORBED_PREFIX`
    # beside it was public — an accident of authoring order, not a rule. Once both
    # became defaults of a public constructor parameter the asymmetry had a cost: a
    # caller wanting "the shipped closing plus one sentence" had no public name and
    # had to import a private one, which three akgentic-agent test modules did.
    import akgentic.tool.mailbox as mailbox_module

    for name in ("ABSORBED_PREFIX", "ARRIVAL_CLOSING"):
        assert name in mailbox_module.__all__
        assert isinstance(getattr(mailbox_module, name), str)


def test_the_no_id_closing_is_not_promoted_with_it() -> None:
    # The pair split, and only one half had a caller. A listing that offers no id
    # may not promise a read whatever anyone configures, so the no-id closing
    # reaches no parameter — there is nobody to give a name to.
    import akgentic.tool.mailbox as mailbox_module

    assert "_CLOSING_WITHOUT_IDS" not in mailbox_module.__all__
    assert not hasattr(mailbox_module, "_CLOSING_WITHOUT_IDS")


def test_the_promoted_constant_is_the_default_of_both_public_surfaces() -> None:
    # The reason it is public at all. If either default is ever inlined or
    # re-privatised, the name stops earning its export and this goes red.
    from inspect import signature

    from akgentic.tool.mailbox import render_arrival_notice

    assert (
        signature(render_arrival_notice).parameters["closing_with_ids"].default == ARRIVAL_CLOSING
    )
    assert signature(MailboxCapability).parameters["arrival_closing"].default == ARRIVAL_CLOSING
