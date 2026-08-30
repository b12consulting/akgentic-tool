"""Mailbox-driven run control: the vocabulary and the hooks that act on it.

The capability has more than one duty. Before every model request it purges a
pending cancel from the mailbox and raises on it, and it renders and enqueues
the mid-run arrival notice for mail that landed while the run was in flight.
After every ``read_mailbox`` call it absorbs the message the model named and
injects that message's own rendering. And at every node boundary it withdraws
its own notice once the run has reached its end, so the doorbell never costs
the answer it interrupted. The mailbox is the single input to all of them,
which is what makes them one capability rather than four.

**The agent renders; the card does not.** ``read_mailbox`` is a signal that
carries an id across and acknowledges it — it reads nothing, consumes nothing
and renders nothing. It used to do all three, and its renderer took each
message's body as ``getattr(message, "content", "")``: correct for exactly the
one message class it was written against, and silently empty for every other,
so a class declaring its own fields was consumed, rendered blank and never
reached its own handler. Rendering a message is the message's job
(``MailboxMessage.rendering()``, the base a deliverable class extends), and
delivering one is this capability's.

**Card and capability sit together, in one package.** This module lived in
``akgentic.agent.capabilities`` first, on the argument that ``BaseAgent`` builds
it unconditionally so it must not depend on a card that may be absent. Two
things retired that argument. ``BaseAgent`` auto-inserts a default
``MailboxTool`` when the config carries none, so a card is never absent; and
``akgentic-tool`` is a hard dependency of ``akgentic-agent``, so a capability
shipped here is never unavailable to it. What remains true is the property the
argument was protecting — cancellation cannot be de-configured — and it is
protected by the wiring being unconditional, not by the module's address.

Living beside the card is what the design actually wants. The vocabulary
(``is_cancel``, ``render_arrival_notice``), the ``/stop`` string it recognises,
the ``read_mailbox`` signal it acts on, the whitelist deciding which handlers
show a preview, and the two injected prompt strings are one subject; splitting
them across two repositories meant every mailbox change was two stories, two
PRs and a release in between. The card is now handed to the capability whole
(see :meth:`MailboxCapability.__init__`), so adding a field is one change here
and none in ``akgentic-agent``.

``akgentic-agent`` keeps the *enforcement*: it builds this capability for every
agent and catches ``RunInterruptedError`` in ``act()``. It no longer holds the
implementation, and no longer carries an ``akgentic.agent.capabilities``
package at all — these names are imported from here.

The card serves no ``LLM_CONTEXT``: mailbox awareness reaches the model through
the mid-run arrival notice below alone.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Protocol, runtime_checkable

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, AgentNode, NodeResult, ValidatedToolArgs
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import ToolDefinition
from pydantic_graph import End

from akgentic.core.messages import CancelMessage, Message
from akgentic.tool.mailbox.mailbox import MailboxTool
from akgentic.tool.mailbox.message import MailboxMessage

logger = logging.getLogger(__name__)

READ_MAILBOX_TOOL = "read_mailbox"
"""The tool whose completed calls this capability turns into injected content."""

MESSAGE_ID_ARG = "message_id"
"""The ``read_mailbox`` argument naming the message the model takes on.

A cross-repository contract, read **by name** off the validated arguments of a
completed tool call. ``akgentic-tool`` owns the signature; an argument this
capability cannot find is a silent no-op, which is what lets the two halves land
in sequence rather than atomically.
"""


@runtime_checkable
class MailboxAccess(Protocol):
    """The mailbox contract this capability needs from the agent it observes.

    Narrower than ``MailboxToolObserver``, the card's own observer protocol, and
    kept separate from it: this capability needs three mailbox methods, while
    that protocol describes everything a card may ask of its owner. ``Akgent``
    satisfies both structurally, so nothing inherits either.

    ``@runtime_checkable`` checks member presence, not signatures, so widening
    this Protocol does not fail loudly for a fake that misses a method: every
    implementer — the agent and each test fake — must be swept by hand.
    """

    def get_mailbox(self) -> list[Message]:
        """Peek at the pending messages, removing none of them."""
        ...

    def consume_mailbox(self, message_ids: list[uuid.UUID]) -> list[Message]:
        """Remove the named messages from the mailbox, recording each removal."""
        ...

    def current_message(self) -> Message | None:
        """The message whose handler is running, or ``None`` when idle."""
        ...


class RunInterruptedError(Exception):
    """The current run was cancelled by a /stop or CancelMessage.

    Raised by ``MailboxCapability.before_model_request`` at the next step
    boundary once a cancel is pending in the mailbox (ADR-040 §5). Carries a
    message only. It must never escape a message handler: an escape ends the
    turn through the actor failure path (``Akgent._handle_failure`` — an
    ErrorMessage to the orchestrator; actor death under stock pykka) instead
    of the designed clean end. ``act()`` owns the catch — it absorbs this
    error, notifies the human once and returns a default instance of the
    caller's output type — so the *run* dies while the *agent* carries on
    cleanly, and no handler needs a catch of its own.
    """


class MailboxRenderError(Exception):
    """The offer filter marked a message offerable that cannot render a preview.

    An internal-invariant guard on **our own filter**, never on the feature. The
    everyday "this run cannot handle that message" case is a silent non-offer —
    the message is listed without an id and the model simply has no affordance
    to ask for it. Reaching this exception means the filter admitted a message
    that is not a ``MailboxMessage``, which is a defect here rather than a
    configuration a deployment could produce.

    Unreachable in correct code, and therefore verifiable only by mutation:
    delete the ``MailboxMessage`` condition from the filter and a plain message
    of the handler's own class reaches the render.
    """


def is_cancel(msg: Message) -> bool:
    """Whether ``msg`` asks the recipient to abandon its current run.

    ``True`` for a ``CancelMessage`` instance, or for a message whose content
    strips to a string whose first whitespace-delimited token is exactly
    ``/stop`` — so ``"  /stop now"`` cancels and ``"/stopwatch"`` does not.
    A message without usable string content is simply ``False``.
    """
    if isinstance(msg, CancelMessage):
        return True
    content = getattr(msg, "content", "")
    if not isinstance(content, str):
        return False
    tokens = content.split(maxsplit=1)
    return bool(tokens) and tokens[0] == "/stop"


PREVIEW_LIMIT = 120
"""How much of a message's content the arrival notice previews, in characters.

A property of the *notice*, not of any message class, which is why it sits here
with the code that renders one. A message's ``rendering_preview()`` reads it to
decide where to cut; that is the message honouring the notice's budget, not the
message owning it.
"""

UNOFFERABLE_LINE = "- Message cannot be handled in the run"
"""What a message this run cannot take on renders as — no id, no content."""

_CLOSING_WITH_IDS = (
    "Call `read_mailbox` with one of the ids above to take that message on now — worth doing "
    "if it may add to or change what you are working on, since a correction only helps before "
    "the work is finished. Otherwise finish your current work first — you will get them just "
    "after."
)

_CLOSING_WITHOUT_IDS = "Finish your current work first — you will get them just after."


def render_arrival_notice(
    new_messages: list[Message],
    offerable_ids: set[uuid.UUID],
    closing_with_ids: str = _CLOSING_WITH_IDS,
) -> str:
    """Doorbell for messages that arrived mid-run (ADR-010 §5, ADR-040 §5).

    A count line, one line per message **in the order given**, then the closing
    pointer. Every message the caller hands over is listed, offerable or not:
    dropping one would hide an arrival the human can see, and the notice's job
    is to say what is waiting.

    What varies is the *id*. A message whose id is in ``offerable_ids`` renders
    as its own ``rendering_preview()`` followed by ``(id: …)``, which is the only
    way the model can name it in a ``read_mailbox`` call. Everything else
    renders as :data:`UNOFFERABLE_LINE` — visible, but unaskable. That missing
    id *is* the constraint: not an error, not a validation, not a refusal the
    model can argue with; the affordance simply is not offered.

    The closing line follows the same rule. It points at ``read_mailbox`` only
    when at least one id is on offer, because promising a read for a listing
    that carries no id would be an instruction the model cannot follow. When it
    does point, it also gives the one reason that decides the timing: a message
    that may add to or change the work in flight is worth taking on before that
    work is finished, and is worth nothing after. The reassurance that unread
    mail arrives as its own turn is true either way and is kept in both.

    **The closing line arrives as a parameter, and the prefix does not.** That
    asymmetry is deliberate, not an oversight: this is a module-level *function*,
    so constructor injection cannot reach it the way it reaches
    :data:`ABSORBED_PREFIX` on :class:`MailboxCapability`. The caller passes its
    configured closing here instead, and the default keeps every existing
    two-argument call rendering exactly what it always did.
    ``_CLOSING_WITHOUT_IDS`` takes no parameter — a listing carrying no id may
    not promise a read whatever a deployment configures.

    Args:
        new_messages: The messages to announce, in reception order.
        offerable_ids: Ids the offer filter admitted. Every one of them must
            belong to a ``MailboxMessage``.
        closing_with_ids: The closing line for a listing that offers at least one
            id. Defaults to the module constant, which is what a directly-called
            render has always used.

    Returns:
        The rendered notice, or ``""`` for an empty list.

    Raises:
        MailboxRenderError: An id was marked offerable for a message that
            cannot render a preview — the filter is broken. Never raised for a
            message that was simply not offered.
    """
    if not new_messages:
        return ""
    count = len(new_messages)
    noun = "message" if count == 1 else "messages"
    lines = [f"{count} new {noun} arrived:"]
    lines.extend(_message_line(message, offerable_ids) for message in new_messages)
    offered = any(message.id in offerable_ids for message in new_messages)
    lines.append(closing_with_ids if offered else _CLOSING_WITHOUT_IDS)
    return "\n".join(lines)


def _message_line(message: Message, offerable_ids: set[uuid.UUID]) -> str:
    """One notice line: a preview plus an id, or the unofferable line."""
    if message.id not in offerable_ids:
        return UNOFFERABLE_LINE
    if not isinstance(message, MailboxMessage):
        raise MailboxRenderError(
            f"{type(message).__name__} was offered for a mid-run read but is not a "
            f"MailboxMessage; the offer filter admitted a message it must exclude."
        )
    return f"- {message.rendering_preview()} (id: {message.id})"


ABSORBED_PREFIX = (
    "Additional work, taken on mid-run. It does NOT replace what you were already asked "
    "to do. It may be a separate request, in which case answer both before this run ends, "
    "one message each in your output; or it may add to or correct the request already in "
    "flight, in which case one message answers both. When unsure, answer separately."
)
"""What an absorbed message is prefixed with when it is injected.

``rendering()`` renders a message the way its own handler receives it —
imperative and self-contained ("You received a request from @X. A reply is
expected."). Injected mid-run that reads as a *new assignment*, and the model
answers it instead of what it was already doing. Observed in the field: an agent
that had just written a report answered only the newer question, and the report
answer reached nobody. The first sentence is the clause that stops that failure
and is not to be reworded.

**The output obligation after it is a choice, not an assertion.** A mid-run
arrival is either a separate request — two answers owed, one message each — or
an addition to, or correction of, the request already in flight, where one
answer covers both. The capability cannot tell which: it has not read the
message, and a classification made here would be invisible and unrecoverable,
where one made by the model is right there in the output. So the string states
both cases and lets the model choose. The second case is not an edge — the offer
filter admits a pending message only when its class is *exactly* the handled
message's, so a message eligible for a mid-run read is by construction the kind
most likely to be a follow-up on the same thread.

**The default lives in the string, and it is "answer separately."** Not in this
docstring, not in a comment: the model reads the string. The two failure modes
are not symmetric — a redundant second message is noise, a swallowed report
reaches nobody — so the doubt is biased toward the cheap failure.

The prefix is the capability's, not the message's. **Rendering a message is the
message's job; delivering one is this capability's**, and framing a delivery is
part of delivering it — so every class that grows a ``rendering()`` inherits
this for free.
"""


class MailboxCapability(AbstractCapability[Any]):
    """Mailbox-driven run cancellation and mid-run arrival notice (ADR-040 §5).

    One instance per agent, built unconditionally by ``BaseAgent`` — never
    contributed by a card, so cancellation cannot be de-configured by omitting
    ``MailboxTool``. The agent owns both the *vocabulary* (``is_cancel``,
    ``render_arrival_notice``, defined in this module) and the *enforcement*:
    an agent with no ``MailboxTool`` has no card to borrow a predicate from,
    and must still be interruptible.

    ``before_model_request`` fires before EVERY model request inside the REACT
    loop, bracketing every tool call and reasoning step — exactly the
    granularity cancellation needs. On each firing, in order:

    1. Cancel check — every pending message matching ``is_cancel`` is purged
       from the mailbox through ``consume_mailbox``, and then
       ``RunInterruptedError`` is raised. The mailbox is the cancellation's
       single source of truth: no flag, no consumed marker — recognising the
       cancel and consuming it are one act, performed here, at recognition.
       So a cancel never gets a turn of its own after interrupting a run, and
       the human hears about it once, through the interruption. A cancel that
       reaches a handler is by construction the *idle* case: nothing was
       running for this hook to have seen it.
    2. Arrival notice — pending messages not yet announced in this run are
       announced through one ``ctx.enqueue(notice, priority="asap")`` call,
       pydantic-ai's supported injection path. The auto-injected, outermost
       ``PendingMessageDrainCapability`` drains the queue at the *next* step
       boundary: the notice lands in that model request, in the durable
       history and in the ``LlmMessageEvent`` stream by design — the event
       store is the audit trail that the doorbell rang. When the run would
       otherwise end first, the notice is **withdrawn** rather than delivered
       — the third duty below. The hook constructs no message of its own and
       never mutates an existing message's parts (they are shared with durable
       history).

    A **second hook**, ``after_node_run``, carries a third duty that only the
    run's end can trigger. The drain's end-of-run redirect — one final model
    request, so an already-enqueued notice is delivered rather than lost — is
    the one case where the doorbell is not worth its price: it discards the
    run's own ``End(FinalResult)``, so an answer the agent had already written
    is never returned by ``run_sync`` and reaches nobody. ``after_node_run``
    therefore withdraws the notice from the queue once the run has reached its
    end, and the message arrives as its own turn instead — the fallback
    ADR-010 §5 already specifies. It is possible only on that hook: ``after_*``
    walks the capability chain **backwards**, so this capability runs ahead of
    the outermost drain rather than behind it.

    Each announced message is offered an **id** only if ``offerable_ids``
    admits it; everything else is listed without one and is therefore visible
    but unaskable. ``after_tool_execute`` is the other half of that bargain: it
    turns an id the model names back into the message's own rendering.

    Announced-id tracking is run-local: the instance lives for the agent's
    lifetime, so ``before_run`` clears the set at each run start. A backlog
    re-announced next run is acceptable; a leak of announced ids across runs
    is not. The preview whitelist is the opposite — resolved once from the
    card at agent init and constant for the agent's life.
    """

    def __init__(
        self,
        observer: MailboxAccess,
        card: MailboxTool,
        *,
        absorbed_prefix: str = ABSORBED_PREFIX,
        arrival_closing: str = _CLOSING_WITH_IDS,
    ) -> None:
        """Wire the capability to one agent's mailbox, its card and its wording.

        **The card still crosses the boundary whole, and this class reads what it
        needs off it.** ``BaseAgent`` hands the card over and inspects none of it.
        Passing ``preview_handlers`` / ``arrival_notice`` as separate arguments
        shipped first and is the reason that seam exists: it put *which fields the
        mailbox needs and how to tolerate an older card* into ``BaseAgent`` — none
        of it the agent's business, and all of it to be re-edited on the next field
        added. The consumer knows what it consumes. ``read_mailbox`` is read here
        for exactly that reason, and it is a switch: it decides whether the doorbell
        rings at all.

        **The wording is not the card's, and that half was reversed deliberately.**
        The prefix and the closing line lived on the card as two string fields for
        one story. The catalog dumps a card with a plain ``model_dump(mode="json")``
        and no ``exclude_defaults``, so the literal default was written into every
        persisted entry — each team frozen with its own private copy of prose that
        is expected to keep improving, and an improvement reaching only teams
        created afterwards. No deployment ever turned the knob; every one paid for
        the copy. So the wording is a constructor parameter defaulting to the module
        constant beside it, and an upgrade moves every existing team at once.

        Both parameters are **keyword-only and defaulted**, so ``BaseAgent``'s
        two-argument construction is untouched. They are assigned straight through:
        an ``""`` passed here is a caller's explicit choice and is honoured as one,
        where an ``""`` arriving from a catalog entry was a mistake worth
        substituting a default for.

        Args:
            observer: The agent whose mailbox this reads.
            card: The agent's ``MailboxTool`` — the config's own, or the
                auto-inserted default. Required.
            absorbed_prefix: What an absorbed message is prefixed with when it is
                injected mid-run. Defaults to :data:`ABSORBED_PREFIX`.
            arrival_closing: The arrival notice's closing line for a listing that
                offers at least one id. Defaults to the shipped closing.
                ``_CLOSING_WITHOUT_IDS`` takes no parameter — a listing offering no
                id may not promise a read whatever a caller configures.

        **Cancellation consults nothing here.** The purge-and-raise in
        ``before_model_request`` runs ahead of every value read in this
        constructor, so a capability built from a card with ``read_mailbox=False``
        and empty strings is still interruptible. That is ADR-040 §5's rule and no
        card field or parameter can de-configure it.
        """
        self._observer = observer
        self._announced_ids: set[uuid.UUID] = set()
        self._notice_enqueue_ids: set[str] = set()
        # ``bool(...)`` resolves the card's ``Param | bool`` convention: a
        # ``ReadMailbox`` instance is truthy, ``False`` is not — so
        # ``read_mailbox=False`` suppresses the doorbell entirely, which is what a
        # run without that tool needs.
        self._arrival_notice = bool(card.read_mailbox)
        self._absorbed_prefix = absorbed_prefix
        self._arrival_closing = arrival_closing

    async def before_run(self, ctx: RunContext[Any]) -> None:
        """Forget which arrivals the previous run announced.

        Announced-id tracking is run-local, and this instance lives for the
        agent's lifetime — so the set must be cleared once per run. pydantic-ai
        calls this hook exactly there, which is why the reset is not the agent's
        job: ``act()`` used to call a public ``reset_run_tracking()`` before
        ``run_sync``, an obligation invisible from this class and silently
        droppable by any other caller that starts a run. Here it cannot be
        skipped, because starting a run *is* what triggers it.

        Observe-only by contract, which is all this needs — it mutates the
        capability's own state and nothing pydantic-ai owns.

        Args:
            ctx: The run context. Unused: the reset is unconditional.
        """
        self._announced_ids.clear()
        self._notice_enqueue_ids.clear()

    def offerable_ids(self, pending: list[Message]) -> set[uuid.UUID]:
        """Which of ``pending`` this run may be offered an id for.

        A message is offerable when **all three** hold:

        1. it is a ``MailboxMessage`` — so it renders and previews, both required;
        2. its class is *exactly* the class of the message being handled;
        3. it is not a cancel.

        **Conditions 1 and 2 are the whole opt-in, and both are expressed in the
        type.** A class opts out of mid-run reads by not extending
        ``MailboxMessage`` at all — ``TriageMessage`` in ``akgentic-agent``'s
        exemplar is the worked case: it declares ``rendering()`` for ``act()``,
        joins no mailbox, and is never offered. There is deliberately no *second*
        way to opt out, such as a previewable base with an optional preview:
        two mechanisms for one rule is how a message ends up listed with an id it
        cannot honour.

        There used to be a fourth condition — a card field,
        ``mailbox_preview_handlers``, naming by dotted path which handlers' runs
        could be offered the mailbox. It is gone and nothing replaced it: 1 and 2
        already decide exactly what it decided, and it cost a dotted-path
        resolution, four ``ValueError`` shapes and a typo that surfaced only at
        agent init. The coarse switch survives on the card as
        ``read_mailbox=False``, which suppresses the notice outright.

        Condition 2 is what keeps routing correct. Same class means same
        handler means same output type, so an absorbed message is answered in
        the shape its own handler would have produced. An exact class check,
        not ``isinstance``: a subclass routes to its own handler.

        Condition 3 exists because a ``/stop`` arrives as an ordinary
        ``AgentMessage``, which *does* have a preview. Offering its id would let
        the model read its way out of being cancelled.

        Returns an empty set while idle — there is no current message to match
        against, so nothing can be offered.
        """
        current = self._observer.current_message()
        if current is None:
            return set()
        return {
            message.id
            for message in pending
            if isinstance(message, MailboxMessage)
            and type(message) is type(current)
            and not is_cancel(message)
        }

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Purge and raise on a pending cancel, else announce mail that arrived.

        The purge runs *before* the raise, and its result is deliberately not
        branched on. ``consume_mailbox`` ignores ids that are no longer queued,
        so an empty return means "already gone", never "no cancel" — the
        recognition has already happened and the run must die either way. It
        also emits the ``HandledMessage`` per removal itself, which is why
        nothing is emitted here: the telemetry belongs to the primitive so that
        no call site can forget it, and none can double it.
        """
        pending = self._observer.get_mailbox()
        cancels = [message for message in pending if is_cancel(message)]
        if cancels:
            self._observer.consume_mailbox([message.id for message in cancels])
            raise RunInterruptedError(
                "The current run was cancelled by a queued /stop or CancelMessage."
            )
        # The doorbell is configurable; cancellation is not. This gate sits BELOW
        # the purge-and-raise deliberately — an agent that never announces mail is
        # still interruptible by a queued /stop or CancelMessage. Moving it above
        # would turn a notice setting into "this agent cannot be cancelled", and
        # no notice-shaped test would catch that.
        if not self._arrival_notice:
            return request_context
        new_messages = [m for m in pending if m.id not in self._announced_ids]
        if new_messages:
            notice = render_arrival_notice(
                new_messages, self.offerable_ids(new_messages), self._arrival_closing
            )
            # The enqueue id is kept so ``after_node_run`` can withdraw *this* entry and
            # nothing else. Withdrawal must key on the enqueue site, never on the rendered
            # text: matching by content would couple the withdrawal to the notice's wording.
            enqueue_id = ctx.enqueue(notice, priority="asap")
            if enqueue_id is not None:
                self._notice_enqueue_ids.add(enqueue_id)
            self._announced_ids.update(message.id for message in new_messages)
        return request_context

    async def after_node_run(
        self, ctx: RunContext[Any], *, node: AgentNode[Any], result: NodeResult[Any]
    ) -> NodeResult[Any]:
        """Withdraw the arrival notice when the run has reached its end.

        **The doorbell must not cost the answer it interrupted.** An ``'asap'`` enqueue is
        always one step late — ``PendingMessageDrainCapability`` is mounted ``outermost``
        and ``before_*`` hooks walk the chain forwards, so that step's drain has already
        run by the time :meth:`before_model_request` enqueues. When the model produces its
        final output on that same step, the graph returns ``End(FinalResult)`` and the
        drain's own ``after_node_run`` **discards the End** and redirects into one more
        model request so the queued content is not lost. The run then produces a *second*
        final result, and ``run_sync`` returns only that one: the answer the agent had
        already written reaches nobody.

        That redirect is right for content with no other delivery path, and wrong for
        ours. The message behind the notice is still sitting in the actor mailbox and gets
        its own turn whatever happens here — which is exactly the fallback ADR-010 §5
        already specifies. So the notice is withdrawn rather than delivered, the ``End``
        survives untouched, and the turn ends with its output intact.

        This hook is where that is possible at all: ``after_*`` walks the chain
        **backwards**, so this capability runs *ahead* of the outermost drain and can empty
        the queue before it looks. The result is returned unchanged in every case — this
        hook never converts an ``End``, never creates one, and never redirects.

        **Only the notice is withdrawn.** :meth:`after_tool_execute` enqueues a message it
        has already *consumed* from the mailbox; withdrawing that would lose the message
        outright, with no queue left holding it. Its enqueue id is deliberately never
        recorded. Entries belonging to any other producer are left alone too, so the drain
        still redirects for them.

        ``_announced_ids`` is deliberately **not** rolled back: it is run-local, the run is
        ending, and ``before_run`` clears it. Re-announcing next run is the documented and
        acceptable outcome.

        Args:
            ctx: The run context, whose ``pending_messages`` is the queue to filter.
                pydantic-ai iterates that queue only between graph nodes — in
                ``before_model_request`` and here — so mutating it in place is supported.
            node: The node that just ran. Unused: only the result decides.
            result: The next node, or the ``End`` that ends the run.

        Returns:
            ``result``, always and unmodified.
        """
        if not isinstance(result, End) or ctx.pending_messages is None:
            return result
        if not self._notice_enqueue_ids:
            return result
        ctx.pending_messages[:] = [
            pending
            for pending in ctx.pending_messages
            if pending.enqueue_id not in self._notice_enqueue_ids
        ]
        return result

    async def after_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: Any,
    ) -> Any:
        """Turn a completed ``read_mailbox`` call into the message it named.

        The tool is a *signal*: it takes an id, touches no mailbox and renders
        nothing. This hook is what makes its acknowledgement true — it consumes
        exactly the named message and enqueues that message's own
        ``rendering()`` at ``"asap"``, the same durable drain the arrival
        notice uses. The tool's return value is passed through untouched; the
        content arrives as its own injected turn rather than as a tool result.

        Every failure is a **silent no-op**, deliberately. An absent, empty,
        malformed or unknown id consumes nothing and enqueues nothing, and the
        run carries on. That is what lets the tool half and this half land in
        sequence rather than atomically: against an older ``read_mailbox`` that
        takes no arguments there is no id to find, and the correct behaviour is
        to do nothing rather than to raise inside someone's run.
        ``MailboxRenderError`` guards the offer filter and has no business here.

        **The enqueue id here is discarded on purpose, and that is load-bearing.**
        :meth:`after_node_run` withdraws every id it was given; this content must
        never be among them. The message has already been consumed from the
        mailbox by the line above, so the queue is the only thing still holding
        it — withdrawing it would lose it outright, with nothing left to deliver
        it as its own turn. Unreachable today, because a step that called a tool
        returns a ``ModelRequestNode`` and never an ``End``, so this content is
        always drained normally. It is an invariant rather than a live path, and
        it is written down because the code that guarantees it is not the code
        that will be edited next.

        Args:
            ctx: The run context, whose ``enqueue`` is documented safe from a
                capability hook.
            call: The completed tool call; only its name is consulted.
            tool_def: The tool's definition. Unused.
            args: The call's validated arguments, read by name.
            result: The tool's own return value, returned unchanged.

        Returns:
            ``result``, always and unmodified.
        """
        if call.tool_name != READ_MAILBOX_TOOL:
            return result

        raw_id = args.get(MESSAGE_ID_ARG)
        try:
            message_id = uuid.UUID(str(raw_id))
        except (ValueError, AttributeError, TypeError):
            logger.info("read_mailbox named no usable message id (%r); nothing absorbed", raw_id)
            return result

        # Peek, render, and only THEN consume. Consuming first would delete a
        # message that turns out to render nothing: gone from the mailbox, never
        # enqueued, and with no queue left holding it — lost outright, silently.
        # Leaving it queued costs nothing and is the behaviour ADR-010 §5 already
        # specifies for mail the run does not take on: it arrives as its own turn.
        target = next((m for m in self._observer.get_mailbox() if m.id == message_id), None)
        if target is None:
            logger.info("read_mailbox named id %s, which is no longer queued", message_id)
            return result

        if not isinstance(target, MailboxMessage):
            logger.info(
                "%s is not a MailboxMessage and cannot render; left in the mailbox to arrive "
                "as its own turn",
                type(target).__name__,
            )
            return result

        rendering = target.rendering()
        self._observer.consume_mailbox([message_id])
        ctx.enqueue(f"{self._absorbed_prefix}\n\n{rendering}", priority="asap")
        return result
