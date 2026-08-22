# MailboxTool

The agent's own mailbox as a capability. An agent's mailbox is its actor inbox: while one turn is
being processed, every message told to it queues up behind the one in flight. This card exposes
that queue on two channels — an on-demand **consuming** read, and the `/stop` cancellation surface
— through a **local peek**: the card creates no actor and performs no proxy round trip.

```python
from akgentic.tool import MailboxTool
```

| | |
|---|---|
| Module | `akgentic.tool.mailbox` |
| Actor | none — this card owns no actor and creates none |
| Channels used | `TOOL_CALL`, `COMMAND` |
| Optional extras | none |

---

## The ToolCard

```python
class MailboxTool(ToolCard):
    read_mailbox: ReadMailbox | bool = True
    stop: Stop | bool = True
```

Two capabilities, one channel each by default, both fields following the same `Param | bool`
convention: `True` (the default) enables the capability with its param's defaults, a param
instance may narrow the channels, and `False` removes **exactly that capability** and nothing
else. Neither param carries a field beyond `expose` — configuration read at factory bind time,
never tool-call schema.

## ToolCard fields

| Field | Type | Default | Channel default | Meaning |
|---|---|---|---|---|
| `read_mailbox` | `ReadMailbox \| bool` | `True` | `{TOOL_CALL}` | The on-demand consuming read, exposed to the model. |
| `stop` | `Stop \| bool` | `True` | `{COMMAND}` | The `stop` command — the `/stop` string surface. |

Every capability is gated twice: the param must resolve, **and** its `expose` set must contain the
channel the serving hook covers (`get_tools()` for `TOOL_CALL`, `get_commands()` for `COMMAND`). A
capability exposed on a channel this card does not serve is dropped silently — the package-wide
rule, not a mailbox quirk.

---

## The two channels

### `TOOL_CALL` — `read_mailbox()`

Renders sender, type, the reply protocol each message expects and its **full content**, numbered,
oldest first. An empty mailbox returns a sentence saying so — never `""`, which reads as a
malfunction to the model.

**The read consumes what it renders, and that contract lives in the docstring the model reads.**
Everything listed has been removed from the mailbox and will not arrive again as its own turn, so
the model is expected to deal with it inside the current run. Anything left unread stays queued
and arrives as its own turn later.

This inverts the card's original non-consuming contract, and the inversion is the point: a peek
that promised redelivery let the model answer a pending message mid-run and then answer it a
second time when the message got its turn. Consumption removes that failure by construction
rather than by docstring.

Two properties of the implementation carry the guarantee:

- **A cancel is neither rendered nor consumed.** A `CancelMessage`, or a message whose first
  whitespace-delimited token is exactly `/stop`, is filtered out of both halves and stays queued.
  The mailbox is the cancellation's single source of truth; consuming a cancel would let the model
  read its way out of being cancelled. `/stopwatch` and `please /stop` are ordinary mail. When
  every pending message is a cancel, the read returns the empty-mailbox sentence — correct: the
  model must not learn about the cancel here.
- **What renders is what consumption *returned*, never the peeked list.** `consume_mailbox` skips
  envelopes carrying a `reply_to` and ignores ids dequeued in between, so a peek is a superset.
  Rendering the superset would show the model a message it did not absorb — the exact hazard being
  removed. The cost is that a `reply_to` envelope is invisible to the read; it still arrives as its
  own turn, so nothing is lost.

The removal telemetry (one `HandledMessage` per message) is emitted by `consume_mailbox` itself.
The card emits nothing: no call site can forget it, and none can double it.

**The reply protocol per message.** Each rendered message carries the framing the agent's
`receiveMsg_AgentMessage` would have applied — *"You received a `request` from `@Alice`. A reply is
expected: respond to @Alice with the result."* — for each of the five known types (`request`,
`response`, `instruction`, `notification`, `acknowledgment`). The type is read duck-typed, so a
message carrying none (a bare `UserMessage`, a `CancelMessage`) renders sender, type name and
content with no protocol line and no error.

The canonical table is `REPLY_PROTOCOLS` in `akgentic.agent.output_models`, which this package may
not import. The duplication is real and unavoidable across the package boundary — nothing keeps the
two copies in sync.

This is in-life code, so the raising accessor form applies: calling `read_mailbox` after the
owning agent stopped raises `ToolObserverGone` — a defined outcome, not a crash.

### `COMMAND` — `stop()`

Canonical name `stop`, string surface `/stop`, zero arguments, registered in the command registry
and therefore announced to every frontend for free via `CommandsAnnouncedEvent`.

**Idle semantics: the handler answers.** It returns `"There is no run to cancel."` Commands
dispatch while the agent is idle, and a cancel that reaches a handler is by construction the idle
case — the agent purges a mid-run cancel at the moment it recognises it, so one can never be
dequeued into a handler. There is therefore exactly one thing a dispatched `/stop` can mean, and
saying it beats silence: a human who typed `/stop` and heard nothing back cannot tell a no-op from
a failure.

Its real effect is mid-run, and that effect is not implemented here: a `/stop` message sitting in a
busy agent's mailbox is recognised by the agent-side cancel hook, which is `akgentic-agent`'s
enforcement. Nothing in this card raises, tracks, or interrupts.

The registry's `dispatch` and `_invoke` keep their `str | None` signature. *A command may decide it
has nothing to say* is a general primitive of the command system; `/stop` merely stops being its
first user.

---

## Where the cancel vocabulary lives

**Not here.** `is_cancel` and `render_arrival_notice` used to be exported from
`akgentic.tool.mailbox`; they now live in `akgentic-agent`, beside the enforcement that uses them,
in `akgentic/agent/capabilities/mailbox_capability.py`.

The reason is ownership, not tidiness: `BaseAgent` builds its cancel capability **unconditionally**,
precisely so that an agent configured with no `MailboxTool` at all is still interruptible. A
predicate shipping with this card cannot serve that case — there is no card to import it from.

The card does hold a **private** exclusion filter, so that its consuming read leaves a cancel
alone. That is not a second vocabulary and is not exported: it runs the opposite way round — the
card protecting the agent's source of truth rather than the agent depending on the card. It must
stay at least as broad as the agent's predicate, because over-excluding costs a message its place
in one read while under-excluding silently kills an interrupt.

What this card keeps is its own surface: the `read_mailbox` tool and the `/stop` command
**registration**. Recognising the `/stop` string as a cancellation is the agent's job, and it does
that without importing anything from here.

## The observer protocol

`MailboxToolObserver` is this tool's own contract, living beside the tool rather than in `core/`
(a domain-specific observer belongs to its domain). It extends `ToolObserver` with two methods:

```python
@runtime_checkable
class MailboxToolObserver(ToolObserver, Protocol):
    def get_mailbox(self) -> list[Message]: ...
    def consume_mailbox(self, message_ids: list[uuid.UUID]) -> list[Message]: ...
```

`get_mailbox()` peeks over the owning agent's inbox, oldest first, dequeuing nothing — but it is no
longer a promise of redelivery, since `read_mailbox` goes on to consume most of what it returns.
`consume_mailbox()` performs the removal and returns what it actually removed; it is
caller-idempotent on ids no longer queued. `Akgent` provides both, so no agent-side change was
needed to satisfy the widened protocol. Conformance is a documented precondition rather than a
runtime gate: observers are duck-typed, so a non-conforming one fails at first use.

---

## Configuration

```python
from akgentic.tool import MailboxTool
from akgentic.tool.core import TOOL_CALL
from akgentic.tool.mailbox import ReadMailbox, Stop

MailboxTool()                     # both capabilities on (the default)
MailboxTool(read_mailbox=False)   # /stop only — no on-demand read
MailboxTool(stop=False)           # no cancellation surface
MailboxTool(read_mailbox=ReadMailbox(expose={TOOL_CALL}))  # explicit param form
```

### Failure modes worth knowing

- **An empty mailbox is not an error.** `read_mailbox` returns its empty-mailbox sentence, and so
  does a mailbox holding nothing but cancels.
- **A collected observer degrades, in the channel-appropriate way.** `read_mailbox` raises
  `ToolObserverGone` (in-life code, raising form); the `stop` closure captures nothing
  observer-shaped at all and outlives its agent without pinning it.
- **A removed field fails silently.** `ToolCard` keeps Pydantic's default `extra="ignore"`, so
  `MailboxTool(mailbox_status=True)` — or a catalog entry persisted before that field was deleted
  — constructs happily and drops the value on the floor rather than raising.
- **The reply-protocol table can drift.** It is a copy of the agent's, kept in sync by nothing.

## What is inert until the agent epics land

Auto-adding this card to every agent, the cancel vocabulary, its enforcement hook,
`RunInterruptedError` and the mid-run arrival-notice injection are all agent-side (that package's
Epic 20). Until then, adding the card does **not** enable cancellation — `/stop` is announced,
dispatches to the idle answer, and interrupts nothing.

`read_mailbox` works as soon as the card is wired to an agent that satisfies the observer
protocol.

### Import paths

```python
from akgentic.tool import MailboxTool
from akgentic.tool.mailbox import (
    MailboxTool,
    MailboxToolObserver,
    ReadMailbox,
    Stop,
)
```

Only `MailboxTool` is re-exported from the package root — the import a deployment writes. The
params and the observer protocol come from `akgentic.tool.mailbox`; `MailboxToolObserver` in
particular is the one symbol the agent package imports from this path.

---

See the [package README](../../../../README.md) for the `ToolCard` / `ToolFactory` machinery, the
channel system, and the authoring guide this card is the worked example of.
