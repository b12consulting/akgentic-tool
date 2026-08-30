# MailboxTool

The agent's own mailbox as a capability. An agent's mailbox is its actor inbox: while one turn is
being processed, every message told to it queues up behind the one in flight. This card gives that
queue two channels — an on-demand **signal** the model calls to name one waiting message by id, and
the `/stop` cancellation surface a human or a program drives: the card creates no actor and performs
no proxy round trip.

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

Two switches and nothing else. The wording a mid-run arrival reads with is not here — see
[The injected prompt text](#the-injected-prompt-text) for where it went. Which messages may be
absorbed mid-run is decided by the message *type*, not by a card field either: see
[Which messages may be absorbed mid-run](#which-messages-may-be-absorbed-mid-run).

## ToolCard fields

| Field | Type | Default | Channel default | Meaning |
|---|---|---|---|---|
| `read_mailbox` | `ReadMailbox \| bool` | `True` | `{TOOL_CALL}` | The on-demand signal, exposed to the model. |
| `stop` | `Stop \| bool` | `True` | `{COMMAND}` | The `stop` command — the `/stop` string surface. |

Every capability is gated twice: the param must resolve, **and** its `expose` set must contain the
channel the serving hook covers (`get_tools()` for `TOOL_CALL`, `get_commands()` for `COMMAND`). A
capability exposed on a channel this card does not serve is dropped silently — the package-wide
rule, not a mailbox quirk.

---

## The two channels

### `TOOL_CALL` — `read_mailbox(message_id)`

```python
def read_mailbox(message_id: str) -> str: ...
```

**A signal, not a read.** The call takes the id of one message waiting in the agent's mailbox,
checks the owning agent is still alive, and returns a short acknowledgement. That is the whole
body: it reads nothing, consumes nothing and renders nothing.

**It deliberately does not validate the id.** There is no useful thing the card could do with a bad
one — it holds no content to withhold and can offer the model no error it could act on — and
looking an id up would put `get_mailbox` back into a card that has just stopped calling it.

**The absorption contract survives; what changed is the mechanism, not the promise.** Naming a
message takes it on in the current run: it will not be delivered again as its own turn, so the
model is expected to deal with it now. Anything left unnamed stays queued and arrives as its own
turn later. That contract lives in the docstring the model reads, which is the only place it can
be taught.

**What makes the acknowledgement true is `MailboxCapability`, in this package.** It reads the id
back off the completed tool call through `after_tool_execute`, consumes that one message, and
injects its content into the run; the message renders itself via `rendering()`. None of that
is the *card's* — carrying configuration and serving channels is one job, acting on the mailbox is
another — but both now ship together, so there is no cross-release window between them.

That division is also why an acknowledgement is a *complete* return rather than a stub. The model
already holds the message's preview from the arrival notice, and its content arrives through the
capability's injection. Returning the body here as well would make the card a second carrier of one
fact — the same duplication that got the card's `LLM_CONTEXT` half deleted in epic 34.

#### `message_id` is a contract with another repository

The documented signature is `read_mailbox(message_id: str) -> str`, and the parameter name is part
of the contract: `MailboxCapability` reads the id out of the completed tool call's arguments **by
name**, through the `MESSAGE_ID_ARG` constant. Rename one side only and the suite stays green while
in production the model names a message and silently receives nothing — no exception, no log line.

Both sides now live in this package, so a rename is a local change rather than a coordinated
two-repository release — but it is still a rename of *two* things, and the type system will not
tell you if you do one.

This is in-life code, so the raising accessor form applies: calling `read_mailbox` after the owning
agent stopped raises `ToolObserverGone` — a defined outcome, not a crash. The liveness check is
load-bearing rather than defensive here, because an acknowledgement from a card whose agent has
stopped is a false one: the capability that would have acted on it went with the agent.

### `COMMAND` — `stop()`

Canonical name `stop`, string surface `/stop`, zero arguments, registered in the command registry
and therefore announced to every frontend for free via `CommandsAnnouncedEvent`.

**Idle semantics: the handler answers.** It returns `"There is no run to cancel."` Commands
dispatch while the agent is idle, and a cancel that reaches a handler is by construction the idle
case — the agent purges a mid-run cancel at the moment it recognises it, so one can never be
dequeued into a handler. There is therefore exactly one thing a dispatched `/stop` can mean, and
saying it beats silence: a human who typed `/stop` and heard nothing back cannot tell a no-op from
a failure.

Its real effect is mid-run, and it is not the *card* that delivers it: a `/stop` sitting in a busy
agent's mailbox is recognised by `MailboxCapability.before_model_request`, which purges it and
raises. Nothing in this card raises, tracks, or interrupts. `BaseAgent` builds that capability for
every agent and `act()` catches the error — the enforcement stays `akgentic-agent`'s, which is what
keeps cancellation impossible to de-configure.

The registry's `dispatch` and `_invoke` keep their `str | None` signature. *A command may decide it
has nothing to say* is a general primitive of the command system; `/stop` merely stops being its
first user.

---

## Where the cancel vocabulary lives

**Here, in `capability.py`, beside the card.** `is_cancel` and `render_arrival_notice` are exported
from `akgentic.tool.mailbox` along with `MailboxCapability` itself.

They spent a while in `akgentic-agent`, on the argument that `BaseAgent` builds the capability
**unconditionally** — so an agent configured with no `MailboxTool` is still interruptible — and a
predicate shipping with the card could not serve that case, there being no card to import it from.
Two things retired that argument: `BaseAgent` auto-inserts a default `MailboxTool` when the config
carries none, so the card is never absent; and this package is a hard dependency of
`akgentic-agent`, so what ships here is never unavailable there.

The property the argument protected is untouched. Cancellation still cannot be de-configured,
because it rests on the *wiring* being unconditional — `BaseAgent` builds the capability for every
agent and `act()` catches `RunInterruptedError` — not on which package the code sits in. What the
move buys is that card, capability and vocabulary are one subject in one repository: a mailbox
change is one story instead of two, with no release in between.

The card holds no cancel predicate of its own, not even a private one. Keeping a cancel out of what
the model is offered is `akgentic-agent`'s offer rule, on the same side of the boundary as the
enforcement it protects. This card keeps only its own surface: the `read_mailbox` signal and the
`/stop` command **registration**. Recognising the `/stop` string as a cancellation is the agent's
job, and it does that without importing anything from here.

## The observer protocol

`MailboxToolObserver` is this tool's own contract, living beside the tool rather than in `core/`
(a domain-specific observer belongs to its domain). It extends `ToolObserver` with two methods:

```python
@runtime_checkable
class MailboxToolObserver(ToolObserver, Protocol):
    def get_mailbox(self) -> list[Message]: ...
    def consume_mailbox(self, message_ids: list[uuid.UUID]) -> list[Message]: ...
```

`get_mailbox()` peeks over the owning agent's inbox, oldest first, dequeuing nothing.
`consume_mailbox()` performs the removal and returns what it actually removed; it is
caller-idempotent on ids no longer queued, and it skips envelopes carrying a `reply_to`, so its
return is a subset of what was asked for. The removal telemetry — one `HandledMessage` per message
— is emitted by `consume_mailbox` itself, so no call site can forget it and none can double it.

**Neither method is called by this card.** The protocol is unchanged and both methods are still
needed, but their caller is `MailboxCapability` rather than the card: it peeks to build the
arrival notice that offers the model a message id, and consumes the one message the model went on
to name. `Akgent` provides both, so no agent-side change was needed to satisfy the protocol.
Conformance is a documented precondition rather than a runtime gate: observers are duck-typed, so
a non-conforming one fails at first use.

---

## Configuration

```python
from akgentic.tool import MailboxTool
from akgentic.tool.core import TOOL_CALL
from akgentic.tool.mailbox import ReadMailbox, Stop

MailboxTool()                     # both capabilities on (the default)
MailboxTool(read_mailbox=False)   # /stop only — no on-demand signal
MailboxTool(stop=False)           # no cancellation surface
MailboxTool(read_mailbox=ReadMailbox(expose={TOOL_CALL}))  # explicit param form
```

### Which messages may be absorbed mid-run

**Decided by the type, not by a setting.** A class extends `MailboxMessage` to declare it can
travel through a mailbox:

```python
from akgentic.tool.mailbox import MailboxMessage

class AgentMessage(MailboxMessage):
    def rendering(self) -> str: ...
    def rendering_preview(self) -> str: ...
```

Both methods are **required**, and the base raises `NotImplementedError` for either one a subclass
leaves unanswered. A message that can be delivered can be listed, so there is no coherent class that
renders but declines a preview — and an optional preview returning `None` would be a second way of
saying "not offerable", competing with the rule that already decides it.

Opting out is one mechanism only: **do not extend `MailboxMessage`.** A class that declares
`rendering()` alone still satisfies `LlmRenderable` and can be handed to `act()`; it is simply never
offered an id in the arrival notice. `TriageMessage` in `akgentic-agent`'s exemplar is the worked
case.

The offer filter then adds two conditions of its own: the pending message's class must be *exactly*
the class being handled (same class means same handler means same output type, so an absorbed
message is answered in the shape its own handler would have produced), and it must not be a cancel.

> **A `mailbox_preview_handlers` field used to do this**, naming handler classes by dotted path,
> resolved at agent init with four `ValueError` shapes for a malformed entry. It is gone and nothing
> replaced it: the type and the exact-class match already decided exactly what it decided, at no
> configuration cost. The coarse switch survives as `read_mailbox=False`, which suppresses the
> notice outright.

### The injected prompt text

**Not on this card. It is `MailboxCapability`'s, one module over in the same package.**

```python
MailboxCapability(
    observer,
    card,
    absorbed_prefix="Additional work, taken on mid-run. It does NOT replace …",
    arrival_closing="Call `read_mailbox` with one of the ids above …",
)
```

Two keyword-only constructor parameters on `MailboxCapability` (`capability.py`), each defaulting
to the module constant beside it — `ABSORBED_PREFIX` and the shipped closing. `absorbed_prefix`
frames a message the model took on through `read_mailbox`; `arrival_closing` is the last line of
the notice announcing mail that landed while a run was in flight. Both are assigned straight
through, so an explicitly passed `""` is honoured as the caller's choice rather than replaced by
the default.

**Who can actually pass them: code, not a catalog.** `BaseAgent` builds the capability with two
arguments and passes neither parameter, so every deployment runs the constants and improving a
sentence is a change to `capability.py` that reaches every existing team on upgrade. That is the
point of the move rather than a gap in it — the parameters exist so the wording has one home and a
caller constructing the capability directly can still override it, not so an operator can tune
prose per team.

> **One clause is load-bearing, and an override can delete it.** `ABSORBED_PREFIX` opens with
> *"It does NOT replace what you were already asked to do"*. It is there because of an observed
> failure: an agent that had just finished a report took on a newer mid-run question, answered only
> that, and the report reached nobody. Reword freely; keep something that says the arrival is
> *additional* work.

> **The wording briefly lived here, as two `str` card fields, and was moved back out.** The
> argument for the card was about *reading* a catalog entry: a literal default shows an operator
> what the agent is currently being told, where a `null` shows nothing. What it did not weigh is
> the *write* side — the catalog dumps a card with a plain `model_dump(mode="json")` and no
> `exclude_defaults`, and `BaseAgent` auto-inserts a `MailboxTool()` when the config has none. So
> every persisted team froze its own private copy of prose that is expected to keep improving, and
> an improvement reached only teams created after it. No deployment ever turned the knob; every
> one paid for the copy. An entry written while the fields existed still loads — `ToolCard` keeps
> Pydantic's default `extra="ignore"`, so the two stale keys are dropped on validation and no
> migration is owed.

**This card carries no prompt text at all**, which is now true by construction rather than by
discipline: it serves no `LLM_CONTEXT`, and its whole serialization is `read_mailbox`, `stop` and
the model discriminator. **The whole card is still handed to `MailboxCapability`**, which reads
`read_mailbox` off it to decide whether the doorbell rings at all; `BaseAgent` passes the card and
unpacks nothing. That seam is unchanged — only the wording left it.

There is deliberately **no parameter** for the closing line of a listing that offers no id. A
listing carrying no id offers no read, so there is no timing to advise on and nothing to configure;
that line stays the constant `_CLOSING_WITHOUT_IDS`.

### Failure modes worth knowing

- **A bad id is not an error.** `read_mailbox` does not look the id up, so an id that names nothing
  returns the same acknowledgement as one that names a message. What it acknowledges is then
  nothing at all — no message is absorbed, and any real mail still arrives as its own turn.
- **A collected observer degrades, in the channel-appropriate way.** `read_mailbox` raises
  `ToolObserverGone` (in-life code, raising form); the `stop` closure captures nothing
  observer-shaped at all and outlives its agent without pinning it.
- **A removed field fails silently.** `ToolCard` keeps Pydantic's default `extra="ignore"`, so
  `MailboxTool(mailbox_status=True)` — or a catalog entry persisted before that field was deleted
  — constructs happily and drops the value on the floor rather than raising.
- **A whitelist typo fails loudly, but only at wiring time.** A card sitting in a catalog with an
  unresolvable entry is perfectly loadable; it is the agent that will not start.

## The cross-package half

Auto-adding this card to every agent, the cancel vocabulary, its enforcement hook,
`RunInterruptedError` and the mid-run arrival-notice injection are all agent-side (that package's
Epic 20). `MailboxCapability.before_model_request` is what sees a queued cancel, purges it through
`consume_mailbox` and raises; `act()` absorbs the error, so the run dies while the agent carries
on. Cancellation therefore does not **depend on** this card — that capability is built
unconditionally, and an agent configured without a `MailboxTool` is interruptible just the same.
What the card contributes is the string surface: `/stop` registered, announced via
`CommandsAnnouncedEvent`, and answered when it dispatches while idle.

The delivery half of `read_mailbox` — consuming the named message and injecting its content, the
message rendering itself, and the offer rule deciding which messages are shown with an id — is
`MailboxCapability`, in this package, beside the card.

### Release coupling

**The signal and its delivery ship together, which is the point of them sharing a package.** They
were split across two repositories once, and the split had a real cost: releasing this package
ahead of the agent half left `read_mailbox(message_id)` acknowledging and delivering nothing, while
an agent on older code emitted a notice telling the model to call `read_mailbox` with no argument —
which the new signature rejected. Inert but noisy: retries burned, tokens spent, nothing lost,
self-healing once the other release landed. That window no longer exists.

What remains is a one-directional floor: `akgentic-agent` imports `MailboxCapability` from here and
passes it a card, so it needs a release of this package carrying both. The reverse is not true —
this package depends on `akgentic-core` alone and knows nothing of `akgentic-agent`.

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
