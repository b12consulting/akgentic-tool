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
    mailbox_preview_handlers: list[str] | None = None
    absorbed_prefix: str = "Additional work, taken on mid-run. It does NOT replace …"
    arrival_closing: str = "Call `read_mailbox` with one of the ids above …"
```

Two capabilities, one channel each by default, both fields following the same `Param | bool`
convention: `True` (the default) enables the capability with its param's defaults, a param
instance may narrow the channels, and `False` removes **exactly that capability** and nothing
else. Neither param carries a field beyond `expose` — configuration read at factory bind time,
never tool-call schema.

`mailbox_preview_handlers` is not a capability but a deployment setting; it is documented under
[Configuration](#the-mailbox-preview-whitelist). The last two are deployment settings as well —
prompt text this card **carries and never reads**, documented under
[The injected prompt text](#the-injected-prompt-text).

## ToolCard fields

| Field | Type | Default | Channel default | Meaning |
|---|---|---|---|---|
| `read_mailbox` | `ReadMailbox \| bool` | `True` | `{TOOL_CALL}` | The on-demand signal, exposed to the model. |
| `stop` | `Stop \| bool` | `True` | `{COMMAND}` | The `stop` command — the `/stop` string surface. |
| `mailbox_preview_handlers` | `list[str] \| None` | `None` | — | Dotted paths of the *handler* message classes whose runs show the mailbox preview. `None` means every handler shows it; `[]` means none does. Resolved at wiring time. |
| `absorbed_prefix` | `str` | the wording `akgentic-agent` injects today | — | What a message absorbed through `read_mailbox` is prefixed with when it is injected. See [The injected prompt text](#the-injected-prompt-text). |
| `arrival_closing` | `str` | the wording `akgentic-agent` injects today | — | The mid-run arrival notice's closing line, for a listing that offers at least one id. See [The injected prompt text](#the-injected-prompt-text). |

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

**What makes the acknowledgement true lives in `akgentic-agent`.** That package's mailbox
capability reads the id back off the completed tool call through `after_tool_execute`, consumes
that one message, and injects its content into the run; the message renders itself via
`render_for_llm()`. None of that is this card's, and none of it has shipped yet — see
[Release coupling](#release-coupling) below.

That division is also why an acknowledgement is a *complete* return rather than a stub. The model
already holds the message's preview from the arrival notice, and its content arrives through the
agent's injection. Returning the body here as well would make the card a second carrier of one
fact — the same duplication that got the card's `LLM_CONTEXT` half deleted in epic 34.

#### `message_id` is a contract with another repository

The documented signature is `read_mailbox(message_id: str) -> str`, and the parameter name is part
of the contract: `akgentic-agent`'s capability reads the id out of the completed tool call's
arguments **by name**. Rename it to `msg_id` or `id` and this package's suite stays green, the
agent's suite stays green, and in production the model names a message and silently receives
nothing — no exception, no log line. Neither test suite can detect the mismatch.

Renaming it is therefore a coordinated two-repository change with a joint release, never a local
tidy-up.

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
needed, but their caller is now `akgentic-agent`'s mailbox capability: it peeks to build the
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

### The mailbox preview whitelist

```python
MailboxTool(mailbox_preview_handlers=["akgentic.agent.messages.AgentMessage"])
```

> The example path is what a deployment actually writes, and it is deliberately not something this
> package can resolve on its own: `akgentic-agent` is a dependency of the deployment, not of
> `akgentic-tool`. That is exactly why the setting is a string resolved at runtime rather than a
> class — and why this value must not be copied into a test in this package, where the class is
> absent from CI.

`mailbox_preview_handlers` narrows *when* the model is offered its mailbox. Four things about it:

- **It names the handler's message class** — the message the agent is *currently handling* — not
  the mail waiting in the box. An entry says "runs that started from one of these show the
  preview", never "show me messages of this type".
- **`None` (the default) means every handler shows the preview.**
- **`[]` means none does.** It is a different value from `None` and is never coerced back to it: an
  empty list is a deployment saying *no handler*, and it is honoured as written.
- **Every entry is resolved when the card is wired to an observer** — agent init, via the
  `observer()` override. A bad entry raises `ValueError` naming the offending string, so a typo
  fails before the agent's first run rather than going quiet for its whole life.

The `ValueError` covers four shapes, all configuration defects:

| Shape | Example |
|---|---|
| The path carries no module part | `"AgentMessage"` |
| The module is not importable | `"akgentic.agent.mesages.AgentMessage"` |
| The module has no such attribute | `"akgentic.agent.messages.AgnetMessage"` |
| The path resolves to something that is not a `Message` subclass | `"akgentic.tool.mailbox.MailboxTool"` |

**Constructing or deserializing the card never raises.** Validation is a wiring-time event on
purpose: a field validator would make a card carrying a perfectly valid entry impossible to load in
any process where that class happens not to be importable — a catalog reader, a serialization round
trip. Reading the card is always safe; wiring it to an agent is where a typo surfaces.

Building the preview itself, and deciding which of the pending messages are offered with an id, is
`akgentic-agent`'s. This card only carries the list.

### The injected prompt text

```python
MailboxTool(
    absorbed_prefix=(
        "Additional work, taken on mid-run. It does NOT replace what you were already asked "
        "to do. Answer both before this run ends, one message each, unless the new message "
        "is plainly a correction of the one in flight."
    )
)
```

Two `str` fields carrying the wording a mid-run mailbox arrival reads with: `absorbed_prefix`
frames a message the model took on through `read_mailbox`, and `arrival_closing` is the last line
of the notice announcing mail that landed while a run was in flight. Both **default to the wording
`akgentic-agent` injects today** — the default is the text itself rather than `None`, so a catalog
entry shows an operator what the agent is currently being told and can be judged before it is
edited. Tuning a sentence is then a catalog edit rather than a code change, a release across two
packages and a redeploy.

> **The knob can break delivery, and it is worth knowing how.** `absorbed_prefix` opens with
> *"It does NOT replace what you were already asked to do"*, and a deployment may delete that
> clause. It is there because of an observed failure: an agent that had just finished a report
> took on a newer mid-run question, answered only that, and the report reached nobody. Rewrite the
> wording freely; keep something that says the arrival is *additional* work.

**This card reads neither field.** It carries them, exactly as it carries
`mailbox_preview_handlers`: nothing here renders prompt text, and a card constructed with either
field set produces byte-identical tools, commands and context states to a default one.
`akgentic-agent`'s mailbox capability picks both up at capability construction, and reads them
**defensively**, so a card published before these fields existed still works and the two halves may
be released in either order.

There is deliberately **no third field** for the closing line of a listing that offers no id. A
listing carrying no id offers no read, so there is no timing to advise on and nothing to configure;
that line stays a constant in `akgentic-agent`.

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
message rendering itself, and the offer rule that decides which messages are shown with an id —
is likewise `akgentic-agent`'s, and is **not yet released**.

### Release coupling

**This package must not be released alone.** Released ahead of the agent half,
`read_mailbox(message_id)` returns an acknowledgement and delivers nothing, and an agent still
running the older code emits an arrival notice telling the model to call `read_mailbox` with no
argument at all — which the new signature rejects.

The window is **inert but noisy**: retries are burned and tokens spent on a call the model cannot
satisfy, but nothing is lost. The card consumes nothing, so every unread message still arrives as
its own turn — the designed fallback. It self-heals the moment the agent release lands. Merge both
halves, release this package first, and release `akgentic-agent` immediately after with its floor
raised.

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
