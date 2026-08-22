# MailboxTool

The agent's own mailbox as a capability. An agent's mailbox is its actor inbox: while one turn is
being processed, every message told to it queues up behind the one in flight. This card exposes
that queue on three channels at once — live status as structured context state, an on-demand
non-consuming read, and the `/stop` cancellation surface — through a **local peek**: the card
creates no actor and performs no proxy round trip.

```python
from akgentic.tool import MailboxTool
```

| | |
|---|---|
| Module | `akgentic.tool.mailbox` |
| Actor | none — this card owns no actor and creates none |
| Channels used | `LLM_CONTEXT`, `TOOL_CALL`, `COMMAND` |
| Optional extras | none |

---

## The ToolCard

```python
class MailboxTool(ToolCard):
    mailbox_status: MailboxStatus | bool = True
    read_mailbox: ReadMailbox | bool = True
    stop: Stop | bool = True
```

Three capabilities, one channel each by default, every field following the same `Param | bool`
convention: `True` (the default) enables the capability with its param's defaults, a param
instance may narrow the channels, and `False` removes **exactly that capability** and nothing
else. None of the three params carries a field beyond `expose` — configuration read at factory
bind time, never tool-call schema.

## ToolCard fields

| Field | Type | Default | Channel default | Meaning |
|---|---|---|---|---|
| `mailbox_status` | `MailboxStatus \| bool` | `True` | `{LLM_CONTEXT}` | The pending-mailbox snapshot as structured context state. |
| `read_mailbox` | `ReadMailbox \| bool` | `True` | `{TOOL_CALL}` | The non-consuming on-demand peek, exposed to the model. |
| `stop` | `Stop \| bool` | `True` | `{COMMAND}` | The `stop` command — the `/stop` string surface. |

Every capability is gated twice: the param must resolve, **and** its `expose` set must contain the
channel the serving hook covers (`get_context_states()` for `LLM_CONTEXT`, `get_tools()` for
`TOOL_CALL`, `get_commands()` for `COMMAND`). A capability exposed on a channel this card does not
serve is dropped silently — the package-wide rule, not a mailbox quirk.

---

## The three channels

### `LLM_CONTEXT` — live mailbox status

`get_context_states()` returns one provider, named `mailbox_state`, built by
`make_mailbox_state_provider` on the card's bound `None`-returning observer accessor. The provider
contract is the standard one:

- **It never raises.** Any failure — including a collected observer — returns `None`, logged, and
  the turn's context is built without the mailbox block.
- **It captures the accessor, never the observer**, so it cannot pin a stopped agent.

The state it produces is `MailboxState`, a list of `MailboxRow(sender, message_type, preview)`
where `preview` is the first line of the message content truncated to ~120 characters, taken
literally — a content whose first line is empty yields an empty preview, with no fallback to later
lines. `render_full()` reads "N message(s) pending from …, consider wrapping up the current
thread" and is `""` when the mailbox is empty. `render_delta(previous)` names **arrivals only**
and is `None` when nothing arrived: messages that left the mailbox became their own turns, and
narrating a departure would be double delivery.

### `TOOL_CALL` — `read_mailbox()`

Renders sender, type and **full content** of every pending message, numbered, oldest first. An
empty mailbox returns a sentence saying so — never `""`, which reads as a malfunction to the
model.

**The redelivery contract is the load-bearing part, and it lives in the docstring the model
reads:** the read is a non-consuming peek, so every message listed is still delivered to the agent
as its own turn after the current run ends. Reading is an act of prioritisation — deciding whether
to wrap up early — never of consumption, and never a licence to answer a pending message from
inside the current run. A model that "answers" a pending message mid-run produces a duplicated
answer when that message arrives as its own turn.

This is in-life code, so the raising accessor form applies: calling `read_mailbox` after the
owning agent stopped raises `ToolObserverGone` — a defined outcome, not a crash.

### `COMMAND` — `stop()`

Canonical name `stop`, string surface `/stop`, zero arguments, registered in the command registry
and therefore announced to every frontend for free via `CommandsAnnouncedEvent`.

**Idle semantics: the handler reports nothing.** Commands dispatch while the agent is idle, so by
the time `stop()` runs there is nothing to cancel — it returns `None`, which the registry
propagates unchanged and the caller reads as *handled, say nothing*. Its real effect is mid-run,
and that effect is not implemented here: a `/stop` message sitting in a busy agent's mailbox is
recognised by the agent-side cancel hook, which is `akgentic-agent`'s enforcement. Nothing in this
card raises, tracks, or interrupts.

A reply would double-report the outcome: a cancel that mattered was already reported by the
interruption path, and one that arrived too late has nothing to add.

---

## Where the cancel vocabulary lives

**Not here.** `is_cancel` and `render_arrival_notice` used to be exported from
`akgentic.tool.mailbox`; they now live in `akgentic-agent`, beside the enforcement that uses them,
in `akgentic/agent/capabilities/mailbox_capability.py`.

The reason is ownership, not tidiness: `BaseAgent` builds its cancel capability **unconditionally**,
precisely so that an agent configured with no `MailboxTool` at all is still interruptible. A
predicate shipping with this card cannot serve that case — there is no card to import it from.

What this card keeps is its own surface: the `MailboxState` provider, the `read_mailbox` tool, and
the `/stop` command **registration**. Recognising the `/stop` string is the agent's job, and it
does that without importing anything from here.

## The observer protocol

`MailboxToolObserver` is this tool's own contract, living beside the tool rather than in `core/`
(a domain-specific observer belongs to its domain). It extends `ToolObserver` with one method:

```python
@runtime_checkable
class MailboxToolObserver(ToolObserver, Protocol):
    def get_mailbox(self) -> list[Message]: ...
```

`get_mailbox()` is a **non-consuming peek** over the owning agent's inbox, oldest first.
`BaseAgent` satisfies the protocol structurally — no agent-side change is needed for the wiring.
Conformance is a documented precondition rather than a runtime gate: observers are duck-typed, so
a non-conforming one fails at first use.

---

## Configuration

```python
from akgentic.tool import MailboxTool
from akgentic.tool.core import COMMAND, LLM_CONTEXT
from akgentic.tool.mailbox import MailboxStatus, Stop

MailboxTool()                     # all three capabilities on (the default)
MailboxTool(read_mailbox=False)   # status + /stop only — no on-demand peek
MailboxTool(stop=False)           # no cancellation surface
MailboxTool(mailbox_status=MailboxStatus(expose={LLM_CONTEXT}))  # explicit param form
```

### Failure modes worth knowing

- **An empty mailbox is not an error anywhere.** The state renders `""` (say nothing), the delta
  is `None`, and `read_mailbox` returns its empty-mailbox sentence.
- **A collected observer degrades, in the channel-appropriate way.** The context-state provider
  returns `None` (never raises); `read_mailbox` raises `ToolObserverGone` (in-life code, raising
  form); the `stop` closure captures nothing observer-shaped at all and outlives its agent
  without pinning it.
- **A silent `/stop` is the designed outcome, not a broken command.** `dispatch("/stop")` returns
  `None` rather than a string; a caller that string-renders every dispatch result would show the
  literal word `None` to the user.
- **The preview is spec-literal.** A message whose first content line is empty or
  whitespace-only yields that whitespace as its row's preview — there is no fallback to later
  lines.

## What is inert until the agent epics land

The card is complete on its side, and two of its three capabilities activate only when
`akgentic-agent` catches up:

- `MailboxState` reaches **no model** until the agent-side context delivery loop lands (that
  package's Epic 19) — the provider is collected today only by code that also serves the other
  `LLM_CONTEXT` cards.
- Auto-adding this card to every agent, the cancel vocabulary, its enforcement hook,
  `RunInterruptedError` and the mid-run arrival-notice injection are all agent-side (that
  package's Epic 20). Until then, adding the card does **not** enable cancellation — `/stop` is
  announced, dispatches silently, and interrupts nothing.

`read_mailbox` works as soon as the card is wired to an agent that satisfies the observer
protocol.

### Import paths

```python
from akgentic.tool import MailboxTool
from akgentic.tool.mailbox import (
    MailboxRow,
    MailboxState,
    MailboxStatus,
    MailboxTool,
    MailboxToolObserver,
    ReadMailbox,
    Stop,
    make_mailbox_state_provider,
)
```

Only `MailboxTool` is re-exported from the package root — the import a deployment writes. The
params, the state models and the observer protocol come from `akgentic.tool.mailbox`;
`MailboxToolObserver` in particular is the one symbol the agent package imports from this path.

---

See the [package README](../../../../README.md) for the `ToolCard` / `ToolFactory` machinery, the
channel system, and the authoring guide this card is the worked example of.
