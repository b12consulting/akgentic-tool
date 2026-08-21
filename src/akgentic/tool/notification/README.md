# NotificationTool

Lets an agent schedule a message **to itself**, delivered after a delay — to defer its own
attention, check a long-running result later, or nudge itself if nothing has happened by then.

```python
from akgentic.tool import NotificationTool
```

| | |
|---|---|
| Module | `akgentic.tool.notification.tool` |
| Actor | `NotificationActor`, singleton named `#NotificationTool` |
| Channels used | `TOOL_CALL`, `COMMAND` |
| Optional extras | none |

---

## The ToolCard

```python
class NotificationTool(ToolCard):
    message_class: str = "akgentic.agent.messages.AgentMessage"
    max_delay_seconds: int = 300

    register_notification: RegisterNotification | bool = True
    pending_notification: PendingNotifications | bool = True
    cancel_notification: CancelNotification | bool = True

    _notification_proxy: NotificationActor | None = PrivateAttr(default=None)
```

**Two settings, three capabilities.** `message_class` and `max_delay_seconds` configure delivery;
each capability field decides whether that capability exists and which channels it reaches.

**Validation happens before binding.** `observer()` resolves `message_class` *first*, then calls
`getChildrenOrCreate(NotificationActor, …)`. A misconfigured card therefore cannot leave a live
singleton behind — it raises `ValueError` at wiring time, and the team never starts with a
notification actor that would fail at fire time.

**Each capability captures the owner's address at bind time, as data.** The address is an
`ActorAddress`, never a proxy, so a pending entry cannot pin its owning agent in memory. That
capture is what scopes cancellation, and what scopes listing by default.

---

## ToolCard fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `message_class` | `str` | `"akgentic.agent.messages.AgentMessage"` | Dotted import path of the class delivered when a notification comes due. See [the contract](#the-message_class-contract). |
| `max_delay_seconds` | `int` | `300` | Largest delay an agent may schedule. Reaches the model **through the tool schema**, substituted into the docstring, not hardcoded in the source. |
| `register_notification` | `RegisterNotification \| bool` | `True` | The scheduling capability. |
| `pending_notification` | `PendingNotifications \| bool` | `True` | The listing capability. |
| `cancel_notification` | `CancelNotification \| bool` | `True` | The cancellation capability. |

---

## Capability parameters

All three param models add **no fields of their own** — they carry the inherited `expose` and
`instructions`, and all three default to both channels.

| Param model | `expose` default | Callable |
|---|---|---|
| `RegisterNotification` | `{TOOL_CALL, COMMAND}` | `register_notification(content: str, delay_seconds: int) -> str` |
| `PendingNotifications` | `{TOOL_CALL, COMMAND}` | `pending_notification(all: bool = False) -> str` |
| `CancelNotification` | `{TOOL_CALL, COMMAND}` | `cancel_notification(notification_id: int) -> str` |

`PendingNotifications` is plural because the singular name is already taken by
`PendingNotification`, the persisted state model this capability lists.

### `register_notification(content, delay_seconds)`

Schedules `content` for delivery in `delay_seconds`. Returns a confirmation carrying the
notification id, which is what `cancel_notification` takes. `delay_seconds` must be between 1 and
`max_delay_seconds`; anything outside raises `RetriableError`, so an over-long delay reaches the
model as a correctable mistake rather than a failure.

### `pending_notification(all=False)`

One line per entry, with the time remaining on each:

```
Pending notifications:
- id 3: check the CI run (in 74 seconds)
- @Analyst412 id 5: re-read the brief (in 210 seconds)   ← only with all=True
```

`all=True` lists **every team member's** entries, each line prefixed with `@owner`; the owner name
comes off the entry's stored address, so it survives a resume and an owner fired before its
notification came due.

`all` widens what you can *see*, never what you can *cancel*.

### `cancel_notification(notification_id)`

Cancels one of **your own** pending entries. The captured owner is always passed, so cancelling
another agent's id — including one just read through `all=True` — fails exactly as cancelling an
unknown id does: `RetriableError`, and the entry stays pending for its real owner.

---

## The `message_class` contract

The path must resolve to a `Message` subclass that

1. is importable,
2. declares `content` and `type` model fields, and
3. **accepts `type="notification"`** — the value delivery writes.

All four failure modes raise `ValueError` at `observer()` bind time, never when a notification
comes due. The third check is a real probe: the resolver validates the class against
`{"content": "notification probe", "type": "notification"}`, so a `type` narrowed to a `Literal`,
an enum, a pattern or a validator that excludes `"notification"` is caught at wiring time rather
than producing one log line per fire. The probe content is non-empty on purpose, so a class that
merely forbids empty `content` is not rejected over a payload it never sees.

Naming the class by string rather than importing it is what keeps this package free of any
dependency on the package that owns it: the deployment picks the delivery class, and the card
resolves it at wiring time. The same string is stored in `NotificationConfig` and re-resolved by
the actor on start, so a resumed team delivers the same class.

---

## Delivery semantics

**The actor sends as itself.** The delivered `sender` is `#NotificationTool` and the message's
`type` is `"notification"`. The send is a tell, so a busy agent never blocks the actor.

**Granularity is ±1 s.** The actor scans for due entries once per `TICK_INTERVAL_S` (1.0 s).

**Delivery waits for an absent owner, briefly.** An owner off the team — an agent between hire and
start, or a resumed team whose agents have not re-registered — is retried for up to
`DELIVERY_GRACE_S` (300 s) and delivered as soon as it is back. Past that window, and for a send
that fails while the owner *is* on the team, the entry is logged and dropped rather than retried.

**Stop and resume need no re-arm.** A pending entry stores an absolute `fire_at`, not a remaining
delay. An entry whose delay expired while the team was down is simply due on the first tick after
the resume, and is delivered on the first tick at which its owner is back.

**The pending set self-drains.** Every entry is capped by `max_delay_seconds` and removed on fire
or on cancel, so `NotificationState.pending` needs no capacity bound.

---

## Configuration

### The `/`-command surface

The `COMMAND` channel is the human command surface, so with the shipped defaults:

```
/register_notification "check CI" 120     schedules an entry from the command line
/pending_notification                     lists that agent's own entries
/pending_notification all=true            lists the whole team's
/cancel_notification 3                    cancels entry 3
```

### Recipes

```python
from akgentic.tool import NotificationTool
from akgentic.tool.core import COMMAND, TOOL_CALL
from akgentic.tool.notification import CancelNotification, RegisterNotification

NotificationTool()                              # AgentMessage delivery, 300 s cap

NotificationTool(max_delay_seconds=60)          # tighter cap

NotificationTool(message_class="acme_core.messages.ReminderMessage")

NotificationTool(
    register_notification=RegisterNotification(
        expose={COMMAND},                       # a human schedules; the LLM cannot
        instructions="Only for CI checks.",     # appended to the tool description
    ),
    cancel_notification=CancelNotification(expose={TOOL_CALL}),
    pending_notification=False,                 # capability removed from both channels
)
```

### Failure modes worth knowing

- `observer()` raises `ValueError` for an unusable `message_class`, and for an observer with no
  orchestrator.
- Using a capability before `observer()` ran raises `RuntimeError` — a wiring bug, deliberately
  not retriable.
- Once the owning agent stops, every capability raises
  `RetriableError("Notifications are unavailable; the agent is shutting down.")`. The observer is
  held weakly, so the tool never keeps a stopped agent alive.

### Import paths

```python
from akgentic.tool import NotificationTool
from akgentic.tool.notification import (
    RegisterNotification, PendingNotifications, CancelNotification,
    NotificationActor, NotificationConfig, NotificationState, PendingNotification,
    DEFAULT_MESSAGE_CLASS, NOTIFICATION_ACTOR_NAME, TICK_INTERVAL_S, resolve_message_class,
)
```

---

See the [package README](../../../../README.md) for the `ToolCard` / `ToolFactory` machinery, the
channel system, and the tool-actor conventions.
