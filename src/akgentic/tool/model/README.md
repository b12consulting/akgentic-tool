# ModelTool

Runtime model switching as a capability. An agent that can read its own roster and move to another
entry in it can escalate itself when a task turns out to be harder than the one it was configured
for, and a human can move it back with one command. This card gives that three surfaces — the
listing and the switch on `TOOL_CALL` and `COMMAND`, and the model in force published as
`LLM_CONTEXT` state — and it creates no actor and performs no proxy round trip.

```python
from akgentic.tool import ModelTool
```

| | |
|---|---|
| Module | `akgentic.tool.model` |
| Actor | none — this card owns no actor and creates none |
| Channels used | `TOOL_CALL`, `COMMAND`, `LLM_CONTEXT` |
| Optional extras | none |
| Auto-injected | **no** — see [Not auto-injected](#not-auto-injected) |

---

## The ToolCard

```python
class ModelTool(ToolCard):
    list_models: ListModels | bool = True
    switch_model: SwitchModel | bool = True
    active_model: ActiveModel | bool = True
```

Three capabilities, three params, all following the same `Param | bool` convention: `True` (the
default) enables the capability with its param's defaults, a param instance may narrow the channels,
and `False` removes **exactly that capability** and nothing else. Beyond `expose`, a param carries
only `instructions: str | None = None`, inherited from `BaseToolParam` and appended under a
structured header to that capability's docstring — on the tool form and the command form alike. Both
fields are configuration read at factory bind time, never tool-call schema.

**`ActiveModel(instructions=…)` is accepted and has no effect.** `instructions` appends to a
*docstring*, and an `LLM_CONTEXT` provider publishes a `ContextState`, not a docstring — there is
nothing for it to append to. It constructs, round-trips and persists with no error and no outcome.
This is the package's standing shape for `LLM_CONTEXT`-only capabilities rather than a quirk of this
card (`TeamTool`'s roster and role-catalog params take it the same way), so it is documented rather
than rejected. Put guidance for the model in the agent's own instructions instead.

Two of the three share their channels, which is what distinguishes this card's shape from
`MailboxTool`'s one-capability-per-channel layout: `list_models` and `switch_model` are each served
on both `TOOL_CALL` and `COMMAND`, because the same act is meaningful pulled by the model and pushed
by a human.

## ToolCard fields

| Field | Type | Default | Channel default | Meaning |
|---|---|---|---|---|
| `list_models` | `ListModels \| bool` | `True` | `{TOOL_CALL, COMMAND}` | The roster listing — what this agent may switch within. |
| `switch_model` | `SwitchModel \| bool` | `True` | `{TOOL_CALL, COMMAND}` | The switch itself, and the only writer of `ToolState.active_model`. |
| `active_model` | `ActiveModel \| bool` | `True` | `{LLM_CONTEXT}` | The `active_model_state` provider publishing the model in force. Its inherited `instructions` is a silent no-op — see above. |

Every capability is gated twice: the param must resolve, **and** its `expose` set must contain the
channel the serving hook covers (`get_tools()` for `TOOL_CALL`, `get_commands()` for `COMMAND`,
`get_context_states()` for `LLM_CONTEXT`). A capability exposed on a channel this card does not
serve is dropped silently — the package-wide rule, not a model-switch quirk.

---

## The three channels

### `TOOL_CALL` and `COMMAND` — `list_models()`

```python
def list_models() -> str: ...
```

Zero arguments. Renders one line per roster entry, **in the roster's own order** — the order is the
roster's decision, and re-sorting here would make the listing disagree with every other view of the
same configuration. The line grammar is fixed:

```
{key} (context: {context_length|undeclared})
{key} (context: {context_length|undeclared}) [ACTIVE]
```

The second form is the entry currently in force: the marker is a space followed by a literal
`[ACTIVE]`, appended to the same line and nothing else. `undeclared` stands in for an entry whose
`context_length` is `None`. Nothing turn-varying enters the text, so two calls against an unchanged
roster are byte-identical.

**An empty roster is not an error.** It renders a fixed sentinel —
`"This agent has no model roster, so there is nothing to switch within."` — rather than an empty
string or a raise, so a model that called the listing gets an answer it can act on.

### `TOOL_CALL` and `COMMAND` — `switch_model(model)`

```python
def switch_model(model: str) -> str: ...
```

One argument, the roster key exactly as `list_models` prints it. The call asks the observer to make
that entry the model in force, and — **only if that succeeds** — records the key in
`ToolState.active_model`, where a restart finds it.

#### The parameter is `model` on this side and `key` on the observer's

This is deliberate and it is visible from the command line. The card's callable is
`switch_model(model: str) -> str`; `ModelSwitchToolObserver.switch_model` takes `key`. Two
contracts, two names. The command registry derives its schema from the **callable's** signature, so:

```
/switch_model openai:gpt-5.2              # binds — positional
/switch_model model=openai:gpt-5.2        # binds — keyword, the advertised name
/switch_model key=openai:gpt-5.2          # binds the WHOLE token positionally, then fails downstream
```

The positional form works because the colon survives `shlex` and the split-on-first-`=` keyword
rule, so `openai:gpt-5.2` arrives as one token.

**The third form is not caught by the registry, which is the part worth knowing.** A token counts as
a keyword only when the text before its first `=` is a known parameter name — deliberately, so a
value containing `=` is never silently swallowed. `key` is the observer's name, not the card's, so
it is not a parameter here: the whole token `key=openai:gpt-5.2` is classified **positional** and
binds to `model`, reaching the observer verbatim and being rejected downstream as an unknown roster
key. Don't write it — but if you are writing a frontend or a script, expect the refusal to come back
from the roster rather than as an unknown-keyword error, and read the parameter name off the
announced `CommandDescriptor` rather than off the observer protocol.

### `LLM_CONTEXT` — the `active_model_state` provider

`get_context_states()` yields one zero-argument provider whose `__name__` is `active_model_state`.
It returns an `ActiveModelState`, or `None` when the state is unavailable. It **never raises**
(ADR-037 §3): a failure is logged and becomes `None`.

**It renders the roster's `active` flag, never `ToolState.active_model`.** This is the one thing
about the card that surprises people, and it is a correctness rule rather than an implementation
detail: the slot is a persisted *preference* that the agent layer re-applies on restore, so
rendering it would show a stale key as though it were the model actually answering. The provider
walks `observer.list_model_rows()` and publishes the row marked `active`. When no row is marked —
including the empty-roster case — it publishes nothing.

The provider name is **load-bearing twice**: `ToolFactory` aggregates providers under the callable's
`__name__`, and it is the key the baseline is persisted under in `ToolState.context_baselines`. Two
providers sharing a `__name__` raise `ValueError` at team-creation time.

---

## The domain vocabulary

### `ModelRow` — a projection, not a record

```python
class ModelRow(SerializableBaseModel):
    key: str
    provider: str
    model: str
    active: bool
    context_length: int | None
```

| Field | Type | Meaning |
|---|---|---|
| `key` | `str` | The roster key, `f"{provider}:{model}"` — what `switch_model` takes and what `ToolState.active_model` stores. |
| `provider` | `str` | The provider the entry resolves through. |
| `model` | `str` | The provider-side model name. |
| `active` | `bool` | Whether this entry is the one currently in force. |
| `context_length` | `int \| None` | The entry's declared context window, or `None` when it declares none. |

**A row is rebuilt from the roster on every call and never stored.** The only thing this feature
persists is `ToolState.active_model`, a single key.

**`context_length` carries no default, on purpose.** A projection has no legacy payload a default
could protect: every `ModelRow` in existence was built moments ago by the observer, so an omitted
field is a construction bug and should fail loudly rather than silently read as `None`.
`ToolState.active_model` **does** default, because there the default is load-bearing — a payload
persisted before the field existed must still restore. The asymmetry is the rule, not an
inconsistency: **defaults protect stored payloads, and a projection has none.**

`ModelRow` is deliberately **not** a `ContextState`. A row is not diffable state; the `LLM_CONTEXT`
state describing the model in force belongs to the card and is `ActiveModelState`.

### `ActiveModelState` — the key, and nothing else

```python
class ActiveModelState(ContextState):
    key: str

    def render_full(self) -> str: ...
    def render_delta(self, previous: Self) -> str | None: ...
```

| Renderer | Output |
|---|---|
| `render_full()` | `**Active model:** {key}` |
| `render_delta(previous)` | `**Active model changed:** {previous.key} → {key}`, or `None` when the key did not move |

One field is a deliberate narrowing. This state is persisted as a baseline inside
`ToolState.context_baselines`, so every column it carries is paid for on every checkpoint of every
agent; one string keeps that O(1). Nothing is lost: provider, model name and context window are all
derivable from the key by whoever holds the roster — and the roster is `akgentic-llm`'s, a package
this one may not import anyway.

---

## The observer protocol

`ModelSwitchToolObserver` is this tool's own contract, living beside the tool rather than in `core/`
(a domain-specific observer belongs to its domain — `ModelTool` is its only consumer). It is a
**sibling** of `ActorToolObserver`, not a widening of it: every observer that offers no model switch
keeps satisfying the base protocol unchanged, and no existing fake or agent gained a member when
this card shipped.

```python
@runtime_checkable
class ModelSwitchToolObserver(ActorToolObserver, Protocol):
    def list_model_rows(self) -> list[ModelRow]: ...
    def switch_model(self, key: str) -> str: ...
```

`list_model_rows()` projects the roster, one row per entry, rebuilt per call. `switch_model(key)`
makes one entry the model in force and returns a human-readable confirmation.

*Cross-package claim, not verifiable here:* **the implementation belongs in `akgentic-agent`**, the
one package that may import both this one and `akgentic-llm` and can therefore project the roster's
own configuration model onto `ModelRow`. It is **not there yet** — no `list_model_rows` or
`ModelSwitchToolObserver` implementation ships in `akgentic-agent` as of this epic, so the card is
inert until a later epic lands the agent-side observer. Conformance is a documented precondition
rather than a runtime gate: observers are duck-typed, so a non-conforming one fails at first use,
exactly as before.

### Why `ModelRow` exists at all

`akgentic-tool` imports `akgentic-core` **only**. The roster's own configuration model belongs to
`akgentic-llm`, which this package must never name — so the contract could not be written in terms
of it. The card declares its own serializable projection instead, and the package that may import
both does the mapping. That is a general rule for card authors, written up in the
[authoring guide](../../../../README.md#building-a-feature-as-a-card); `ModelRow` is its worked case.

---

## Configuration

```python
from akgentic.tool import ModelTool
from akgentic.tool.core import COMMAND
from akgentic.tool.model import SwitchModel

ModelTool()                                                # all three capabilities on (the default)
ModelTool(switch_model=False)                              # read-only: the roster, no switch
ModelTool(active_model=False)                              # no LLM_CONTEXT block
ModelTool(switch_model=SwitchModel(expose={COMMAND}))      # humans may switch, the model may not
```

The last form is the useful one for a governed deployment: the model can still *see* what it is
running on and read the roster, while moving between entries stays a human act.

### Not auto-injected

`BaseAgent` auto-adds `TeamTool` and `MailboxTool`. It does **not** add this card. Granting every
agent the standing power to change its own model is a cost and governance decision that belongs to
whoever writes the card list, not to a framework default — a self-escalating agent is a bill, and an
agent that quietly moved off the model it was evaluated on is a support case.

*Cross-package claim, not verifiable here:* nothing under `src/` in this package can enforce that.
`BaseAgent` and its default card list are `akgentic-agent`'s, and there is no default-card list in
this package to omit the card from. This is stated as the consumer contract.

### Failure modes worth knowing

- **A refused switch leaves the slot untouched.** The observer's refusal — an unknown key, an entry
  that will not build, a bound the llm layer enforces — surfaces as `RetriableError` from the
  callable, and `ToolState.active_model` is not written. The write is deliberately the **last**
  statement of the call, after the observer returns, so the recorded key is one the llm layer
  actually accepted. A `RetriableError` raised from below is re-raised unchanged; anything else is
  wrapped, because there is no stable exception type to name across a package boundary this one may
  not import.
- **A collected observer degrades, in the channel-appropriate way.** `list_models` and
  `switch_model` are in-life code and raise `ToolObserverGone` once the owning agent has stopped —
  a defined outcome, not a crash. The `LLM_CONTEXT` provider may outlive its agent, so it returns
  `None` instead and never raises.
- **An empty roster is answered, not failed.** The listing returns its sentinel and the context
  provider publishes nothing.
- **A removed field fails silently.** `ToolCard` keeps Pydantic's default `extra="ignore"`, so a
  catalog entry naming a capability that no longer exists constructs happily and drops the value.

---

## Persistence: `ToolState.active_model`

A successful switch writes the roster key into `observer.state.tool_state.active_model`. The agent's
existing state checkpoints persist it — this card emits no event, sends no notification and has no
save path of its own.

**The dereference happens at the write, after the observer call — never hoisted.** The full
`observer.state.tool_state` chain is evaluated at the moment of the write, on every call. Reading it
into a local *before* `observer.switch_model(...)` is a silent defect and not merely a style
preference: `init_state()` replaces the agent's state object wholesale on restore, and a restore can
happen inside that call, so a pre-computed carrier writes into an abandoned object. Nothing raises,
nothing logs, and the preference is simply lost. Two guard tests exist, one per forbidden form: one
replaces the carrier **between** calls (catching a bind-time capture), the other replaces it
**during** the observer call (catching the per-call hoist, which the first one cannot see).

`None` in the slot means the agent expresses no preference and the config's declared active entry
wins — which is also what a payload persisted before the field existed restores to, so the recovery
direction stays harmless and needs no migration step.

*Cross-package claim, not verifiable here:* re-applying a persisted key on restore, and dropping a
key that is no longer in the roster with a log line so the declared entry wins and the restore never
fails, are both the agent/llm layer's (`akgentic-llm`'s ADR-018 §4). Nothing under `src/` in this
package reads the slot back.

*Cross-package claim, not verifiable here:* a switch between heterogeneous models does **not**
sanitize the message history, so switching mid-conversation is best-effort. Characterising that
behaviour belongs to `akgentic-llm`, which owns the run loop; this card makes no guarantee about it.

---

## Import paths

```python
# All seven symbols live here.
from akgentic.tool.model import (
    ActiveModel,
    ActiveModelState,
    ListModels,
    ModelRow,
    ModelSwitchToolObserver,
    ModelTool,
    SwitchModel,
)

# Three of the seven are re-exported from the package root as well.
from akgentic.tool import ModelRow, ModelSwitchToolObserver, ModelTool
```

`akgentic.tool.model` carries the whole surface. The package root re-exports **three** of it:
`ModelTool` — the import a deployment writes — plus `ModelRow` and `ModelSwitchToolObserver`, the
two contract symbols `akgentic-agent` needs to implement the protocol and project the roster. That
is the same placement `TeamManagementToolObserver` has, and it is asserted rather than incidental:
both are named in both `__all__` lists.

The three capability params and `ActiveModelState` are **not** at the root — a
`from akgentic.tool import ListModels` or `ActiveModelState` will not resolve. Reach for those
through `akgentic.tool.model`.

---

See the [package README](../../../../README.md) for the `ToolCard` / `ToolFactory` machinery, the
channel system, and the authoring guide whose projection rule this card is the worked example of.
