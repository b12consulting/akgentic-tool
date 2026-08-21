# SkillTool

A library of domain guidance — a refund procedure, an escalation policy, a report playbook — split
by **size and volatility**. The menu (one line per skill) is small and immutable, so it goes into
the frozen system prefix; the bodies are large and optional, so they arrive at the tail on demand as
ordinary tool returns. Until this card the only home for such guidance was an agent's backstory,
where every agent paid for every playbook on every turn.

```python
from akgentic.tool import SkillTool
```

| | |
|---|---|
| Module | `akgentic.tool.skill.tool` |
| Actor | none — this card owns no actor and creates none |
| Channels used | `SYSTEM_PROMPT`, `TOOL_CALL`, `COMMAND` |
| Optional extras | none |

---

## The ToolCard

```python
class SkillTool(ToolCard):
    skills: Skills | bool = True
```

**One field, one capability, and no private attribute of its own.** The inherited weak observer is
the only one there is. That is the design rather than an omission: what the model has been given is
what sits in its message history, which the runtime owns and event-sources already, so a card
tracking a parallel "loaded set" could only ever drift from it.

`True` — the default — means *enabled with defaults*, and the default library is empty: the card
contributes an empty menu and a `use_skill` that knows no names. That is harmless, and it is the
shape an operator sees before configuring anything. A `Skills` instance supplies the entries and may
narrow the channels; `False` removes the capability entirely, contributing nothing on any of the
three channels and raising nothing.

**Nothing here renders a template.** Skill bodies are operator-written catalog data and travel
through serialization, so a body containing a brace must never reach `str.format` — it would raise,
or worse, resolve an attribute. The card concatenates, and that is all.

---

## ToolCard fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `skills` | `Skills \| bool` | `True` | The one capability. `True` enables it with an empty library — an empty menu and a `use_skill` that knows no names. A `Skills` instance supplies the entries and narrows the channels. `False` removes the capability from all three channels. |

## Capability parameter — `Skills`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `skills` | `list[SkillEntry]` | `[]` | The entries, in the order they appear in the menu — declaration order, never sorted. Names are unique within one library: a duplicate raises at **model validation**, not at call time, because the second entry could never be reached by `use_skill`. |
| `expose` | `set[Channels]` | `{SYSTEM_PROMPT, TOOL_CALL, COMMAND}` | **All three channels** — the only capability in this package that ships on every one of them. Each carries what it is good at: the menu in the cached prefix, the body on demand, and the same menu behind a command a human can call. |
| `instructions` | `str \| None` | `None` | Inherited from `BaseToolParam`. Appended to the docstring of whichever callable the channel produces — `use_skill()` on the tool channel, `skills()` on the command channel. |

## `SkillEntry`

| Field | Type | Meaning | What it costs |
|---|---|---|---|
| `name` | `str`, min length 1 | The handle the model passes to `use_skill`, e.g. `"refund-policy"`. | Paid **every turn** — it is one half of a menu line. |
| `description` | `str`, min length 1 | The one line that goes in the menu. It is the only thing the model sees until it asks. | Paid **every turn** — the other half of the menu line. |
| `content` | `str`, min length 1 | The body, delivered on demand. It never reaches the prefix. | Paid **only when a conversation asks for it**, once per call, at the tail. |

All three reject the empty string at construction, through `Field(min_length=1)`. An entry that says
nothing is a misconfiguration rather than an empty skill, and catching it at construction is what
keeps the menu meaningful.

The asymmetry in the cost column is the whole point of the card: `name` and `description` are the
recurring cost and `content` is not, which is why the menu is **O(skills)** and never O(content).
That holds by construction rather than by care — the renderer reads `name` and `description` and
never touches `content`.

---

## The three channels

### `SYSTEM_PROMPT` — the menu

A header line, then one `name — description` line per entry, in declaration order:

```
**Skills available to you** — call use_skill(name) to load one.
refund-policy — How refunds are approved, and the thresholds that need a second signature.
escalation — When to escalate to a human, and what the handover must contain.
```

An **empty library contributes the empty string**, not a bare header: a header alone would advertise
nothing.

### `TOOL_CALL` — `use_skill(name)`

Returns `"{name}\n\n{content}"` — **the body itself, as the tool result, in the same turn.** Not an
acknowledgement, not a promise of a later delivery. The model called it because it needs the body for
the answer it is composing right now, so anything arriving on the next turn arrives after the answer
it was needed for.

Re-calling it on the same name simply returns the body again. There is no "already loaded" case to
special-case, because there is no record of one — and after a fold, the repeat call is exactly the
intended recovery rather than a redundancy to suppress.

### `COMMAND` — `skills()`

Registers `/skills`, which renders **the same menu, from the same renderer** the system prompt uses.
That is what makes it an honest answer to *what was this agent actually given?* rather than a second
rendering that might differ. It takes no arguments and marks nothing as loaded — there is nothing to
mark.

## The loading model

**One rule, and one recovery.** A loaded body lives in the conversation until a compaction or the
sliding window removes it; after that the model re-calls `use_skill` for whatever it still needs. The
menu is in the frozen prefix and survives that, which is why re-calling always works.

**A restart is not a special case.** Bodies replay with the message history, and if a later fold
drops them the same reload applies. There are not two horizons here, only one.

That is also why the menu's placement is **load-bearing rather than merely economical**: it is the
whole recovery path. A design that put the menu at the tail would be one compaction away from an
agent that no longer knows its skills exist.

---

## Configuration

### The `/`-command surface

`COMMAND` is exposed by default, so a configured card registers:

```
/skills                 prints the menu exactly as the agents received it
```

Its name is the closure's `__name__`, which `ToolFactory.get_command_registry()` uses as the
canonical command name.

### Recipes

```python
from akgentic.tool.core import COMMAND, SYSTEM_PROMPT, TOOL_CALL
from akgentic.tool.skill import SkillEntry, Skills, SkillTool

REFUND = SkillEntry(
    name="refund-policy",
    description="How refunds are approved, and the thresholds that need a second signature.",
    content="Refunds under 100 EUR are approved by the agent handling the case. …",
)
ESCALATION = SkillEntry(
    name="escalation",
    description="When to escalate to a human, and what the handover must contain.",
    content="Escalate whenever the customer asks for a person, or after two failed fixes. …",
)

# The normal case: all three channels.
SkillTool(skills=Skills(skills=[REFUND, ESCALATION]))

# Prompt + tool, no human command surface.
SkillTool(skills=Skills(skills=[REFUND, ESCALATION], expose={SYSTEM_PROMPT, TOOL_CALL}))

# Command only — a human can list the library, the agents are told nothing and can load nothing.
SkillTool(skills=Skills(skills=[REFUND, ESCALATION], expose={COMMAND}))

# Extra guidance, appended to the use_skill() docstring on the tool channel.
SkillTool(skills=Skills(
    skills=[REFUND, ESCALATION],
    expose={SYSTEM_PROMPT, TOOL_CALL},
    instructions="Load the relevant skill before answering a policy question.",
))

SkillTool()                    # enabled with defaults: an empty library on all three channels
SkillTool(skills=False)        # capability removed from all three channels
```

### Failure modes worth knowing

- **An unknown name raises `RetriableError` listing the names that do exist**, so the model corrects
  itself in-loop rather than failing the turn. Against an **empty** library it still raises, and says
  so explicitly (`none — no skills are configured`) rather than trailing off after "Available
  skills:".
- **A duplicate name raises at model validation, not at call time.** The message names the offending
  handle. The second entry could never be reached by `use_skill`, so failing at construction is the
  only place the mistake is visible.
- **An empty library is legal and silent.** It contributes an empty menu and a `use_skill` that knows
  no names — the shape a card has before anyone configures it, and therefore not something to raise
  on.
- **A `description` that undersells its skill means the skill is never used, and nothing surfaces
  that.** The menu line is the only thing the model sees until it asks, so description quality is an
  operator concern with no automatic feedback.
- **`expose={SYSTEM_PROMPT}` alone advertises a tool that is not bound.** The header pins the words
  "call `use_skill(name)` to load one", so a prompt-only card tells the model to call something no
  channel provides. The prompt channel presumes the tool channel; expose them together unless you
  specifically want a library the model can read about and not open.
- **A `description` is validated non-empty but not single-line.** The menu's shape is one line per
  entry, so a newline inside a description — or a ` — ` inside a name — makes the model read the
  overflow as another skill.

### Import paths

```python
from akgentic.tool import SkillTool
from akgentic.tool.skill import SkillEntry, Skills, SkillTool
```

Only `SkillTool` is re-exported from the package root — that is the import a card author writes.
Configuring a library takes one more import, from `akgentic.tool.skill`, for `SkillEntry` and
`Skills`. That is the same split `MetadataTool` and `NotificationTool` follow.

---

See the [package README](../../../../README.md) for the `ToolCard` / `ToolFactory` machinery, the
channel system, and the tool-actor conventions.
