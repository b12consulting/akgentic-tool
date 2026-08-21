# MetadataTool

Renders the team's **business context** — the model the deployment wrote with
`Orchestrator.set_metadata()` — into every agent's system prompt, from one operator-written
template. The orchestrator already holds that model; until this card nothing surfaced it to the
LLM, so deployments copied the same facts into every role's backstory, where they duplicated and
drifted.

```python
from akgentic.tool import MetadataTool
```

| | |
|---|---|
| Module | `akgentic.tool.metadata.tool` |
| Actor | none — this card owns no actor and creates none |
| Channels used | `SYSTEM_PROMPT`, `COMMAND` |
| Optional extras | none |

---

## The ToolCard

```python
class MetadataTool(ToolCard):
    render_metadata: RenderMetadata | bool = False

    _orchestrator_proxy: Orchestrator | None = PrivateAttr(default=None)
    _rendered: str | None = PrivateAttr(default=None)
    _names: list[str] = PrivateAttr(default_factory=list)
```

**One field, one capability — and the one capability in this package that ships off.** `False`, the
default, removes the capability entirely, so a card nobody has configured contributes nothing and
raises nothing. A `RenderMetadata` instance turns it on: it supplies the template and narrows the
channels. `True` stays legal and still means *enabled with defaults*, but since there is no default
template it fails at `observer()`, which is where a missing template should be noticed rather than
at the first turn.

**Nothing runtime is a Pydantic field.** The orchestrator proxy is not serializable, and neither is
the rendered block: a card restored from configuration renders afresh against *its own* team. All
three runtime handles — the proxy, the block and the parsed placeholder names — live in
`PrivateAttr`, so `model_dump()` carries the configured capability and nothing else.

**Loud at wiring, degrading at render.** A malformed or unresolvable placeholder raises in
`observer()`, next to the mistake. Nothing on the render path raises: a prompt callable that throws
kills the turn far from its cause, so a block that cannot be produced contributes `""` and says so
in the log.

---

## ToolCard fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `render_metadata` | `RenderMetadata \| bool` | `False` | The one capability, and the package's one exception to its default-on convention: every other capability here is meaningful unconfigured — a roster renders itself, a graph summarises itself — whereas this card's whole content is an operator-written `template` no framework could supply. So it **contributes nothing until a `RenderMetadata` gives it one**, and an unconfigured card raises nothing. `True` enables it with defaults and therefore **fails at wiring**, there being no default template. |

## Capability parameter — `RenderMetadata`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `template` | `str`, min length 1 | *required* | The text to render, carrying `{field}` placeholders naming fields of the team's metadata model. Bare identifiers only — see [the grammar](#the-template-grammar). **The result is a snapshot:** it is rendered once, at the first render that *succeeds*, and a later `set_metadata` is not reflected. A deployment whose business context genuinely changes mid-life does not want this capability. |
| `header` | `str \| None` | `None` | An optional header line, rendered bold on its own line above the text. An empty string renders as *no header* rather than as an empty bold line. |
| `instructions` | `str \| None` | `None` | Inherited from `BaseToolParam`. Appended to the `team_metadata()` docstring. |
| `expose` | `set[Channels]` | `{SYSTEM_PROMPT, COMMAND}` | Inherited, with a widened default: both non-LLM channels — the prompt the agents read, and the command a human can call to see exactly what they were given. |

A template that says nothing is a misconfiguration rather than an empty block, so the empty string
is rejected by Pydantic at construction, before `observer()` is ever reached.

---

## The template grammar

A placeholder is a **bare field name** and nothing else:

| Form | Example | Accepted |
|---|---|---|
| Bare identifier | `{fiscal_year}` | yes |
| Escaped braces | `{{not a placeholder}}` | yes — literal text, never a placeholder |
| No placeholder at all | `This team works on nothing in particular.` | yes |
| Dotted path | `{account.name}` | no |
| Index | `{items[0]}` | no |
| Conversion | `{fiscal_year!r}` | no |
| Format spec | `{fiscal_year:>10}` | no |
| Auto-numbered | `{}` | no |
| Positional | `{0}` | no |

Two reasons, not one. `str.format` on an operator-supplied template is an **attribute walk** —
`{a.__class__}` is legal Python formatting — and a name that cannot be checked against the metadata
model's `model_fields` is a name that must not be accepted. The narrow grammar removes the first
surface and is what makes the second check possible.

Every rejected form above fails one rule: `str.isidentifier()` is false for `account.name`,
`items[0]`, `""` (what an auto-numbered `{}` parses to) and `"0"` alike. Conversions and format
specs are rejected separately, because the name in front of them *is* a valid identifier.

A name used twice is collected once and substituted twice. Values are rendered with `str()`, so a
non-string field needs no special handling.

## Two validation points

**Grammar is checked unconditionally at `observer()`.** It depends on the template alone, so there
is never a reason to defer it.

**Names are checked at `observer()` only when the team already holds metadata.** `set_metadata` may
legitimately run after the agents start, so when `get_metadata()` returns `None` at wiring time the
name check moves to the first render — where it can no longer raise, and surfaces as a log line
instead.

Both raise `ValueError` when they fail at wiring. The name check's message names the offending
placeholder, the metadata model, and the fields that model *does* declare, so the correction is
visible without opening the deployment's own code.

## Degradation at render

The render path never raises:

| At render | Contributes | Log |
|---|---|---|
| metadata present, all names resolve | the rendered block | — |
| `get_metadata()` returns `None` | `""` | WARNING naming the card |
| metadata present, a name is missing | `""` | ERROR naming the placeholder and the model type |
| anything else fails | `""` | `logger.exception` — the traceback is kept |

The second and third are expected and self-describing, so a message-only log carries everything
worth knowing. The fourth is a genuine surprise, and a prompt callable is the one place a traceback
cannot be recovered afterwards — so it keeps its stack.

**Only a successful render is cached.** A degraded render caches nothing and is retried on the next
turn, which is what makes the deferred name check worth having: a `set_metadata` landing moments
after start-up still produces its block, rather than freezing the agent at `""` for the session.

---

## Configuration

### The `/`-command surface

`COMMAND` is the human command surface, and `RenderMetadata` exposes it by default — so a card that
has been given a template registers:

```
/team_metadata          prints the block exactly as the agents received it
```

The command takes no arguments and reads the same snapshot the prompt does, which is what makes it
an honest answer to *what was this agent actually told?* rather than a second rendering that might
differ. Its name is the closure's `__name__`, which `ToolFactory.get_command_registry()` uses as
the canonical command name.

### Recipes

```python
from akgentic.tool.core import COMMAND, SYSTEM_PROMPT
from akgentic.tool.metadata import MetadataTool, RenderMetadata

# The normal case: metadata written before the team starts.
MetadataTool(render_metadata=RenderMetadata(
    header="Team context",
    template="Fiscal year: {fiscal_year}. Engagement: {engagement}. Region: {region}.",
))

# Metadata written after the agents start: names cannot be checked at wiring, so they
# are checked at the first render, and the block appears on the turn after set_metadata.
MetadataTool(render_metadata=RenderMetadata(template="Engagement: {engagement}."))

# Command only — a human can ask, the prompt carries nothing.
MetadataTool(render_metadata=RenderMetadata(
    template="Engagement: {engagement}.",
    expose={COMMAND},
))

# Prompt only — the agents get the block, no command surface is registered.
MetadataTool(render_metadata=RenderMetadata(
    template="Engagement: {engagement}.",
    expose={SYSTEM_PROMPT},
))

# Extra guidance, appended to the team_metadata() docstring — which only the COMMAND
# channel carries, so keep that channel exposed for it to be visible anywhere.
MetadataTool(render_metadata=RenderMetadata(
    template="Engagement: {engagement}.",
    instructions="Quote these facts verbatim; never infer beyond them.",
))

MetadataTool(render_metadata=False)          # capability removed from both channels
```

### Failure modes worth knowing

- **Opting in with `render_metadata=True` raises at wiring.** `True` means *defaults*, and there is
  no default template — so it is an explicit mistake rather than a shipped one; the card is turned
  on by handing it a `RenderMetadata`, never by a bare `True`.
- **An observer with no orchestrator raises `ValueError` at wiring** — there is nothing to read the
  metadata from.
- **A card that was never wired degrades rather than raising.** The render path catches its own
  "`observer()` must run first" error into the same `""` path, with an ERROR in the log.
- **A dotted path is rejected, and the fix is not to nest.** Flatten the value onto your own
  metadata model — a computed field or a plain string field beside the others — and name it as a
  bare identifier. The card reads `model_fields` and `getattr` and knows nothing else about the
  shape.
- **A disabled card validates nothing.** With `render_metadata=False` there is no template to
  check, so a broken one cannot be reached and no proxy is bound.
- **Re-wiring drops the previous team's block.** `ToolFactory` attaches an observer to every card
  it is handed, so `observer()` clears both the proxy and the snapshot on every binding; a card
  bound a second time renders afresh for the second team.

### Import paths

```python
from akgentic.tool import MetadataTool
from akgentic.tool.metadata import MetadataTool, RenderMetadata
```

Only `MetadataTool` is re-exported from the package root — that is the import a card author writes.
Configuring the template takes one more import, from `akgentic.tool.metadata`.

---

See the [package README](../../../../README.md) for the `ToolCard` / `ToolFactory` machinery, the
channel system, and the tool-actor conventions.
