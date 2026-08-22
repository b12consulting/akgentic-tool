# akgentic-tool

[![CI](https://github.com/b12consulting/akgentic-tool/actions/workflows/ci.yml/badge.svg)](https://github.com/b12consulting/akgentic-tool/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/gpiroux/c0f2e0aa0a8c184ee8823dd4feefddd5/raw/coverage.json)](https://github.com/b12consulting/akgentic-tool/actions/workflows/ci.yml)

Tool infrastructure and domain tools for the [Akgentic](https://github.com/b12consulting/akgentic-framework)
multi-agent framework (open-source bundle). Define, compose, and expose capabilities to LLM agents through a unified
channel system — as tool calls, system prompt injections, structured context state, or programmatic
commands.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
  - [ToolCard](#toolcard)
  - [ToolFactory](#toolfactory)
  - [BaseToolParam: Configuration, Not Schema](#basetoolparam-configuration-not-schema)
  - [Channel System](#channel-system)
- [Migration: moved import paths](#migration-moved-import-paths)
- [Observers: How a Tool Acts on the System](#observers-how-a-tool-acts-on-the-system)
- [Tool State Events](#tool-state-events)
- [Tool Actors](#tool-actors)
- [Deferred Results: Never Block a Tool Actor](#deferred-results-never-block-a-tool-actor)
- [Building a feature as a card](#building-a-feature-as-a-card)
- [Tool Catalog](#tool-catalog)
  - [WorkspaceTool](#workspacetool)
  - [PlanningTool](#planningtool)
  - [KnowledgeGraphTool](#knowledgegraphtool)
  - [VectorStoreTool](#vectorstoretool)
  - [SearchTool](#searchtool)
  - [TeamTool](#teamtool)
  - [MetadataTool](#metadatatool)
  - [NotificationTool](#notificationtool)
  - [SkillTool](#skilltool)
  - [MailboxTool](#mailboxtool)
  - [MCPTool](#mcptool)
  - [ExecTool](#exectool)
- [Error Handling](#error-handling)
- [Optional Extras](#optional-extras)
- [Development](#development)
- [License](#license)

## Overview

`akgentic-tool` is the capability layer between the Akgentic actor system and the LLM agents
running inside it. It provides:

- **Abstract contracts** — `ToolCard` and `BaseToolParam` define the serializable configuration
  model every tool follows; `ToolFactory` aggregates multiple cards into agent-ready callables
- **Channel system** — each capability declares whether it surfaces as a `TOOL_CALL` (LLM
  invokes it), a `SYSTEM_PROMPT` (rendered once into the frozen system block), an `LLM_CONTEXT`
  (structured context state appended into the context tail per turn, as a delta), or a `COMMAND`
  (programmatic call from another agent or the orchestrator)
- **Observer protocols** — `ToolObserver`, `ActorToolObserver`, and `TeamManagementToolObserver`
  give tools access to the actor system, event emission, and team lifecycle hooks.
  > **Migration note (ADR-018):** `ToolCallEvent` has been removed from `akgentic-tool`. Tool call
  > observability is now handled by `akgentic-llm`. Import `ToolCallEvent` and `ToolReturnEvent`
  > from `akgentic.llm.event` instead.
- **RetriableError** — tools signal recoverable failures; `ToolFactory` translates them to the
  framework-specific retry exception without coupling tool logic to pydantic-ai
- **Domain tools** — twelve production-ready tool implementations covering workspace I/O, task
  planning, knowledge graph, web search, team management, the team's business context, on-demand
  skill guidance, the agent's own mailbox and run-cancellation surface, vector-store configuration,
  MCP server integration, sandboxed shell execution, and self-scheduled notifications

```
ToolCard(s)
    │
    ▼
ToolFactory
    │
    ├── get_tools()           → list[Callable]  ─────▶ LLM ReAct loop
    ├── get_system_prompts()  → list[Callable]  ─────▶ injected into LLM context
    ├── get_context_states()  → list[Callable[[], ContextState | None]] ▶ per-turn context deltas
    ├── get_context_updater() → ContextUpdater  ─────▶ composes one delta block per turn
    ├── get_commands()        → dict[type, Callable] ▶ orchestrator / other agents
    └── get_toolsets()        → list[Any]       ─────▶ pydantic-ai toolset objects
```

`get_toolsets()` is typed `list[Any]` because its elements are runtime pydantic-ai
objects: an `MCPToolset`, or a `PrefixedToolset` wrapping one when the connection
configures a `tool_prefix`.

## Installation

Published on PyPI. Python 3.12 or newer.

```bash
uv add akgentic-tool
# or
pip install akgentic-tool
```

That is the whole install. `akgentic-core`, `pydantic-ai`, `tavily-python` and
`httpx` come with it as ordinary dependencies — no workspace checkout, no
submodules.

### Installing Extras

The base install gives you the `ToolCard` / `ToolFactory` machinery and every
tool's text path. Each extra adds one optional surface — see
[Optional Extras](#optional-extras) below for what degrades without it:

```bash
# Semantic search for planning and knowledge graph (numpy + OpenAI embeddings)
uv add "akgentic-tool[vector_search]"

# Weaviate backend for the vector store (weaviate-client)
uv add "akgentic-tool[weaviate]"

# Binary file reading for workspace_read (PDF, DOCX, XLSX, PPTX via MarkItDown)
uv add "akgentic-tool[docs]"

# Image resizing for workspace_view (Pillow)
uv add "akgentic-tool[vision]"

# Everything
uv add "akgentic-tool[vector_search,weaviate,docs,vision]"
```

### As part of the framework bundle

`akgentic-framework` is the meta-distribution that pins every akgentic package
at versions built and tested together. Install `akgentic-tool` through it when
you want the release-wide pin rather than a single package:

```bash
pip install "akgentic-framework[tool]"   # this package + its closure, release-pinned
pip install "akgentic-framework[all]"    # the whole framework
```

### Working on the package itself

To develop `akgentic-tool` rather than use it, clone the open-source bundle
[akgentic-framework](https://github.com/b12consulting/akgentic-framework), which
carries every package together as submodules:

```bash
git clone git@github.com:b12consulting/akgentic-framework.git
cd akgentic-framework
git submodule update --init
# uncomment the two "SOURCE MODE" blocks in pyproject.toml
uv sync
```

Source mode resolves `akgentic-*` to the local checkouts, editable.

## Quick Start

Attach tools to an agent configuration:

```python
from akgentic.tool import (
    ToolFactory,
    WorkspaceTool,
)
from akgentic.tool.planning import PlanningTool
from akgentic.tool.search import SearchTool

# Build a factory with multiple tools
factory = ToolFactory(
    tool_cards=[
        WorkspaceTool(),          # full read/write workspace access
        PlanningTool(),           # shared team task board
        SearchTool(),             # Tavily web search + fetch
    ],
    observer=agent,               # ActorToolObserver (provided by BaseAgent)
    retry_exception=ModelRetry,   # pydantic-ai retry — injected by BaseAgent
)

# Get callables ready for pydantic-ai agent registration
tools = factory.get_tools()            # LLM-callable functions
prompts = factory.get_system_prompts() # static prompt injections
states = factory.get_context_states()  # structured context state, delivered as deltas
commands = factory.get_commands()      # programmatic calls from orchestrator
```

Grant read-only workspace access to a reviewer agent:

```python
WorkspaceTool(read_only=True)
```

Custom planning configuration with semantic search:

```python
PlanningTool(
    get_planning=GetPlanning(filter_by_agent=False),  # show all tasks, not just own
)
```

## Architecture

The package follows a two-layer design: a **core layer** of abstract contracts and a **domain
layer** of independent tool implementations. Domain submodules never import each other — cross-
tool composition happens at the agent level.

```
┌──────────────────────────────────────────────────────────────────┐
│  Domain Tools                                                    │
│  workspace │ planning │ knowledge_graph │ search │ team          │
│  vector_store │ mcp │ sandbox │ notification                     │
├──────────────────────────────────────────────────────────────────┤
│  Core Layer: ToolCard, BaseToolParam, ToolFactory, Channels      │
│              RetriableError, Observer protocols                   │
├──────────────────────────────────────────────────────────────────┤
│  Vector infrastructure (optional): VectorIndex, EmbeddingService │
├──────────────────────────────────────────────────────────────────┤
│  akgentic-core (Pykka actors, ActorAddress, Orchestrator)        │
└──────────────────────────────────────────────────────────────────┘
```

### ToolCard

`ToolCard` is the base class for all tool configurations. It is a Pydantic model — fully
serializable, round-trippable through `model_dump()` / `model_validate()`.

**Serialization rules (Golden Rule 1b):**
- All fields must use serializable types (primitives, `BaseModel` subclasses, enums, collections)
- `ConfigDict(arbitrary_types_allowed=True)` is **forbidden** on any `ToolCard` subclass
- Runtime state (actor proxies, filesystem handles) goes in `PrivateAttr` — excluded from serialization

```python
class MyTool(ToolCard):
    config_value: str = "default"
    _runtime_handle: Handle | None = PrivateAttr(default=None)  # not serialized

    def observer(self, observer: ActorToolObserver) -> "MyTool":
        self._observer = observer
        self._runtime_handle = setup_handle()
        return self

    def get_tools(self) -> list[Callable]:
        handle = self._runtime_handle

        def my_tool(input: str) -> str:
            """Do something with input."""
            try:
                return handle.process(input)
            except ValueError as e:
                raise RetriableError(f"Invalid input: {e}")

        return [my_tool]
```

#### Context state: the `get_context_states()` hook

A stateful card exposes its volatile prompt content as **structured context state** on the
`LLM_CONTEXT` channel, through `get_context_states()`. The hook returns zero-arg **providers** —
callables returning a `ContextState` snapshot, or `None` when the state is unavailable (collected
observer, stopped actor). Providers **never raise**; the default is `[]`. It sits alongside
`get_system_prompts()`, which stays for **static** text that belongs in the cached prefix.

`ContextState` is a small contract:

- `render_full()` — the whole state, as the model should first see it; `""` when there is nothing
  to say.
- `render_delta(previous)` — what changed since `previous`; `None` when nothing did. The caller
  guarantees `previous` is the **same concrete type**, and a first-seen state is always rendered
  full — `render_delta` is never asked to diff against nothing.

Composition lives **in this package**: `ContextUpdater` reads every provider, diffs each against its
persisted baseline, and returns the block. The agent layer's one remaining job is to **append** what
it gets back (ADR-037 §6-8) — wiring that `akgentic-agent`'s Epic 21 rebuilds against a released
`akgentic-tool` carrying this contract. That is a documented merge-order dependency, not a defect of
this package. See *Persisted baselines and the `ContextUpdater`* below.

**The channel a capability declares is the card author's promise** — `SYSTEM_PROMPT` means
*rendered once, cached prefix*; `LLM_CONTEXT` means *per-turn delta at the tail*. Choose carefully:
a capability exposed on a channel its card cannot serve is **dropped silently** — no error, no
warning. A `GetPlanning` left on `SYSTEM_PROMPT` after the card stopped overriding
`get_system_prompts()` would simply vanish from the prompt.

Two mechanics back the hook:

- **Aggregation** — `ToolFactory.get_context_states()` iterates cards in dependency order. A
  provider's key is its callable `__name__` (the same convention `get_command_registry()` uses),
  and two providers sharing a name raise `ValueError` at build time, naming both owning cards —
  never silent shadowing.
- **Persisted-payload migration** — a card persisted with an explicit
  `expose: ["system_prompt", ...]` before its content moved would resolve to a channel the card no
  longer serves. `normalize_system_prompt_to_llm_context` rewrites that persisted exposure to
  `LLM_CONTEXT`, attached as a `field_validator` **per param class, on moved params only** — never
  globally, which would silently drag static content out of the cached prefix.

#### Persisted baselines and the `ContextUpdater`

The per-provider baselines are persisted, so a restored agent resumes delta delivery instead of
re-sending a full snapshot on its first turn. They live in `ToolState` — `context_baselines`
(`dict[str, ContextState]`, keyed by provider `__name__`) and `context_update_seq` (`int`) — carried
by the agent's own state object and written out by the agent's existing state checkpoints. This
package ships the slot, the engine and the contract; the state field that carries them is
`akgentic-agent`'s half of the same decision, on the merge-order dependency noted above.

**They are a cache, never a record.** The durable record of what the model was told is the message
history; the baselines only spare the engine from repeating it. This package therefore has **no save
path** — no event, no retry, no write-ahead anything — because it needs none.

**The engine.** `ToolFactory.get_context_updater()` builds a `ContextUpdater` from the factory's
observer and the same providers `get_context_states()` yields — so a duplicate provider `__name__`
fails through this path too, and a factory whose observer is missing or is not an
`ActorToolObserver` raises `ValueError` at team-creation time rather than going silently inert. Its
surface is two methods:

- `compose_update(messages)` returns at most **one** `**Context update N**` block per turn, or
  `None` when nothing changed. It **never appends** — the append stays with the agent layer.
- `reset()` zeroes both fields. That is the `/clear` path, and the only legitimate zeroing of the
  counter: it matches a history whose markers were wiped along with it.

**The trust gate.** Baselines are honoured only while the last marker they claim to have delivered
is still visible in the history. If the persisted counter has fallen behind the markers, the
baselines are kept and the next delta simply re-states what the missed blocks said — a repeat, never
an omission. If the last delivered marker is no longer visible at all (compaction, trimming, an
out-of-band wipe), the baselines are dropped and the next block is a full snapshot. A wrong, stale
or missing save therefore degrades to a wider delta or a full snapshot, never to a lost update.

> **The trap: never cache a `ToolState` reference.** Read it through the observer —
> `observer.state.tool_state` — on **every** call. The agent replaces its state object wholesale on
> restore, so a `ToolState`, a carrier, or a strong observer reference stored anywhere outside the
> agent's state goes silently stale. It is the first thing a review of this area looks for.

Both types are importable from the package root:

```python
from akgentic.tool import ContextUpdater, ToolState
```

### ToolFactory

Aggregates multiple `ToolCard` instances into flat lists. When `retry_exception` is set, wraps
every tool callable with a converter that catches `RetriableError` and re-raises it as the
framework-specific exception (e.g., pydantic-ai's `ModelRetry`).

```python
ToolFactory(
    tool_cards=[tool_a, tool_b],
    observer=agent,
    retry_exception=ModelRetry,
)
```

Beside the five aggregators it also builds one runtime object: `get_context_updater()` returns a
`ContextUpdater` over this factory's observer and its context-state providers — see *Persisted
baselines and the `ContextUpdater`* above.

### BaseToolParam: Configuration, Not Schema

Every custom field on a `BaseToolParam` subclass must be read at factory bind time and influence
the tool's runtime behavior — as a closure variable, function default, or observer setup value.
Fields that merely mirror the LLM-facing function signature are dead code: they look configurable
but have no effect.

**Rule:** if a developer writes `MyParam(field=value)`, that value must influence runtime behavior.
LLM-facing parameters belong exclusively on the factory-produced function signature.

```python
# CORRECT — field captured at bind time, controls behavior
class GetPlanning(BaseToolParam):
    filter_by_agent: bool = True  # read by factory, stored in closure

# CORRECT — no custom fields, search params live on the function signature
class SearchGraph(BaseToolParam):
    expose: set[Channels] = {TOOL_CALL, COMMAND}

# WRONG — fields duplicate function signature but are never read
class BadParam(BaseToolParam):
    status: str | None = None  # never consumed by factory
```

### Channel System

The `Channels` enum (`TOOL_CALL`, `SYSTEM_PROMPT`, `LLM_CONTEXT`, `COMMAND`) controls how a
capability is surfaced. Each `BaseToolParam` subclass declares its `expose` set. A single
capability can appear on multiple channels simultaneously.

| Channel | Consumer | Invocation |
|---|---|---|
| `TOOL_CALL` | LLM agent | Called by the LLM during the ReAct loop |
| `SYSTEM_PROMPT` | LLM context | Rendered once into the frozen system block |
| `LLM_CONTEXT` | LLM context | Structured context state appended into the context tail per turn, as a delta |
| `COMMAND` | Orchestrator / agents / humans | Called programmatically, or by `/`-prefixed text through `CommandRegistry.dispatch` |

The two prompt-bearing channels carry different promises, and picking between them is the card
author's contract with the runtime:

- **`SYSTEM_PROMPT`** — rendered **once** into the frozen system block, never re-rendered, part of
  the cached prefix. Static content only.
- **`LLM_CONTEXT`** — structured context state appended into the context tail **per turn, as a
  delta**. The channel for volatile tool state that must not invalidate the cached prefix. That
  per-turn delta is composed by the `ContextUpdater` against the baselines persisted in `ToolState`
  on the agent's state.

```python
class GetPlanning(BaseToolParam):
    expose: set[Channels] = {LLM_CONTEXT, COMMAND}    # context state, never a direct LLM call

class GetPlanningTask(BaseToolParam):
    expose: set[Channels] = {TOOL_CALL, COMMAND}      # LLM + programmatic
```

`BaseToolParam.instructions` appends runtime guidance to a tool's docstring without modifying
source — useful for injecting team-specific constraints at configuration time.

## Migration: moved import paths

Two modules were reorganised: `akgentic.tool.event` was split by audience, and
`akgentic.tool.vector` moved next to the code built on it. Both old paths keep a
compatibility façade, which emits a `DeprecationWarning` on **attribute access** — not at
import time, so code that touches none of the moved symbols is never warned. **No removal
release is scheduled** for either façade. The Internal-tier courtesy entries that no stored
payload can name have been **withdrawn**: those old paths now raise `ImportError`.

**Importing from the `akgentic.tool` package root needs no migration at all.** That surface
is unchanged, and reaching a symbol through it emits no warning. The root is also the
supported home of the global observers — `from akgentic.tool import ToolObserver` has
always worked and still does.

### Still shimmed — resolves, with a `DeprecationWarning`

| Old path | New home | Tier |
|---|---|---|
| `akgentic.tool.event.ToolStateEvent` | `akgentic.tool.core.event` | Stable |
| `akgentic.tool.event.CommandArg` | `akgentic.tool.core.event` | Stable |
| `akgentic.tool.event.CommandDescriptor` | `akgentic.tool.core.event` | Stable |
| `akgentic.tool.event.CommandsAnnouncedEvent` | `akgentic.tool.core.event` | Stable |
| `akgentic.tool.vector.VectorEntry` | `akgentic.tool.vector_store.vector` | Internal |

These stay shimmed because the façades are load-bearing beyond source compatibility, and
persisted data is what makes them so. `__model__` markers are written for Pydantic models
and dataclasses, and resolving one is `import_module` plus `getattr` on the path recorded
at write time — so any row stored before a move keeps naming the old path for as long as
that row exists. `VectorEntry` is on this list for that reason alone, and it is the only
entry of its module that qualifies. The event symbols additionally have a source consumer:
a sibling package imports `CommandsAnnouncedEvent` from the old path.

### Withdrawn — raises `ImportError`, move now

| Old path | Import instead |
|---|---|
| `akgentic.tool.event.ToolObserver` | `akgentic.tool` (root) or `akgentic.tool.core.observer` |
| `akgentic.tool.event.ActorToolObserver` | `akgentic.tool` (root) or `akgentic.tool.core.observer` |
| `akgentic.tool.event.TeamManagementToolObserver` | `akgentic.tool` (root) or `akgentic.tool.team.observer` |
| `akgentic.tool.vector.EmbeddingService` | `akgentic.tool` (root) or `akgentic.tool.vector_store.vector` |
| `akgentic.tool.vector.VectorIndex` | `akgentic.tool` (root) or `akgentic.tool.vector_store.vector` |

The observers are Stable-tier **symbols** — but the promise attaches to the package root,
their supported surface, not to every path they historically resolved from. Their `event.py`
residence was an accident of the pre-split layout. The `akgentic.tool.vector` module was
Internal-tier in its entirety, and everything in it that cannot appear in a stored payload
went with it — a plain class is never written to a `__model__` marker, so nothing in a
database can ask for one back.

### What the two tiers mean

The tier is not a measure of how important a symbol is. It says **whether its import path is
something you may build against**, and the two answers carry different promises.

**Stable — a supported surface.** These are the contracts a custom `ToolCard` author outside
this package writes against: the `akgentic.tool` package root, the core abstractions
(`ToolCard`, `BaseToolParam`, `ToolFactory`, `Channels`, `CommandRegistry`), the *global*
observers `ToolObserver` and `ActorToolObserver` **at the package root**, `ToolStateEvent`,
and the command-discovery models. That surface is part of the API: if a symbol on it moves,
it is shimmed, and the shim is kept.

**Internal — not a surface.** These belong to one specific tool: `TeamManagementToolObserver`
is `TeamTool`'s contract, the vector primitives are `vector_store`'s. They move freely with the
tool that owns them. A shim entry for one is a **courtesy, not a guarantee** — it exists because
removing a working import for no reason is rude, not because the path was ever promised.
Treating it as a promise would freeze this package's internal structure by accident, which is
exactly what the split was done to avoid.

An internal symbol may also be **removed outright**, not merely moved — and then there is no
shim and no warning. `ToolStatePayload` went first: an alias for the knowledge graph's delta
type that stopped annotating anything once `ToolStateEvent.payload` was typed structurally.
The courtesy entries above were withdrawn the same way: the observer entries in `event.py`,
and every entry of `akgentic.tool.vector` except `VectorEntry`. Importing a withdrawn path
raises `ImportError` rather than warning.

That exception is the general rule rather than a special case, and it is worth stating on its
own: **a symbol's tier governs its import path, not its wire format.** Internal-tier says no
caller may build against the path. It says nothing about the rows already on disk that name
it, and those rows are not a caller — they cannot be migrated by editing an import, and they
outlive the release that moved the class. So a persisted model keeps its old path regardless
of tier, and the courtesy that may be withdrawn from it is only the courtesy owed to source.

If one of your imports is in the *Withdrawn* table, it has already stopped working — move it
to the path in its *Import instead* column.

## Observers: How a Tool Acts on the System

A `ToolCard` is **inert**. It is fully serializable configuration, and that is a hard rule rather
than a default: every field must round-trip through Pydantic, so a card cannot hold an actor proxy,
a connection, or an open file. Taken literally, a card has no way to reach the running system at
all.

The **observer** is the inversion that resolves this. At wiring time `ToolFactory` calls
`observer()` on every card, handing it the agent that owns it. From that moment the observer is the
tool's **only** channel to the runtime — and because it arrives after construction and is never a
field, the card stays serializable. Everything a tool does to the system, it does through the
observer.

### Three levels — ask for the least you need

The observer is a `Protocol`, and there are three, each extending the one above it:

| Protocol | What it adds | What that lets a tool do |
|---|---|---|
| `ToolObserver` | `notify_event(event)` | Emit a domain event onto the orchestrator's stream. Nothing more. |
| `ActorToolObserver` | `myAddress`, `orchestrator`, `team_id`, `state`, `proxy_ask(...)` | Reach any actor by address — including a singleton tool actor. Reach the persisted tool-layer slot, `state.tool_state`, read live on every call. |
| `TeamManagementToolObserver` | `createActor(...)`, `on_hire(...)`, `on_fire(...)` | Create actors, and change the team's membership. |

Each capability is gated by the level above it: a tool that only emits events cannot reach an
actor, and a tool that reaches actors cannot hire anyone. Declare the narrowest level your tool
genuinely uses. The third level is domain-specific rather than general — `TeamTool` is its only
consumer in this package — which is why it lives beside that tool instead of on the core surface.

### Narrow in an accessor, not in the signature

One trap catches the obvious reading of "ask for the level you need". **Do not narrow the
`observer()` parameter.** `ToolFactory` attaches one observer to every card uniformly, so a card
demanding a richer parameter type is not substitutable for its base — a Liskov violation that a
type checker will happily let you write and the factory will break at runtime.

Keep the base parameter type, and narrow in your own accessor:

```python
class MyTool(ToolCard):
    # A proxy is not serializable: runtime handles are private attributes, never fields.
    _activity_proxy: TeamActivityActor | None = PrivateAttr(default=None)

    def observer(self, observer: ToolObserver) -> "MyTool":   # base type, always
        super().observer(observer)
        obs = self._team_observer()                           # narrow here instead
        # The observer hands you the orchestrator's *address*; ask it for a proxy first.
        orchestrator = obs.proxy_ask(obs.orchestrator, Orchestrator)
        address = orchestrator.getChildrenOrCreate(...)
        self._activity_proxy = obs.proxy_ask(address, TeamActivityActor)
        return self

    def _team_observer(self) -> TeamManagementToolObserver:
        return cast(TeamManagementToolObserver, self._observer)
```

`TeamTool` and `PlanningTool` both ship exactly this shape.

### The observer is held weakly

`ToolCard` stores the observer through a **weak reference**. A tool, its closures and its command
registry must never keep a stopped agent alive, and a strong reference in any one of them would do
it. Closures are the easy mistake, which is why they capture the *accessor* rather than the agent.

The consequence to plan for: **using a tool after its owning agent has stopped raises
`ToolObserverGone`.** That is a defined outcome, not a crash — the framework telling you the owner
is gone. There are two accessors for exactly this reason: one raises `ToolObserverGone`, the other
returns `None`. Synchronous in-life code uses the raising form; a closure that may outlive its
agent uses the `None`-returning one and handles the `None`.

Do not stash the observer in a field of your own to avoid this. It would not be serializable, and
it would reintroduce the strong reference the weak one exists to prevent.

## Tool State Events

A stateful tool actor's state is the point of it — the plan, the graph, the index. Clients want to
follow that state as it changes, and `ToolStateEvent` is how a tool actor tells them: by
broadcasting **what changed**, not what it now holds.

```python
ToolStateEvent(tool_id="#KnowledgeGraphTool", seq=7, payload=delta)
```

- **`tool_id`** — the name of the emitting tool actor, `#`-prefixed like the actor itself. A client
  following several stateful tools in one team routes on it.
- **`seq`** — a **per-tool monotonic** counter starting at 1. Per-tool, not per-team: two tool
  actors each number their own stream independently. A consumer detects a missed event by watching
  it.
- **`payload`** — the delta itself.

The envelope inherits `team_id`, `timestamp`, `id`, `sender` and `display_type` from the framework's
`Message` base without overriding any of them, so it travels on the ordinary event stream.

### Why deltas, and why there is no snapshot protocol

A tool actor's state can be large, and it changes in small increments. Republishing all of it on
every mutation would be wasteful in the ordinary case and useless in the interesting one — a client
that wants to show *"three entities were added"* cannot recover that from two snapshots without
diffing them itself.

So the event carries the increment, and it rides the path the orchestrator already has:
`notify_event` puts it on the orchestrator's event stream, which is recorded in team history. A
client that joins late does not ask for a snapshot — **there is no snapshot request and no snapshot
message** — it replays the history it would have replayed anyway and applies the deltas in order.
Tool state reconstructs itself out of the normal replay path, which is why the mechanism needs no
protocol of its own.

### `payload` is structurally typed

`payload` is declared as *any* serializable model, not as a union of the concrete delta types. That
is deliberate: a union naming the knowledge graph's delta and its peers would make the
package-global envelope depend on every domain that emits one — exactly the dependency the package
layout forbids.

The concrete class is not lost. Serialization tags the payload with a `__model__` marker naming its
class, so a consumer deserializes the real object and **discriminates on the object, not on the
envelope** — an `isinstance` check, in Python terms. Your own delta type needs no registration and
no entry in any union; it needs only to be a serializable model.

One caveat if you ever move a delta class between modules: that marker records the class's module
path, so it moves when the class moves. The payload's own fields are unaffected.

### The emit-before-return contract

A mutation method emits its event **before it returns** — and before it raises, if it collected
errors along the way. A caller that gets a return value knows the event is already on its way; a
caller that gets an exception still gets the events for the work that did succeed.

The knowledge graph is the shipped example. `update_graph` applies its entity and relation changes,
builds a delta from what it actually added, modified and removed, then:

```python
delta = KnowledgeGraphStateEvent(
    entities_added=created_entities,
    entities_modified=modified_entities,
    entities_removed=deleted_entity_ids,
    relations_added=created_relations,
    relations_removed=merged_relations_removed,
)

self.state.notify_state_change()

if self._delta_is_non_empty(delta):
    self._state_event_seq += 1
    self.notify_event(
        ToolStateEvent(tool_id=KG_ACTOR_NAME, seq=self._state_event_seq, payload=delta)
    )

if errors:
    raise RetriableError("Update errors: " + "; ".join(errors))
return "Done"
```

Two details worth copying:

- **An empty delta emits nothing.** "Emit before return" is not "emit unconditionally" — a call
  that changed nothing is not a state change, and broadcasting it would make every consumer filter
  noise.
- **`seq` advances only when an event is actually emitted**, inside the guard. Numbering therefore
  has no gaps for suppressed empty deltas — which is what makes a gap meaningful to a consumer.

## Tool Actors

Most tools are stateless: the card holds configuration, the callable does its work and returns. A
few are not. A plan, a knowledge graph and a vector index are **shared, mutable state that outlives
any single tool call**, and the framework gives that state a home — a **tool actor**, one per team,
that every agent carrying the card talks to.

Six ship in this package today: `#VectorStore`, `#PlanningTool`, `#KnowledgeGraphTool`,
`#SandboxActor`, `#TeamActivity` and `#NotificationTool`.

### One per team, and what that buys

**Shared state.** Ten agents carrying `PlanningTool` do not get ten plans. They get ten proxies to
one `#PlanningTool`, so when the researcher marks a task done the writer sees it. Give each agent
its own copy and the tool stops meaning anything — agents would be coordinating through state they
cannot both see.

**Centralised processing.** One embedding path, one sandbox, one store, rather than N. The
expensive machinery is built once, and configuration that must agree — which model embeds, which
sandbox mode is permitted — is decided in one place instead of being replicated per agent and left
to drift.

**No locks.** An actor processes one message at a time, so a tool actor's mutations cannot
interleave. Two agents updating the graph in the same instant are serialised by the mailbox, not by
anything you write, which is why a tool actor's methods can read-modify-write without a mutex. The
same one-thread property is why the next section exists: it also means a slow method blocks
everyone queued behind it.

**State that persists itself.** A tool actor's state reaches the team's event store without the
tool arranging it — the actor calls `notify_state_change()`, and the framework snapshots the state
and restores it when the team resumes. Persistence here is a property of being an actor, not
something a tool implements.

### Binding one: `getChildrenOrCreate`, never check-then-create

A tool binds its actor through `orchestrator_proxy.getChildrenOrCreate(...)`, which is idempotent:
it returns the existing singleton or creates it, in one step. The actor is created as a child of
the orchestrator — the team's single orchestrator owning it is what guarantees unicity.

The obvious alternative is a bug. "Ask whether it exists, create it if it does not" is two messages
with a window between them — two agents wiring the same tool at startup both look, both find
nothing, and both create. That is not a theoretical race. It produced duplicate singletons, which
is the exact failure the singleton pattern exists to prevent, arrived at by the code written to
prevent it.

### The `#` prefix is a teardown invariant

Every tool actor's name starts with `#`, and that is not a naming convention you may opt out of.
The orchestrator decides what counts as a tool actor by that prefix, and drives a two-phase stop
with it: regular members first, tool actors only once no regular member remains — which is what
stops a tool actor being torn down while an agent is still calling it. If your tool creates an
actor, prefix its name.

What the prefix does and does not buy is covered in
[Teardown: why the `#` prefix is not cosmetic](#teardown-why-the--prefix-is-not-cosmetic), below.

## Deferred Results: Never Block a Tool Actor

A tool actor is a **team singleton with one thread**. If a method that callers reach via
`proxy_ask` performs slow external work — an LLM call, a document conversion, a sandbox run, any
network round-trip — that actor is occupied for the whole call and **every other team member queuing
on it is blocked**. The obvious mitigation does not work: a Pykka `timeout=` on the ask abandons the
future without cancelling the work, so the actor stays occupied and its mailbox backs up.

The pattern: a **cache actor** that never performs slow work, **short-lived workers** that do, and a
**bounded caller-side poll**.

```
tool closure                     #CacheActor                   #defer-<key> (worker)
     │                                 │                                   │
     │── get(key) ─ask────────────────▶│  dict lookup, O(1)                │
     │◀──────────────────── None ──────│                                   │
     │── request(key, payload) ─ask ──▶│  not cached, not in-flight        │
     │                                 │──── createActor + tell ──────────▶│
     │                                 │                                   │
     │       … poll_deferred: N × (sleep, get(key)) …                      │  blocking call
     │                                 │◀───── deliver(key, value) tell ───│
     │◀──────────────────── value ─────│                                   │  self.stop()
```

The cache actor's thread is held only for dict lookups, so N members query it concurrently while one
production is in flight. The caller waits on its own thread — which is why the poll budget is bounded
and a degraded answer always exists.

```python
import uuid

from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.tool.core.deferred import DeferredResultActor, DeferredWorker, poll_deferred

class SummaryCache(DeferredResultActor[BaseConfig, BaseState, uuid.UUID, str]):
    def worker_class(self) -> type[DeferredWorker]:
        return SummarizerWorker

# In the tool closure — `cache` is an ask proxy, and it is the only proxy there is:
summary = cache.get(message_id)              # ask — O(1) dict lookup
if summary is None:
    cache.request(message_id, payload)       # TELL-shaped, called on the ask proxy
    summary = poll_deferred(lambda: cache.get(message_id), attempts=5, delay=0.4)
if summary is None:
    summary = text[:200] + "…"               # degraded answer, always available
```

The four type parameters are `ConfigType`, `StateType`, the hashable cache key `K`, and the produced
value `V` — the first two because `Akgent` already declares them. `deferred` is deliberately **not**
on the `akgentic.tool.core` façade; import `akgentic.tool.core.deferred` directly.

Calling `request` on the **ask** proxy is not a partial adoption of the mechanism. `request` adds to
the in-flight set, spawns a worker and tells it the payload — all O(1) on the cache actor's thread,
so the ask never waits on external work. A tool closure holds an ask proxy and nothing else:
`ActorToolObserver` exposes no tell proxy.

### Seven rules — all of them, or none

1. **The cache actor never performs the slow call.** It spawns, caches, and answers `get`.
2. **One worker per key, short-lived, self-stopping.** Never reused, never accumulates state.
3. **The worker's actor name MUST start with `#`.** See the teardown note below.
4. **De-duplicate through the in-flight set.** Three callers, one key ⇒ one external call.
5. **Failures are cached negatively.** A failed key does not respawn a worker on every poll; retry
   policy is a TTL on the negative entry, never an uncapped respawn.
6. **The cache is capped (LRU).** An uncapped cache on a team singleton leaks for the life of the team.
7. **Callers poll with a bounded budget and always have a degraded answer.** An unbounded ask —
   with or without a timeout — is forbidden.

### Teardown: why the `#` prefix is not cosmetic

Every actor announces itself to the orchestrator on start, so **a spawned worker is a visible team
member**. The orchestrator stops tool actors only once no *non-tool* member remains, and it decides
what is a tool actor by the `#` name prefix.

What the prefix does **not** buy is a faster teardown. A worker is a *child* of its cache actor, and
`stop_children(blocking=True)` waits for it under **either** name — so a worker mid-call holds its
parent's stop open whatever it is called, total teardown time is the same with or without the prefix,
and the stop backstop fires in both cases or neither.

What the prefix buys is **sibling release**. Named `#defer-…`, a worker is a tool actor, so phase 2
tears down unrelated tool actors — `#PlanningTool`, `#KnowledgeGraphTool`, … — in parallel. Named
`summarize-abc123` it counts as a regular member, and every one of those siblings serializes behind a
worker it has nothing to do with.

Because the parent's stop blocks on its children either way, every worker must bound its own external
call with an explicit timeout below the orchestrator's stop backstop — and hand that budget to its
I/O client. A Python thread cannot be cancelled, so a timeout that does not reach the client is
decoration.

## Building a feature as a card

A `BaseAgent` feature is not a method added to `BaseAgent` — **it is a `ToolCard` whose
capabilities declare, channel by channel, how they reach the agent's world.** The agent grows
behaviour by hosting cards, not by accreting methods, so authoring a feature starts with one
question per capability: which channel actually serves it?

### The four-channel contract

Each channel is a promise about how a capability reaches the model or the humans around it — and
each fails in its own way when misused:

| Channel | Promise | Failure mode |
|---|---|---|
| `SYSTEM_PROMPT` | Rendered **once** into the frozen prefix — static instructions, cached | Volatile content here invalidates every agent's prefix cache on every turn — the anti-pattern ADR-037 removed |
| `LLM_CONTEXT` | Structured state, diffed and appended per turn at the tail — volatile awareness | A provider that raises breaks context construction — providers return `None`, never raise |
| `TOOL_CALL` | Model-invoked, pull — deliberate action | The model only calls what the docstring teaches; the docstring **is** the contract |
| `COMMAND` | Human- or code-invoked by name, push — control | Dispatch replies only — a command is not an enforcement point |

**The silent drop.** A capability exposed on a channel its card cannot serve is dropped with **no
error and no warning** — the factory aggregates only what the card's hooks return, and nothing
cross-checks that `expose` and the hooks agree. A `GetPlanning` left on `SYSTEM_PROMPT` after the
card stopped overriding `get_system_prompts()` would simply vanish from the prompt — the drop is
real enough that the params the epic-31 migration moved each carry the per-param
`normalize_system_prompt_to_llm_context` validator to rewrite exactly that persisted exposure.
This is why a card's wiring tests assert the actual provider / tool / command lists rather than
trusting the declaration.

**The routing rule.** A capability is served exactly when both gates pass: its param **resolves**
(`True` or a param instance — `False` removes it), **and** its `expose` set contains the channel
the serving hook covers. Every hook applies that same two-step gate: `get_context_states()` serves
`LLM_CONTEXT`, `get_tools()` serves `TOOL_CALL`, `get_commands()` serves `COMMAND`, and
`get_system_prompts()` serves `SYSTEM_PROMPT`.

The machinery behind each row is documented above and not restated here — the
[Channel System](#channel-system) section covers the enum and the two prompt-bearing promises, and
[Context state: the `get_context_states()` hook](#context-state-the-get_context_states-hook)
covers the provider contract (`render_full` / `render_delta`, aggregation, the `__name__` key).

### Worked example: `MailboxTool`, one capability per row

`MailboxTool` (`akgentic.tool.mailbox`) routes **each of its capabilities onto its own channel**,
which makes it a compact reference shape for routing. Two capabilities, two params, one channel
each:

| Capability | Param (default) | Channel | Serving hook | Gate |
|---|---|---|---|---|
| On-demand signal | `read_mailbox: ReadMailbox \| bool = True` | `TOOL_CALL` | `get_tools()` | param resolves **and** `TOOL_CALL` in its `expose` |
| Run cancellation | `stop: Stop \| bool = True` | `COMMAND` | `get_commands()` | param resolves **and** `COMMAND` in its `expose` |

**`read_mailbox` → `TOOL_CALL`.** Taking on a waiting message is a deliberate act, so the model
pulls it: the card offers a signal that names one message by id, and the model calls it when it
means to handle that message now. The docstring carries the load-bearing contract — naming a
message absorbs it into the current run, so it will not arrive again as its own turn, while
anything left unnamed stays queued — and the docstring is the only place that contract can be
taught, which is precisely why the capability belongs on a pull channel rather than a push one.
What the model does not get here is content: the call returns an acknowledgement, and the message
itself reaches the run through `akgentic-agent`'s half.

**`stop` → `COMMAND`.** Cancellation is control, pushed by a human or a program, so it is a named
command — string surface `/stop`, announced to every frontend for free via
`CommandsAnnouncedEvent`. Dispatched while the agent is idle it answers *"There is no run to
cancel."*, because that is by construction the only case a dispatched `/stop` can be: the agent
purges a mid-run cancel at recognition, so one never survives to reach a handler. **The card owns
the registration, the agent owns both the vocabulary and the enforcement** — recognising the
`/stop` string and the mid-run interrupt both live in `akgentic-agent` (its Epic 20), because
cancellation must work even on an agent configured without this card, which has no card to import
a predicate from.

## Tool Catalog

Twelve tool cards ship with the package. Each one has its own README next to the code, covering the
`ToolCard` definition, every field and every nested capability parameter, and the full
configuration surface — environment, extras, actor wiring and failure modes. The entries below
are the index; the detail lives beside the module it documents.

| Tool | Module | What it does | Reference |
|---|---|---|---|
| `WorkspaceTool` | `akgentic.tool.workspace` | Team-scoped filesystem: read, list, glob, grep, view, write, edit, patch | [README](src/akgentic/tool/workspace/README.md) |
| `PlanningTool` | `akgentic.tool.planning` | Shared task board backed by the `#PlanningTool` actor | [README](src/akgentic/tool/planning/README.md) |
| `KnowledgeGraphTool` | `akgentic.tool.knowledge_graph` | Entities and relations with hybrid keyword + semantic search | [README](src/akgentic/tool/knowledge_graph/README.md) |
| `VectorStoreTool` | `akgentic.tool.vector_store` | Configuration-only card owning the shared embedding store | [README](src/akgentic/tool/vector_store/README.md) |
| `SearchTool` | `akgentic.tool.search` | Web search, fetch and crawl via Tavily | [README](src/akgentic/tool/search/README.md) |
| `TeamTool` | `akgentic.tool.team` | Hire, fire, roster, role profiles, and who is busy right now | [README](src/akgentic/tool/team/README.md) |
| `MetadataTool` | `akgentic.tool.metadata` | The team's business context, rendered once into every agent's prefix | [README](src/akgentic/tool/metadata/README.md) |
| `NotificationTool` | `akgentic.tool.notification` | Delayed messages an agent schedules to itself | [README](src/akgentic/tool/notification/README.md) |
| `SkillTool` | `akgentic.tool.skill` | A library of skills: the menu in the prefix, the bodies on demand | [README](src/akgentic/tool/skill/README.md) |
| `MailboxTool` | `akgentic.tool.mailbox` | A signal naming one message in the agent's own mailbox, and the `/stop` cancel surface | [README](src/akgentic/tool/mailbox/README.md) |
| `MCPTool` | `akgentic.tool.mcp` | External MCP servers as pydantic-ai toolsets | [README](src/akgentic/tool/mcp/README.md) |
| `ExecTool` | `akgentic.tool.sandbox` | Sandboxed shell execution in the team workspace | [README](src/akgentic/tool/sandbox/README.md) |

### WorkspaceTool

Sandboxed read/write access to a shared team filesystem — `workspace_read`, `workspace_list`,
`workspace_glob`, `workspace_grep`, `workspace_view` on the read side; `workspace_write`,
`workspace_edit`, `workspace_multi_edit`, `workspace_patch`, `workspace_delete`,
`workspace_mkdir` on the write side. One class covers both modes via a `read_only: bool` gate.
All paths are anchored to `<AKGENTIC_WORKSPACES_ROOT>/<workspace_id or team_id>` and traversal
out of that root is rejected.

```python
from akgentic.tool import WorkspaceTool

WorkspaceTool()                                      # full access (default)
WorkspaceTool(read_only=True)                        # read tools only
WorkspaceTool(workspace_id="shared")                 # shared workspace across teams
WorkspaceTool(read_only=True, workspace_glob=False)  # fine-grained capability control
```

Binary reads (PDF, DOCX, XLSX, PPTX) need `akgentic-tool[docs]`; image resizing for
`workspace_view` needs `akgentic-tool[vision]`. Both degrade rather than fail.

**[Full reference → `src/akgentic/tool/workspace/README.md`](src/akgentic/tool/workspace/README.md)** —
every capability parameter, the `DocumentReader` two-pass extraction, resource seeding, sidecar
caching and the edit-matching cascade.

### PlanningTool

Shared actor-based task board for multi-agent teams. A singleton `PlanActor` (named
`#PlanningTool`) is created by the orchestrator as one of its children — get-or-create semantics
guarantee unicity — and persists across all agents' tool calls. The plan is exposed as structured
context state on `LLM_CONTEXT`, delivered into the context tail as per-turn deltas and scoped to
each agent's own tasks by default.

```python
from akgentic.tool.planning import GetPlanning, PlanningTool

PlanningTool()                                                  # default config
PlanningTool(get_planning=GetPlanning(filter_by_agent=False))   # show all tasks
PlanningTool(vector_store=False)                                # keyword-only search
```

Semantic search needs `akgentic-tool[vector_search]` and a `VectorStoreTool` in the team; without
either it degrades to keyword-only.

**[Full reference → `src/akgentic/tool/planning/README.md`](src/akgentic/tool/planning/README.md)** —
task model constraints, the four capabilities and their channels, collection configuration and
the `depends_on` contract.

### KnowledgeGraphTool

Persistent actor-based knowledge graph for structured entity and relationship storage with hybrid
keyword + semantic search. The graph summary is exposed as structured context state on
`LLM_CONTEXT` — delivered into the context tail as deltas — and scales as *O(types + roots)*
rather than *O(entities)*, so a large graph stays affordable as context.

```python
from akgentic.tool.knowledge_graph import KnowledgeGraphTool

KnowledgeGraphTool()
KnowledgeGraphTool(read_only=True)
```

Requires `akgentic-tool[vector_search]` — the dependency is checked at wiring time even when
`vector_store=False`.

**[Full reference → `src/akgentic/tool/knowledge_graph/README.md`](src/akgentic/tool/knowledge_graph/README.md)** —
the mutation and query models, search modes and expansion flags, scoring, and the state-delta
events.

### VectorStoreTool

Configuration-only companion card for the `VectorStoreActor` singleton — it exposes **no LLM
tools, system prompts, or commands** (`get_tools()` returns `[]`). Its sole runtime job is to
ensure the singleton exists when the observer attaches. Consumer cards (`PlanningTool`,
`KnowledgeGraphTool`) never create the actor themselves: they look it up by name and declare a
conditional `depends_on: ["VectorStoreTool"]`, so `ToolFactory`'s topological sort wires this card
first.

```python
from akgentic.tool.vector_store import VectorStoreTool

VectorStoreTool()                                  # "#VectorStore", OpenAI embeddings
VectorStoreTool(vector_store_name="#VectorStore-RAG", embedding_provider="azure")
```

Collections are configured on the *consumer* card (`collection: CollectionConfig`), and Weaviate
connection settings are deliberately not fields on any card — they are infrastructure, injected by
the deployment layer.

**[Full reference → `src/akgentic/tool/vector_store/README.md`](src/akgentic/tool/vector_store/README.md)** —
`CollectionConfig` in full, the service protocol, asynchronous embedding, and multi-store setups.

### SearchTool

Web search and content fetching via the [Tavily](https://tavily.com/) API: `web_search`,
`web_fetch` and `web_crawl`. Every capability parameter becomes the *default value* of the
corresponding tool argument, so configuration biases the model without removing its judgement.

```python
from akgentic.tool.search import SearchTool, WebCrawl

SearchTool()
SearchTool(web_crawl=WebCrawl(max_depth=2, limit=50))
```

Requires the `TAVILY_API_KEY` environment variable. A missing or invalid key never raises — the
tool returns a message telling the model it is unavailable.

**[Full reference → `src/akgentic/tool/search/README.md`](src/akgentic/tool/search/README.md)** —
every Tavily parameter with its accepted range, and how `crawl_instructions` differs from the
inherited `instructions`.

### TeamTool

Exposes team management capabilities (hire/fire agents, roster, role profiles) to the LLM, and
answers *who is working right now, and on what*. The roster and the hireable-role catalog are
exposed as structured context state on `LLM_CONTEXT`, delivered into the context tail as deltas.
Used by `BaseAgent` in `akgentic-agent` to let orchestrator-level agents extend the team at
runtime. Requires a `TeamManagementToolObserver`.

```python
from akgentic.tool.team import ActivitySummarizer, GetTeamActivity, TeamTool

TeamTool()                                      # hire/fire/roster/profiles + team_activity()
                                                #   (truncation only — no actor, no model call)
TeamTool(get_team_activity=False)               # team management only
TeamTool(get_team_activity=GetTeamActivity(     # + summaries on demand; #TeamActivity is created
    summarizer=ActivitySummarizer(model="openai:gpt-5.2-mini"),
))
```

`team_activity` defaults to on because the truncate-only report is derived from telemetry the
orchestrator already keeps — it costs nothing. The `#TeamActivity` cache actor is created **only**
when a summarizer is configured, and the callable's signature follows the configuration:
`summarize_over` is absent from the schema when nothing could produce a summary.

**[Full reference → `src/akgentic/tool/team/README.md`](src/akgentic/tool/team/README.md)** —
the hire/fire channel split, partial-success reporting, the two activity gates, and how "busy" is
derived from telemetry.

### MetadataTool

Renders the team's **business context** — the model the deployment wrote with
`Orchestrator.set_metadata()` — into every agent's system prompt, from one operator-written
template. Without it the same facts get copied into every role's backstory, where they duplicate
and drift away from the authoritative copy. The card owns no actor and holds no state beyond the
block it rendered. Alone among the cards here its capability ships **off**, there being no template
a framework could supply: a `RenderMetadata` turns it on, and a card left unconfigured contributes
nothing rather than failing.

```python
from akgentic.tool.core import COMMAND
from akgentic.tool.metadata import MetadataTool, RenderMetadata

MetadataTool(render_metadata=RenderMetadata(
    header="Team context",
    template="Fiscal year: {fiscal_year}. Engagement: {engagement}.",
))

MetadataTool(render_metadata=RenderMetadata(     # command only — nothing in the prompt
    template="Fiscal year: {fiscal_year}.",
    expose={COMMAND},
))
```

Placeholders are **bare field names** of the team's metadata model — no dotted paths, indices,
conversions or format specs — and a template that breaks that rule raises `ValueError` at wiring
time, next to the mistake. A name the model does not declare raises there too, but only when the
team already holds metadata: `set_metadata` may legitimately run after the agents start, so
otherwise the name check moves to the first render, where it degrades to an empty block and an
ERROR in the log rather than raising.

**The block is a snapshot.** It is rendered **once**, at the first render that *succeeds*, and a
later `set_metadata` is **not** reflected. That is deliberate, not a limitation waiting to be
fixed: re-reading per turn would make the system prompt volatile and one write would invalidate
every agent's prefix cache. (A *degraded* render caches nothing, so metadata that arrives just
after start-up still produces its block on a later turn.) A deployment whose business context
genuinely changes mid-life does not want this card.

`expose` defaults to `{SYSTEM_PROMPT, COMMAND}`: the prompt the agents read, and `team_metadata()`
for a human who wants to see exactly what they were given. `get_tools()` is **always** empty — the
model is never handed a tool to fetch metadata, which would cost a round trip for content that
never changes and require the model to know it should ask.

**[Full reference → `src/akgentic/tool/metadata/README.md`](src/akgentic/tool/metadata/README.md)** —
the template grammar in full, both validation points, the degradation table, and the recipes for
metadata set before and after the team starts.

### NotificationTool

Lets an agent schedule a message **to itself**, delivered after a delay — to defer its own
attention, check a long-running result later, or nudge itself if nothing has happened by then. A
team singleton (named `#NotificationTool`) holds the pending entries and delivers them.

```python
from akgentic.tool import NotificationTool

NotificationTool()                            # AgentMessage delivery, 300 s cap
NotificationTool(max_delay_seconds=60)        # tighter cap
NotificationTool(message_class="acme_core.messages.ReminderMessage")
```

Ownership is scoped per agent: listing can be widened to the whole team with
`pending_notification(all=True)`, but cancel authority never widens with it. Entries store an
absolute due time, so a delay that expired while the team was stopped simply fires on resume.

**[Full reference → `src/akgentic/tool/notification/README.md`](src/akgentic/tool/notification/README.md)** —
the `message_class` validation contract, delivery and grace semantics, and the `/`-command
surface.

### SkillTool

A library of domain guidance — a refund procedure, an escalation policy, a report playbook — split
by size and volatility. The **menu** (one `name — description` line per skill) is small and
immutable, so it goes into the frozen system prefix. The **bodies** are large and optional, so they
arrive at the tail on demand, as ordinary tool returns. Without the split every agent pays for every
playbook on every turn, and the instructions that matter for the turn compete with seven that do
not.

```python
from akgentic.tool import SkillTool
from akgentic.tool.core import SYSTEM_PROMPT, TOOL_CALL
from akgentic.tool.skill import SkillEntry, Skills

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

SkillTool(skills=Skills(skills=[REFUND, ESCALATION]))   # all three channels (default)

SkillTool(skills=Skills(                                # prompt + tool only: no /skills command
    skills=[REFUND, ESCALATION],
    expose={SYSTEM_PROMPT, TOOL_CALL},
))

SkillTool(skills=False)                                 # capability removed entirely
```

**Three channels, each carrying what it is good at.** `SYSTEM_PROMPT` carries the menu — a header
line, then one `name — description` line per skill. `TOOL_CALL` carries `use_skill(name)`.
`COMMAND` registers `skills()`, the same menu rendered for a human, from the same renderer, so it is
an honest answer to *what was this agent actually given?*

**`use_skill` returns the body as its tool result, in the same turn** — `"{name}\n\n{content}"`, not
an acknowledgement and not a next-turn delivery. The model asked because it needs the body for the
answer it is composing, so anything arriving on the next turn arrives after the answer it was needed
for. An unknown name raises `RetriableError` listing the available names, so the model corrects
itself in-loop.

**One loading rule.** A body lives in the conversation until a compaction or the sliding window
drops it; after that the model re-calls `use_skill`. The menu is in the frozen prefix and survives
that, which is why re-calling always works — a restart is **not** a special case. That placement is
load-bearing rather than merely economical: a menu at the tail would be one compaction away from an
agent that no longer knows its skills exist.

The prefix cost is **O(skills), not O(content)** — only `name` and `description` are rendered, so
body size is paid solely by the conversations that ask for it. The card holds **no state**: it keeps
no loaded set, so calling `use_skill` on the same name again is the intended recovery rather than a
redundancy to suppress.

**[Full reference → `src/akgentic/tool/skill/README.md`](src/akgentic/tool/skill/README.md)** —
every field of `SkillEntry` and `Skills` with what it costs, the menu quoted as rendered, the
loading model, and the failure modes worth knowing.

### MailboxTool

The agent's own mailbox as a capability, on two channels: `read_mailbox` on `TOOL_CALL` — a
**signal** naming one waiting message by id — and the `/stop` cancellation surface on `COMMAND`.
The card creates no actor and performs no proxy round trip.

```python
from akgentic.tool import MailboxTool

MailboxTool()                     # both capabilities on (the default)
MailboxTool(read_mailbox=False)   # /stop only — no on-demand signal
MailboxTool(stop=False)           # no cancellation surface
MailboxTool(mailbox_preview_handlers=["akgentic.agent.messages.AgentMessage"])
```

Reading absorbs: naming a message takes it on in the current run, so it will not also arrive as its
own turn — which is what stops the agent answering the same message twice — while anything left
unnamed stays queued and arrives later. The *mechanism* is what changed, not the promise: the card
itself reads, consumes and renders nothing, and returns an acknowledgement. Consuming the named
message and injecting its content is `akgentic-agent`'s, and that half has not yet shipped, so this
package must not be released alone. `mailbox_preview_handlers` names which handlers' runs are
offered the mailbox at all; every entry is resolved at agent init, and a typo raises `ValueError`
there. The `stop` command registers the `/stop` surface only — dispatched while the agent is idle it
answers *"There is no run to cancel."*, and both recognising the string and the mid-run enforcement
are the agent's, in `akgentic-agent` (its Epic 20). Nothing in this card raises, tracks or
interrupts.

**[Full reference → `src/akgentic/tool/mailbox/README.md`](src/akgentic/tool/mailbox/README.md)** —
the two capability params, the absorption contract and why the id is not validated, the
`mailbox_preview_handlers` whitelist and its four failure shapes, the `message_id` cross-repository
contract, where the cancel vocabulary lives, the observer protocol, and the cross-package half that
`akgentic-agent` owns — including the release coupling.

### MCPTool

Integrates external [Model Context Protocol](https://modelcontextprotocol.io) servers as native
pydantic-ai toolsets over three transports: `streamable-http` (default), `sse`, and `stdio`.
`MCPTool` takes exactly one connection, on a required singular `connection` field.

```python
from akgentic.tool.mcp import MCPTool, MCPHTTPConnectionConfig, MCPStdioConnectionConfig

MCPTool(connection=MCPHTTPConnectionConfig(url="https://mcp.acme.example/api/v1/endpoint"))
MCPTool(connection=MCPStdioConnectionConfig(stdio_command="uvx", stdio_args=["acme-mcp-server"]))
```

`get_tools()` is always empty — MCP capabilities reach the agent through `get_toolsets()`. The
transport is always taken from the config, never inferred from the URL, so `transport="sse"` must
be requested explicitly.

**[Full reference → `src/akgentic/tool/mcp/README.md`](src/akgentic/tool/mcp/README.md)** —
both connection models field by field, the SSE timeout subtlety, tool prefixing, diagnostics and
the OAuth helpers.

### ExecTool

Sandboxed shell command execution inside the team workspace. A single `SandboxActor` is spawned
per team and reused across all `ExecTool` calls; the backend is selected via the `mode` field,
with `auto` probing the host (`bwrap` → `seatbelt` → `docker` → `local`).

```python
from akgentic.tool import ExecTool

ExecTool()                        # auto mode (default)
ExecTool(mode="docker")           # persistent container per team
ExecTool(workspace_id="shared")   # share the workspace directory with WorkspaceTool
```

Commands are checked against the `ALLOWED_COMMANDS` allowlist — first token only. Disallowed
commands and backend failures come back as strings, never as exceptions. Set
`AKGENTIC_SANDBOX_IMAGE` to use a pre-built Docker image instead of the auto-built one.

**[Full reference → `src/akgentic/tool/sandbox/README.md`](src/akgentic/tool/sandbox/README.md)** —
the four backends compared (isolation, timeouts, rlimits, network), the Docker image lifecycle,
and how to register a backend of your own.

## Error Handling

`RetriableError` (defined in `akgentic.tool.errors`) is the single signal for recoverable
failures. Tools raise it with a clear, actionable message. `ToolFactory` translates it to
the framework-specific retry exception (e.g., pydantic-ai `ModelRetry`) via injection —
tool logic stays framework-agnostic.

```python
from akgentic.tool.errors import RetriableError

def my_tool(path: str) -> str:
    """Read a file."""
    try:
        return backend.read(path)
    except FileNotFoundError:
        raise RetriableError(f"File not found: {path}")
    except PermissionError:
        raise RetriableError("Path escapes workspace root — use a relative path")
```

**Rule:** no raw Python exception should escape a tool callable. An unhandled exception
produces no tool response and stalls the agent's ReAct loop.

| Exception | Treatment |
|---|---|
| `FileNotFoundError` | Wrap as `RetriableError("File not found: {path}")` |
| `PermissionError` (path escape) | Wrap as `RetriableError("Path escapes workspace root ...")` |
| `re.error` (bad regex) | Wrap as `RetriableError("Invalid regex pattern: {error}")` |
| `RuntimeError` (uninitialised state) | Let propagate — programming error, not an LLM error |

## Optional Extras

| Extra | Packages | Enables |
|---|---|---|
| `vector_search` | `openai>=1.0.0`, `numpy>=1.26.0` | Semantic search in `PlanningTool` and `KnowledgeGraphTool` |
| `weaviate` | `weaviate-client>=4.9.0` | Weaviate backend for the vector store |
| `docs` | `markitdown[pdf,docx,xlsx,xls,pptx,outlook]>=0.1` | Binary file reading in `workspace_read` |
| `vision` | `Pillow>=10.0` | Image resizing + sidecar cache in `workspace_view` |

No extra is required at import time. When one is absent the affected feature either falls
back or fails with an actionable message: planning falls back to keyword-only search, image
resizing is skipped with a one-time warning, workspace binary reads raise `ValueError` with
an install hint, and selecting the Weaviate backend raises `ImportError` with install
instructions.

## Development

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
uv sync --all-extras
```

### Commands

```bash
# Run tests
uv run pytest tests/

# Run tests with coverage
uv run pytest tests/ --cov=akgentic.tool --cov-fail-under=80

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Type check
uv run mypy src/
```

### CI Pipeline

Every pull request runs the full quality gate via GitHub Actions
(`.github/workflows/ci.yml`):

The repository is checked out standalone and `akgentic-*` dependencies resolve
from PyPI, so CI runs the same repo-relative commands listed above:

| Step | Command | Gate |
|---|---|---|
| Type check | `mypy src/` (strict, Python 3.12) | Zero errors |
| Lint | `ruff check src/` | Zero errors |
| Tests | `pytest tests/ --cov=akgentic.tool --cov-branch --cov-fail-under=80` | All pass, ≥ 80% branch coverage |

The CI badge at the top of this README reflects the current state of `master`. PRs are
blocked from merging until all three steps are green.

### Project Structure

```
src/akgentic/tool/
    __init__.py               # Public API
    py.typed                  # PEP 561 typing marker
    core/
    │   __init__.py           # Façade: ToolCard, BaseToolParam, ToolFactory, Channels
    │   channels.py           # Channels enum: TOOL_CALL, SYSTEM_PROMPT, LLM_CONTEXT, COMMAND
    │   context_state.py      # ContextState ABC — diffable prompt state for LLM_CONTEXT
    │   state.py              # ToolState — the persisted per-agent tool-layer slot
    │   params.py             # BaseToolParam, normalize_system_prompt_to_llm_context
    │   card.py               # ToolCard
    │   dependencies.py       # Topological ordering of cards by depends_on
    │   commands.py           # CommandRegistry
    │   context_update.py     # ContextUpdater — one Context update block per turn
    │   factory.py            # ToolFactory
    │   event.py              # ToolStateEvent, CommandArg, CommandDescriptor,
    │   │                     #   CommandsAnnouncedEvent — package-global contracts
    │   observer.py           # ToolObserver, ActorToolObserver — the global observers;
    │                         #   ToolStateCarrier — the structural carrier of ToolState
    │   └── deferred.py       # DeferredResultActor, DeferredWorker, poll_deferred
    │                         #   NOT on the façade — import akgentic.tool.core.deferred
    errors.py                 # RetriableError
    event.py                  # Compatibility façade only — the symbols that lived here
    │                         #   moved to core/, team/ and knowledge_graph/.
    │                         #   See "Migration: moved import paths"
    vector.py                 # Compatibility façade only — moved to
    │                         #   vector_store/vector.py. See the migration table
    vector_store/
    │   README.md           # VectorStoreTool reference — fields, CollectionConfig, backends
    │   vector.py             # VectorEntry, EmbeddingService, VectorIndex
    │   │                     #   [optional: vector_search extra]
    │   protocol.py           # VectorStore Protocol, VectorStoreConfig, data models
    │   inmemory.py           # InMemory backend
    │   weaviate.py           # Weaviate backend [optional: weaviate extra]
    │   actor.py              # VectorStoreActor singleton
    │   embedding_actor.py    # EmbeddingActor (non-blocking embedding); spawned as
    │                         #   "#embed-<collection>-<request_id>" (teardown
    │                         #   invariant — see Deferred Results)
    │   └── tool.py           # VectorStoreTool ToolCard
    planning/
    │   README.md           # PlanningTool reference — capabilities, task models, wiring
    │   planning_actor.py     # Task models, PlanConfig, PlanActor
    │   state.py              # TaskRow, PlanningState — planning context state + deltas
    │   └── planning.py       # PlanningTool ToolCard
    knowledge_graph/
    │   README.md           # KnowledgeGraphTool reference — params, search modes, deltas
    │   models.py             # Entity, Relation, CRUD + query models
    │   event.py              # Re-exports KnowledgeGraphStateEvent, this domain's delta
    │   kg_actor.py           # KnowledgeGraphActor
    │   state.py              # RootRow, KnowledgeGraphSummaryState — summary state + deltas
    │   └── kg_tool.py        # KnowledgeGraphTool ToolCard
    search/
    │   README.md           # SearchTool reference — Tavily parameters and ranges
    │   └── search.py         # SearchTool (Tavily)
    team/
    │   README.md           # TeamTool reference — hire/fire channels, activity gates
    │   team.py               # TeamTool — hire/fire/roster/profiles + get_team_activity
    │   state.py              # TeamMemberRow/TeamRosterState, RoleRow/RoleCatalogState
    │   observer.py           # TeamManagementToolObserver — TeamTool's own contract
    │   └── activity.py       # team_activity models, GetTeamActivity,
    │                         #   ActivitySummarizer, TeamActivityActor, SummarizerWorker
    metadata/
    │   README.md           # MetadataTool reference — template grammar, snapshot contract
    │   __init__.py           # Public exports: MetadataTool, RenderMetadata
    │   └── tool.py           # MetadataTool ToolCard + RenderMetadata; no actor, no state
    notification/
    │   README.md           # NotificationTool reference — message_class contract, delivery
    │   __init__.py           # Public exports: NotificationTool, its capability params,
    │                         #   NotificationActor, models
    │   models.py             # PendingNotification, NotificationConfig,
    │                         #   NotificationState, resolve_message_class
    │   actor.py              # NotificationActor singleton "#NotificationTool" + tick loop
    │   └── tool.py           # NotificationTool ToolCard + RegisterNotification,
    │                         #   PendingNotifications, CancelNotification
    skill/
    │   README.md           # SkillTool reference — the three channels, the loading model
    │   __init__.py           # Public exports: SkillEntry, Skills, SkillTool
    │   └── tool.py           # SkillTool ToolCard + SkillEntry, Skills, MENU_HEADER;
    │                         #   no actor, no state
    mailbox/
    │   README.md           # MailboxTool reference — params, the signal contract, /stop
    │   __init__.py           # Public exports: the card, its params, the observer protocol
    │   observer.py           # MailboxToolObserver — get_mailbox + consume_mailbox
    │   params.py             # ReadMailbox, Stop
    │   └── mailbox.py        # MailboxTool ToolCard — read_mailbox signal, preview
    │                         #   whitelist, idle /stop; no actor, no proxy round trip
    mcp/
    │   README.md           # MCPTool reference — transports, timeouts, diagnostics
    │   mcp.py                # MCPTool, connection configs
    │   └── oauth_handler.py  # OAuth 2.0 flow
    workspace/
        README.md           # WorkspaceTool reference — every capability parameter
        workspace.py          # Workspace Protocol, Filesystem, get_workspace()
        edit.py               # EditMatcher (7-strategy), FilePatch, parse_patch
        readers.py            # DocumentReader (Pydantic BaseModel), TEXT_EXTENSIONS
        └── tool.py           # WorkspaceTool ToolCard
    sandbox/
        README.md           # ExecTool reference — backends compared, allowlist, image
        __init__.py           # Public exports: ExecTool, SandboxActor subclasses, models
        actor.py              # SandboxActor (abstract), SandboxConfig, ALLOWED_COMMANDS
        local.py              # LocalSandboxActor (subprocess, resource limits)
        docker.py             # DockerSandboxActor (persistent container per team)
        seatbelt.py           # SeatbeltSandboxActor (macOS Apple Seatbelt)
        bwrap.py              # BwrapSandboxActor (Linux bubblewrap)
        tool.py               # ExecTool ToolCard, SANDBOX_ACTOR_CLASSES registry
        └── sandbox.Dockerfile # Bundled image definition for akgentic-sandbox:latest
tests/                        # Tests organised by domain
```

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://github.com/b12consulting/akgentic-tool/blob/master/LICENSE).

> **Dual licensing & CLA** — Akgentic is available under the AGPL-3.0 open-source license. A commercial license is also planned for organizations that require alternative terms. Contact [Yuma](https://www.weareyuma.com/en/contact) for more information. External contributions will be accepted once a Contributor License Agreement (CLA) is in place. Until then, please hold off on submitting pull requests.
