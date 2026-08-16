# akgentic-tool

[![CI](https://github.com/b12consulting/akgentic-tool/actions/workflows/ci.yml/badge.svg)](https://github.com/b12consulting/akgentic-tool/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/gpiroux/c0f2e0aa0a8c184ee8823dd4feefddd5/raw/coverage.json)](https://github.com/b12consulting/akgentic-tool/actions/workflows/ci.yml)

Tool infrastructure and domain tools for the [Akgentic](https://github.com/b12consulting/akgentic-framework)
multi-agent framework (open-source bundle). Define, compose, and expose capabilities to LLM agents through a unified
channel system — as tool calls, system prompt injections, or programmatic commands.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Migration: moved import paths](#migration-moved-import-paths)
- [Deferred Results: Never Block a Tool Actor](#deferred-results-never-block-a-tool-actor)
- [Channel System](#channel-system)
- [Tool Catalog](#tool-catalog)
  - [WorkspaceTool](#workspacetool)
  - [PlanningTool](#planningtool)
  - [KnowledgeGraphTool](#knowledgegraphtool)
  - [SearchTool](#searchtool)
  - [TeamTool](#teamtool)
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
  invokes it), a `SYSTEM_PROMPT` (injected into context before each LLM call), or a `COMMAND`
  (programmatic call from another agent or the orchestrator)
- **Observer protocols** — `ToolObserver`, `ActorToolObserver`, and `TeamManagementToolObserver`
  give tools access to the actor system, event emission, and team lifecycle hooks.
  > **Migration note (ADR-018):** `ToolCallEvent` has been removed from `akgentic-tool`. Tool call
  > observability is now handled by `akgentic-llm`. Import `ToolCallEvent` and `ToolReturnEvent`
  > from `akgentic.llm.event` instead.
- **RetriableError** — tools signal recoverable failures; `ToolFactory` translates them to the
  framework-specific retry exception without coupling tool logic to pydantic-ai
- **Domain tools** — eight production-ready tool implementations covering workspace I/O, task
  planning, knowledge graph, web search, team management, vector-store configuration, MCP server
  integration, and sandboxed shell execution

```
ToolCard(s)
    │
    ▼
ToolFactory
    │
    ├── get_tools()          → list[Callable]  ─────▶ LLM ReAct loop
    ├── get_system_prompts() → list[Callable]  ─────▶ injected into LLM context
    ├── get_commands()       → dict[type, Callable] ▶ orchestrator / other agents
    └── get_toolsets()       → list[Any]       ─────▶ pydantic-ai toolset objects
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
prompts = factory.get_system_prompts() # dynamic context injections
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
│  vector_store │ mcp │ sandbox                                    │
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

## Migration: moved import paths

Two modules were reorganised: `akgentic.tool.event` was split by audience, and
`akgentic.tool.vector` moved next to the code built on it. **Every old path below still
works.** Each one now resolves through a compatibility façade that emits a
`DeprecationWarning` on **attribute access** — not at import time, so code that touches
none of these symbols is never warned. **No removal release is scheduled.**

**Importing from the `akgentic.tool` package root needs no migration at all.** That surface
is unchanged, and reaching a symbol through it emits no warning.

| Old path | New home | Tier |
|---|---|---|
| `akgentic.tool.event.ToolStateEvent` | `akgentic.tool.core.event` | Stable |
| `akgentic.tool.event.CommandArg` | `akgentic.tool.core.event` | Stable |
| `akgentic.tool.event.CommandDescriptor` | `akgentic.tool.core.event` | Stable |
| `akgentic.tool.event.CommandsAnnouncedEvent` | `akgentic.tool.core.event` | Stable |
| `akgentic.tool.event.ToolObserver` | `akgentic.tool.core.observer` | Stable |
| `akgentic.tool.event.ActorToolObserver` | `akgentic.tool.core.observer` | Stable |
| `akgentic.tool.event.TeamManagementToolObserver` | `akgentic.tool.team.observer` | Internal |
| `akgentic.tool.event.ToolStatePayload` | `akgentic.tool.knowledge_graph.event` | Internal |
| `akgentic.tool.vector.VectorEntry` | `akgentic.tool.vector_store.vector` | Internal |
| `akgentic.tool.vector.EmbeddingService` | `akgentic.tool.vector_store.vector` | Internal |
| `akgentic.tool.vector.VectorIndex` | `akgentic.tool.vector_store.vector` | Internal |
| `akgentic.tool.vector._check_vector_search_dependencies` | `akgentic.tool.vector_store.vector` | Internal |

### What the two tiers mean

The tier is not a measure of how important a symbol is. It says **whether its import path is
something you may build against**, and the two answers carry different promises.

**Stable — a supported surface.** These are the contracts a custom `ToolCard` author outside
this package writes against: the `akgentic.tool` package root, the core abstractions
(`ToolCard`, `BaseToolParam`, `ToolFactory`, `Channels`, `CommandRegistry`), the *global*
observers `ToolObserver` and `ActorToolObserver`, `ToolStateEvent`, and the command-discovery
models. Their import paths are part of the API. If one moves, it is shimmed, and the shim is
kept.

**Internal — not a surface.** These belong to one specific tool: `TeamManagementToolObserver`
is `TeamTool`'s contract, `ToolStatePayload` is the knowledge graph's, the vector primitives
are `vector_store`'s. They move freely with the tool that owns them. Their rows above are a
**courtesy, not a guarantee** — the shim entry exists because removing a working import for
no reason is rude, not because the path was ever promised. Treating it as a promise would
freeze this package's internal structure by accident, which is exactly what the split was
done to avoid.

If one of your imports is in the Internal tier, move it now rather than relying on the row.

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

## Channel System

The `Channels` enum (`TOOL_CALL`, `SYSTEM_PROMPT`, `COMMAND`) controls how a capability is
surfaced. Each `BaseToolParam` subclass declares its `expose` set. A single capability can
appear on multiple channels simultaneously.

| Channel | Consumer | Invocation |
|---|---|---|
| `TOOL_CALL` | LLM agent | Called by the LLM during the ReAct loop |
| `SYSTEM_PROMPT` | LLM context | Injected as dynamic content before each LLM call |
| `COMMAND` | Orchestrator / agents | Called programmatically via `proxy_call` |

```python
class GetPlanning(BaseToolParam):
    expose: set[Channels] = {SYSTEM_PROMPT, COMMAND}  # never a direct LLM call

class GetPlanningTask(BaseToolParam):
    expose: set[Channels] = {TOOL_CALL, COMMAND}      # LLM + programmatic
```

`BaseToolParam.instructions` appends runtime guidance to a tool's docstring without modifying
source — useful for injecting team-specific constraints at configuration time.

## Tool Catalog

### WorkspaceTool

Sandboxed read/write access to a shared team filesystem. A single `WorkspaceTool` class covers
both read-only and full access via a `read_only: bool` field.

```python
from akgentic.tool import WorkspaceTool

WorkspaceTool()                          # full access (default)
WorkspaceTool(read_only=True)            # read tools only
WorkspaceTool(workspace_id="shared")     # shared workspace across teams
WorkspaceTool(read_only=True, workspace_glob=False)  # fine-grained capability control
```

| Tool | Description |
|---|---|
| `workspace_read` | Read file with line-number pagination; auto-converts PDF, DOCX, XLSX, images to Markdown |
| `workspace_list` | List directory (flat or ASCII tree by depth) |
| `workspace_glob` | Find files by glob pattern with `{py,ts}` brace expansion; results sorted by mtime |
| `workspace_grep` | Regex search across files; uses `rg` if available, falls back to Python |
| `workspace_view` | View image as `BinaryContent` for LLM vision (PNG, JPG, WebP, GIF, BMP) |
| `workspace_write` | Overwrite or create a file; auto-detects CRLF/LF line endings |
| `workspace_edit` | Surgical find-and-replace with 7-strategy cascade (exact → fuzzy, threshold 0.85) |
| `workspace_multi_edit` | Apply multiple `EditItem` operations across files in one call |
| `workspace_patch` | Apply unified diff patch (GNU format) |
| `workspace_delete` | Delete a file |
| `workspace_mkdir` | Create directory tree (parents included, idempotent) |

The workspace root is resolved from `AKGENTIC_WORKSPACES_ROOT` (default `./workspaces`). All
path operations validate against the root — traversal attacks (`../`) raise `RetriableError`.

**Binary file reading** (requires `akgentic-tool[docs]`): `workspace_read` transparently
handles PDF, DOCX, XLSX, PPTX, and images via MarkItDown. A sidecar cache (`.report.pdf.md`)
avoids re-extraction on subsequent reads.

**Image viewing** (requires `akgentic-tool[vision]`): `workspace_view` delivers raw pixels
to the model's vision endpoint. Images are optionally resized (default `max_dimension=1568`)
with a sidecar cache for the resized version.

### PlanningTool

Shared actor-based task board for multi-agent teams. A singleton `PlanActor` (named
`#PlanningTool`) lives in the orchestrator and persists across all agents' tool calls.

```python
from akgentic.tool.planning import PlanningTool

PlanningTool()                                    # default config
PlanningTool(get_planning=GetPlanning(filter_by_agent=False))  # show all tasks
```

| Capability | Default channel | Description |
|---|---|---|
| `get_planning` | `SYSTEM_PROMPT`, `COMMAND` | Team plan injected into LLM context; scoped to calling agent by default |
| `get_planning_task` | `TOOL_CALL`, `COMMAND` | Look up a single task by integer ID |
| `update_planning` | `TOOL_CALL` | Batch create / update / delete tasks in one call |
| `search_planning` | `TOOL_CALL`, `COMMAND` | Filter tasks by status, owner, creator, or natural-language query |

**Task model constraints:** `description` max 300 chars; `output` max 150 chars (auto-truncated
if exceeded — no `ValidationError`). Constraints are stated explicitly in the tool schema so
LLMs respect them before composing a call.

**Semantic search** (requires `akgentic-tool[vector_search]`): task descriptions are embedded
on create/update. `search_planning(query=...)` runs keyword UNION semantic search (cosine ≥ 0.5,
top_k=20). Degrades gracefully to keyword-only when vector deps are absent.

```python
# System prompt output example (filter_by_agent=True)
"""
**Team planning:** 5 tasks total
Owners: @Alice: 3 | @Bob: 1 | unassigned: 1

**Your tasks** (owner or creator: @Alice):
- ID 3 [started] Implement auth module (Owner: @Alice, Creator: @Alice)
- ID 7 [pending] Review PR #42 — Output: pending (Owner: @Bob, Creator: @Alice)

Use get_planning_task(id) for exact ID lookup or search_planning(...) to filter tasks.
"""
```

### KnowledgeGraphTool

Persistent actor-based knowledge graph for structured entity and relationship storage with
hybrid keyword + semantic search.

```python
from akgentic.tool.knowledge_graph import KnowledgeGraphTool

KnowledgeGraphTool()
```

Exposes `get_graph`, `update_graph`, and `search_graph` capabilities. Entities and relations
are stored in a `KnowledgeGraphActor`. Semantic search uses the shared `VectorIndex`
infrastructure (requires `akgentic-tool[vector_search]`).

### SearchTool

Web search and content fetching via the [Tavily](https://tavily.com/) API.

```python
from akgentic.tool.search import SearchTool

SearchTool()
```

| Tool | Description |
|---|---|
| `web_search` | Tavily search — returns titles, URLs, and snippets |
| `web_fetch` | Fetch and extract clean text from a URL (Tavily extract) |
| `web_crawl` | Crawl a URL and return structured content |

Requires `TAVILY_API_KEY` environment variable.

### TeamTool

Exposes team management capabilities (hire/fire agents, roster view) to the LLM, and — opt-in —
answers *who is working right now, and on what*. Used by `BaseAgent` in `akgentic-agent` to enable
orchestrator-level agents to dynamically extend the team.

```python
from akgentic.tool.team import ActivitySummarizer, GetTeamActivity, TeamTool

TeamTool()                                      # hire/fire/roster/profiles — no actor
TeamTool(get_team_activity=True)                # + who_is_working(), truncation only, still no actor
TeamTool(get_team_activity=GetTeamActivity(     # + summaries on demand; #TeamActivity is created
    summarizer=ActivitySummarizer(model="openai:gpt-5.2-mini"),
))
```

Requires a `TeamManagementToolObserver` (provided by `BaseAgent`). Surfaces agent roster and
available profiles as a system prompt; `hire_members(roles)` and `fire_members(names)` as tool calls.
The single-member `hire_member(role, name=None)` and `fire_member(name)` are `COMMAND`-channel
variants, not tool calls.

#### Team activity — `who_is_working`

`get_team_activity` **defaults to `False`**, so an existing card keeps its behaviour and its surface
byte-for-byte. Two independent gates then decide what turning it on costs:

| Configuration | `who_is_working` | `#TeamActivity` actor | model call | `summarize_over` in the schema |
|---|---|---|---|---|
| `get_team_activity=False` *(default)* | not exposed | not created | never | n/a |
| `get_team_activity=True`, `summarizer=None` | exposed, truncates | **not created** | never | **absent** |
| `summarizer=ActivitySummarizer(...)` | exposed, summarizes | created | on demand | present |

The `#TeamActivity` cache actor is created **only** when `get_team_activity` resolves truthy **and**
its `summarizer` is not `None`. The actor exists solely to cache summaries, so with the capability on
and no summarizer there is nothing to cache: `who_is_working` answers by truncation and **no actor is
created at all**.

The signature follows the configuration rather than being fixed. Without a summarizer the callable is
`who_is_working() -> TeamActivityReport`, and `summarize_over` is **absent from the tool schema** —
not merely defaulted off — so the model cannot request a summary nothing could produce. With one
configured it becomes `who_is_working(summarize_over: int | None = None)`, and **`summarize_over=None`
still performs zero model calls**: long task text is truncated to `max_task_chars`. Passing an integer
is the opt-in — only longer tasks go through the deferred-result cache above, keyed by `message_id` so
a follow-up call costs nothing. The threshold *is* the consent; there is no eager warming.

Busy members are derived from the orchestrator's own telemetry: an agent with a `ReceivedMessage` and
no matching `ProcessedMessage` is mid-handler, and the task text comes from the corresponding
`SentMessage`. Three behaviours worth knowing:

- **Busy means exactly one open message.** Actors are sequential, so the open count is structurally
  0 or 1; a higher count is reported as `suspect` rather than as plain "working", and never dropped.
- **Stale entries are dropped.** A resumed team replays telemetry that can be permanently unbalanced
  (a message received before the stop, processed never). Anything open longer than
  `stale_after_seconds` (default 300 s) is excluded rather than reported as a phantom worker.
- **The caller, tool actors, and the user proxy never appear.** The caller is excluded by `agent_id`,
  so a rename cannot slip it through; a human proxy waiting on input is not working.

`GetTeamActivity` also carries `expose` (`TOOL_CALL`, `COMMAND`) and `max_task_chars` (default 200),
the budget for reported task text; `ActivitySummarizer` carries `poll_attempts` (5) and
`poll_delay_seconds` (0.4). Its `model` is a pydantic-ai model spec string rather than the framework's
`ModelConfig`, because `akgentic-tool` does not depend on `akgentic-llm` — so those tokens are
produced outside `ReactAgent` and are counted by neither its cost accounting nor its usage limits.

### MCPTool

Integrates external [Model Context Protocol](https://modelcontextprotocol.io) servers as native
pydantic-ai toolsets over three transports: `streamable-http` (default), `sse`, and `stdio`.

`MCPTool` takes exactly one connection, on a required singular `connection` field:

```python
from akgentic.tool.mcp import MCPTool, MCPHTTPConnectionConfig

# Remote server over streamable HTTP (the default transport)
MCPTool(
    connection=MCPHTTPConnectionConfig(
        url="https://mcp.acme.example/api/v1/endpoint",
    )
)
```

The transport is always taken from the config, never inferred from the URL. pydantic-ai's
own inference only recognises URLs ending in `/sse`, which would silently downgrade any SSE
endpoint published on another path — so `sse` must be requested explicitly:

```python
# Server-Sent Events — `transport="sse"` is required, a /sse suffix is not enough
MCPTool(
    connection=MCPHTTPConnectionConfig(
        url="https://mcp.acme.example/api/v1/endpoint",
        transport="sse",
        bearer_token="...",       # sent as an Authorization header on the transport
        read_timeout=900.0,       # also governs how long the event stream tolerates silence
    )
)
```

```python
from akgentic.tool.mcp import MCPTool, MCPStdioConnectionConfig

# Local server launched as a subprocess — `stdio_command` is required
MCPTool(
    connection=MCPStdioConnectionConfig(
        stdio_command="uvx",
        stdio_args=["acme-mcp-server"],
        tool_prefix="acme",       # applied via the toolset's prefixed() wrapper
    )
)
```

`get_tools()` is always empty — MCP capabilities reach the agent through `get_toolsets()`,
which returns a single toolset and lets pydantic-ai handle schema resolution and dispatch.
Setting `tool_prefix` wraps that toolset in a `PrefixedToolset`.

For servers that answer `401` with an MCP `WWW-Authenticate` challenge, `mcp/oauth_handler.py`
runs a browser-based authorization flow. Note that it stops at the **authorization code** —
exchanging that code for an access token is not implemented, so the returned value is the code
itself. The helpers are not wired into `MCPTool`; call them yourself and pass the result as
`bearer_token`.

### ExecTool

Sandboxed shell command execution inside the team workspace. A single `SandboxActor` is spawned
per team and reused across all `ExecTool` calls. The backend is selected via the `mode` field.

```python
from akgentic.tool.sandbox.tool import ExecTool

ExecTool()                        # auto mode (default — probe: bwrap → seatbelt → docker → local)
ExecTool(mode="local")            # local mode (subprocess, no filesystem isolation)
ExecTool(mode="bwrap")            # Linux bubblewrap (filesystem namespace isolation)
ExecTool(mode="seatbelt")         # macOS Apple Seatbelt (sandbox-exec profile)
ExecTool(mode="docker")           # persistent Docker container per team
ExecTool(workspace_id="shared")   # share workspace directory with WorkspaceTool
```

**Sandbox modes:**

| Mode | Platform | Isolation | Requirement |
|---|---|---|---|
| `local` | Any | None — subprocess only | No extra tools needed |
| `bwrap` | Linux | Filesystem namespace (bubblewrap) | `bwrap` on PATH |
| `seatbelt` | macOS | Apple Seatbelt profile (`sandbox-exec`) | `sandbox-exec` on PATH |
| `docker` | Any | Persistent container per team | Docker daemon on PATH |
| `auto` | Any | Best available (probe order: bwrap → seatbelt → docker → local) | Automatic |

**Allowed commands** (enforced by `ALLOWED_COMMANDS` allowlist — first token only):

`python`, `python3`, `pytest`, `ruff`, `mypy`, `git`, `uv`, `pip`, `cat`, `ls`, `find`,
`grep`, `mkdir`, `cp`, `mv`, `rm`, `echo`, `touch`, `curl`, `wget`, `make`, `bash`, `sh`,
`node`, `npm`, `npx`

**Auto-mode probe order (`_resolve_auto_mode()`):** When `mode="auto"`, the function probes
the host at `ExecTool.observer()` call time in the following order: `bwrap` (Linux bubblewrap)
→ `seatbelt` (macOS `sandbox-exec`) → `docker` → `local` (fallback, no isolation). If `local`
is selected as the fallback, a `DeprecationWarning` is emitted to alert that no isolation
backend was found.

**Platform notes:**

- **RLIMIT_AS on Darwin:** The `local` mode sets `RLIMIT_AS` (virtual address space limit) to
  512 MB on Linux but skips this resource limit on macOS/Darwin, where `RLIMIT_AS` is not
  reliably enforceable. CPU time and file size limits are applied on all platforms.
- **Seatbelt DeprecationWarning:** `SeatbeltSandboxActor._start_sandbox()` emits a
  `DeprecationWarning` because `sandbox-exec` is deprecated since macOS 10.15 Catalina and
  may be removed in a future macOS release. The seatbelt mode is intended for macOS developer
  workstations only.

**Docker sandbox image:** The `docker` mode (and `auto` when it resolves to Docker) runs containers
from the image `akgentic-sandbox:latest` (the `SANDBOX_IMAGE` constant in `sandbox/docker.py`). The
image is **built automatically on first use** by `DockerSandboxActor._ensure_image()` from the bundled
`sandbox.Dockerfile` (Python 3.12 + pytest/ruff/mypy, uv, Node.js 18) — no manual step is required.
The build runs once; Docker's layer cache makes later container starts instant.

**Pre-built / CI image (`AKGENTIC_SANDBOX_IMAGE`):** Set `AKGENTIC_SANDBOX_IMAGE=<name>` to use a
pre-built or registry image. When set, the auto-build check is skipped and that image is used directly
— recommended for CI and production, where the image is pre-built and pushed to a registry.

To pre-build the image manually (optional — e.g. to warm the cache before first use):

```bash
docker build \
  -f packages/akgentic-tool/src/akgentic/tool/sandbox/sandbox.Dockerfile \
  -t akgentic-sandbox:latest \
  packages/akgentic-tool/src/akgentic/tool/sandbox
```

**Error handling:** All errors from the sandbox backend surface as a `SandboxError` string
returned to the LLM (never raised). Disallowed commands return a `CommandNotAllowedError`
string listing the allowed commands.

```python
# Example tool response for a disallowed command:
# "CommandNotAllowedError: Command 'curl' is not in the allowed commands list.
#  Allowed: ['bash', 'cat', 'cp', ...]"

# Example tool response for a backend failure:
# "SandboxError: TimeoutExpired: Command 'python main.py' timed out after 30s"
```

**`SANDBOX_ACTOR_CLASSES` registry:** The backend registry is a mutable `dict[str, type[SandboxActor]]`
exposed at `akgentic.tool.sandbox.tool.SANDBOX_ACTOR_CLASSES`. Infrastructure packages (e.g.,
`akgentic-infra`) can inject additional backends at import time before any `ExecTool` is
constructed:

```python
from akgentic.tool.sandbox.tool import SANDBOX_ACTOR_CLASSES
from my_infra.e2b_actor import E2BSandboxActor

SANDBOX_ACTOR_CLASSES["e2b"] = E2BSandboxActor  # now available as ExecTool(mode="e2b")
```

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
    │   channels.py           # Channels enum: TOOL_CALL, SYSTEM_PROMPT, COMMAND
    │   params.py             # BaseToolParam
    │   card.py               # ToolCard
    │   dependencies.py       # Topological ordering of cards by depends_on
    │   commands.py           # CommandRegistry
    │   factory.py            # ToolFactory
    │   event.py              # ToolStateEvent, CommandArg, CommandDescriptor,
    │   │                     #   CommandsAnnouncedEvent — package-global contracts
    │   observer.py           # ToolObserver, ActorToolObserver — the global observers
    │   └── deferred.py       # DeferredResultActor, DeferredWorker, poll_deferred
    │                         #   NOT on the façade — import akgentic.tool.core.deferred
    errors.py                 # RetriableError
    event.py                  # Compatibility façade only — the symbols that lived here
    │                         #   moved to core/, team/ and knowledge_graph/.
    │                         #   See "Migration: moved import paths"
    vector.py                 # Compatibility façade only — moved to
    │                         #   vector_store/vector.py. See the migration table
    vector_store/
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
    │   planning_actor.py     # Task models, PlanConfig, PlanActor
    │   └── planning.py       # PlanningTool ToolCard
    knowledge_graph/
    │   models.py             # Entity, Relation, CRUD + query models
    │   event.py              # ToolStatePayload — this domain's delta payload typing
    │   kg_actor.py           # KnowledgeGraphActor
    │   └── kg_tool.py        # KnowledgeGraphTool ToolCard
    search/
    │   └── search.py         # SearchTool (Tavily)
    team/
    │   team.py               # TeamTool — hire/fire/roster/profiles + get_team_activity
    │   observer.py           # TeamManagementToolObserver — TeamTool's own contract
    │   └── activity.py       # who_is_working models, GetTeamActivity,
    │                         #   ActivitySummarizer, TeamActivityActor, SummarizerWorker
    mcp/
    │   mcp.py                # MCPTool, connection configs
    │   └── oauth_handler.py  # OAuth 2.0 flow
    workspace/
        workspace.py          # Workspace Protocol, Filesystem, get_workspace()
        edit.py               # EditMatcher (7-strategy), FilePatch, parse_patch
        readers.py            # DocumentReader (Pydantic BaseModel), TEXT_EXTENSIONS
        └── tool.py           # WorkspaceTool ToolCard
    sandbox/
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
