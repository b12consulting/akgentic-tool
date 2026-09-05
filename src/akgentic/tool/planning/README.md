# PlanningTool

A shared task board for a multi-agent team: one actor holds the plan, every agent reads and
writes it, and the plan is exposed as structured context state delivered into each agent's context
as per-turn deltas, so nobody has to ask what the team is doing.

```python
from akgentic.tool.planning import PlanningTool
```

| | |
|---|---|
| Module | `akgentic.tool.planning.planning` |
| Actor | `PlanActor`, singleton named `#PlanningTool` |
| Channels used | `LLM_CONTEXT`, `TOOL_CALL`, `COMMAND` |
| Depends on | `VectorStoreTool` — conditionally, see [`vector_store`](#vector_store) |
| Optional extras | `[vector_search]` for semantic search |

---

## The ToolCard

```python
class PlanningTool(ToolCard):
    # Vector-search wiring
    vector_store: bool | str = True
    collection: CollectionConfig = CollectionConfig()
    search_top_k: int = 10
    search_score_threshold: float = 0.5

    # Capabilities
    get_planning: GetPlanning | bool = True
    get_planning_task: GetPlanningTask | bool = True
    update_planning: UpdatePlanning | bool = True
    search_planning: SearchPlanning | bool = True

    @property
    def depends_on(self) -> list[str]:
        return ["VectorStoreTool"] if self.vector_store is not False else []
```

**The card is a thin proxy; the plan lives in an actor.** `observer()` asks the orchestrator for
`getChildrenOrCreate(PlanActor, config=PlanConfig(...))` — get-or-create, so every agent carrying
a `PlanningTool` binds to the *same* `#PlanningTool` singleton and sees the same task list. All
four capabilities are closures over that actor's ask proxy.

**`depends_on` is a property, not a field.** It returns `["VectorStoreTool"]` only when
`vector_store` is not `False`, so `ToolFactory`'s topological sort wires `VectorStoreTool` first
when it is needed and does not demand one when the tool runs keyword-only. Because it is a
property it never appears in `model_dump()` and cannot be set through `model_validate`.

---

## ToolCard fields

### `vector_store`

| Value | Effect |
|---|---|
| `True` *(default)* | Bind to the default `#VectorStore` actor. `depends_on` requires a `VectorStoreTool` in the team. |
| `"#VectorStore-RAG"` (any `str`) | Bind to that named singleton, created by `VectorStoreTool(vector_store_name=...)`. |
| `False` | Degraded mode: no vector wiring, no `depends_on`, `search_planning` runs keyword-only. |

The card never creates the vector store actor — `VectorStoreTool` owns it. `PlanActor` looks it up
by name during its own `on_start`. If the lookup fails, or `[vector_search]` is not installed, the
tool degrades to keyword-only search rather than failing.

### `collection`

A `CollectionConfig` forwarded to `VectorStoreActor.create_collection("planning", …)` when
`PlanActor` starts.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `dimension` | `int` | `1536` | Embedding dimensionality; must match the embedding model. |
| `backend` | `"inmemory" \| "weaviate"` | `"inmemory"` | `weaviate` requires `akgentic-tool[weaviate]`. |
| `tenant` | `str \| None` | `None` | Weaviate tenant id for multi-tenancy — usually the team id. |

### `search_top_k` / `search_score_threshold`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `search_top_k` | `int` | `10` | Default number of semantic hits. Overridable per call via `search_planning(top_k=…)`. |
| `search_score_threshold` | `float` | `0.5` | Minimum cosine similarity for a semantic hit. Overridable per call. Higher than `KnowledgeGraphTool`'s `0.3`: a task board wants precision, graph exploration wants recall. |

Both are propagated to `PlanConfig`, so the actor applies them when the per-call argument is
`None`.

---

## Capability parameters

### `GetPlanning` — the plan itself

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{LLM_CONTEXT, COMMAND}` | **Not a tool call by default.** The plan is context — structured context state delivered as per-turn deltas, not something the model must remember to fetch. Add `TOOL_CALL` to also offer it as a callable. |
| `filter_by_agent` | `bool` | `True` | When `True` the rendered plan lists only tasks the calling agent owns *or* created (and that have an owner). The team summary — totals and per-owner breakdown — is always shown. `False` lists every task. |

A card persisted with an explicit `expose: ["system_prompt", ...]` from before the move is
revalidated onto `LLM_CONTEXT` by the attached `normalize_system_prompt_to_llm_context` validator.

`filter_by_agent=True` is what keeps the context cost of a 200-task board bounded: every agent sees
the same two summary lines plus its own slice.

```
**Team planning:** 5 tasks total
Owners: @Alice: 3 | @Bob: 1 | unassigned: 1

**Your tasks** (owner or creator: @Alice):
- ID 3 [started] Implement auth module (Owner: @Alice, Creator: @Alice)
- ID 7 [pending] Review PR #42 — Output: pending (Owner: @Bob, Creator: @Alice)

Use get_planning_task(id) for exact ID lookup or search_planning(...) to filter tasks.
```

### `GetPlanningTask` — `get_planning_task(task_id: int)`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{TOOL_CALL, COMMAND}` | |

Exact lookup by integer id. No extra fields.

### `UpdatePlanning` — `update_planning(update: UpdatePlan)`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{TOOL_CALL}` | **Tool call only** — writing the plan is not offered on the `COMMAND` channel, and `get_commands()` does not register it even if you add `COMMAND` to `expose`. |

One batched mutation carries creates, updates and deletes:

```python
class UpdatePlan(BaseModel):
    create_tasks: list[TaskCreate] = []
    update_tasks: list[TaskUpdate] = []
    delete_tasks: list[int] = []
```

| Model | Fields |
|---|---|
| `TaskCreate` | `id: int`, `status`, `description` (**max 300 chars**), `owner: str` (empty ⇒ unassigned), `dependencies: list[int]` |
| `TaskUpdate` | `id: int` plus optional `status`, `description` (max 300), `output` (**max 150, silently truncated**), `owner`, `dependencies` |
| `Task` | `TaskCreate` + `output: str`, `creator: str`, `updated_at: datetime` |

`status` is `"pending" | "started" | "completed" | "abort"`.

`description` over 300 characters is a **validation error** the model must correct; `output` over
150 characters is **truncated to 147 characters plus `...`** by a `before` validator, so a verbose
result never fails a call. Both limits are stated in the tool docstring so the model respects them
before composing the call.

`creator` is never supplied by the model — the closure stamps it from the calling agent's address.

### `SearchPlanning` — `search_planning(...)`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{TOOL_CALL, COMMAND}` | |

```python
search_planning(
    query: str | None = None,
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid",
    status: TaskStatus | None = None,
    owner: str | None = None,
    creator: str | None = None,
    top_k: int | None = None,             # None -> search_top_k
    score_threshold: float | None = None, # None -> search_score_threshold
) -> list[str]
```

All filters are AND-combined; omitting everything returns the full list. The field filters are
applied **before** scoring, so `top_k` is never spent on tasks they would discard. Results are
tagged with how they were found — `(keyword match)`, `(semantic: 0.85)`, `(hybrid: 0.90)`; the
number is the raw cosine, not the fused score, since fusion is relative to the result set.

`top_k` caps the returned list, defaulting to `search_top_k`. The vector store itself is queried
for more than that, because fusion and the field filters both drop candidates.

`mode="keyword"` performs **no embedding call at all**, which is the mode to use when the query is
a known substring and you do not want to pay for an embedding.

---

## Configuration

### Wiring order

```python
from akgentic.tool.planning import PlanningTool
from akgentic.tool.vector_store import VectorStoreTool

ToolFactory([PlanningTool(), VectorStoreTool()], observer=agent)
# -> topologically sorted to [VectorStoreTool, PlanningTool]; order in the list is irrelevant
```

Listing `PlanningTool` with `vector_store=True` and **no** `VectorStoreTool` raises `ValueError`
at factory construction — fail fast at team creation, not at the first search.

### Recipes

```python
PlanningTool()                                               # defaults

PlanningTool(get_planning=GetPlanning(filter_by_agent=False))  # everyone sees every task

PlanningTool(get_planning=GetPlanning(expose={LLM_CONTEXT, TOOL_CALL, COMMAND}))
                                                             # also fetchable on demand

PlanningTool(vector_store=False)                             # keyword-only, no vector store needed

PlanningTool(update_planning=False)                          # read-only board for an observer agent

PlanningTool(                                                # persistent, multi-tenant board
    collection=CollectionConfig(backend="weaviate", tenant="team-42"),
    search_score_threshold=0.65,
)

PlanningTool(hybrid_alpha=0.3)                               # trust exact wording over similarity
```

> **The environment picks the backend; you only override it.** The connection is read from the
> environment at `observer()` time, never from the card — a catalog entry must not carry a cluster
> URL and an API key as plain configuration:
>
> ```bash
> export AKGENTIC_WEAVIATE_URL="https://your-cluster.weaviate.network"
> export AKGENTIC_WEAVIATE_API_KEY="..."          # omit for an unauthenticated cluster
> ```
>
> **Exporting the URL is what turns Weaviate on**, and a `CollectionConfig` that names no backend
> then defaults to `weaviate` rather than to the in-memory index. An exported but empty variable
> counts as unset. Requires `akgentic-tool[weaviate]`.
>
> Naming `backend="weaviate"` with no URL exported **raises at team creation** — the card asked for
> durable, shared, tenant-isolated storage, and a process-local index is the wrong answer to a
> question the deployment already settled, not a lesser one. Drop the setting to opt into memory.

### Semantic search

With `[vector_search]` installed, task descriptions are embedded on create and update and stored
in the `planning` collection of the bound `VectorStoreActor`.

`mode="hybrid"` fuses the keyword and semantic legs with the shared rule, Weaviate's
`relativeScoreFusion` at `alpha = 0.7`: `alpha * norm(cosine) + (1 - alpha) * keyword`. A strong
semantic hit therefore outranks a keyword-only one, and a task confirmed by both outranks either.
Set `hybrid_alpha` below `0.5` to put exact wording first. The rule and its consequences are
documented once in
[the vector store README](../vector_store/README.md#hybrid-search-lives-here-not-in-the-backends).

`score_threshold` gates the semantic leg only, on the raw cosine before fusion, so a keyword match
is never dropped by it.

Without the extra, or with `vector_store=False`, `search_planning` still answers — keyword and
field filters only. There is no error and no warning at call time; the degradation is by design.

### Import paths

```python
from akgentic.tool.planning import GetPlanning, GetPlanningTask, PlanningTool, UpdatePlanning
from akgentic.tool.planning.planning import SearchPlanning        # not re-exported from the package
from akgentic.tool.planning.planning_actor import Task, TaskCreate, TaskUpdate, UpdatePlan
```

---

See the [package README](../../../../README.md) for the `ToolCard` / `ToolFactory` machinery, the
channel system, and the tool-actor conventions (`#`-prefixed names, `getChildrenOrCreate`).
