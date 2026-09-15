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
| Depends on | nothing — the card creates its own store, see [`vector_store`](#vector_store) |
| Optional extras | `[vector_search]` for semantic search |

---

## The ToolCard

```python
class PlanningTool(ToolCard):
    # Vector-search wiring
    vector_store: VectorStoreParam | bool = True
    search_top_k: int = 10
    search_score_threshold: float = 0.5

    # Capabilities
    get_planning: GetPlanning | bool = True
    get_planning_task: GetPlanningTask | bool = True
    update_planning: UpdatePlanning | bool = True
    search_planning: SearchPlanning | bool = True
```

**The card is a thin proxy; the plan lives in an actor.** `observer()` asks the orchestrator for
`getChildrenOrCreate(PlanActor, config=PlanConfig(...))` — get-or-create, so every agent carrying
a `PlanningTool` binds to the *same* `#PlanningTool` singleton and sees the same task list. All
four capabilities are closures over that actor's ask proxy.

**This card declares no `depends_on`, because it owns its own storage.** `observer()` creates the
store actor itself — when the backend needs one — immediately before it creates the `PlanActor`
that will look it up, so the ordering that a dependency edge between two cards used to enforce is
now two lines in one method. There is no second card to add to the team and nothing for
`ToolFactory`'s topological sort to order.

---

## ToolCard fields

### `vector_store`

A `VectorStoreParam` or a `bool`, saying where the plan's vectors live and how they are embedded,
in one of three shapes:

| Written | Means |
|---|---|
| `True` (the default) | an enabled store with default settings |
| `False` | no store at all — no store actor, no embedding, semantic search absent |
| `VectorStoreParam(…)` | an enabled store configured explicitly |

The card keeps what the author wrote, verbatim, and normalises it at the point of use:
`resolve_store_param` turns `True` into a fresh `VectorStoreParam()` and `False` into `None`, so
everything below the card sees only two shapes. The resolved value is forwarded to `PlanConfig` and
is what `PlanActor` resolves its storage engine from at `on_start`, calling
`create_collection("planning", …)` on whatever it resolves.

**Why the bool is not expanded on the card.** `VectorStoreParam.backend` resolves from the
environment *per instantiation*, so coercing `True` into a param at validation time would write the
build environment's backend into a stored catalog record whose author wrote `true` — and catalogs
are routinely promoted between tiers.

**The backend decides whether an actor is involved at all.** An actor-state backend — the in-memory
index, whose data *is* the store actor's state — gets a store actor, created by this card's
`observer()` before the `PlanActor` that looks it up. A cluster backend gets none: the data lives on
the cluster, so `PlanActor` builds the backend through the registered factory and talks to the
process's shared client directly.

If the store cannot be resolved, or the collection cannot be created, or `[vector_search]` is not
installed, the tool degrades to keyword-only search rather than failing — one WARNING, and
`search_planning` still answers from its keyword leg.

**Only half of what this field used to mean is gone.** Before epic 49 it was a `bool | str` doing
two jobs: the string named *which* `VectorStoreActor` to look up, and the bool said whether to have
a store at all. The lookup is gone for good — a card writing `vector_store: "#VectorStore-RAG"`
fails validation, because the actor it addressed no longer exists — and the configuration that used
to live on a separate `collection` field now lives in the param, so a persisted card carrying
`collection:` silently takes the default. The **bool** is not gone: it is the opt-out described
above, and every stored card writing `vector_store: false` keeps loading.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `dimension` | `int` | `1536` | Embedding dimensionality; must be the native width of a known `embedding_model`, refused at bind otherwise. |
| `backend` | `str` | `default_backend()` | Any **registered** backend name — `inmemory`, `weaviate`, `qdrant`, or one a deployment registers itself. Not a closed union: the set is the registry's, so a name nobody registered fails the build rather than silently taking a branch. The two cluster backends need `akgentic-tool[weaviate]` / `[qdrant]`. |
| `tenant` | `str \| None` | `None` | Tenant id for multi-tenancy — usually the team id. Native on Weaviate; a payload field on Qdrant. |
| `params` | `dict[str, Any]` | `{}` | Backend-native settings passed through untouched. Schemaless by nature, so it is the one place `Any` is the honest type. |
| `embedding_model` | `str` | `"text-embedding-3-small"` | The model that produces the collection's vectors. |
| `embedding_provider` | `Literal["openai", "azure"]` | `"openai"` | The embedding API provider. |

**`default_backend()` is resolved per instantiation, not at import.** It asks every registered
backend whether the environment has provisioned it, so a card that names no backend lands wherever
the deployment actually is: a configured cluster is deployed to be used, and a collection with no
opinion should not quietly receive a process-local index that disappears with the actor.

**`planning` is a team-scoped collection**, so every search and every removal the store issues
carries this team's id as a predicate. That is not true of every collection in the package — the
workspace's `workspace_chunks` is shared across teams, because its rows belong to a filesystem tree
rather than to a team — but it is true of this one, and it is declared by name in
`vector_store/protocol.py` rather than by anything a catalog author can set.

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

ToolFactory([PlanningTool()], observer=agent)
# -> no ordering to arrange: the card creates its own store inside observer()
```

There is no second card to list and no order to get wrong. What still fails fast at team creation
is a card naming a backend the environment has not provisioned — a cluster URL that is not
exported, or a dimension contradicting the embedding model — which `observer()` refuses with a
`ValueError` rather than degrading at the first search.

### Recipes

```python
PlanningTool()                                               # defaults

PlanningTool(get_planning=GetPlanning(filter_by_agent=False))  # everyone sees every task

PlanningTool(get_planning=GetPlanning(expose={LLM_CONTEXT, TOOL_CALL, COMMAND}))
                                                             # also fetchable on demand

PlanningTool(update_planning=False)                          # read-only board for an observer agent

PlanningTool(                                                # persistent, multi-tenant board;
    vector_store=VectorStoreParam(backend="weaviate", tenant="team-42"),
    search_score_threshold=0.65,                             # no store actor is created for it
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
> **Exporting the URL is what turns Weaviate on**, and a `VectorStoreParam` that names no backend
> then defaults to `weaviate` rather than to the in-memory index. An exported but empty variable
> counts as unset. Requires `akgentic-tool[weaviate]`.
>
> Naming `backend="weaviate"` with no URL exported **raises at team creation** — the card asked for
> durable, shared, tenant-isolated storage, and a process-local index is the wrong answer to a
> question the deployment already settled, not a lesser one. Drop the setting to opt into memory.

### Semantic search

With `[vector_search]` installed, task descriptions are embedded on create and update and stored
in the `planning` collection of whatever storage engine the card's `vector_store` resolves — the
store actor on an actor-state backend, the backend itself on a cluster one.

`mode="hybrid"` fuses the keyword and semantic legs with the shared rule, Weaviate's
`relativeScoreFusion` at `alpha = 0.7`: `alpha * norm(cosine) + (1 - alpha) * keyword`. A strong
semantic hit therefore outranks a keyword-only one, and a task confirmed by both outranks either.
Set `hybrid_alpha` below `0.5` to put exact wording first. The rule and its consequences are
documented once in
[the vector store README](../vector_store/README.md#hybrid-search-lives-here-not-in-the-backends).

`score_threshold` gates the semantic leg only, on the raw cosine before fusion, so a keyword match
is never dropped by it.

Without the extra, or whenever the store cannot be resolved or its collection cannot be created,
`search_planning` still answers — keyword and field filters only. There is no error and no warning
at call time; the degradation is by design.

**To switch the vector store off deliberately, write `vector_store=False`.** Nothing is probed at
bind — a card naming an unprovisioned cluster and then switched off does not fail its team's build —
no store actor is created, and nothing is embedded. The capability is **not** unregistered, because
most of it still works:

```python
PlanningTool(vector_store=False)   # keyword and hybrid search still answer from the task list
```

| Mode | With `vector_store=False` |
|---|---|
| `"keyword"` | unchanged — substring matching over the task list |
| `"hybrid"` | unchanged — the keyword leg answers, the semantic leg is empty |
| `"vector"` | returns `[SEMANTIC_DISABLED]`, one sentence saying semantic search is off |

The sentence matters: an empty list would read as "no task matches" rather than "there is no index".
A store that was *asked for* and could not be built keeps returning empty results, so a real
misconfiguration is still visible rather than hidden behind a reassuring sentence. The two cases are
also distinguishable in the logs — a declined store logs one line at bind saying it is off by
configuration.

### Import paths

```python
from akgentic.tool.planning import GetPlanning, GetPlanningTask, PlanningTool, UpdatePlanning
from akgentic.tool.planning.planning import SearchPlanning        # not re-exported from the package
from akgentic.tool.planning.planning_actor import Task, TaskCreate, TaskUpdate, UpdatePlan
```

---

See the [package README](../../../../README.md) for the `ToolCard` / `ToolFactory` machinery, the
channel system, and the tool-actor conventions (`#`-prefixed names, `getChildrenOrCreate`).
