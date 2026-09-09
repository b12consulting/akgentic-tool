# KnowledgeGraphTool

A shared, persistent knowledge graph for a team: entities and typed relations, mutated in
batches, searched by keyword, vector or both, and summarised into structured context state
delivered to every agent as per-turn deltas.

```python
from akgentic.tool.knowledge_graph import KnowledgeGraphTool
```

| | |
|---|---|
| Module | `akgentic.tool.knowledge_graph.kg_tool` |
| Actor | `KnowledgeGraphActor`, singleton named `#KnowledgeGraphTool` |
| Channels used | `LLM_CONTEXT`, `TOOL_CALL`, `COMMAND` |
| Depends on | nothing — the card creates its own store, see [`vector_store`](#vector_store) |
| Required extra | `[vector_search]` — checked at `observer()` time, see [below](#the-dependency-check-is-unconditional) |

---

## The ToolCard

```python
class KnowledgeGraphTool(ToolCard):
    # Vector-search wiring
    vector_store: VectorStoreParam = VectorStoreParam()
    search_top_k: int = 10
    search_score_threshold: float = 0.3
    hybrid_alpha: float = 0.7

    # Capabilities
    get_graph: GetGraph | bool = True
    update_graph: UpdateGraph | bool = True
    search: SearchGraph | bool = True

    # Write gate
    read_only: bool = False
```

**Same shape as `PlanningTool`, different domain.** `observer()` calls
`getChildrenOrCreate(KnowledgeGraphActor, config=KnowledgeGraphConfig(...))`, so every agent
carrying the card binds to the one `#KnowledgeGraphTool` singleton and shares one graph. The card
holds no graph state; its methods are formatting helpers over the actor's ask proxy.

**`read_only` gates only `update_graph`.** `get_graph` and `search_graph` are built regardless.

---

## ToolCard fields

### `vector_store`

A `VectorStoreParam`: the one object saying where the graph's vectors live and how they are
embedded. It is forwarded to `KnowledgeGraphConfig` and is what `KnowledgeGraphActor` resolves its
storage engine from at `on_start`, calling `create_collection("knowledge_graph", …)` on whatever it
resolves.

**The backend decides whether an actor is involved at all.** An actor-state backend — the in-memory
index, whose data *is* the store actor's state — gets a store actor, created by this card's
`observer()` before the `KnowledgeGraphActor` that looks it up. A cluster backend gets none: the
data lives on the cluster, so the actor builds the backend through the registered factory and talks
to the process's shared client directly. This card declares no `depends_on`; the ordering a
dependency edge used to enforce between two cards is now two lines in one method.

**This field used to mean something else.** Before epic 49 it was a `bool | str` naming *which*
`VectorStoreActor` to look up, and the configuration lived on a separate `collection` field. Both
are gone: a persisted card carrying `vector_store: true` now fails validation rather than being
ignored, and one carrying `collection:` silently takes the default.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `dimension` | `int` | `1536` | Embedding dimensionality; must be the native width of a known `embedding_model`, refused at bind otherwise. |
| `backend` | `str` | `default_backend()` | Any **registered** backend name — `inmemory`, `weaviate`, `qdrant`, or one a deployment registers itself. Not a closed union: the set is the registry's, so a name nobody registered fails the build rather than silently taking a branch. The two cluster backends need `akgentic-tool[weaviate]` / `[qdrant]`. |
| `tenant` | `str \| None` | `None` | Tenant id for multi-tenancy. Native on Weaviate; a payload field on Qdrant. |
| `params` | `dict[str, Any]` | `{}` | Backend-native settings passed through untouched. Schemaless by nature, so it is the one place `Any` is the honest type. |
| `embedding_model` | `str` | `"text-embedding-3-small"` | The model that produces the collection's vectors. |
| `embedding_provider` | `Literal["openai", "azure"]` | `"openai"` | The embedding API provider. |

**`default_backend()` is resolved per instantiation, not at import.** It asks every registered
backend whether the environment has provisioned it, so a card that names no backend lands wherever
the deployment actually is.

**`knowledge_graph` is a team-scoped collection**, so every search and every removal the store issues
carries this team's id as a predicate. That is declared by collection *name* in
`vector_store/protocol.py`, not by any field on this card — the workspace's `workspace_chunks` is the
one collection in the package that is shared across teams, because its rows belong to a filesystem
tree rather than to a team.

### `search_top_k` / `search_score_threshold` / `hybrid_alpha`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `search_top_k` | `int` | `10` | Default hit count; a `SearchQuery.top_k` overrides it. |
| `search_score_threshold` | `float` | `0.3` | Minimum **raw** cosine similarity, applied to the vector leg before fusion; `SearchQuery.score_threshold` overrides it. Deliberately lower than `PlanningTool`'s `0.5` — graph exploration favours recall. A keyword hit is never dropped by it. |
| `hybrid_alpha` | `float` (0–1) | `0.7` | Weight of the vector leg when fusing; the keyword leg gets `1 - alpha`. Below `0.5` a keyword match outranks a strong semantic hit. See [the fusion rule](../vector_store/README.md#hybrid-search-lives-here-not-in-the-backends). |

### `read_only`

`True` removes `update_graph` from the tool list. Useful for a reader agent on a graph another
agent curates.

---

## Capability parameters

### `GetGraph` — the graph as context

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{LLM_CONTEXT, COMMAND}` | **Not a tool call by default.** The summary is structured context state delivered as per-turn deltas. Add `TOOL_CALL` to also expose `get_graph()` as a callable that returns the *full* rendering. |
| `prompt_include_schema` | `bool` | `True` | Include the sorted entity-type and relation-type lists in the summary context state. |
| `prompt_include_roots` | `bool` | `True` | Include entities marked `is_root=True`, with their descriptions. |

A card persisted with an explicit `expose: ["system_prompt", ...]` from before the move is
revalidated onto `LLM_CONTEXT` by the attached `normalize_system_prompt_to_llm_context` validator.

The two `prompt_*` fields shape the **summary context state only** — they are carried on the state
as fields and gate their blocks in both the full rendering and the deltas; the `TOOL_CALL` and
`COMMAND` renderings always emit the full graph.

That distinction is the point. The summary is designed to scale as *O(types + roots)*, not
*O(entities + relations)*:

```
**Knowledge Graph Summary:**
Entities: 412 | Relations: 980
Entity types: Concept, Document, Person
Relation types: authored, cites, mentions
Root entities:
- Engagement brief (Document): the scope agreed in March

Use the get_graph tool to explore the full graph or subgraphs.
```

Turning both flags off reduces the summary to the two count lines. `is_root` is therefore a
context-budget decision as much as a modelling one: root entities are the entry points an agent is
told about without being told everything.

### `UpdateGraph` — `update_graph(update: ManageGraph)`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{TOOL_CALL}` | Tool call only; `get_commands()` never registers it. |

One batched mutation:

```python
class ManageGraph(SerializableBaseModel):
    create_entities: list[EntityCreate] = []
    update_entities: list[EntityUpdate] = []
    delete_entities: list[str] = []          # by name; cascades to relations
    create_relations: list[RelationCreate] = []
    delete_relations: list[RelationDelete] = []
```

Applied in a fixed order: **create entities → create relations → update entities → delete
relations → delete entities**. Creating an entity and a relation that references it in the same
call therefore works.

| Model | Fields |
|---|---|
| `EntityCreate` | `name` (the deduplication key), `entity_type`, `description`, `observations: list[str]`, `is_root: bool` |
| `EntityUpdate` | `name` (lookup key, required) + optional `description`, `entity_type`, `add_observations`, `remove_observations`, `is_root`. Only non-`None` fields are applied. |
| `RelationCreate` | `from_entity`, `to_entity`, `relation_type`, `description` |
| `RelationDelete` | `from_entity`, `to_entity`, `relation_type` — the unique triple |

`Entity` and `Relation` each carry an auto-generated `id: uuid.UUID`, which is the stable handle
used by the vector index. Entities are addressed by **name** in the API and by **UUID** in the
index; deleting an entity cascades to every relation touching it.

### `SearchGraph` — `search_graph(query: SearchQuery)`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{TOOL_CALL, COMMAND}` | |

```python
class SearchQuery(SerializableBaseModel):
    query: str
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid"
    top_k: int | None = None              # None -> search_top_k
    score_threshold: float | None = None  # None -> search_score_threshold
    include_neighbors: bool = False
    include_edges: bool = False
    find_paths: bool = False
```

| Field | Effect |
|---|---|
| `mode` | `"hybrid"` fuses keyword and semantic hits; `"keyword"` is substring-only and makes **no embedding call**; `"vector"` is cosine similarity only, and returns raw cosines rather than fused scores. |
| `include_neighbors` | Adds the 1-hop neighbours of every entity hit to `SearchResult.neighbors`. |
| `include_edges` | Adds every relation connected to an entity hit to `SearchResult.connected_relations`. |
| `find_paths` | BFS shortest paths between the top 5 entity hits, capped at 10 pairs, into `SearchResult.paths` as alternating `[Entity, Relation, Entity, …]` lists. |

All three expansion flags default to `False`; each one multiplies the size of the answer the model
reads back, so turn them on for a question that is actually about structure.

Searching covers **entities and relations**. Keyword matching is a case-insensitive substring test
across `name`, `description` and `entity_type` for entities, and `relation_type`, `description`,
`from_entity` and `to_entity` for relations — it runs in-process over graph state, not in the
vector store.

`mode="hybrid"` fuses the two legs with the shared rule, Weaviate's `relativeScoreFusion` at
`alpha = 0.7`: `alpha * norm(cosine) + (1 - alpha) * keyword`, scores in `[0, 1]`, highest first.
A strong semantic hit therefore outranks a keyword-only one; lower `hybrid_alpha` to invert that.
With no embeddings available the query degrades to keyword-only — no error, no warning at call
time. The rule, and the reasons behind it, are documented once in
[the vector store README](../vector_store/README.md#hybrid-search-lives-here-not-in-the-backends).

---

## Configuration

### The dependency check is unconditional

`observer()` calls `_check_kg_dependencies()` **before** anything else. That check requires
`numpy` and `openai` — the `[vector_search]` extra — and raises `ImportError` with the install
command when either is missing. It runs whatever the card's `vector_store` names, so unlike
`PlanningTool` this card cannot be used at all without the extra:

```bash
uv add "akgentic-tool[vector_search]"
```

The check is on the card, not on the storage: a cluster backend does not exempt it.

### Wiring order

```python
from akgentic.tool.knowledge_graph import KnowledgeGraphTool

ToolFactory([KnowledgeGraphTool()], observer=agent)
# -> no ordering to arrange: the card creates its own store inside observer()
```

There is no second card to list and no order to get wrong. What still fails fast at team creation
is a card naming a backend the environment has not provisioned, which `observer()` refuses with a
`ValueError`.

### Recipes

```python
KnowledgeGraphTool()                                  # defaults

KnowledgeGraphTool(read_only=True)                    # reader agent

KnowledgeGraphTool(get_graph=GetGraph(                # minimal context footprint
    prompt_include_schema=False,
    prompt_include_roots=False,
))

KnowledgeGraphTool(get_graph=GetGraph(                # also fetchable on demand
    expose={LLM_CONTEXT, TOOL_CALL, COMMAND},
))

KnowledgeGraphTool(                                   # persistent, tenant-isolated;
    vector_store=VectorStoreParam(backend="weaviate", tenant="team-42"),
    search_score_threshold=0.45,                      # no store actor is created for it
    search_top_k=25,
    hybrid_alpha=0.3,                                 # trust exact names over similarity
)
```

**There is no way to switch the vector store off on this card.** `vector_store` is a
`VectorStoreParam`, not a `bool`, so a card passing `False` fails validation. This card requires
the `[vector_search]` extra unconditionally in any case.

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

### State events

The actor emits `KnowledgeGraphStateEvent` deltas — `entities_added` / `entities_modified` /
`entities_removed` and the relation equivalents — inside a `ToolStateEvent` envelope on the
orchestrator's event stream. Every collection defaults to empty, so one mutation category can be
reported without the others. It is a delta, never a snapshot: consumers accumulate.

```python
from akgentic.tool import KnowledgeGraphStateEvent   # lazily re-exported
```

The lazy re-export matters: importing it eagerly from `akgentic.tool` would pull the
`[vector_search]` chain into every bare `import akgentic.tool`.

### Import paths

```python
from akgentic.tool.knowledge_graph import (
    GetGraph, KnowledgeGraphTool, SearchGraph, UpdateGraph,          # card + params
    Entity, EntityCreate, EntityUpdate, ManageGraph,                 # mutation models
    Relation, RelationCreate, RelationDelete,
    GetGraphQuery, GraphView, PathStep, SearchQuery, SearchResult,   # query models
    KnowledgeGraphActor, KnowledgeGraphConfig, KG_ACTOR_NAME,        # actor
)
```

`GetGraphQuery` is the actor-level subgraph query (`entity_names`, `relation_types`, `depth`,
`roots_only`, `path`). The LLM-facing `get_graph` callable does not expose it — it always sends
`GetGraphQuery()` and renders the full graph. Reach for it when calling the actor directly.

---

See the [package README](../../../../README.md) for the `ToolCard` / `ToolFactory` machinery, the
channel system, and the tool-state-event contract.
