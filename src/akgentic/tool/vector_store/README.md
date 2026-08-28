# VectorStoreTool

The configuration card for the team's shared embedding store. It exposes **no LLM tools at all** —
its only job is to make sure the `VectorStoreActor` singleton exists before the cards that use it
are wired.

```python
from akgentic.tool.vector_store import VectorStoreTool
```

| | |
|---|---|
| Module | `akgentic.tool.vector_store.tool` |
| Actor | `VectorStoreActor`, singleton named `#VectorStore` by default |
| Channels used | **none** — `get_tools()`, `get_system_prompts()`, `get_commands()` and `get_toolsets()` all return empty |
| Consumers | `PlanningTool`, `KnowledgeGraphTool` |
| Optional extras | `[vector_search]` (numpy + openai), `[weaviate]` |

---

## The ToolCard

```python
class VectorStoreTool(ToolCard):
    vector_store_name: str = "#VectorStore"
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: Literal["openai", "azure"] = "openai"

    def observer(self, observer: ActorToolObserver) -> None:
        super().observer(observer)
        if observer.orchestrator is None:
            raise ValueError("VectorStoreTool requires access to the orchestrator.")
        orchestrator_proxy = observer.proxy_ask(observer.orchestrator, Orchestrator)
        orchestrator_proxy.getChildrenOrCreate(
            VectorStoreActor,
            config=VectorStoreConfig(
                name=self.vector_store_name,
                role=VS_ACTOR_ROLE,
                embedding_model=self.embedding_model,
                embedding_provider=self.embedding_provider,
                weaviate_url=os.environ.get("AKGENTIC_WEAVIATE_URL") or None,
                weaviate_api_key=os.environ.get("AKGENTIC_WEAVIATE_API_KEY") or None,
            ),
        )
```

**Why a card with no tools.** Two cards need the same actor and neither should own it. Putting
creation in a third card and having the consumers declare `depends_on: ["VectorStoreTool"]` makes
the ordering explicit and checkable: `ToolFactory` topologically sorts the cards, wires this one
first, and raises `ValueError` at team-creation time if a consumer asks for a store nobody
provides. Consumers then look the actor up by name during their own `on_start` — they never call
`getChildrenOrCreate` themselves.

`getChildrenOrCreate` is idempotent, so attaching several observers, or wiring several consumers,
resolves to the same actor rather than racing to create duplicates.

---

## ToolCard fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `vector_store_name` | `str` | `"#VectorStore"` | Singleton actor name. Several named stores can coexist in one team — give each a distinct name and point each consumer at the one it wants with `vector_store="#VectorStore-RAG"`. The `#` prefix is the package's tool-actor convention and is load-bearing for teardown; keep it. |
| `embedding_model` | `str` | `"text-embedding-3-small"` | Embedding model identifier. Must agree with the `dimension` in every `CollectionConfig` pointed at this store (`text-embedding-3-small` ⇒ 1536). |
| `embedding_provider` | `Literal["openai", "azure"]` | `"openai"` | Selects `openai.OpenAI()` or `openai.AzureOpenAI()`. The client is constructed lazily on the first `embed()` call, so a team that never searches never needs credentials. Both read their configuration from the standard `OPENAI_*` / `AZURE_OPENAI_*` environment variables. |

That is the whole card. Three primitives, no nested param models, no capability toggles — there is
nothing to expose on a channel.

### What is deliberately *not* here

**Weaviate connection settings.** `VectorStoreConfig` carries `weaviate_url` and
`weaviate_api_key`, but the card does not surface them: they are infrastructure, not something a
catalog entry should carry. A card persisted in a catalog would otherwise store a cluster URL and
an API key as plain configuration.

The card reads them from the environment instead, at `observer()` time — `AKGENTIC_WEAVIATE_URL`
and `AKGENTIC_WEAVIATE_API_KEY`. **Exporting a URL is what turns Weaviate on**; leave it unset and
every collection stays on the in-memory backend, whatever a `CollectionConfig` asks for (selecting
`backend="weaviate"` without a URL logs a warning and leaves the backend unavailable). An exported
but *empty* variable counts as unset — `or None` — so a deployment template that always exports
the name does not read as a cluster at `""`.

**The team id.** Also not a field, and deliberately not configurable: it is `VectorStoreActor.team_id`,
which the actor system propagates. A card cannot be trusted to say which team it belongs to —
the same card is reused across every team in a catalog.

**Collections.** The store is a container of named collections, and each consumer owns its own —
`PlanningTool` creates `planning`, `KnowledgeGraphTool` creates `knowledge_graph`. The
`CollectionConfig` therefore lives on the *consumer* card, not here.

**Keyword search.** The backends answer pure similarity queries — `search` takes a vector and
nothing else, and neither backend is ever asked for text. The lexical half of a hybrid search runs
in the calling actor, over its own authoritative state. See below for why.

---

## Hybrid search lives here, not in the backends

`akgentic.tool.vector_store.hybrid` owns the one rule that combines keyword and vector hits. Both
`KnowledgeGraphActor` and `PlanActor` search through it, so they rank identically.

```python
from akgentic.tool.vector_store.hybrid import hybrid_search

result = hybrid_search(
    keyword_keys,            # ref_ids the caller's own keyword phase found
    vs_proxy, "planning", "deployment plan",
    top_k=10,
    score_threshold=0.5,     # minimum *raw* cosine, before normalisation
    alpha=0.7,
)
result.ranked          # [(ref_id, fused score)], best first, NOT cut to top_k
result.vector_scores   # raw cosine per ref_id, for callers that render a score
```

### The rule

Weaviate's `relativeScoreFusion` — the default behind `collection.query.hybrid()` — reproduced
in Python:

```
score = alpha * norm(cosine) + (1 - alpha) * keyword
```

Each leg is min-max normalised onto `[0, 1]`. The keyword leg is an *indicator*, not a normalised
score: the lexical match is a substring test, so every keyword hit is equally good and normalising
a flat list yields `1.0` throughout. Three outcomes follow:

| Hit found by | Score at `alpha = 0.7` |
|---|---|
| vector only | `0.7 × norm(cosine)` — `0.7` for the strongest hit in the set, `0.0` for the weakest |
| keyword only | `0.3` |
| both | the sum |

`alpha` defaults to `0.7`, the value the Weaviate client sends. **This weights semantics above
lexical matching:** a strong vector hit at `0.7` outranks a keyword-only hit at `0.3`. Set
`hybrid_alpha` below `0.5` on `PlanningTool` or `KnowledgeGraphTool` to invert that, which is the
right call when your lexical matches are the precise ones — exact ids, product names, error codes.

### Two consequences worth knowing before you tune it

**Normalisation is relative, so scores compare only within one query.** A result set whose cosines
are `0.9 / 0.7 / 0.5` fuses exactly like one at `0.5 / 0.3 / 0.1`. The weakest vector hit always
normalises to `0.0` however good its absolute cosine was — most visible on small result sets. This
is Weaviate's behaviour, kept deliberately so a collection moved to a cluster does not reorder.

**`score_threshold` gates the vector leg only, on the raw cosine, before normalisation.** That
keeps its absolute meaning, and means a keyword hit can never be dropped by it. The two are on
different scales by design.

### Why not push it into the backend?

Weaviate can do hybrid natively and the in-memory index cannot, so putting the rule behind the
backend seam would make the same data rank differently on the two backends — dev on in-memory,
production on Weaviate. It also could not reproduce today's recall: the embedded text is
`f"{entity.name}: {entity.description}"`, which omits `entity_type`, and ingest is best-effort, so
BM25 over the stored `text` would silently miss what the in-process scan finds. Revisit when the
store is an authoritative index of the graph rather than a lossy projection of it.

---

## Collection configuration (on the consumer card)

```python
PlanningTool(collection=CollectionConfig(backend="weaviate", tenant="team-42"))
```

| `CollectionConfig` field | Type | Default | Meaning |
|---|---|---|---|
| `dimension` | `int` (≥1) | `1536` | Embedding vector dimensionality. Must match `embedding_model`. |
| `backend` | `"inmemory" \| "weaviate"` | `"inmemory"` | `inmemory` is a numpy cosine index inside the actor; `weaviate` delegates to a cluster and requires `akgentic-tool[weaviate]`. |
| `persistence` | `"actor_state" \| "workspace"` | `"actor_state"` | **inmemory backend only.** `actor_state` keeps vectors in the actor's persisted state; `workspace` writes them to a file. |
| `workspace_path` | `str \| None` | `None` | The file path used when `persistence="workspace"`. |
| `tenant` | `str \| None` | `None` | Weaviate tenant id for multi-tenancy — normally the workspace or team id. |

---

## Runtime shape

### The actor

`VectorStoreActor` implements the `VectorStoreService` protocol:

| Method | Purpose |
|---|---|
| `create_collection(name, config)` | Create or reconfigure a named collection. Called by each consumer's actor on start. |
| `add(collection, entries)` | Ingest `VectorEntry` records. Entries arriving without a vector are embedded asynchronously. |
| `remove(collection, ref_ids)` | Drop entries by reference id. |
| `search(collection, query_vector, top_k)` | Cosine search, returning a `SearchResult`. |
| `embed(texts)` | Embed a batch directly. |

`SearchResult` carries `hits: list[SearchHit]` (`ref_type`, `ref_id`, `text`, `score`), a
`status` of `ready` / `indexing` / `error`, and `indexing_pending` — the number of entries still
being embedded. A non-zero `indexing_pending` is why a just-written task can be missing from a
semantic search a moment later and present a moment after.

`VectorEntry` links an embedding back to its source: `ref_type` (a free-form domain label —
`"entity"`, `"relation"`, a planning label), `ref_id` (a UUID string), `text`, `vector`.

### Embedding happens off the actor thread

Entries needing an embedding are handed to an `EmbeddingActor`, which answers with an
`EmbeddingResult` or an `EmbeddingError`. The store actor never blocks on the OpenAI call, so a
slow or failing embedding endpoint degrades search freshness rather than freezing every tool call
routed through the store.

### The in-memory index

`VectorIndex` keeps a pre-allocated numpy matrix that grows geometrically, with each row's L2 norm
computed at insertion. `search_cosine` is then a single BLAS pass over zero-copy views —
sub-millisecond for 10 000 entries at 1536 dimensions. `remove` compacts the buffers.

### Every Weaviate object carries its team

`WeaviateBackend` declares a fourth schema property, `team_id`, alongside `ref_type` / `ref_id` /
`text`, and stamps it onto every object it writes. The value is the owning `VectorStoreActor`'s
`team_id` — propagated by the actor system, never configured, never on a card:

```python
WeaviateBackend(url=..., api_key=..., team_id=str(actor.team_id))
```

**Why it is there.** An in-memory collection dies with its actor; a Weaviate collection does not.
When a team is deleted its vectors stay in the cluster, and until now nothing on the object said
who had produced them — there was no filter that could find them, so they were unreachable
garbage accumulating in a shared cluster. `team_id` is the handle a cleanup process needs.

A backend built without a `team_id` still writes the property, as the empty string, so the schema
is uniform and a sweep never has to reason about objects that predate the field or come from an
unattributed writer.

`team_id` is written, never read back: `SearchHit` does not expose it and search is unaffected.

### Reaping a deleted team

Two methods sit outside the `VectorStoreService` protocol because they are cluster
administration, not vector storage — the in-memory backend has no equivalent and needs none:

| Method | Purpose |
|---|---|
| `list_collections()` | Every collection name in the cluster, read from Weaviate rather than from this backend's own bookkeeping. |
| `delete_by_team(collection, team_id)` | Delete every object in one collection stamped with `team_id`. Returns the number deleted. Raises `ValueError` if the cluster has no such collection. |

Both work on a backend that created nothing — which is the point, since the sweeper runs after
the team and its actors are gone:

```python
from akgentic.tool.vector_store.weaviate import WeaviateBackend

backend = WeaviateBackend(url=WEAVIATE_URL, api_key=WEAVIATE_API_KEY)
try:
    for team_id in deleted_team_ids:
        for collection in backend.list_collections():
            deleted = backend.delete_by_team(collection, team_id)
            log.info("reaped %d objects from %s for team %s", deleted, collection, team_id)
finally:
    backend.close()
```

On a **multi-tenant** collection the delete is scoped to the backend's tenant, so pass the tenant
too when the deployment maps one tenant per team — `WeaviateBackend(url=..., tenant=team_id)`.
`tenant` and `team_id` are independent: the tenant partitions storage, `team_id` is a property on
the object, and a backend given both stamps its own `team_id` rather than the tenant name.

The sweeper process itself — what decides a team is deleted, and on what schedule — lives in the
deployment layer, not in this package.

---

## Configuration

### Extras

```bash
uv add "akgentic-tool[vector_search]"   # numpy + openai — required for any embedding at all
uv add "akgentic-tool[weaviate]"        # weaviate-client — only for backend="weaviate"
```

Without `[vector_search]` the consumer cards degrade to keyword-only search
(`KnowledgeGraphTool` excepted: it checks the dependency in `observer()` and raises). Selecting
`backend="weaviate"` without `[weaviate]`, or without a `weaviate_url`, logs a warning and leaves
the backend unavailable rather than crashing the team.

### Recipes

```python
VectorStoreTool()                                        # "#VectorStore", OpenAI embeddings

VectorStoreTool(embedding_provider="azure")              # Azure OpenAI deployment

VectorStoreTool(                                         # a second, independent store
    vector_store_name="#VectorStore-RAG",
    embedding_model="text-embedding-3-large",
)

# Point one consumer at the named store, leave the other on the default
ToolFactory([
    VectorStoreTool(),
    VectorStoreTool(vector_store_name="#VectorStore-RAG"),
    PlanningTool(),                                       # -> "#VectorStore"
    KnowledgeGraphTool(vector_store="#VectorStore-RAG"),   # -> the large-embedding store
], observer=agent)
```

A larger embedding model needs a matching `dimension` on the consumer's collection:

```python
KnowledgeGraphTool(
    vector_store="#VectorStore-RAG",
    collection=CollectionConfig(dimension=3072),          # text-embedding-3-large
)
```

### Import paths

```python
from akgentic.tool.vector_store import (
    VectorStoreTool, VectorStoreActor, VectorStoreConfig, VS_ACTOR_NAME,
    CollectionConfig, CollectionStatus, SearchHit, SearchResult,
    VectorEntry, VectorIndex, EmbeddingService, InMemoryBackend, WeaviateBackend,
)
```

`WeaviateBackend` is `None` when `weaviate-client` is not installed — the import never fails, so
guard on the value rather than on the import.

---

See the [package README](../../../../README.md) for the `ToolCard` / `ToolFactory` machinery and
the dependency-ordering contract (`depends_on`).
