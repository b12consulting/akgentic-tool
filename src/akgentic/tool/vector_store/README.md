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
`weaviate_api_key`, but the card does not surface them: they are infrastructure, injected by the
deployment layer (from `AKGENTIC_WEAVIATE_URL` / `AKGENTIC_WEAVIATE_API_KEY`), not something a
catalog entry should carry. A card persisted in a catalog would otherwise store a cluster URL and
an API key as plain configuration.

**Collections.** The store is a container of named collections, and each consumer owns its own —
`PlanningTool` creates `planning`, `KnowledgeGraphTool` creates `knowledge_graph`. The
`CollectionConfig` therefore lives on the *consumer* card, not here.

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
