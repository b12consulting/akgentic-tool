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
| Optional extras | `[vector_search]` (numpy + openai), `[weaviate]`, `[qdrant]` |

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
| `embedding_model` | `str` | `"text-embedding-3-small"` | Embedding model identifier. The consumer's `VectorStoreParam` declares both `embedding_model` and `dimension`, and a known model whose native width disagrees with the declared dimension is refused at bind (`text-embedding-3-small` ⇒ 1536). This store still embeds every collection with its own model; a consumer declaring a different one is warned once at `create_collection`. |
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
every collection stays on the in-memory backend, whatever a `VectorStoreParam` asks for (selecting
`backend="weaviate"` without a URL logs a warning and leaves the backend unavailable). An exported
but *empty* variable counts as unset — `or None` — so a deployment template that always exports
the name does not read as a cluster at `""`.

**The team id.** Also not a field, and deliberately not configurable: it is `VectorStoreActor.team_id`,
which the actor system propagates. A card cannot be trusted to say which team it belongs to —
the same card is reused across every team in a catalog.

**The backend choice, mostly.** Which backend a collection uses follows the environment rather than
a field — see *The environment picks the backend* below. A card only overrides it.

**Collections.** The store is a container of named collections, and each consumer owns its own —
`PlanningTool` creates `planning`, `KnowledgeGraphTool` creates `knowledge_graph`. The
`VectorStoreParam` therefore lives on the *consumer* card, not here.

**Keyword search.** The backends answer pure similarity queries — `search` takes a vector and
nothing else, and neither backend is ever asked for text. The lexical half of a hybrid search runs
in the calling actor, over its own authoritative state. See below for why.

---

## The environment picks the backend

```bash
export AKGENTIC_WEAVIATE_URL="https://your-cluster.weaviate.network"
export AKGENTIC_WEAVIATE_API_KEY="..."          # omit for an unauthenticated cluster
```

**Exporting the URL is what turns Weaviate on.** `VectorStoreParam.backend` resolves through
`default_backend()` at instantiation: `weaviate` when a cluster URL is set, `inmemory` otherwise.
A card that names no backend therefore lands wherever the deployment actually is — a cluster is
deployed to be used, and a collection with no opinion should not quietly get a process-local index
that disappears with the actor.

Weaviate is one *registered* backend, not a hard-coded branch. `default_backend()` delegates to
the [backend registry](#pluggable-backends): it returns the first registered, environment-provisioned
backend that is `selectable_as_default`, falling back to `inmemory`. Weaviate is registered before
Qdrant, so an installation that exports both cluster URLs keeps the historical Weaviate default;
export only `AKGENTIC_QDRANT_URL` and the default becomes `qdrant`.

An exported but *empty* variable counts as unset, so a deployment template that always exports the
name does not read as a cluster at `""`. Resolution happens per instantiation, not at import, so a
process that exports the variable late still sees it.

**One client per cluster per process.** Every consumer that resolves the same cluster is handed the
same `weaviate.WeaviateClient`, from `get_client(url, api_key)` in `vector_store/client.py`. The
cache is keyed on the parsed connection — host, port, scheme and API key, so `http://localhost:8080`
and `http://LOCALHOST:8080/` are one cluster and two API keys are two — guarded by a lock so that two
actor threads resolving one cluster at once open one connection, and emptied by `close_all()`, which
is registered with `atexit` on the first successful connect. A `WeaviateBackend` takes that client;
it never connects and never closes.

### Naming a cluster that is not there is an error

```python
PlanningTool(collection=VectorStoreParam(backend="weaviate"))   # with no URL exported
# ValueError: PlanningTool configures backend='weaviate' but AKGENTIC_WEAVIATE_URL is not set.
#   Export AKGENTIC_WEAVIATE_URL (and AKGENTIC_WEAVIATE_API_KEY for an authenticated cluster),
#   or drop the backend setting to use the in-memory index.
```

`require_weaviate_configured` runs in each consumer card's `observer()`, **before any actor is
created**, so the team fails to build rather than starting up half-wired. This is deliberately not
a degradation: a card that says `weaviate` has asked for durable, shared, tenant-isolated storage,
and an in-memory index is the wrong answer to a question the deployment already settled — silently
substituting it loses data that everything downstream assumes is persisted.

A card that names *no* backend never reaches the guard: without a cluster the default already
resolved to `inmemory`, so there is nothing to contradict.

> **A catalog entry records the resolved value.** `VectorStoreParam()` dumped on a machine with a
> cluster writes `backend: weaviate`, and loading that entry where no URL is exported raises. That
> is the intended behaviour — the entry is asking for a cluster — but it is why an environment
> promoting catalogs between tiers must export the variable in every tier that runs them.

| Helper | Purpose |
|---|---|
| `weaviate_url()` | Cluster URL, or `None`. Empty counts as unset. |
| `weaviate_api_key()` | API key, or `None`. |
| `default_backend()` | Highest-priority provisioned backend from the registry; `"inmemory"` when none. |
| `require_weaviate_configured(config, card_name)` | Raises `ValueError` when *config* names Weaviate and no URL is set. |
| `require_backend_configured(config, card_name)` | Backend-agnostic guard: dispatches to the named backend's own `require_configured`. Prefer this in new consumer cards. |

---

## Pluggable backends

A backend is any object satisfying the `VectorStoreService` protocol (`create_collection`, `add`,
`remove`, `search`). Backends register themselves by name; the actor resolves and routes to them
through the registry and **never switches on a backend string**, so adding one needs no edit to the
actor or the protocol. `inmemory`, `weaviate`, and `qdrant` are just the three built-ins, each
registered the same way a third party would register its own.

### The capability table

`resolve_default_backend()` and the actor read a backend's declared capabilities from its
`BackendSpec` instead of special-casing it:

| `BackendSpec` field | Type | Default | What it decides |
|---|---|---|---|
| `name` | `str` | — | Identifier matched against `VectorStoreParam.backend`. |
| `factory` | `Callable[[BackendContext], VectorStoreService]` | — | Builds the instance. Import the client library *inside* the factory so registering never forces the optional dependency. |
| `persists_in_actor_state` | `bool` | `False` | `True` for stores whose data lives *in* the actor and must be snapshotted on every mutation. The backend must also implement `ActorStateBackend` (`get_state` / `restore_state`). `False` is for external stores that own their persistence. |
| `selectable_as_default` | `bool` | `True` | Whether `resolve_default_backend()` may pick it for a collection that names none. `inmemory` sets this `False` so it is only ever the fallback. |
| `is_configured` | `Callable[[], bool]` | always `True` | `True` when the environment has provisioned it (e.g. a URL is exported). Drives default resolution. |
| `require_configured` | `Callable[[str], None]` | no-op | Raises `ValueError` with remediation when the backend is named but unprovisioned. Reached from `require_backend_configured`. |

| Built-in | `persists_in_actor_state` | `selectable_as_default` | Provisioned by |
|---|---|---|---|
| `inmemory` | `True` | `False` | always available |
| `weaviate` | `False` | `True` | `AKGENTIC_WEAVIATE_URL` |
| `qdrant` | `False` | `True` | `AKGENTIC_QDRANT_URL` |

### Adding a backend

```python
from akgentic.tool.vector_store.protocol import VectorStoreService
from akgentic.tool.vector_store.registry import (
    BackendContext, BackendSpec, register_backend,
)

class PineconeBackend:                       # satisfies VectorStoreService structurally
    def __init__(self, ctx: BackendContext) -> None:
        import pinecone                       # lazy: optional dependency stays optional
        self._team_id = ctx.team_id           # stamp objects for team-scoped cleanup
        ...
    def create_collection(self, name, config): ...
    def add(self, collection, entries): ...
    def remove(self, collection, ref_ids): ...
    def search(self, collection, query_vector, top_k, query=None): ...

register_backend(BackendSpec(
    name="pinecone",
    factory=PineconeBackend,
    is_configured=lambda: bool(os.environ.get("PINECONE_API_KEY")),
))
```

A card then selects it with `VectorStoreParam(backend="pinecone")`; nothing in the actor changes.
`register_backend` / `unregister_backend` / `is_registered` / `available_backends` manage the
registry (tests use `unregister_backend` to clean up). Prefer **subclassing** a built-in when you
only need to bend one behaviour — `WeaviateBackend` and `QdrantBackend` expose overridable hooks
(`_build_filter`, `_search_kwargs` / `_near_vector_kwargs`) so a subclass can inject native query
options without reimplementing ingestion or team-scoping.

### Per-query parameters (`VectorQuery`)

`search` takes an optional `VectorQuery` so a caller can refine one query without touching the
backend or the collection. Every field is optional and a backend applies what it understands and
ignores the rest, so the same query stays portable across backends of differing capability.

| `VectorQuery` field | Type | Meaning |
|---|---|---|
| `filters` | `dict[str, Any] \| None` | Exact-match metadata constraints, e.g. `{"ref_type": "entity"}`. A value may be a scalar or a list (match-any). Applied by `inmemory`, `weaviate`, and `qdrant`. |
| `score_threshold` | `float \| None` | Drop hits whose raw cosine score is below this floor. |
| `params` | `dict[str, Any] \| None` | Backend-native knobs passed through untouched — HNSW `ef`, an exact-search toggle, a certainty floor — recognised only by the backend that defines them. |

```python
from akgentic.tool.vector_store.protocol import VectorQuery

actor.search("kg", vector, top_k=5, query=VectorQuery(
    filters={"ref_type": "entity"},
    score_threshold=0.75,
    params={"search_params": {"hnsw_ef": 128}},  # honoured by qdrant, ignored by inmemory
))
```

Omitting `query` calls the historical three-argument `search`, so pre-parameter backends keep
working unchanged.

### Qdrant setup

```bash
uv add "akgentic-tool[qdrant]"                # qdrant-client
export AKGENTIC_QDRANT_URL="https://your-cluster.qdrant.io"
export AKGENTIC_QDRANT_API_KEY="..."          # omit for an unauthenticated instance
```

`QdrantBackend` mirrors the Weaviate model: every point carries its owning `team_id` in the payload,
searches and removes are team-scoped (a team-less backend raises rather than leaking across teams),
and point ids are derived from the team, effective tenant, and `ref_id`, with `ref_id` kept in the
payload for filtering. Naming `backend="qdrant"` without a URL or the `[qdrant]` dependency fails
at card build time via `require_backend_configured`, exactly like Weaviate — a card asking for a
durable store is never silently downgraded to the in-memory index.

---


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
PlanningTool(collection=VectorStoreParam(backend="weaviate", tenant="team-42"))
```

| `VectorStoreParam` field | Type | Default | Meaning |
|---|---|---|---|
| `dimension` | `int` (≥1) | `1536` | Embedding vector dimensionality. Must be the native width of a known `embedding_model`; a mismatch is refused when the card binds. |
| `backend` | `str` | **follows the environment** | A registered backend name (`inmemory` / `weaviate` / `qdrant` / any third-party). Defaults via `default_backend()`: the highest-priority provisioned store, else `inmemory`. External backends require their extra (e.g. `akgentic-tool[qdrant]`). See [Pluggable backends](#pluggable-backends). |
| `tenant` | `str \| None` | `None` | Weaviate tenant id for multi-tenancy — normally the workspace or team id. |
| `embedding_model` | `str` | `"text-embedding-3-small"` | The model that produces this collection's vectors. Known models (`text-embedding-3-small` 1536, `text-embedding-3-large` 3072, `text-embedding-ada-002` 1536) pin `dimension`; an unknown name — an Azure deployment — is accepted at any dimension. |
| `embedding_provider` | `Literal["openai", "azure"]` | `"openai"` | The embedding API provider. |

---

## Runtime shape

### The actor

`VectorStoreActor` implements the `VectorStoreService` protocol:

| Method | Purpose |
|---|---|
| `create_collection(name, config)` | Create or reconfigure a named collection. Called by each consumer's actor on start. |
| `add(collection, entries, requester=None, request_ref=None)` | Ingest `VectorEntry` records. Entries arriving without a vector are embedded asynchronously. |
| `remove(collection, ref_ids, scope=None, path_prefix=None)` | Drop entries by reference id, narrowed by the predicates. |
| `search(collection, query_vector, top_k, scope=None, path_prefix=None, query=None)` | Cosine search, returning a `SearchResult`. `scope` / `path_prefix` narrow within a team; pass an optional `VectorQuery` to filter, threshold, or forward backend-native params. |
| `embed(texts)` | Embed a batch directly. |

`SearchResult` carries `hits: list[SearchHit]` (`ref_type`, `ref_id`, `text`, `score`, plus
`scope` / `path` / `ordinal` when the entry set them), a `status` of `ready` / `indexing`, and
`indexing_pending` — the number of entries still being embedded. A non-zero `indexing_pending` is
why a just-written task can be missing from a semantic search a moment later and present a moment
after. `error` remains in the enum for a backend-level fault, but no single failed batch reaches
it: a failure is reported to the caller that asked for the batch and leaves the collection alone.

`VectorEntry` links an embedding back to its source: `ref_type` (a free-form domain label —
`"entity"`, `"relation"`, a planning label), `ref_id` (a UUID string), `text`, `vector`, and the
optional `scope` / `path` / `ordinal` a chunking producer sets.

### One collection, partitioned by `scope` and `path`

`scope` partitions a collection *within* one team, the same way `team_id` partitions it across
teams — one class for the whole deployment, narrowed by a property predicate rather than by a
class per producer. `path` is the entry's source path, filtered by prefix; `ordinal` is a chunk's
position within its source, returned on a hit for reassembly and never filtered on. All three
default to `None`, and an entry that sets none of them is written exactly as it was before they
existed. Both predicates are applied **before** `top_k`, in the cluster on Weaviate and before the
cut in memory, so a scoped search returns a full `top_k` of its own entries.

### Embedding happens off the actor thread

Entries needing an embedding are handed to an `EmbeddingActor`, which answers with an
`EmbeddingResult` or an `EmbeddingError`. The store actor never blocks on the OpenAI call, so a
slow or failing embedding endpoint degrades search freshness rather than freezing every tool call
routed through the store.

Each asynchronous `add` opens its own request, and a result or an error settles only that
request: two concurrent adds into one collection never settle each other, and a failure fails one
batch. A caller that passes a `requester` is told `EmbeddingCompleted` when its request settles —
either way, with `error` set on failure — carrying back the `request_ref` it gave, since `add` is
reached by `tell` and cannot return the id the store mints internally.

### The in-memory index

`VectorIndex` keeps a pre-allocated numpy matrix that grows geometrically, with each row's L2 norm
computed at insertion. `search_cosine` is then a single BLAS pass over zero-copy views —
sub-millisecond for 10 000 entries at 1536 dimensions. `remove` compacts the buffers.

### Every Weaviate object carries its team

`WeaviateBackend` declares a `team_id` schema property alongside `ref_type` / `ref_id` / `text`
(and, since the workspace dimension, `scope` / `path` / `ordinal`), and stamps it onto every
object it writes. Unlike those three, `team_id` is always stamped. The value is the owning
`VectorStoreActor`'s `team_id` — propagated by the actor system, never configured, never on a
card:

```python
WeaviateBackend(client=get_client(url, api_key), team_id=str(actor.team_id))
```

**Why it is there.** An in-memory collection dies with its actor; a Weaviate collection does not.
When a team is deleted its vectors stay in the cluster, and until now nothing on the object said
who had produced them — there was no filter that could find them, so they were unreachable
garbage accumulating in a shared cluster. `team_id` is the handle a cleanup process needs.

A backend built without a `team_id` still writes the property, as the empty string, so the schema
is uniform and a sweep never has to reason about objects that predate the field or come from an
unattributed writer.

**And read back on every query.** Collection names are module constants — `knowledge_graph`,
`planning` — so every team on a cluster shares the same two collections, and multi-tenancy is off
unless a deployment turns it on. `team_id` is therefore also the predicate:

| Method | Filter |
|---|---|
| `add` | stamps `team_id` |
| `search` | `team_id == <backend's own>`, passed to the cluster as `filters=` so it applies **before** `limit` |
| `remove` | `ref_id IN (...)` **AND** `team_id == <backend's own>` |
| `delete_by_team(collection, team_id)` | `team_id == <argument>` — the backend's own is deliberately *not* anded on |
| `list_collections()` | none; a collection name identifies no team |

`remove` needs both legs. `ref_id` alone deletes the matching object of every team on the cluster,
and reference ids collide across teams by construction — planning ids are small integers, so
completing task `3` would reach every team's task `3`. The team leg alone deletes the collection.

**A backend with no `team_id` cannot query at all** — `search` and `remove` raise `ValueError`
rather than filtering on something. Filtering on `""` would not be a safe default: `""` is a real
value in the data, written by `add` for a team-less writer, so a query filtering on it would answer
*as* the unattributed team — an identity the caller never claimed. There is no safe guess here, so
the backend refuses instead of making one. `team_id=""` is refused on the same grounds.

Writing without a team is still allowed, and the asymmetry is deliberate: `""` on a stored object is
a value a sweeper can find and act on, whereas `""` in a *query* is an invented identity.

A sweeper is the one script that legitimately has no `team_id`, and the example under *Reaping a
deleted team* below builds one without: `list_collections` and `delete_by_team` are the two methods
that carry no team predicate, so neither is affected by the rule above.

In a running deployment the raise is unreachable — `VectorStoreActor` always passes
`str(self.team_id)`, and an actor's `team_id` is a UUID defaulted at construction. It guards
hand-built backends.

`SearchHit` still does not expose `team_id`; the boundary is applied in the query, not reported in
the result.

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
from akgentic.tool.vector_store import close_all, get_client
from akgentic.tool.vector_store.weaviate import WeaviateBackend

backend = WeaviateBackend(client=get_client(WEAVIATE_URL, WEAVIATE_API_KEY))
try:
    for team_id in deleted_team_ids:
        for collection in backend.list_collections():
            deleted = backend.delete_by_team(collection, team_id)
            log.info("reaped %d objects from %s for team %s", deleted, collection, team_id)
finally:
    close_all()
```

A backend has no `close()` of its own: it does not own the connection. `close_all()` is
process-level — it closes every client this process has cached, for every cluster — so it belongs
at the end of a script or at process shutdown, where it also runs by itself through `atexit`, and it
must not run while teams are live in the same process.

On a **multi-tenant** collection the delete is scoped to the backend's tenant, so pass the tenant
too when the deployment maps one tenant per team — `WeaviateBackend(client=..., tenant=team_id)`.
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
uv add "akgentic-tool[qdrant]"          # qdrant-client   — only for backend="qdrant"
```

Without `[vector_search]` the consumer cards degrade to keyword-only search
(`KnowledgeGraphTool` excepted: it checks the dependency in `observer()` and raises). Selecting
`backend="weaviate"` without `[weaviate]`, or without a `weaviate_url`, leaves the backend
unavailable. `backend="qdrant"` without `[qdrant]` or `AKGENTIC_QDRANT_URL` fails during consumer
card construction with an actionable installation or configuration message.

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

A larger embedding model needs a matching `dimension` on the consumer's collection, declared beside
the model that produces it — `dimension=3072` with the default model is refused at bind:

```python
KnowledgeGraphTool(
    vector_store="#VectorStore-RAG",
    collection=VectorStoreParam(dimension=3072, embedding_model="text-embedding-3-large"),
)
```

### Import paths

```python
from akgentic.tool.vector_store import (
    VectorStoreTool, VectorStoreActor, VectorStoreConfig, VS_ACTOR_NAME,
    VectorStoreParam, CollectionStatus, SearchHit, SearchResult, VectorQuery,
    VectorEntry, VectorIndex, EmbeddingService,
    VectorStoreService, InMemoryBackend, WeaviateBackend, QdrantBackend,
    BackendContext, BackendSpec, register_backend, unregister_backend,
    is_registered, available_backends, get_backend_spec,
    default_backend, resolve_default_backend, require_backend_configured,
)
```

Backend classes remain importable when their optional client is absent. Construction raises an
`ImportError` with the corresponding extra to install. Qdrant consumer cards fail earlier through
`require_backend_configured`; Weaviate's guard currently validates its URL and dependency failure
is reported when the backend is constructed.

---

See the [package README](../../../../README.md) for the `ToolCard` / `ToolFactory` machinery and
the dependency-ordering contract (`depends_on`).
