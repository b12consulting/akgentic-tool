# The vector store

The package's storage engine and the two helpers that decide who needs one. **There is no
configuration card any more**: each consumer — `PlanningTool`, `KnowledgeGraphTool`,
`WorkspaceTool` — carries its own `VectorStoreParam` and resolves its own engine, so a store's
settings live on the card that uses it rather than in a singleton someone had to remember to add.

```python
from akgentic.tool.vector_store import ensure_store_actor, needs_store_actor
```

| | |
|---|---|
| Module | `akgentic.tool.vector_store` |
| Actor | `VectorStoreActor`, singleton named `#VectorStore` — **created only for an actor-state backend** |
| Consumers | `PlanningTool`, `KnowledgeGraphTool`, `WorkspaceTool` |
| Optional extras | `[vector_search]` (numpy + openai), `[weaviate]`, `[qdrant]` |

---

## Who gets an actor

`BackendSpec.persists_in_actor_state` answers one question: **does this backend need an actor at
all.** In memory the actor's state *is* the database, so the actor is what holds the data and one
must exist. On a cluster the data lives elsewhere and an actor would hold nothing but a socket, so
the consumer calls the backend's client directly.

Two helpers read that flag, and both halves of the wiring must agree:

```python
def needs_store_actor(param: VectorStoreParam) -> bool:
    """Whether this backend keeps its data inside an actor."""


def ensure_store_actor(param: VectorStoreParam, orchestrator_proxy: Orchestrator) -> None:
    """Create the store actor for *param*, if that backend needs one."""
```

A consumer card calls `ensure_store_actor` in its `observer()`, **before** it creates its own
consumer actor, because that actor resolves the store during its `on_start`:

```python
param = resolve_store_param(self.vector_store)            # None when the card declined a store
if param is not None:
    require_backend_configured(param, "PlanningTool")
    require_dimension_matches(param, "PlanningTool")
...
if param is not None:
    ensure_store_actor(param, orchestrator_proxy)         # nothing, on a cluster
orchestrator_proxy.getChildrenOrCreate(PlanActor, config=PlanConfig(vector_store=param, ...))
```

**A card that declined a store owes none of those three obligations.** The probe is skipped as well
as the creation, so a stored card that names `weaviate` explicitly and is *then* switched off does
not take its whole team down for a cluster nothing will ever open.

The consumer actor then resolves an **engine**, not a proxy. The four methods it calls —
`create_collection`, `add`, `remove`, `search` — are exactly `VectorStoreService`, which the store
actor and every backend satisfy with identical signatures, so nothing below the branch can tell
them apart:

```python
if needs_store_actor(param):
    addr = orch_proxy.get_team_member(VS_ACTOR_NAME)
    store = self.proxy_ask(addr, VectorStoreActor)
else:
    store = get_backend_spec(param.backend).factory(BackendContext(...))
store.create_collection(PLAN_COLLECTION, param)
```

**The ordering the deleted card's `depends_on` edge enforced is now intra-card.** No `ToolCard` in
this package overrides `depends_on`, and a consumer alone is a complete team.

**An unknown backend name raises**, out of `get_backend_spec`. A card naming a backend nobody
registered fails the build rather than silently taking one branch.

---

## Where the settings live

Every setting a store needs is on `VectorStoreParam`, carried by the consumer as its
`vector_store` field: `backend`, `dimension`, `tenant`, `params`, `embedding_model` and
`embedding_provider`. **The consumer's `embedding_model` is the one that embeds** — the store
writes vectors and produces none.

### Three shapes on the card, two below it

`PlanningTool` and `KnowledgeGraphTool` declare `vector_store: VectorStoreParam | bool`, because the
card is the author's surface and an author needs to be able to decline a store outright.
`WorkspaceTool` keeps a narrow `VectorStoreParam`: it already carries that opt-out under the larger
`_rag_enabled()`, and a second switch would be contradictable.

`resolve_store_param` in `protocol.py` is the single place the three collapse into two, for the same
reason `needs_store_actor` is a single place — two halves of the wiring must get the same answer:

```python
def resolve_store_param(value: VectorStoreParam | bool) -> VectorStoreParam | None:
    if value is True:
        return VectorStoreParam()
    if value is False:
        return None
    return value
```

`False` becomes `None` because `None` is what every consumer's store slot already means: `_vs_proxy
is None`, `_resolve_store(...) -> VectorStoreService | None`, and `store is None` in
`semantic_scores`. A literal `False` carried downstream would give every consumer three shapes to
test for, and a `VectorStoreParam` carrying a disabled flag would be a live object meaning "not
live" — one that still names a backend, so every guard here would have to learn to ignore an object
it was handed.

An explicit param passes through **by identity**, so a param two cards share through a catalog
`__ref__` is not silently duplicated.

### What is deliberately *not* here

**Cluster connection settings.** Neither a card nor `VectorStoreConfig` carries a cluster URL or
an API key: they are infrastructure, not something a catalog entry should carry. A card persisted
in a catalog would otherwise store a cluster URL and an API key as plain configuration.
`VectorStoreConfig` declares no field beyond `BaseConfig`'s — the actor's name and role.

Each backend's registered factory reads its own pair from the environment instead —
`AKGENTIC_WEAVIATE_URL` and `AKGENTIC_WEAVIATE_API_KEY` for Weaviate, `AKGENTIC_QDRANT_URL` and
`AKGENTIC_QDRANT_API_KEY` for Qdrant. **Exporting a URL is what turns that cluster on**; leave it
unset and a card that names no backend stays on the in-memory one, while a card that names the
cluster fails its build (see below). An exported but *empty* variable counts as unset — `or None`
— so a deployment template that always exports the name does not read as a cluster at `""`.

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

**One client per cluster per process, for every cluster backend.** `vector_store/client.py` holds
one cache, one lock and one `atexit` registration, shared by Weaviate and Qdrant alike. The key is
`ClusterKey(backend, host, port, secure, api_key)` — the backend name leads, so two backends
answering on one host:port are two clients and never one; `http://localhost:8080` and
`http://LOCALHOST:8080/` are one cluster, and two API keys are two. Qdrant passes
`default_port=6333` so `http://h` and `http://h:6333` collapse onto one key.

**The cache names no vendor.** `get_client(key, connect)` takes the connect callable from the
backend, which is the only place a client library is constructed, so adding a third cluster backend
adds no branch there. Both `WeaviateBackend` and `QdrantBackend` take a connected client; neither
connects and neither closes one. `close_all()` at process exit is the only closer.

### Naming a cluster that is not there is an error

```python
PlanningTool(vector_store=VectorStoreParam(backend="weaviate"))  # with no URL exported
# ValueError: PlanningTool configures backend='weaviate' but AKGENTIC_WEAVIATE_URL is not set.
#   Export AKGENTIC_WEAVIATE_URL (and AKGENTIC_WEAVIATE_API_KEY for an authenticated cluster),
#   or drop the backend setting to use the in-memory index.
```

`require_backend_configured` runs in each consumer card's `observer()`, **before any actor is
created**, and asks the named backend's own `require_configured` probe — here the Weaviate
module's — so the team fails to build rather than starting up half-wired. This is deliberately not
a degradation: a card that says `weaviate` has asked for durable, shared, tenant-isolated storage,
and an in-memory index is the wrong answer to a question the deployment already settled — silently
substituting it loses data that everything downstream assumes is persisted.

A card that names *no* backend never reaches the guard: without a cluster the default already
resolved to `inmemory`, so there is nothing to contradict.

> **A catalog entry records the resolved value.** `VectorStoreParam()` dumped on a machine with a
> cluster writes `backend: weaviate`, and loading that entry where no URL is exported raises. That
> is the intended behaviour — the entry is asking for a cluster — but it is why an environment
> promoting catalogs between tiers must export the variable in every tier that runs them.

| Helper | Where | Purpose |
|---|---|---|
| `default_backend()` | package root | Highest-priority provisioned backend from the registry; `"inmemory"` when none. |
| `require_backend_configured(config, card_name)` | package root | Backend-agnostic guard: dispatches to the named backend's own `require_configured`. |
| `weaviate_url()` | `vector_store.backends.weaviate` | Weaviate cluster URL, or `None`. Empty counts as unset. |
| `weaviate_api_key()` | `vector_store.backends.weaviate` | Weaviate API key, or `None`. |
| `WEAVIATE_URL_ENV`, `WEAVIATE_API_KEY_ENV` | `vector_store.backends.weaviate` | The two variable names. |

The Weaviate helpers live in the Weaviate backend module, not the package root, exactly as Qdrant's
`qdrant_url()` / `qdrant_api_key()` live in its own: a backend's environment belongs to the backend.

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
searches and removes on a team-scoped collection are team-scoped (a team-less backend raises rather
than leaking across teams), and point ids are derived from the team, effective tenant, and `ref_id`
by `stable_object_id` in `protocol.py` — the derivation Weaviate's object ids share — with `ref_id`
kept in the payload for filtering. A team-less writer on the shared `workspace_chunks` collection —
the workspace — stamps `""` and derives from an empty team, so the same chunk keeps one point
id whichever actor wrote it. Naming `backend="qdrant"` without a URL or the `[qdrant]` dependency fails
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
PlanningTool(vector_store=VectorStoreParam(backend="weaviate", tenant="team-42"))
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
| `add(collection, entries)` | Write pre-embedded `VectorEntry` records. An entry with an empty vector raises `ValueError`; a backend fault raises `RetriableError` rather than being swallowed. |
| `remove(collection, ref_ids, scope=None, path_prefix=None)` | Drop entries by reference id, narrowed by the predicates. `scope` is **required** on a collection shared across teams. |
| `search(collection, query_vector, top_k, scope=None, path_prefix=None, query=None)` | Cosine search, returning a `SearchResult`. `scope` / `path_prefix` narrow within a team, and `scope` is **required** on a collection shared across teams; pass an optional `VectorQuery` to filter, threshold, or forward backend-native params. |

`SearchResult` carries `hits: list[SearchHit]` (`ref_type`, `ref_id`, `text`, `score`, plus
`scope` / `path` / `ordinal` when the entry set them) and a `status` of `ready` or `error`. That is
the whole model — `indexing_pending` was removed once the store stopped tracking work in progress,
and `CollectionStatus.indexing` with it: a write either lands on the turn it arrives or raises, so
there is no interval left for a collection to be indexing in. `error` stays, unassigned by anything
in this package today, because a backend-level fault that invalidates a whole collection is what it
is for.

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

### Embedding belongs to the consumer, not to the store

The store embeds nothing. The code that owns a `VectorStoreParam` — the workspace indexer, the
planning actor, the knowledge graph — embeds with the model *its own* param names and hands the
store finished vectors.

The workspace indexer spawns one `EmbeddingWorker` per batch under a `#embed-` name, using
`build_embedding_service`, which is the single place the worker's timeout budget is chosen. The
worker embeds off the actor thread and reports back through the consumer's own
`receiveMsg_EmbeddingResult` / `receiveMsg_EmbeddingError`, carrying the whole entries with their
vectors filled in — so `scope`, `path` and `ordinal` never leave the consumer's objects. The
consumer then writes that batch with a single `add` ask on its own mailbox turn, and a write that
cannot land marks the file `FAILED` with the reason there and then.

Planning and the knowledge graph embed one entry at a time on a call site that already blocks, so
they keep their synchronous shape and simply hold their own `EmbeddingService`. What they gain is a
bounded turn: the blocking call now carries the worker's timeout budget.

### The in-memory index

`VectorIndex` keeps a pre-allocated numpy matrix that grows geometrically, with each row's L2 norm
computed at insertion. `search_cosine` is then a single BLAS pass over zero-copy views —
sub-millisecond for 10 000 entries at 1536 dimensions. `remove` compacts the buffers.

### Every Weaviate object carries its team

`WeaviateBackend` declares a `team_id` schema property alongside `ref_type` / `ref_id` / `text`
(and, since the workspace dimension, `scope` / `path` / `ordinal`), and stamps it onto every
object it writes. Unlike those three, `team_id` is always stamped. The value is the `team_id` the
registered factory is handed in its `BackendContext` — the owning team's id, propagated by the
actor system, never configured, never on a card. The factory reads the cluster from the
environment and builds:

```python
WeaviateBackend(client=_weaviate_client(weaviate_url(), weaviate_api_key()), team_id=context.team_id)
```

**Why it is there.** An in-memory collection dies with its actor; a Weaviate collection does not.
When a team is deleted its vectors stay in the cluster, and until now nothing on the object said
who had produced them — there was no filter that could find them, so they were unreachable
garbage accumulating in a shared cluster. `team_id` is the handle a cleanup process needs.

A backend built without a `team_id` still writes the property, as the empty string, so the schema
is uniform and a sweep never has to reason about objects that predate the field or come from an
unattributed writer. **The workspace is such a writer**: its backend is deliberately built with `team_id=None`, so that
a chunk's identity is a property of the *tree* rather than of whichever actor indexed it — and every
`workspace_chunks` row carries `""`. The mandatory `scope` bounds those rows instead.

**Every object's id is derived, like Qdrant's point ids.** `add` passes each object a `uuid` from
`stable_object_id(team_id, tenant, ref_id)` in `protocol.py` — the backend's team, the tenant the
object is written to, and its `ref_id`. The client replaces an object whose uuid already exists and
mints a UUIDv4 when none is given, so before this every re-add of a chunk left one more copy. On a
team-scoped collection the team is inside the id, so two teams writing one `ref_id` stay two
objects; on the shared collection a team-less writer's id carries `""`, and every lifetime of a tree
writes the same object.

**And read back on every query — but not on every collection.** Collection names are module
constants, so every team on a cluster shares the same collections and multi-tenancy is off unless a
deployment turns it on. Which of them the team predicate applies to is **a property the collection
declares**, not a backend-wide invariant:

| Collection | Team-scoped? | Why |
|---|---|---|
| `planning`, `knowledge_graph` | **yes** | their rows belong to one team |
| `workspace_chunks` | **no** | its rows belong to a *filesystem tree*, and two teams indexing one tree must read the same rows |

The declaration lives in `protocol.py`, beside `check_path_prefix`, so the two cluster backends
cannot drift apart: `SHARED_COLLECTIONS` is a frozen set of names and
`collection_is_team_scoped(collection)` answers from it. An unregistered name is team-scoped — the
safe answer for a collection nobody declared is the isolating one.

**It is declared by name and deliberately not by a field on `VectorStoreParam`.** Every field on that
model is catalog-settable, so a `team_scoped: false` there would let a catalog author silently unscope
`planning` — with no error and no log line. The set is frozen for the same reason: a registry anyone
may add to at runtime is the same hazard in another shape.

| Method | Filter on a team-scoped collection | Filter on a shared collection |
|---|---|---|
| `add` | stamps `team_id`; the object's `uuid` includes the team | stamps the backend's `team_id`, and nothing filters on it — for the workspace, whose backend is built with no team, that is `""` |
| `search` | `team_id == <backend's own>`, plus `scope` / `path` legs, passed to the cluster as `filters=` so it applies **before** `limit` | the `scope` and `path` legs alone; **`scope` is mandatory** |
| `remove` | `ref_id IN (...)` **AND** `team_id == <backend's own>` | `ref_id IN (...)` **AND** `scope == <argument>`; **`scope` is mandatory** |
| `delete_by_team(collection, team_id)` | `team_id == <argument>` — the backend's own is deliberately *not* anded on | **refused**, before any cluster call |
| `list_collections()` | none; a collection name identifies no team | none |

`remove` needs both legs on a team-scoped collection. `ref_id` alone deletes the matching object of
every team on the cluster, and reference ids collide across teams by construction — planning ids are
small integers, so completing task `3` would reach every team's task `3`. The team leg alone deletes
the collection.

**Where the team predicate goes, the scope predicate becomes mandatory.** The team leg was doing two
jobs on the workspace collection, and only one of them was wrong. It was wrong as an *ownership*
boundary. It was incidentally right as a *blast-radius* limit: a caller who forgot `scope` read only
its own team's chunks. Remove the predicate and that second job has no owner, so an omitted `scope`
would read or delete every workspace on the cluster — strictly worse than the bug the team predicate
was added to fix. `search` and `remove` therefore raise `ValueError` for a shared collection with no
`scope`, on **all three** backends. The in-memory one has no team predicate to lose and carries the
guard anyway: one team may hold two `WorkspaceTool` cards on two trees.

**`delete_by_team` refuses a shared collection**, in both cluster backends, before any client call. On
one, `team_id` records only who wrote the row, so a sweeper pointed at `workspace_chunks` would delete
rows another live team is still reading, from a tree that still exists. The guard reads a module-level
fact rather than the backend's own bookkeeping, which is why it still bites on the administrative
backend a sweeper builds — one that has created no collection at all.

**A backend with no `team_id` cannot query a team-scoped collection** — `search` and `remove` raise
`ValueError` rather than filtering on something. Filtering on `""` would not be a safe default: `""`
is a real value in the data, written by `add` for a team-less writer, so a query filtering on it would
answer *as* the unattributed team — an identity the caller never claimed. There is no safe guess
here, so the backend refuses instead of making one. `team_id=""` is refused on the same grounds.

**It can, however, query a shared one**, and that follows from the same argument rather than
weakening it: on a shared collection there is no identity to invent, because the boundary is the
scope. So a team-less backend can search `workspace_chunks` and still cannot search `planning`.

Writing without a team is still allowed, and the asymmetry is deliberate: `""` on a stored object is
a value a sweeper can find and act on, whereas `""` in a *query* is an invented identity.

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
| `delete_by_team(collection, team_id)` | Delete every object in one collection stamped with `team_id`. Returns the number deleted. Raises `ValueError` if the collection is **shared across teams**, or if the cluster has no such collection. |

Both work on a backend that created nothing — which is the point, since the sweeper runs after
the team and its actors are gone. **A sweep must skip the shared collections**, and the backend
enforces that rather than trusting the loop to:

```python
from akgentic.tool.vector_store import close_all
from akgentic.tool.vector_store.protocol import collection_is_team_scoped
from akgentic.tool.vector_store.backends.weaviate import WeaviateBackend, _weaviate_client

backend = WeaviateBackend(client=_weaviate_client(WEAVIATE_URL, WEAVIATE_API_KEY))
try:
    for team_id in deleted_team_ids:
        for collection in backend.list_collections():
            if not collection_is_team_scoped(collection):
                continue  # workspace_chunks belongs to a tree, not to a team
            deleted = backend.delete_by_team(collection, team_id)
            log.info("reaped %d objects from %s for team %s", deleted, collection, team_id)
finally:
    close_all()
```

The `continue` is a courtesy that keeps the log honest. Dropping it does not cause damage: the call
raises `ValueError` naming the collection before it reaches the cluster.

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
`backend="weaviate"` without `AKGENTIC_WEAVIATE_URL` fails during consumer card construction;
without `[weaviate]` the failure comes when the backend is built, as an `ImportError` naming the
extra. `backend="qdrant"` without `[qdrant]` or `AKGENTIC_QDRANT_URL` fails during consumer card
construction with an actionable installation or configuration message.

### Recipes

```python
# In memory: the consumer's own observer() creates the store actor.
PlanningTool()

# A cluster: no actor at all, and a URL is required at bind time.
KnowledgeGraphTool(vector_store=VectorStoreParam(backend="weaviate"))

# Two consumers, two different stores, no card to add and no ordering to declare.
ToolFactory([
    PlanningTool(vector_store=VectorStoreParam(backend="inmemory")),
    KnowledgeGraphTool(
        vector_store=VectorStoreParam(
            backend="weaviate", dimension=3072, embedding_model="text-embedding-3-large"
        )
    ),
], observer=agent)
```

**Naming a second store is gone with the card that made it possible.** A named
`#VectorStore-RAG` singleton was how two consumers reached two different stores; they now do it by
carrying two different `VectorStoreParam`s, which is the same capability with one fewer indirection.

`dimension=3072` with the default model is refused at bind:

```python
KnowledgeGraphTool(
    vector_store=VectorStoreParam(dimension=3072, embedding_model="text-embedding-3-large"),
)
```

### Import paths

```python
from akgentic.tool.vector_store import (
    VectorStoreActor, VectorStoreConfig, VS_ACTOR_NAME,
    ensure_store_actor, needs_store_actor,
    VectorStoreParam, CollectionStatus, SearchHit, SearchResult, VectorQuery,
    VectorEntry, VectorIndex, EmbeddingService,
    EmbeddingWorker, build_embedding_service, embedding_worker_name,
    VectorStoreService, InMemoryBackend, WeaviateBackend, QdrantBackend,
    BackendContext, BackendSpec, register_backend, unregister_backend,
    is_registered, available_backends, get_backend_spec,
    default_backend, resolve_default_backend, require_backend_configured,
)

# A backend's environment belongs to its own module, not the package root.
from akgentic.tool.vector_store.backends.weaviate import (
    WEAVIATE_URL_ENV, WEAVIATE_API_KEY_ENV, weaviate_url, weaviate_api_key,
)
from akgentic.tool.vector_store.backends.qdrant import (
    QDRANT_URL_ENV, QDRANT_API_KEY_ENV, qdrant_url, qdrant_api_key,
)
```

The three backend modules live in `vector_store/backends/`, and each registers itself on import.
Import the backend classes from the package root, which is their one public surface; the
`backends` package itself exports nothing.

Backend classes remain importable when their optional client is absent. Construction raises an
`ImportError` with the corresponding extra to install. Qdrant consumer cards fail earlier through
`require_backend_configured`; Weaviate's guard currently validates its URL and dependency failure
is reported when the backend is constructed.

---

See the [package README](../../../../README.md) for the `ToolCard` / `ToolFactory` machinery and
the dependency-ordering contract (`depends_on`).
