# SearchTool

Web search, single-page extraction and multi-page crawling through the
[Tavily](https://tavily.com/) API — the tool an agent reaches for when the answer is not in the
team's own context.

```python
from akgentic.tool.search import SearchTool
```

| | |
|---|---|
| Module | `akgentic.tool.search.search` |
| Actor | none — the Tavily client is constructed per call |
| Channels used | `TOOL_CALL` |
| Dependency | `tavily-python`, a base dependency (no extra) |
| Environment | `TAVILY_API_KEY` |

---

## The ToolCard

```python
class SearchTool(ToolCard):
    web_search: WebSearch | bool = True
    web_crawl: WebCrawl | bool = True
    web_fetch: WebFetch | bool = True
```

Three capabilities, no wiring. `SearchTool` needs no observer setup and no actor: each callable
constructs a `TavilyClient()` when invoked and lets it read `TAVILY_API_KEY` from the environment.

**Every parameter is exposed twice.** The fields on the param models become the **default values
of the tool function's arguments**, not fixed settings. `WebSearch(max_results=3)` means "default
to 3 results" and the model may still ask for 12. This is the shape to reach for when you want to
bias behaviour without removing the model's judgement; to remove the judgement, disable the
capability or narrow the description with `instructions`.

Two arguments are deliberately outside that contract: **`web_fetch`'s `query` and `web_crawl`'s
`crawl_instructions` are required and have no field.** Both are the *relevance filter* for their
capability, and there is no sensible value to configure in advance — each belongs to the question
*this particular call* is meant to answer. A configured default would make them optional for the
model, which is exactly what they exist to prevent. Everything else is a configurable default.

---

## ToolCard fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `web_search` | `WebSearch \| bool` | `True` | Natural-language query → ranked sources. |
| `web_fetch` | `WebFetch \| bool` | `True` | Known URLs → clean extracted content. |
| `web_crawl` | `WebCrawl \| bool` | `True` | Root URL → content from discovered pages. |

`True` enables with defaults, `False` removes the capability entirely, an instance configures it.

---

## Capability parameters

### `WebSearch` — `web_search(query, max_results=…, search_depth=…)`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{TOOL_CALL}` | |
| `max_results` | `int` | `5` | Default result count. Tavily accepts 0–20. |
| `search_depth` | `"basic" \| "advanced"` | `"basic"` | `basic` balances relevance against latency and credit cost; `advanced` is more relevant, slower and more expensive. Always sent — the tool no longer defers to Tavily's own default. |

Returns Tavily's raw search response — titles, URLs and snippets, **not** page content. It is the
cheap half of the pair: search to find the URLs worth reading, then `web_fetch` to read them.

### `WebFetch` — `web_fetch(urls, query, chunks_per_source=…, timeout=…, extract_depth=…)`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{TOOL_CALL}` | |
| `timeout` | `float` | `30` | Seconds per extraction request. Tavily supports roughly 1–60. |
| `extract_depth` | `"basic" \| "advanced"` | `"basic"` | `basic` is faster and cheaper and is enough for articles and documentation; `advanced` reaches tables and embedded content at higher latency and cost. Always sent. |
| `chunks_per_source` | `int` | `3` | Ceiling on how many relevant passages come back **per URL**. |

Takes a **list** of absolute URLs and returns extracted content for each. Use it when the agent
already has the links — from a `web_search` result, from the user, from a document.

#### The fetch is relevance-filtered, not a page dump

`query` and `chunks_per_source` are what keep the response a size a model can actually read, and
they are the reason this capability is not simply "download these pages":

| Argument | Role |
|---|---|
| `query` (required, per call) | Selects **which** passages of each page are returned. Tavily scores the page against it and keeps the matching parts. |
| `chunks_per_source` (configured, default `3`) | Caps **how many** of those passages come back per URL. |

Without them an extract returns whole pages. Three long documentation pages is then tens of
thousands of tokens of tool output, arriving in one message — enough to crowd out the conversation
the agent is meant to be having, and in the worst case to stall the ReAct loop on its own result
before the model has read any of it. With them the same three URLs cost roughly nine passages.

The cost of the filter is that a bad `query` returns bad content: the tool cannot return what the
query did not ask for. When an agent reports that a page "does not mention" something it plainly
contains, suspect a vague query before suspecting the page — the docstring tells the model as much.

Raise `chunks_per_source` when one page genuinely has to be covered exhaustively. Prefer more URLs
at the default cap over one URL at a high cap; breadth is cheaper than depth here.

Content is always returned as **markdown** — structure the model can use, without the markup the
page carries. This is fixed, not configurable.

### `WebCrawl` — `web_crawl(url, crawl_instructions, timeout=…, max_depth=…, max_breadth=…, limit=…, extract_depth=…)`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{TOOL_CALL}` | |
| `timeout` | `float` | `150` | Seconds for the whole crawl. Tavily supports 10–150. |
| `max_depth` | `int \| None` | `None` | Link depth from the root URL. Tavily supports 1–5. |
| `max_breadth` | `int \| None` | `None` | Links followed per level. Tavily supports 1–500. |
| `limit` | `int \| None` | `None` | Total pages processed before stopping. Must be ≥ 1. |
| `extract_depth` | `"basic" \| "advanced"` | `"basic"` | Extraction depth applied to crawled pages. Always sent. |

The remaining `None` fields — `max_depth`, `max_breadth`, `limit` — are omitted from the Tavily
call rather than sent as null, so Tavily's own defaults apply. `extract_depth` is not among them:
like its counterparts on `web_search` and `web_fetch`, it now always carries a concrete value.

#### The crawl is directed, not exhaustive

`crawl_instructions` is a **required argument with no field**, for the same reason `web_fetch`
requires a `query`: an unguided crawl walks the site and returns whatever it finds. It is the
model's statement of what this crawl is for — *"the authentication section of the API reference"* —
and Tavily uses it to decide which links to follow and what to extract from them.

To bias every crawl a team performs, set `instructions` instead. Telling the model what to look
for is the supported way to influence an argument it is required to choose:

| Name | Layer | Effect |
|---|---|---|
| `crawl_instructions` *(required argument)* | per call | Sent to Tavily. Directs which pages are crawled and what is extracted. Chosen by the model, every call. |
| `instructions` *(inherited field)* | configured | Appended to the tool description the model reads. Never sent anywhere. |

```python
WebCrawl(
    instructions="Only crawl public documentation, and say which section you want.",
)
```

The name is not cosmetic. Both once lived under the single name `instructions`, so a crawl could
not be biased without also rewriting its description, nor described without also biasing it.

Crawling remains the expensive capability: a `max_depth=5, max_breadth=500` crawl can walk
thousands of pages, and `limit` is unset by default. Set `limit` whenever you expose crawling to
an agent that picks its own arguments — required guidance narrows *what* comes back, but it does
not cap *how much*.

---

## A note on stored configuration

These param models are fields of a persisted `Process`: a team's `SearchTool` settings are
written into its `team.yaml` and read back at startup. Tightening a field's type therefore
applies retroactively to data already on disk.

`WebSearch.search_depth`, `WebFetch.extract_depth` and `WebCrawl.extract_depth` were once
`Literal["basic", "advanced"] | None` defaulting to `None`. Teams saved under that schema carry a
literal `null`, and the narrowed annotation rejects it — the failure is not a warning but a refusal
to deserialize, which surfaces as `Corrupted team.yaml` and drops the team at startup.

A `mode="before"` validator on all three fields reads a stored `null` as `"basic"`. `None` meant "no
explicit choice", and the tool's own choice is now `basic`, so the stored intent is preserved
rather than guessed at. **Existing YAML is read forward, never rewritten** — no migration step, and
a team that explicitly chose `advanced` keeps it.

`crawl_instructions` needs no such handling despite having been removed as a field: unknown keys in
stored payloads are ignored, so an old file carrying it simply loads without it.

The general lesson for this package: narrowing a `ToolCard` param type is a data-compatibility
change, not a local one. Widen the reader before you narrow the writer — and note that the three
capabilities were narrowed in three separate steps, each of which broke the same teams again until
its own validator was added.

---

## Configuration

### The API key

```bash
export TAVILY_API_KEY="tvly-..."
```

The key is checked in three places, all of them soft:

1. `get_tools()` logs a warning when the key is absent — the tools are still registered.
2. Each callable re-checks at invocation time and, if the key is missing, **returns a string**
   telling the model the tool is unavailable and to ask the user to configure it.
3. Any exception from Tavily (invalid key, rate limit, outage) is caught, logged at warning level
   and returned as a string.

Nothing here raises. That is the package's error-handling contract in its plainest form: a tool
call must always produce a tool response, because an unhandled exception stalls the ReAct loop.
The cost is that a misconfigured key surfaces as a message inside the conversation rather than as
a startup failure — check the logs for `TAVILY_API_KEY is not set` when an agent keeps reporting
that search is unavailable.

### Recipes

```python
SearchTool()                                     # all three, Tavily defaults

SearchTool(web_crawl=False)                      # search and fetch only — no crawl budget

SearchTool(
    web_search=WebSearch(max_results=10, search_depth="advanced"),
    web_fetch=WebFetch(timeout=60, extract_depth="advanced"),
)

SearchTool(                                      # tighter fetch — smaller tool responses
    web_fetch=WebFetch(chunks_per_source=2),
)

SearchTool(                                      # one page, covered exhaustively
    web_fetch=WebFetch(chunks_per_source=15, extract_depth="advanced"),
)

SearchTool(                                      # bounded crawl
    web_crawl=WebCrawl(max_depth=2, max_breadth=20, limit=50, timeout=60),
)

SearchTool(                                      # narrow the description the model reads
    web_search=WebSearch(
        instructions="Prefer primary sources and official documentation over blog posts.",
    ),
)
```

### Import paths

```python
from akgentic.tool.search import SearchTool, WebCrawl, WebFetch, WebSearch
```

---

See the [package README](../../../../README.md) for the `ToolCard` / `ToolFactory` machinery, the
channel system, and the error-handling contract.
