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
| `search_depth` | `"basic" \| "advanced" \| None` | `None` | `basic` balances relevance against latency and credit cost; `advanced` is more relevant, slower and more expensive. `None` sends nothing and lets Tavily apply its own default. |

Returns Tavily's raw search response — titles, URLs and snippets.

### `WebFetch` — `web_fetch(urls, timeout=…, extract_depth=…)`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{TOOL_CALL}` | |
| `timeout` | `float` | `30` | Seconds per extraction request. Tavily supports roughly 1–60. |
| `extract_depth` | `"basic" \| "advanced" \| None` | `None` | `basic` is faster and cheaper; `advanced` extracts more of the page. `None` ⇒ Tavily's default. |

Takes a **list** of absolute URLs and returns extracted content for each. Use it when the agent
already has the links — from a `web_search` result, from the user, from a document.

### `WebCrawl` — `web_crawl(url, timeout=…, max_depth=…, max_breadth=…, limit=…, crawl_instructions=…, extract_depth=…)`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `expose` | `set[Channels]` | `{TOOL_CALL}` | |
| `timeout` | `float` | `150` | Seconds for the whole crawl. Tavily supports 10–150. |
| `max_depth` | `int \| None` | `None` | Link depth from the root URL. Tavily supports 1–5. |
| `max_breadth` | `int \| None` | `None` | Links followed per level. Tavily supports 1–500. |
| `limit` | `int \| None` | `None` | Total pages processed before stopping. Must be ≥ 1. |
| `crawl_instructions` | `str \| None` | `None` | Natural-language guidance biasing the crawl and extraction toward a topic or section. Reaches Tavily under its own kwarg name, `instructions`. |
| `extract_depth` | `"basic" \| "advanced" \| None` | `None` | Extraction depth applied to crawled pages. |

`crawl_instructions` is **not** the inherited `instructions`, and the difference is the whole
point of the name:

| Field | Effect |
|---|---|
| `crawl_instructions` | Sent to Tavily. Biases which pages are crawled and what is extracted. Invisible to the model except as an argument default. |
| `instructions` *(inherited from `BaseToolParam`)* | Appended to the tool description the model reads. Never sent anywhere. |

```python
WebCrawl(
    crawl_instructions="focus on the API reference",   # biases the crawl
    instructions="Only crawl public documentation.",   # tells the model when to use it
)
```

Every `None` field is omitted from the Tavily call rather than sent as null, so Tavily's own
defaults apply.

Crawling is the expensive capability: a `max_depth=5, max_breadth=500` crawl can walk thousands of
pages. Set `limit` when exposing it to an agent that decides its own arguments.

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
