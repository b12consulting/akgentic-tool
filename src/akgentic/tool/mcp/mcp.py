"""Generic MCP protocol support for akgentic-tool.

This module contains protocol-level concerns only:
- MCP transport configuration
- MCP server toolset creation
- MCP diagnostics (tool listing)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from akgentic.tool.core import ToolCard

MCPHTTPTransport = Literal["streamable-http", "sse"]


class MCPHTTPConnectionConfig(BaseModel):
    """HTTP/SSE transport configuration for an MCP endpoint."""

    url: str = Field(description="MCP endpoint URL for HTTP/SSE transports")
    transport: MCPHTTPTransport = Field(
        default="streamable-http",
        description="MCP HTTP transport type",
    )
    bearer_token: str | None = Field(
        default=None,
        description="Optional bearer token for Authorization header",
    )
    timeout: float = Field(
        default=10.0,
        gt=0,
        description="Connection initialization timeout for MCP server "
        "(translated to pydantic-ai's init_timeout; deliberately not renamed, "
        "because this field name is persisted in catalog entries)",
    )
    read_timeout: float = Field(
        default=300.0,
        gt=0,
        description="Read timeout for MCP transport",
    )
    tool_prefix: str | None = Field(
        default=None,
        description="Optional tool name prefix applied via the toolset's prefixed() wrapper",
    )


class MCPStdioConnectionConfig(BaseModel):
    """stdio transport configuration for an MCP server subprocess."""

    transport: Literal["stdio"] = Field(
        default="stdio",
        description="MCP stdio transport",
    )
    stdio_command: str | None = Field(
        default=None,
        description="Command to launch MCP server in stdio mode (e.g., docker, npx, uvx)",
    )
    stdio_args: list[str] = Field(
        default_factory=list,
        description="Arguments for stdio command",
    )
    stdio_env: dict[str, str] | None = Field(
        default=None,
        description="Environment variables passed to stdio MCP process",
    )
    stdio_cwd: str | None = Field(
        default=None,
        description="Working directory for stdio MCP process",
    )
    stdio_token_env_var: str | None = Field(
        default=None,
        description="If set with bearer_token, inject token into this env var for stdio process",
    )
    bearer_token: str | None = Field(
        default=None,
        description="Optional token that can be injected into stdio env via stdio_token_env_var",
    )
    timeout: float = Field(
        default=10.0,
        gt=0,
        description="Connection initialization timeout for MCP server "
        "(translated to pydantic-ai's init_timeout; deliberately not renamed, "
        "because this field name is persisted in catalog entries)",
    )
    read_timeout: float = Field(
        default=300.0,
        gt=0,
        description="Read timeout for MCP transport",
    )
    tool_prefix: str | None = Field(
        default=None,
        description="Optional tool name prefix applied via the toolset's prefixed() wrapper",
    )


MCPConnectionConfig = MCPHTTPConnectionConfig | MCPStdioConnectionConfig
"""Union type for MCP connection configurations.

Supports both HTTP/SSE and stdio transport types.
"""


class MCPDiagnosticsConfig(BaseModel):
    """Generic diagnostics behavior for MCP probing."""

    max_tools_to_print: int = Field(default=20, ge=1)


def _mcp_auth_headers(bearer_token: str | None) -> dict[str, str] | None:
    """Build HTTP authorization headers from a bearer token.

    Args:
        bearer_token: Optional bearer token for authentication.

    Returns:
        Dictionary with Authorization header if token provided, None otherwise.
    """
    if not bearer_token:
        return None
    return {"Authorization": f"Bearer {bearer_token}"}


def _load_mcp_toolset_class() -> type[Any]:
    """Lazy-load the pydantic-ai MCP toolset class.

    The import stays lazy so `pydantic_ai` remains an implementation detail of this
    module. It is deliberately NOT wrapped in a try/except: this package declares the
    `mcp` extra itself (`pydantic-ai-slim[mcp]` in `pyproject.toml`), so the module is
    present in every correct install and a genuine ImportError must surface with its
    real cause instead of being rewritten into misleading installation advice. Dropping
    that extra would make this import optional — and this docstring wrong.

    Returns:
        The `MCPToolset` class.
    """
    from pydantic_ai.mcp import MCPToolset  # noqa: PLC0415

    return MCPToolset


def _load_mcp_transport_classes() -> tuple[type[Any], type[Any], type[Any]]:
    """Lazy-load the fastmcp transport classes re-exported by pydantic-ai.

    These are module-level names in `pydantic_ai.mcp` (they are not listed in its
    `__all__`). They are imported from there rather than from `fastmcp` directly:
    fastmcp arrives transitively via pydantic-ai and is not a declared dependency
    of this package.

    Returns:
        Tuple of (StdioTransport, SSETransport, StreamableHttpTransport) classes.
    """
    from pydantic_ai.mcp import (  # noqa: PLC0415
        SSETransport,
        StdioTransport,
        StreamableHttpTransport,
    )

    return StdioTransport, SSETransport, StreamableHttpTransport


def _build_stdio_transport(
    connection: MCPStdioConnectionConfig,
    stdio_transport_cls: type[Any],
) -> Any:
    """Map a stdio connection config onto a fastmcp stdio transport.

    Args:
        connection: stdio MCP connection configuration.
        stdio_transport_cls: The `StdioTransport` class to instantiate.

    Returns:
        A configured stdio transport.

    Raises:
        ValueError: If stdio_command is missing.
    """
    if not connection.stdio_command:
        raise ValueError("stdio_command is required for MCPStdioConnectionConfig")

    env = dict(connection.stdio_env or {})
    if connection.stdio_token_env_var and connection.bearer_token:
        env[connection.stdio_token_env_var] = connection.bearer_token

    return stdio_transport_cls(
        command=connection.stdio_command,
        args=connection.stdio_args,
        env=env or None,
        cwd=connection.stdio_cwd,
    )


def _build_mcp_transport(connection: MCPConnectionConfig) -> Any:
    """Map an MCP connection config onto a fastmcp transport object.

    The transport is always constructed explicitly. In particular `transport="sse"`
    must not fall through to URL-based inference, which only detects SSE for URLs
    ending in `/sse` and would silently downgrade a configured SSE endpoint.

    Auth headers are passed to the *transport* constructor, never to the toolset:
    pydantic-ai raises ValueError when `headers=` accompanies a transport object.

    SSE additionally needs `read_timeout` forwarded here as `sse_read_timeout`, because
    that is the only route to the long-lived event stream. The toolset's own
    `read_timeout` reaches just the per-request session timeout, and pydantic-ai's
    transport builder — which would set the stream timeout itself — returns early when
    given a pre-built transport, as it is here. Left unset, fastmcp omits the kwarg and
    `mcp.client.sse`'s 5-minute default silently overrides the configured value.

    The streamable-HTTP branch deliberately gets nothing: upstream deprecates
    `sse_read_timeout` there and derives that timeout from the session's
    `read_timeout_seconds`, which the toolset already supplies.

    Args:
        connection: MCP connection configuration (HTTP/SSE or stdio).

    Returns:
        A configured fastmcp transport instance.

    Raises:
        ValueError: If stdio_command is missing for stdio transport.
    """
    stdio_transport_cls, sse_transport_cls, http_transport_cls = _load_mcp_transport_classes()

    if isinstance(connection, MCPStdioConnectionConfig):
        return _build_stdio_transport(connection, stdio_transport_cls)

    headers = _mcp_auth_headers(connection.bearer_token)
    if connection.transport == "sse":
        return sse_transport_cls(
            url=connection.url,
            headers=headers,
            sse_read_timeout=connection.read_timeout,
        )
    return http_transport_cls(url=connection.url, headers=headers)


def _build_mcp_toolset(connection: MCPConnectionConfig) -> Any:
    """Build the bare, unprefixed MCP toolset for a connection config.

    `timeout` is translated to pydantic-ai's `init_timeout` here rather than renamed on
    the config models, which are catalog-persisted. Both timeouts are always passed
    explicitly so the config's values win over pydantic-ai's own (different) defaults.

    Accepted behaviour change — MCP tool-call retries. The v1 server classes hard-defaulted
    `max_retries=1`; `MCPToolset` defaults it to `None` and falls back to `ctx.max_retries`,
    so MCP tool calls now inherit the agent's retry budget (`Agent(retries=...)`, reached
    from `ReactAgentConfig.runtime_cfg.retries`) instead of always being capped at one.
    This is deliberate and left as-is: the connection configs are catalog-persisted, and
    adding a field to carry the old constant would change their stored shape for a value
    the agent already owns.

    Args:
        connection: MCP connection configuration (HTTP/SSE or stdio).

    Returns:
        An `MCPToolset` with no tool-name prefix applied.

    Raises:
        ValueError: If stdio_command is missing for stdio transport.
    """
    toolset_cls = _load_mcp_toolset_class()
    return toolset_cls(
        _build_mcp_transport(connection),
        init_timeout=connection.timeout,
        read_timeout=connection.read_timeout,
    )


class MCPTool(ToolCard):
    """MCP protocol integration — exposes tools via toolsets, not callables.

    Attributes:
        connection: MCP transport configuration (HTTP/SSE or stdio).
    """

    connection: MCPConnectionConfig

    def get_tools(self) -> list[Callable[..., Any]]:
        """MCP tools come via toolsets, not individual callables."""
        return []

    def get_toolsets(self) -> list[Any]:
        """Create and return an MCP toolset for pydantic-ai agents.

        Builds a single `MCPToolset` over the fastmcp transport selected by the
        connection configuration (stdio, SSE, or streamable-HTTP). When `tool_prefix`
        is configured it is applied via the toolset's `prefixed()` wrapper — pydantic-ai
        removed the constructor keyword.

        Returns:
            List containing a single configured toolset ready to be used in
            pydantic-ai agents.

        Raises:
            ValueError: If stdio_command is missing for stdio transport.
        """
        toolset = _build_mcp_toolset(self.connection)
        if self.connection.tool_prefix:
            return [toolset.prefixed(self.connection.tool_prefix)]
        return [toolset]


async def list_mcp_tools(connection: MCPConnectionConfig) -> list[str]:
    """Connect to an MCP server and return exposed tool names.

    Args:
        connection: MCP connection configuration (HTTP/SSE or stdio).

    Returns:
        List of tool names exposed by the MCP server, as the server itself reports
        them (i.e. without any configured `tool_prefix` applied).

    Raises:
        ValueError: If stdio_command is missing for stdio transport.
        Exception: If connection fails or server is unreachable.
    """
    print("## Creating MCP toolset for diagnostics...")
    # Deliberately built from the bare toolset rather than MCPTool.get_toolsets():
    # a prefixed toolset is a wrapper with no list_tools() and no attribute proxy,
    # so diagnostics would break for exactly the configs that set a tool_prefix.
    server = _build_mcp_toolset(connection)
    print("## Server toolset created, connecting and listing tools...")
    async with server:
        print("## Connected to MCP server, fetching tool list...")
        tools = await server.list_tools()
    return [tool_def.name for tool_def in tools]


async def probe_mcp_connection(
    connection: MCPConnectionConfig,
    *,
    max_tools_to_print: int = 20,
) -> dict[str, Any]:
    """Probe an MCP server and return a compact feasibility summary.

    Args:
        connection: MCP connection configuration to probe.
        max_tools_to_print: Maximum number of tool names to include in result.

    Returns:
        Dictionary with tool_count, tools (list), and feasible (bool).
    """
    tool_names = await list_mcp_tools(connection)
    return {
        "tool_count": len(tool_names),
        "tools": tool_names[:max_tools_to_print],
        "feasible": len(tool_names) > 0,
    }
