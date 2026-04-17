"""Generic MCP protocol support for akgentic-tool.

This module contains protocol-level concerns only:
- MCP transport configuration
- MCP server toolset creation
- MCP diagnostics (tool listing)
- Tool filtering and policy enforcement (ADR-023)
- MCPServerFactory for observer and hook management
"""

from __future__ import annotations

import fnmatch
from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field

from akgentic.tool.core import ToolCard
from akgentic.tool.event import ToolCallEvent, ToolObserver

MCPHTTPTransport: TypeAlias = Literal["streamable-http", "sse"]


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
        description="Connection initialization timeout for MCP server",
    )
    read_timeout: float = Field(
        default=300.0,
        gt=0,
        description="Read timeout for MCP transport",
    )
    tool_prefix: str | None = Field(
        default=None,
        description="Optional tool name prefix applied by pydantic-ai MCP wrapper",
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
        description="Connection initialization timeout for MCP server",
    )
    read_timeout: float = Field(
        default=300.0,
        gt=0,
        description="Read timeout for MCP transport",
    )
    tool_prefix: str | None = Field(
        default=None,
        description="Optional tool name prefix applied by pydantic-ai MCP wrapper",
    )


MCPConnectionConfig: TypeAlias = MCPHTTPConnectionConfig | MCPStdioConnectionConfig


class MCPDiagnosticsConfig(BaseModel):
    """Generic diagnostics behavior for MCP probing."""

    max_tools_to_print: int = Field(default=20, ge=1)


def _mcp_auth_headers(bearer_token: str | None) -> dict[str, str] | None:
    """Build HTTP authorization headers from a bearer token."""
    if not bearer_token:
        return None
    return {"Authorization": f"Bearer {bearer_token}"}


def _load_mcp_server_classes() -> tuple[type[Any], type[Any], type[Any]]:
    """Lazy-load MCP server classes from pydantic-ai."""
    try:
        from pydantic_ai.mcp import (  # noqa: PLC0415
            MCPServerSSE,
            MCPServerStdio,
            MCPServerStreamableHTTP,
        )
    except ImportError as error:  # pragma: no cover - environment-specific
        raise ImportError(
            "MCP support requires pydantic-ai MCP extras. "
            'Install with: pip install "pydantic-ai-slim[mcp]"'
        ) from error

    return MCPServerSSE, MCPServerStreamableHTTP, MCPServerStdio


def _matches_any_pattern(tool_name: str, patterns: set[str] | None) -> bool:
    """Check if a tool name matches any pattern in the set using fnmatch."""
    if patterns is None:
        return True
    return any(fnmatch.fnmatch(tool_name, pattern) for pattern in patterns)


def _extract_tool_metadata(tool_def: Any) -> tuple[str, dict[str, Any]]:
    """Extract a tool name and annotations from a tool definition."""
    if isinstance(tool_def, str):
        return tool_def, {}

    if isinstance(tool_def, dict):
        name = str(tool_def.get("name", ""))
        annotations = tool_def.get("annotations", {})
        return name, annotations if isinstance(annotations, dict) else {}

    name = str(getattr(tool_def, "name", ""))
    metadata = getattr(tool_def, "metadata", None)
    if not isinstance(metadata, dict):
        return name, {}

    annotations = metadata.get("annotations", {})
    return name, annotations if isinstance(annotations, dict) else {}


def _load_filtered_toolset_class() -> type[Any]:
    """Lazy-load the pydantic-ai filtered toolset wrapper."""
    try:
        from pydantic_ai.toolsets.filtered import FilteredToolset  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - environment-specific
        raise ImportError(
            "MCP filtering requires pydantic-ai MCP extras. "
            'Install with: pip install "pydantic-ai-slim[mcp]"'
        ) from error

    return FilteredToolset


def _apply_tool_filters(
    server: Any,
    allowed_tools: set[str] | None,
    blocked_tools: set[str] | None,
    read_only_tools: bool,
    tool_prefix: str | None = None,
) -> Any:
    """Apply filtering policy to an MCP toolset wrapper."""
    if not allowed_tools and not blocked_tools and not read_only_tools:
        return server

    filtered_toolset_class = _load_filtered_toolset_class()

    def filter_fn(_ctx: Any, tool_def: Any) -> bool:
        tool_name, annotations = _extract_tool_metadata(tool_def)
        normalized_name = tool_name
        if tool_prefix and tool_name.startswith(f"{tool_prefix}_"):
            normalized_name = tool_name[len(tool_prefix) + 1 :]

        if allowed_tools and not _matches_any_pattern(normalized_name, allowed_tools):
            return False

        if read_only_tools and not (
            annotations.get("readOnly") or annotations.get("readOnlyHint")
        ):
            return False

        if blocked_tools and _matches_any_pattern(normalized_name, blocked_tools):
            return False

        return True

    return filtered_toolset_class(server, filter_fn)


class MCPServerFactory:
    """Factory for MCP servers with observer-backed process hooks."""

    def __init__(self, get_observer: Callable[[], ToolObserver | None] | None = None) -> None:
        self._get_observer = get_observer or (lambda: None)

    def _build_process_tool_call(self) -> Callable[..., Awaitable[Any]]:
        async def process_tool_call(
            _ctx: Any,
            call_next: Callable[[str, dict[str, Any]], Awaitable[Any]],
            tool_name: str,
            tool_args: dict[str, Any],
        ) -> Any:
            observer = self._get_observer()
            if observer is not None:
                observer.notify_event(
                    ToolCallEvent(tool_name=tool_name, args=[], kwargs=tool_args)
                )
            return await call_next(tool_name, tool_args)

        return process_tool_call

    def create(self, connection: MCPConnectionConfig) -> Any:
        """Create a transport-specific MCP server."""
        if isinstance(connection, MCPStdioConnectionConfig) and not connection.stdio_command:
            raise ValueError("stdio_command is required for MCPStdioConnectionConfig")

        mcp_server_sse, mcp_server_streamable_http, mcp_server_stdio = (
            _load_mcp_server_classes()
        )
        process_tool_call = self._build_process_tool_call()
        headers = _mcp_auth_headers(connection.bearer_token)

        if isinstance(connection, MCPStdioConnectionConfig):
            env = dict(connection.stdio_env or {})
            if connection.stdio_token_env_var and connection.bearer_token:
                env[connection.stdio_token_env_var] = connection.bearer_token

            return mcp_server_stdio(
                command=connection.stdio_command,
                args=connection.stdio_args,
                env=env or None,
                cwd=connection.stdio_cwd,
                tool_prefix=connection.tool_prefix,
                timeout=connection.timeout,
                read_timeout=connection.read_timeout,
                process_tool_call=process_tool_call,
            )

        if connection.transport == "sse":
            return mcp_server_sse(
                url=connection.url,
                headers=headers,
                tool_prefix=connection.tool_prefix,
                timeout=connection.timeout,
                read_timeout=connection.read_timeout,
                process_tool_call=process_tool_call,
            )

        return mcp_server_streamable_http(
            url=connection.url,
            headers=headers,
            tool_prefix=connection.tool_prefix,
            timeout=connection.timeout,
            read_timeout=connection.read_timeout,
            process_tool_call=process_tool_call,
        )


class MCPTool(ToolCard):
    """MCP protocol integration — exposes tools via toolsets, not callables."""

    connection: MCPConnectionConfig
    allowed_tools: set[str] | None = Field(
        default=None,
        description="Allowlist tool patterns (fnmatch); None means all allowed.",
    )
    blocked_tools: set[str] | None = Field(
        default=None,
        description="Blocklist tool patterns (fnmatch); takes precedence.",
    )
    read_only_tools: bool = Field(
        default=False,
        description="If True, only expose tools with readOnly/readOnlyHint.",
    )

    def get_tools(self) -> list[Callable[..., Any]]:
        """MCP tools come via toolsets, not individual callables."""
        return []

    def get_toolsets(self) -> list[Any]:
        factory = MCPServerFactory()
        created_server = factory.create(self.connection)
        filtered_server = _apply_tool_filters(
            created_server,
            self.allowed_tools,
            self.blocked_tools,
            read_only_tools=self.read_only_tools,
            tool_prefix=self.connection.tool_prefix,
        )
        return [filtered_server]


async def list_mcp_tools(connection: MCPConnectionConfig) -> list[str]:
    """Connect to an MCP server and return exposed tool names."""
    tool = MCPTool(name="mcp-probe", description="Probe MCP endpoint", connection=connection)
    toolsets = tool.get_toolsets()
    if not toolsets:
        raise ValueError("MCPTool.get_toolsets() returned empty list")

    server = toolsets[0]
    async with server:
        tools = await server.list_tools()

    return [_extract_tool_metadata(tool_def)[0] for tool_def in tools]


async def probe_mcp_connection(
    connection: MCPConnectionConfig,
    *,
    max_tools_to_print: int = 20,
) -> dict[str, Any]:
    """Probe an MCP server and return a compact feasibility summary."""
    tool_defs = await list_mcp_tools(connection)

    tool_names: list[str] = []
    tool_annotations: list[dict[str, Any]] = []
    for tool_def in tool_defs[:max_tools_to_print]:
        tool_name, annotations = _extract_tool_metadata(tool_def)
        tool_names.append(tool_name)
        tool_annotations.append({"name": tool_name, "annotations": annotations})

    return {
        "tool_count": len(tool_defs),
        "tools": tool_names,
        "tool_annotations": tool_annotations,
        "feasible": len(tool_defs) > 0,
    }
