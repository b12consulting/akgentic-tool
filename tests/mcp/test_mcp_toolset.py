"""Tests for MCP toolset construction on pydantic-ai v2.

These tests are the only gate on this module: `akgentic.tool.mcp.mcp` sits on the mypy
`ignore_errors` override list, so a wrong translation of the connection config onto
pydantic-ai's `MCPToolset` is invisible to every other check. They therefore assert the
*configuration of the constructed object* — transport type, url, headers, command, args,
env, cwd, init timeout, read timeout, prefix — never merely that construction did not raise.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from typing import Any

import pytest
from pydantic_ai.mcp import MCPToolset, SSETransport, StdioTransport, StreamableHttpTransport
from pydantic_ai.toolsets.prefixed import PrefixedToolset

from akgentic.tool.mcp.mcp import (
    MCPHTTPConnectionConfig,
    MCPStdioConnectionConfig,
    MCPTool,
    _build_mcp_toolset,
    _load_mcp_toolset_class,
    _load_mcp_transport_classes,
    _mcp_auth_headers,
    list_mcp_tools,
    probe_mcp_connection,
)

ACME_URL = "https://mcp.acme.example/api/v1/endpoint"
CONTOSO_TOKEN = "contoso-secret-token"


def _transport_of(toolset: Any) -> Any:
    """Return the fastmcp transport the toolset's client was built over."""
    return toolset.client.transport


def _init_timeout_of(toolset: Any) -> float:
    """Return the resolved connection-initialization timeout, in seconds."""
    return toolset.client._init_timeout


def _read_timeout_of(toolset: Any) -> timedelta:
    """Return the resolved session read timeout."""
    return toolset.client._session_kwargs["read_timeout_seconds"]


def _sse_read_timeout_of(toolset: Any) -> timedelta | None:
    """Return the long-lived SSE event-stream timeout, read off the transport itself.

    Deliberately not the session kwargs: `MCPToolset(read_timeout=)` reaches only
    `ClientSession.read_timeout_seconds`, the *per-request* timeout. The stream timeout
    lives on the transport, so a session-level assertion is blind to it.
    """
    return _transport_of(toolset).sse_read_timeout


class _RecordingToolset:
    """Stand-in for `MCPToolset` that records how it was constructed.

    Mirrors the parts of the real class this module relies on: `prefixed()` returns a
    genuine `PrefixedToolset`, which — like upstream — exposes no `list_tools()` and no
    attribute proxy. Diagnostics routed through a prefixed toolset therefore fail here
    exactly as they would at runtime.
    """

    instances: list[_RecordingToolset] = []

    def __init__(self, client: Any, **kwargs: Any) -> None:
        self.client = client
        self.kwargs = kwargs
        self.entered = 0
        self.exited = 0
        _RecordingToolset.instances.append(self)

    def prefixed(self, prefix: str) -> PrefixedToolset[Any]:
        return PrefixedToolset(self, prefix)  # type: ignore[arg-type]

    async def __aenter__(self) -> _RecordingToolset:
        self.entered += 1
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.exited += 1

    async def list_tools(self) -> list[Any]:
        return [_FakeToolDef("read_file"), _FakeToolDef("write_file")]


class _FakeToolDef:
    """Minimal stand-in for an MCP tool definition (only `.name` is read)."""

    def __init__(self, name: str) -> None:
        self.name = name


@pytest.fixture
def recording_toolset(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingToolset]:
    """Replace `MCPToolset` with a recorder; the transport is still built for real."""
    import pydantic_ai.mcp

    _RecordingToolset.instances = []
    monkeypatch.setattr(pydantic_ai.mcp, "MCPToolset", _RecordingToolset)
    return _RecordingToolset


# --------------------------------------------------------------------------------------
# Lazy loaders — the symbols exist, and a genuine ImportError is not rewritten
# --------------------------------------------------------------------------------------


def test_loaders_return_the_v2_symbols() -> None:
    assert _load_mcp_toolset_class() is MCPToolset
    assert _load_mcp_transport_classes() == (StdioTransport, SSETransport, StreamableHttpTransport)


@pytest.mark.parametrize("loader", [_load_mcp_toolset_class, _load_mcp_transport_classes])
def test_import_failure_surfaces_its_real_cause_not_installation_advice(
    loader: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real ImportError must propagate unrewritten.

    `mcp` ships unconditionally with pydantic-ai v2, so the old "install
    pydantic-ai-slim[mcp]" guidance misdiagnosed a removed symbol as a missing package.
    """
    monkeypatch.setitem(sys.modules, "pydantic_ai.mcp", None)

    with pytest.raises(ImportError) as excinfo:
        loader()

    message = str(excinfo.value)
    assert "pydantic_ai.mcp" in message
    assert "pydantic-ai-slim[mcp]" not in message
    assert "extras" not in message


# --------------------------------------------------------------------------------------
# Transport selection and translation
# --------------------------------------------------------------------------------------


def test_default_transport_builds_streamable_http_transport() -> None:
    toolset = _build_mcp_toolset(MCPHTTPConnectionConfig(url=ACME_URL))

    transport = _transport_of(toolset)
    assert isinstance(transport, StreamableHttpTransport)
    assert transport.url == ACME_URL
    assert transport.headers == {}


def test_sse_transport_is_explicit_and_not_url_inferred() -> None:
    """`transport="sse"` must select SSE for a URL that inference would call HTTP.

    pydantic-ai only auto-detects SSE for URLs ending in `/sse`; our configs carry
    arbitrary paths, so relying on inference would silently downgrade the endpoint.
    """
    connection = MCPHTTPConnectionConfig(url=ACME_URL, transport="sse")

    transport = _transport_of(_build_mcp_toolset(connection))

    assert isinstance(transport, SSETransport)
    assert not isinstance(transport, StreamableHttpTransport)
    assert transport.url == ACME_URL


def test_stdio_builds_stdio_transport_with_command_args_and_cwd() -> None:
    connection = MCPStdioConnectionConfig(
        stdio_command="uvx",
        stdio_args=["acme-mcp-server", "--verbose"],
        stdio_cwd="/srv/acme",
        stdio_env={"ACME_REGION": "eu-west-1"},
    )

    transport = _transport_of(_build_mcp_toolset(connection))

    assert isinstance(transport, StdioTransport)
    assert transport.command == "uvx"
    assert transport.args == ["acme-mcp-server", "--verbose"]
    assert transport.cwd == "/srv/acme"
    assert transport.env == {"ACME_REGION": "eu-west-1"}


def test_stdio_missing_command_raises_before_any_toolset_is_built(
    recording_toolset: type[_RecordingToolset],
) -> None:
    with pytest.raises(ValueError, match="stdio_command is required for MCPStdioConnectionConfig"):
        _build_mcp_toolset(MCPStdioConnectionConfig(stdio_args=["--flag"]))

    assert recording_toolset.instances == []


# --------------------------------------------------------------------------------------
# stdio environment marshalling
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stdio_env", "token_env_var", "bearer_token", "expected_env"),
    [
        (None, None, None, None),
        ({}, None, None, None),
        ({"ACME_REGION": "eu-west-1"}, None, None, {"ACME_REGION": "eu-west-1"}),
        (None, "ACME_TOKEN", CONTOSO_TOKEN, {"ACME_TOKEN": CONTOSO_TOKEN}),
        (None, "ACME_TOKEN", None, None),
        (None, None, CONTOSO_TOKEN, None),
        (
            {"ACME_REGION": "eu-west-1"},
            "ACME_TOKEN",
            CONTOSO_TOKEN,
            {"ACME_REGION": "eu-west-1", "ACME_TOKEN": CONTOSO_TOKEN},
        ),
    ],
)
def test_stdio_env_injection(
    stdio_env: dict[str, str] | None,
    token_env_var: str | None,
    bearer_token: str | None,
    expected_env: dict[str, str] | None,
) -> None:
    connection = MCPStdioConnectionConfig(
        stdio_command="docker",
        stdio_env=stdio_env,
        stdio_token_env_var=token_env_var,
        bearer_token=bearer_token,
    )

    transport = _transport_of(_build_mcp_toolset(connection))

    assert transport.env == expected_env


def test_stdio_env_is_a_copy_of_the_config_value() -> None:
    """Injection must not mutate the persisted config's own dict."""
    connection = MCPStdioConnectionConfig(
        stdio_command="docker",
        stdio_env={"ACME_REGION": "eu-west-1"},
        stdio_token_env_var="ACME_TOKEN",
        bearer_token=CONTOSO_TOKEN,
    )

    _build_mcp_toolset(connection)

    assert connection.stdio_env == {"ACME_REGION": "eu-west-1"}


# --------------------------------------------------------------------------------------
# Bearer-token headers
# --------------------------------------------------------------------------------------


def test_mcp_auth_headers_with_token() -> None:
    assert _mcp_auth_headers(CONTOSO_TOKEN) == {"Authorization": f"Bearer {CONTOSO_TOKEN}"}


@pytest.mark.parametrize("token", [None, ""])
def test_mcp_auth_headers_without_token(token: str | None) -> None:
    assert _mcp_auth_headers(token) is None


@pytest.mark.parametrize(
    ("transport_type", "expected_cls"),
    [("streamable-http", StreamableHttpTransport), ("sse", SSETransport)],
)
def test_bearer_token_reaches_the_http_transport(
    transport_type: Any,
    expected_cls: type[Any],
) -> None:
    connection = MCPHTTPConnectionConfig(
        url=ACME_URL,
        transport=transport_type,
        bearer_token=CONTOSO_TOKEN,
    )

    transport = _transport_of(_build_mcp_toolset(connection))

    assert isinstance(transport, expected_cls)
    assert transport.headers == {"Authorization": f"Bearer {CONTOSO_TOKEN}"}


@pytest.mark.parametrize("transport_type", ["streamable-http", "sse"])
def test_headers_are_never_passed_to_the_toolset(
    transport_type: Any,
    recording_toolset: type[_RecordingToolset],
) -> None:
    """pydantic-ai raises ValueError for `headers=` alongside a transport object.

    Headers belong on the transport constructor, so the toolset must receive none.
    """
    connection = MCPHTTPConnectionConfig(
        url=ACME_URL,
        transport=transport_type,
        bearer_token=CONTOSO_TOKEN,
    )

    _build_mcp_toolset(connection)

    (built,) = recording_toolset.instances
    assert "headers" not in built.kwargs
    assert built.client.headers == {"Authorization": f"Bearer {CONTOSO_TOKEN}"}


def test_bearer_token_on_http_config_does_not_raise_on_real_toolset() -> None:
    """Regression guard for the ValueError described above, end to end."""
    connection = MCPHTTPConnectionConfig(url=ACME_URL, bearer_token=CONTOSO_TOKEN)

    toolset = _build_mcp_toolset(connection)

    assert isinstance(toolset, MCPToolset)


# --------------------------------------------------------------------------------------
# Timeout translation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "connection",
    [
        MCPHTTPConnectionConfig(url=ACME_URL, timeout=12.5, read_timeout=77.0),
        MCPHTTPConnectionConfig(url=ACME_URL, transport="sse", timeout=12.5, read_timeout=77.0),
        MCPStdioConnectionConfig(stdio_command="uvx", timeout=12.5, read_timeout=77.0),
    ],
    ids=["streamable-http", "sse", "stdio"],
)
def test_timeout_maps_to_init_timeout_and_read_timeout_passes_through(
    connection: Any,
) -> None:
    toolset = _build_mcp_toolset(connection)

    assert _init_timeout_of(toolset) == 12.5
    assert _read_timeout_of(toolset) == timedelta(seconds=77.0)


def test_config_defaults_win_over_pydantic_ai_defaults() -> None:
    """Both timeouts are always passed explicitly, so v2's own defaults never apply."""
    toolset = _build_mcp_toolset(MCPHTTPConnectionConfig(url=ACME_URL))

    assert _init_timeout_of(toolset) == 10.0
    assert _read_timeout_of(toolset) == timedelta(seconds=300.0)


def test_timeouts_are_passed_as_explicit_keywords(
    recording_toolset: type[_RecordingToolset],
) -> None:
    _build_mcp_toolset(MCPHTTPConnectionConfig(url=ACME_URL, timeout=3.0, read_timeout=9.0))

    (built,) = recording_toolset.instances
    assert built.kwargs["init_timeout"] == 3.0
    assert built.kwargs["read_timeout"] == 9.0
    assert "timeout" not in built.kwargs
    assert "tool_prefix" not in built.kwargs


@pytest.mark.parametrize("read_timeout", [900.0, 45.0], ids=["longer", "shorter"])
def test_sse_stream_timeout_comes_from_the_config(read_timeout: float) -> None:
    """The configured value must govern how long the SSE event stream tolerates silence.

    pydantic-ai would set this itself, but only when it builds the transport from a URL;
    handed a pre-built transport it returns early. fastmcp then leaves `sse_read_timeout`
    at `None` and omits the kwarg entirely, so `mcp.client.sse`'s own 5-minute default
    applies — loosening every configured value below it and truncating every one above.
    """
    connection = MCPHTTPConnectionConfig(
        url=ACME_URL,
        transport="sse",
        read_timeout=read_timeout,
    )

    toolset = _build_mcp_toolset(connection)

    assert _sse_read_timeout_of(toolset) == timedelta(seconds=read_timeout)


def test_streamable_http_keeps_its_session_timeout_and_gains_no_stream_timeout() -> None:
    """The kwarg is SSE-only by design, verified against the installed fastmcp.

    `StreamableHttpTransport` deprecates `sse_read_timeout` — it warns and ignores the
    value — and drives its read timeout from the session's `read_timeout_seconds`, which
    the toolset already supplies. So that branch must stay exactly as it was.
    """
    toolset = _build_mcp_toolset(MCPHTTPConnectionConfig(url=ACME_URL, read_timeout=77.0))

    assert isinstance(_transport_of(toolset), StreamableHttpTransport)
    assert _sse_read_timeout_of(toolset) is None
    assert _read_timeout_of(toolset) == timedelta(seconds=77.0)


# --------------------------------------------------------------------------------------
# MCPTool.get_toolsets — shape and prefixing
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "connection",
    [
        MCPHTTPConnectionConfig(url=ACME_URL),
        MCPHTTPConnectionConfig(url=ACME_URL, transport="sse"),
        MCPStdioConnectionConfig(stdio_command="uvx", stdio_args=["acme-mcp-server"]),
    ],
    ids=["streamable-http", "sse", "stdio"],
)
def test_get_toolsets_returns_exactly_one_bare_toolset(connection: Any) -> None:
    toolsets = MCPTool(connection=connection).get_toolsets()

    assert len(toolsets) == 1
    assert isinstance(toolsets[0], MCPToolset)
    assert not isinstance(toolsets[0], PrefixedToolset)


def test_get_toolsets_applies_tool_prefix_via_prefixed_wrapper() -> None:
    connection = MCPHTTPConnectionConfig(url=ACME_URL, tool_prefix="acme")

    (toolset,) = MCPTool(connection=connection).get_toolsets()

    assert isinstance(toolset, PrefixedToolset)
    assert toolset.prefix == "acme"
    assert isinstance(toolset.wrapped, MCPToolset)
    assert isinstance(_transport_of(toolset.wrapped), StreamableHttpTransport)


def test_get_toolsets_preserves_transport_config_under_a_prefix() -> None:
    connection = MCPStdioConnectionConfig(
        stdio_command="npx",
        stdio_args=["@acme/mcp"],
        tool_prefix="acme",
        timeout=4.0,
        read_timeout=8.0,
    )

    (toolset,) = MCPTool(connection=connection).get_toolsets()

    assert isinstance(toolset, PrefixedToolset)
    transport = _transport_of(toolset.wrapped)
    assert isinstance(transport, StdioTransport)
    assert transport.command == "npx"
    assert transport.args == ["@acme/mcp"]
    assert _init_timeout_of(toolset.wrapped) == 4.0
    assert _read_timeout_of(toolset.wrapped) == timedelta(seconds=8.0)


def test_get_tools_is_empty_because_mcp_exposes_toolsets() -> None:
    tool = MCPTool(connection=MCPHTTPConnectionConfig(url=ACME_URL))

    assert tool.get_tools() == []


def test_get_toolsets_propagates_missing_stdio_command() -> None:
    tool = MCPTool(connection=MCPStdioConnectionConfig(stdio_args=["--flag"]))

    with pytest.raises(ValueError, match="stdio_command is required for MCPStdioConnectionConfig"):
        tool.get_toolsets()


# --------------------------------------------------------------------------------------
# Diagnostics — must keep working when a tool_prefix is configured
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_prefix", [None, "acme"], ids=["no-prefix", "with-prefix"])
async def test_list_mcp_tools_reports_unprefixed_server_tool_names(
    tool_prefix: str | None,
    recording_toolset: type[_RecordingToolset],
) -> None:
    """A prefixed toolset has no `list_tools()`; diagnostics must bypass the wrapper."""
    connection = MCPHTTPConnectionConfig(url=ACME_URL, tool_prefix=tool_prefix)

    names = await list_mcp_tools(connection)

    assert names == ["read_file", "write_file"]
    (built,) = recording_toolset.instances
    assert built.entered == 1
    assert built.exited == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_prefix", [None, "acme"], ids=["no-prefix", "with-prefix"])
async def test_probe_mcp_connection_shape(
    tool_prefix: str | None,
    recording_toolset: type[_RecordingToolset],
) -> None:
    connection = MCPStdioConnectionConfig(stdio_command="uvx", tool_prefix=tool_prefix)

    result = await probe_mcp_connection(connection)

    assert result == {
        "tool_count": 2,
        "tools": ["read_file", "write_file"],
        "feasible": True,
    }


@pytest.mark.asyncio
async def test_probe_mcp_connection_truncates_the_tool_list(
    recording_toolset: type[_RecordingToolset],
) -> None:
    result = await probe_mcp_connection(
        MCPHTTPConnectionConfig(url=ACME_URL),
        max_tools_to_print=1,
    )

    assert result == {"tool_count": 2, "tools": ["read_file"], "feasible": True}


@pytest.mark.asyncio
async def test_probe_reports_infeasible_when_the_server_exposes_no_tools(
    recording_toolset: type[_RecordingToolset],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty tool list is the negative feasibility verdict, and `--cov-branch` misses it.

    `list_mcp_tools`' comprehension is a one-liner, so both endpoints of its loop arc share
    a line and coverage.py records no branch for it — the not-entered direction is invisible
    to branch measurement and has to be pinned behaviourally instead.
    """

    async def _no_tools(self: _RecordingToolset) -> list[Any]:
        return []

    monkeypatch.setattr(_RecordingToolset, "list_tools", _no_tools)

    result = await probe_mcp_connection(MCPHTTPConnectionConfig(url=ACME_URL))

    assert result == {"tool_count": 0, "tools": [], "feasible": False}


def test_real_mcptoolset_still_exposes_list_tools() -> None:
    """Every diagnostics test above runs against `_RecordingToolset`, which defines its own
    `list_tools()`. An upstream rename would therefore leave them all green and break only
    at runtime, so the real class is pinned here."""
    assert hasattr(MCPToolset, "list_tools")


def test_prefixed_toolset_really_has_no_list_tools() -> None:
    """Pins the upstream fact that makes the diagnostics routing necessary."""
    toolset = _build_mcp_toolset(MCPHTTPConnectionConfig(url=ACME_URL))

    prefixed = toolset.prefixed("acme")

    assert isinstance(prefixed, PrefixedToolset)
    assert not hasattr(prefixed, "list_tools")


# --------------------------------------------------------------------------------------
# Config models stay catalog-compatible
# --------------------------------------------------------------------------------------


def test_persisted_http_config_round_trips() -> None:
    stored = {
        "url": ACME_URL,
        "transport": "sse",
        "bearer_token": CONTOSO_TOKEN,
        "timeout": 15.0,
        "read_timeout": 120.0,
        "tool_prefix": "acme",
    }

    assert MCPHTTPConnectionConfig(**stored).model_dump() == stored


def test_persisted_stdio_config_round_trips() -> None:
    stored = {
        "transport": "stdio",
        "stdio_command": "docker",
        "stdio_args": ["run", "-i", "acme/mcp"],
        "stdio_env": {"ACME_REGION": "eu-west-1"},
        "stdio_cwd": "/srv/acme",
        "stdio_token_env_var": "ACME_TOKEN",
        "bearer_token": CONTOSO_TOKEN,
        "timeout": 15.0,
        "read_timeout": 120.0,
        "tool_prefix": "acme",
    }

    assert MCPStdioConnectionConfig(**stored).model_dump() == stored
