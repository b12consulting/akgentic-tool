"""Tests for MCP toolset construction on pydantic-ai v2.

These tests are the only gate on this module: `akgentic.tool.mcp.mcp` sits on the mypy
`ignore_errors` override list, so a wrong translation of the connection config onto
pydantic-ai's `MCPToolset` is invisible to every other check. They therefore assert the
*configuration of the constructed object* — transport type, url, headers, command, args,
env, cwd, init timeout, read timeout, prefix — never merely that construction did not raise.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import pytest
from fastmcp.client.transports import SSETransport, StdioTransport, StreamableHttpTransport
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets.prefixed import PrefixedToolset

from akgentic.tool.mcp.mcp import (
    MCP_ENABLED_ENV,
    MCP_SECRET_PREFIX,
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


def _seconds(value: timedelta | float | None) -> float | None:
    """Normalize a timeout to seconds, whichever way fastmcp chose to store it.

    Not one convention: fastmcp 4.0.0 normalizes the *session* read timeout to a plain
    `float` (it was a `timedelta` up to 3.x) while `SSETransport` still stores
    `sse_read_timeout` as a `timedelta`. Both branches below are therefore live, and
    neither is dead code on any version in range. `src/` passes a float throughout, so
    the container is upstream's business and the tests assert the *value in seconds*.
    """
    return value.total_seconds() if isinstance(value, timedelta) else value


def _read_timeout_of(toolset: Any) -> float | None:
    """Return the resolved session read timeout, in seconds."""
    return _seconds(toolset.client._session_kwargs["read_timeout_seconds"])


def _sse_read_timeout_of(toolset: Any) -> float | None:
    """Return the long-lived SSE event-stream timeout in seconds, off the transport itself.

    Deliberately not the session kwargs: `MCPToolset(read_timeout=)` reaches only
    `ClientSession.read_timeout_seconds`, the *per-request* timeout. The stream timeout
    lives on the transport, so a session-level assertion is blind to it.

    `getattr` because `StreamableHttpTransport` dropped the attribute in fastmcp 4.0.0:
    absent and `None` both mean "this transport carries no stream timeout", which is
    exactly what the streamable-HTTP branch asserts.
    """
    return _seconds(getattr(_transport_of(toolset), "sse_read_timeout", None))


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


@pytest.mark.parametrize(
    ("loader", "source_module"),
    [
        (_load_mcp_toolset_class, "pydantic_ai.mcp"),
        (_load_mcp_transport_classes, "fastmcp.client.transports"),
    ],
)
def test_import_failure_surfaces_its_real_cause_not_installation_advice(
    loader: Any,
    source_module: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real ImportError must propagate unrewritten.

    Both sources are declared dependencies of this package, so the old "install
    pydantic-ai-slim[mcp]" guidance misdiagnosed a removed symbol as a missing package.
    """
    monkeypatch.setitem(sys.modules, source_module, None)

    with pytest.raises(ImportError) as excinfo:
        loader()

    message = str(excinfo.value)
    assert source_module in message
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
    assert _read_timeout_of(toolset) == 77.0


def test_config_defaults_win_over_pydantic_ai_defaults() -> None:
    """Both timeouts are always passed explicitly, so v2's own defaults never apply."""
    toolset = _build_mcp_toolset(MCPHTTPConnectionConfig(url=ACME_URL))

    assert _init_timeout_of(toolset) == 10.0
    assert _read_timeout_of(toolset) == 300.0


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

    assert _sse_read_timeout_of(toolset) == read_timeout


def test_streamable_http_keeps_its_session_timeout_and_gains_no_stream_timeout() -> None:
    """The kwarg is SSE-only by design, verified against the installed fastmcp.

    `StreamableHttpTransport` deprecates `sse_read_timeout` — it warns and ignores the
    value — and drives its read timeout from the session's `read_timeout_seconds`, which
    the toolset already supplies. So that branch must stay exactly as it was.
    """
    toolset = _build_mcp_toolset(MCPHTTPConnectionConfig(url=ACME_URL, read_timeout=77.0))

    assert isinstance(_transport_of(toolset), StreamableHttpTransport)
    assert _sse_read_timeout_of(toolset) is None
    assert _read_timeout_of(toolset) == 77.0


def _sse_read_timeout_deprecations(build: Any) -> list[str]:
    """Return the `sse_read_timeout` deprecation messages emitted while building."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build()
    return [str(w.message) for w in caught if "sse_read_timeout" in str(w.message)]


def test_sse_still_honours_the_kwarg_that_streamable_http_already_discards() -> None:
    """Reading the value back off the transport is not enough on its own.

    If SSE ever stops honouring `sse_read_timeout` the way streamable-HTTP already does,
    the attribute assertion above would go on passing while `mcp.client.sse`'s 5-minute
    default silently returned. So this asserts the *signal*, not the stored value.

    The streamable-HTTP half is the positive control, proving the divergence is real in
    this environment rather than merely unobserved. Upstream has escalated how it refuses
    the kwarg — fastmcp 3.x warned and ignored it, fastmcp 4.0.0 removed it outright — so
    the control accepts either refusal: a deprecation warning, or a `TypeError` naming
    that kwarg. What must not happen is streamable-HTTP accepting it silently, which
    would make the control vacuous and this test meaningless.
    """
    try:
        discarded: list[str] | None = _sse_read_timeout_deprecations(
            lambda: StreamableHttpTransport(url=ACME_URL, sse_read_timeout=900.0)
        )
    except TypeError as exc:
        # fastmcp >= 4.0.0 removed the kwarg: refusal, escalated. The message check keeps
        # the catch narrow — an unrelated signature change (a renamed `url`, a new
        # required argument) also raises TypeError here, and would otherwise read as a
        # refusal and quietly satisfy the control.
        assert "sse_read_timeout" in str(exc), f"TypeError is not about the kwarg: {exc}"
        discarded = None

    honoured = _sse_read_timeout_deprecations(
        lambda: _build_mcp_toolset(
            MCPHTTPConnectionConfig(url=ACME_URL, transport="sse", read_timeout=900.0)
        )
    )

    assert discarded is None or discarded, (
        "streamable-HTTP now accepts sse_read_timeout silently; the control is vacuous"
    )
    assert honoured == []


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
    assert _read_timeout_of(toolset.wrapped) == 8.0


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


# --------------------------------------------------------------------------------------
# ${VAR} placeholders — resolved into the transport, never into the card
# --------------------------------------------------------------------------------------

SECRET_VAR = MCP_SECRET_PREFIX + "CLIENT"
SECRET_VALUE = "contoso-resolved-secret"
REGION_VAR = MCP_SECRET_PREFIX + "REGION"
REGION_VALUE = "eu-west-1"


@pytest.fixture
def secret_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Export the placeholder targets on the 'worker' for the duration of one test."""
    monkeypatch.setenv(SECRET_VAR, SECRET_VALUE)
    monkeypatch.setenv(REGION_VAR, REGION_VALUE)


def _placeholder_stdio_config() -> MCPStdioConnectionConfig:
    return MCPStdioConnectionConfig(
        stdio_command="uvx",
        stdio_args=["acme-mcp-server", "--region=${" + REGION_VAR + "}"],
        stdio_env={"CLIENT_SECRET": "${" + SECRET_VAR + "}", "ACME_PLAIN": "unchanged"},
        stdio_token_env_var="ACME_TOKEN",
        bearer_token="${" + SECRET_VAR + "}",
    )


def _placeholder_http_config(transport: Any = "streamable-http") -> MCPHTTPConnectionConfig:
    return MCPHTTPConnectionConfig(
        url=ACME_URL + "?region=${" + REGION_VAR + "}",
        transport=transport,
        bearer_token="${" + SECRET_VAR + "}",
    )


@pytest.mark.usefixtures("secret_env")
def test_stdio_transport_carries_resolved_env_args_and_token() -> None:
    transport = _transport_of(_build_mcp_toolset(_placeholder_stdio_config()))

    assert transport.env == {
        "CLIENT_SECRET": SECRET_VALUE,
        "ACME_PLAIN": "unchanged",
        "ACME_TOKEN": SECRET_VALUE,
    }
    assert transport.args == ["acme-mcp-server", f"--region={REGION_VALUE}"]


@pytest.mark.usefixtures("secret_env")
@pytest.mark.parametrize("transport_type", ["streamable-http", "sse"])
def test_http_transport_carries_resolved_url_and_token(transport_type: Any) -> None:
    transport = _transport_of(_build_mcp_toolset(_placeholder_http_config(transport_type)))

    assert transport.url == f"{ACME_URL}?region={REGION_VALUE}"
    assert transport.headers == {"Authorization": f"Bearer {SECRET_VALUE}"}


@pytest.mark.usefixtures("secret_env")
@pytest.mark.parametrize(
    "connection_factory",
    [_placeholder_stdio_config, _placeholder_http_config],
    ids=["stdio", "http"],
)
def test_building_never_writes_the_resolved_value_back_onto_the_card(
    connection_factory: Callable[[], Any],
) -> None:
    """The card is serialized into team cards and the event store: the secret must not be."""
    card = MCPTool(connection=connection_factory())
    before = card.model_dump()

    card.get_toolsets()
    after = card.model_dump()

    assert after == before
    assert SECRET_VALUE not in str(after)
    assert REGION_VALUE not in str(after)
    assert "${" + SECRET_VAR + "}" in str(after)


@pytest.mark.parametrize(
    "connection_factory",
    [_placeholder_stdio_config, _placeholder_http_config],
    ids=["stdio", "http"],
)
@pytest.mark.parametrize("exported", [None, ""], ids=["unset", "empty"])
def test_missing_variable_fails_at_build_naming_it_and_no_value(
    connection_factory: Callable[[], Any],
    exported: str | None,
    monkeypatch: pytest.MonkeyPatch,
    recording_toolset: type[_RecordingToolset],
) -> None:
    monkeypatch.setenv(REGION_VAR, REGION_VALUE)
    if exported is None:
        monkeypatch.delenv(SECRET_VAR, raising=False)
    else:
        monkeypatch.setenv(SECRET_VAR, exported)

    with pytest.raises(ValueError, match=f"'{SECRET_VAR}'") as excinfo:
        MCPTool(connection=connection_factory()).get_toolsets()

    assert REGION_VALUE not in str(excinfo.value)
    assert recording_toolset.instances == []


@pytest.mark.parametrize(
    "literal",
    ["plain-token", "$ACME_MCP_CLIENT_SECRET", "${not-a-name}", "${", "price: $5"],
)
def test_strings_without_a_placeholder_pass_unchanged(
    literal: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SECRET_VAR, SECRET_VALUE)
    connection = MCPStdioConnectionConfig(
        stdio_command="uvx",
        stdio_args=[literal],
        stdio_env={"ACME_VALUE": literal},
    )

    transport = _transport_of(_build_mcp_toolset(connection))

    assert transport.args == [literal]
    assert transport.env == {"ACME_VALUE": literal}


@pytest.mark.usefixtures("secret_env")
def test_several_placeholders_in_one_string_all_resolve() -> None:
    connection = MCPHTTPConnectionConfig(
        url=ACME_URL + "?region=${" + REGION_VAR + "}&key=${" + SECRET_VAR + "}",
    )

    transport = _transport_of(_build_mcp_toolset(connection))

    assert transport.url == f"{ACME_URL}?region={REGION_VALUE}&key={SECRET_VALUE}"


@pytest.mark.asyncio
@pytest.mark.usefixtures("secret_env")
async def test_diagnostics_resolve_placeholders_too(
    recording_toolset: type[_RecordingToolset],
) -> None:
    """`list_mcp_tools` builds through the same path, so a probe reaches the real server."""
    await list_mcp_tools(_placeholder_http_config())

    (built,) = recording_toolset.instances
    assert built.client.headers == {"Authorization": f"Bearer {SECRET_VALUE}"}


@pytest.mark.parametrize("field", ["url", "bearer_token"])
def test_a_placeholder_outside_the_prefix_is_refused_even_when_set(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
    recording_toolset: type[_RecordingToolset],
) -> None:
    """A card must not reach the rest of the worker's environment (an LLM key, a DB URL)."""
    monkeypatch.setenv("ACME_DB_PASSWORD", "contoso-db-password")
    kwargs: dict[str, Any] = {"url": ACME_URL, field: "${ACME_DB_PASSWORD}"}
    if field == "url":
        kwargs["url"] = ACME_URL + "?k=${ACME_DB_PASSWORD}"

    with pytest.raises(ValueError, match="'ACME_DB_PASSWORD'") as excinfo:
        MCPTool(connection=MCPHTTPConnectionConfig(**kwargs)).get_toolsets()

    assert MCP_SECRET_PREFIX in str(excinfo.value)
    assert "contoso-db-password" not in str(excinfo.value)
    assert recording_toolset.instances == []


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ACME_DB_PASSWORD", "contoso-db-password"),
        ("ACME_" + MCP_SECRET_PREFIX + "DB", "contoso-db-password"),
        (MCP_SECRET_PREFIX.lower() + "db", "contoso-db-password"),
        (MCP_ENABLED_ENV, "true"),
    ],
    ids=["unrelated", "prefix-not-leading", "prefix-lower-case", "the-switch-itself"],
)
def test_a_stdio_placeholder_outside_the_prefix_is_refused(
    name: str,
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prefix must lead the name, in its exact case; the switch is not readable either."""
    monkeypatch.setenv(name, value)
    connection = MCPStdioConnectionConfig(
        stdio_command="uvx",
        stdio_env={"PASSWORD": "${" + name + "}"},
    )

    with pytest.raises(ValueError, match=f"'{name}'") as excinfo:
        _build_mcp_toolset(connection)

    assert MCP_SECRET_PREFIX + "*" in str(excinfo.value)
    assert value not in str(excinfo.value)


@pytest.mark.usefixtures("secret_env")
def test_names_and_locations_are_never_resolved() -> None:
    """Only fields that can carry a credential are resolved; the rest stay as written."""
    placeholder = "${" + SECRET_VAR + "}"
    connection = MCPStdioConnectionConfig(
        stdio_command=placeholder,
        stdio_cwd=placeholder,
        stdio_env={"ACME_" + placeholder: "plain"},
        stdio_token_env_var=placeholder,
        bearer_token="acme-token",
        tool_prefix=placeholder,
    )

    (toolset,) = MCPTool(connection=connection).get_toolsets()
    transport = _transport_of(toolset.wrapped)

    assert transport.command == placeholder
    assert transport.cwd == placeholder
    assert transport.env == {"ACME_" + placeholder: "plain", placeholder: "acme-token"}
    assert toolset.prefix == placeholder


# --------------------------------------------------------------------------------------
# The worker switch — MCPTool is off unless the operator turns it on
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "connection",
    [
        MCPHTTPConnectionConfig(url=ACME_URL),
        MCPStdioConnectionConfig(stdio_command="uvx", stdio_args=["acme-mcp-server"]),
    ],
    ids=["http", "stdio"],
)
@pytest.mark.parametrize("flag", [None, "", "false", "0", "no", "off", "enabled"])
def test_mcp_is_refused_unless_the_worker_switches_it_on(
    connection: Any,
    flag: str | None,
    monkeypatch: pytest.MonkeyPatch,
    recording_toolset: type[_RecordingToolset],
) -> None:
    if flag is None:
        monkeypatch.delenv(MCP_ENABLED_ENV, raising=False)
    else:
        monkeypatch.setenv(MCP_ENABLED_ENV, flag)

    with pytest.raises(PermissionError, match=MCP_ENABLED_ENV):
        MCPTool(connection=connection).get_toolsets()

    assert recording_toolset.instances == []


@pytest.mark.parametrize("flag", ["true", "TRUE", "True", "1", "yes", " true "])
def test_mcp_builds_when_the_worker_switches_it_on(
    flag: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MCP_ENABLED_ENV, flag)

    (toolset,) = MCPTool(connection=MCPHTTPConnectionConfig(url=ACME_URL)).get_toolsets()

    assert isinstance(toolset, MCPToolset)


@pytest.mark.asyncio
async def test_diagnostics_are_refused_when_mcp_is_off(
    monkeypatch: pytest.MonkeyPatch,
    recording_toolset: type[_RecordingToolset],
) -> None:
    """A probe launches the stdio command too, so it is gated like `get_toolsets()`."""
    monkeypatch.delenv(MCP_ENABLED_ENV, raising=False)
    connection = MCPStdioConnectionConfig(stdio_command="uvx")

    with pytest.raises(PermissionError, match=MCP_ENABLED_ENV):
        await list_mcp_tools(connection)
    with pytest.raises(PermissionError, match=MCP_ENABLED_ENV):
        await probe_mcp_connection(connection)

    assert recording_toolset.instances == []


@pytest.mark.parametrize(
    "connection_factory",
    [_placeholder_stdio_config, _placeholder_http_config],
    ids=["stdio", "http"],
)
@pytest.mark.parametrize("secret_exported", [True, False], ids=["secret-set", "secret-unset"])
def test_the_switch_is_checked_before_any_placeholder_is_read_or_transport_built(
    connection_factory: Callable[[], Any],
    secret_exported: bool,
    monkeypatch: pytest.MonkeyPatch,
    recording_toolset: type[_RecordingToolset],
) -> None:
    """Off means nothing is attempted: no card variable is read and no transport is built.

    With the secret unset, resolving before the switch would surface a `ValueError` instead;
    with it set, the transport constructor would be reached.
    """
    import fastmcp.client.transports as transports

    built: list[dict[str, Any]] = []

    def record(**kwargs: Any) -> None:
        built.append(kwargs)

    for name in ("StdioTransport", "SSETransport", "StreamableHttpTransport"):
        monkeypatch.setattr(transports, name, record)
    monkeypatch.delenv(MCP_ENABLED_ENV, raising=False)
    monkeypatch.setenv(REGION_VAR, REGION_VALUE)
    if secret_exported:
        monkeypatch.setenv(SECRET_VAR, SECRET_VALUE)
    else:
        monkeypatch.delenv(SECRET_VAR, raising=False)

    with pytest.raises(PermissionError, match=MCP_ENABLED_ENV):
        MCPTool(connection=connection_factory()).get_toolsets()

    assert built == []
    assert recording_toolset.instances == []


# --------------------------------------------------------------------------------------
# No resolved secret survives a failed build, and an unused token is never resolved
# --------------------------------------------------------------------------------------

MISSING_VAR = MCP_SECRET_PREFIX + "NOT_EXPORTED"


def _secret_in_traceback_locals(exc: BaseException) -> bool:
    """True if any frame on the traceback holds a local whose repr carries the secret.

    Error reporters (Sentry, Logfire, rich tracebacks) record frame locals, so a resolved
    secret still bound when a later placeholder fails would be shipped with the error.
    """
    tb = exc.__traceback__
    while tb is not None:
        for local in tb.tb_frame.f_locals.values():
            if SECRET_VALUE in repr(local):
                return True
        tb = tb.tb_next
    return False


@pytest.mark.usefixtures("secret_env")
@pytest.mark.parametrize(
    "connection_factory",
    [
        lambda: MCPHTTPConnectionConfig(
            url=ACME_URL + "?k=${" + SECRET_VAR + "}",
            bearer_token="${" + MISSING_VAR + "}",
        ),
        lambda: MCPStdioConnectionConfig(
            stdio_command="uvx",
            stdio_env={"CLIENT_SECRET": "${" + SECRET_VAR + "}"},
            stdio_args=["--key=${" + MISSING_VAR + "}"],
        ),
        lambda: MCPStdioConnectionConfig(
            stdio_command="uvx",
            stdio_env={"CLIENT_SECRET": "${" + SECRET_VAR + "}"},
            stdio_token_env_var="ACME_TOKEN",
            bearer_token="${" + MISSING_VAR + "}",
        ),
    ],
    ids=["http-url-then-token", "stdio-env-then-args", "stdio-env-then-token"],
)
def test_a_late_failing_placeholder_leaves_no_resolved_secret_in_any_frame(
    connection_factory: Callable[[], Any],
    recording_toolset: type[_RecordingToolset],
) -> None:
    with pytest.raises(ValueError, match=f"'{MISSING_VAR}'") as excinfo:
        MCPTool(connection=connection_factory()).get_toolsets()

    assert not _secret_in_traceback_locals(excinfo.value)
    assert recording_toolset.instances == []


@pytest.mark.parametrize("token", ["${" + MISSING_VAR + "}", "${ACME_DB_PASSWORD}"])
def test_an_unused_stdio_token_is_neither_checked_nor_resolved(token: str) -> None:
    """Without `stdio_token_env_var` the token never reaches the server, so it cannot fail."""
    connection = MCPStdioConnectionConfig(stdio_command="uvx", bearer_token=token)

    transport = _transport_of(_build_mcp_toolset(connection))

    assert transport.env is None
