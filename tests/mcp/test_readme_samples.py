"""The MCPTool README samples, transcribed verbatim and executed.

The samples live in `src/akgentic/tool/mcp/README.md`; the package README's Tool Catalog
carries a short entry that links to it. Every runnable Python snippet in that file appears
below exactly as a reader would copy it, so a sample that cannot construct fails CI rather
than failing for whoever pastes it. Before this file existed the published sample raised
`ValidationError` twice over — it passed a plural `connections=[...]` list to a field that
is singular and required.

Construction alone is too weak a gate, for the reason given in `test_mcp_toolset.py`:
`akgentic.tool.mcp.mcp` sits on the mypy `ignore_errors` list, so a wrong translation of the
config onto pydantic-ai's `MCPToolset` is invisible to every other check. These tests
therefore assert the *configuration the sample actually produces* — transport class, url,
headers, init timeout, read timeout, stream timeout, prefix — and pin the surrounding prose
claims that a reader would otherwise take on trust.

Nothing here reads or string-matches the README itself. The samples are transcribed into
Python; asserting on markdown content would test the document instead of the behaviour.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pydantic_ai.mcp import MCPToolset, SSETransport, StdioTransport, StreamableHttpTransport
from pydantic_ai.toolsets.prefixed import PrefixedToolset

from .test_mcp_toolset import (
    _init_timeout_of,
    _read_timeout_of,
    _sse_read_timeout_of,
    _transport_of,
)

# The README's own placeholder host, kept in step with the rest of the MCP suite.
README_URL = "https://mcp.acme.example/api/v1/endpoint"


# --------------------------------------------------------------------------------------
# Sample 1 — remote server over streamable HTTP (the default transport)
# --------------------------------------------------------------------------------------


def test_readme_streamable_http_sample_builds_its_documented_toolset() -> None:
    from akgentic.tool.mcp import MCPHTTPConnectionConfig, MCPTool

    tool = MCPTool(
        connection=MCPHTTPConnectionConfig(
            url="https://mcp.acme.example/api/v1/endpoint",
        )
    )

    (toolset,) = tool.get_toolsets()
    assert isinstance(toolset, MCPToolset)
    assert not isinstance(toolset, PrefixedToolset)

    transport = _transport_of(toolset)
    assert isinstance(transport, StreamableHttpTransport)
    assert transport.url == README_URL
    assert transport.headers == {}

    # The documented defaults, which the sample relies on by omitting them.
    assert _init_timeout_of(toolset) == 10.0
    assert _read_timeout_of(toolset) == 300.0


# --------------------------------------------------------------------------------------
# Sample 2 — Server-Sent Events, requested explicitly
# --------------------------------------------------------------------------------------


def test_readme_sse_sample_builds_its_documented_toolset() -> None:
    from akgentic.tool.mcp import MCPHTTPConnectionConfig, MCPTool

    tool = MCPTool(
        connection=MCPHTTPConnectionConfig(
            url="https://mcp.acme.example/api/v1/endpoint",
            transport="sse",
            bearer_token="...",  # sent as an Authorization header on the transport
            read_timeout=900.0,  # also governs how long the event stream tolerates silence
        )
    )

    (toolset,) = tool.get_toolsets()
    transport = _transport_of(toolset)

    assert isinstance(transport, SSETransport)
    assert not isinstance(transport, StreamableHttpTransport)
    assert transport.url == README_URL

    # "sent as an Authorization header on the transport" — on the transport, not the toolset.
    assert transport.headers == {"Authorization": "Bearer ..."}

    # "also governs how long the event stream tolerates silence" — both timeouts, not one.
    assert _read_timeout_of(toolset) == 900.0
    assert _sse_read_timeout_of(toolset) == 900.0


# --------------------------------------------------------------------------------------
# Sample 3 — local subprocess over stdio, with a tool prefix
# --------------------------------------------------------------------------------------


def test_readme_stdio_sample_builds_its_documented_toolset() -> None:
    from akgentic.tool.mcp import MCPStdioConnectionConfig, MCPTool

    tool = MCPTool(
        connection=MCPStdioConnectionConfig(
            stdio_command="uvx",
            stdio_args=["acme-mcp-server"],
            tool_prefix="acme",  # applied via the toolset's prefixed() wrapper
        )
    )

    # "Setting tool_prefix wraps that toolset in a PrefixedToolset."
    (toolset,) = tool.get_toolsets()
    assert isinstance(toolset, PrefixedToolset)
    assert toolset.prefix == "acme"
    assert isinstance(toolset.wrapped, MCPToolset)

    transport = _transport_of(toolset.wrapped)
    assert isinstance(transport, StdioTransport)
    assert transport.command == "uvx"
    assert transport.args == ["acme-mcp-server"]

    assert _init_timeout_of(toolset.wrapped) == 10.0
    assert _read_timeout_of(toolset.wrapped) == 300.0


# --------------------------------------------------------------------------------------
# Prose claims the samples depend on
# --------------------------------------------------------------------------------------


def test_readme_claim_mcptool_takes_exactly_one_singular_connection() -> None:
    """The defect this story removes: the published sample passed `connections=[...]`.

    The field is singular and required, so the plural form failed twice over — a missing
    keyword *and* an unexpected one. Both halves are pinned so neither can quietly come back.
    """
    from akgentic.tool.mcp import MCPHTTPConnectionConfig, MCPTool

    with pytest.raises(ValidationError):
        MCPTool()  # type: ignore[call-arg]

    with pytest.raises(ValidationError):
        MCPTool(connections=[MCPHTTPConnectionConfig(url=README_URL)])  # type: ignore[call-arg]


def test_readme_claim_an_sse_suffix_alone_does_not_select_sse() -> None:
    """The transport is always taken from the config, never inferred from the URL.

    This is the exact trap the old sample fell into: a `/sse` URL with no `transport=`
    connects over streamable HTTP. If URL inference were ever reinstated, the README's
    explanation of why `transport="sse"` is mandatory would become false.
    """
    from akgentic.tool.mcp import MCPHTTPConnectionConfig, MCPTool

    tool = MCPTool(connection=MCPHTTPConnectionConfig(url="https://mcp.acme.example/sse"))

    (toolset,) = tool.get_toolsets()
    transport = _transport_of(toolset)

    assert isinstance(transport, StreamableHttpTransport)
    assert not isinstance(transport, SSETransport)


def test_readme_claim_get_tools_is_always_empty() -> None:
    """`get_tools()` is always empty — MCP capabilities reach the agent through toolsets."""
    from akgentic.tool.mcp import MCPHTTPConnectionConfig, MCPStdioConnectionConfig, MCPTool

    assert MCPTool(connection=MCPHTTPConnectionConfig(url=README_URL)).get_tools() == []
    assert MCPTool(connection=MCPStdioConnectionConfig(stdio_command="uvx")).get_tools() == []
