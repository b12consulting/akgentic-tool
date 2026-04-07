"""Tests for akgentic.tool.mcp.mcp — protocol-level MCP support."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from akgentic.tool.event import ToolCallEvent, ToolObserver
from akgentic.tool.mcp.mcp import (
    MCPHTTPConnectionConfig,
    MCPServerFactory,
    MCPStdioConnectionConfig,
    MCPTool,
    _apply_tool_filters,
    _extract_tool_metadata,
    _matches_any_pattern,
    _mcp_auth_headers,
    list_mcp_tools,
    probe_mcp_connection,
)


class _ToolDefObj:
    """Fake tool-definition object mimicking pydantic-ai tool shapes."""

    def __init__(self, name: str, annotations: dict[str, Any] | None = None) -> None:
        self.name = name
        self.metadata = {"annotations": annotations or {}}


class _NoMetadataObj:
    """Object with a name attribute but no metadata attribute."""

    name = "nameless"


class _RecordingObserver:
    """Minimal ToolObserver that records received events."""

    def __init__(self) -> None:
        self.events: list[ToolCallEvent] = []

    def notify_event(self, event: object) -> None:
        if isinstance(event, ToolCallEvent):
            self.events.append(event)


assert isinstance(_RecordingObserver(), ToolObserver)  # satisfies Protocol at import time


class TestMCPHTTPConnectionConfig:
    def test_minimal_defaults(self) -> None:
        cfg = MCPHTTPConnectionConfig(url="http://localhost:8080")
        assert cfg.url == "http://localhost:8080"
        assert cfg.transport == "streamable-http"
        assert cfg.bearer_token is None
        assert cfg.timeout == 10.0
        assert cfg.read_timeout == 300.0
        assert cfg.tool_prefix is None

    def test_sse_transport(self) -> None:
        cfg = MCPHTTPConnectionConfig(url="http://x", transport="sse")
        assert cfg.transport == "sse"

    def test_bearer_token_stored(self) -> None:
        cfg = MCPHTTPConnectionConfig(url="http://x", bearer_token="tok123")
        assert cfg.bearer_token == "tok123"

    def test_tool_prefix_stored(self) -> None:
        cfg = MCPHTTPConnectionConfig(url="http://x", tool_prefix="fs")
        assert cfg.tool_prefix == "fs"

    def test_custom_timeouts(self) -> None:
        cfg = MCPHTTPConnectionConfig(url="http://x", timeout=5.0, read_timeout=60.0)
        assert cfg.timeout == 5.0
        assert cfg.read_timeout == 60.0

    def test_timeout_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            MCPHTTPConnectionConfig(url="http://x", timeout=0)

    def test_read_timeout_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            MCPHTTPConnectionConfig(url="http://x", read_timeout=0)


class TestMCPStdioConnectionConfig:
    def test_defaults(self) -> None:
        cfg = MCPStdioConnectionConfig()
        assert cfg.transport == "stdio"
        assert cfg.stdio_command is None
        assert cfg.stdio_args == []
        assert cfg.stdio_env is None
        assert cfg.stdio_cwd is None
        assert cfg.bearer_token is None
        assert cfg.stdio_token_env_var is None

    def test_full_config(self) -> None:
        cfg = MCPStdioConnectionConfig(
            stdio_command="docker",
            stdio_args=["run", "-i", "my-mcp"],
            stdio_env={"MY_KEY": "val"},
            stdio_cwd="/tmp",
            bearer_token="secret",
            stdio_token_env_var="MY_TOKEN",
        )
        assert cfg.stdio_command == "docker"
        assert cfg.stdio_args == ["run", "-i", "my-mcp"]
        assert cfg.stdio_env == {"MY_KEY": "val"}
        assert cfg.stdio_cwd == "/tmp"
        assert cfg.bearer_token == "secret"
        assert cfg.stdio_token_env_var == "MY_TOKEN"

    def test_timeout_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            MCPStdioConnectionConfig(timeout=0)

    def test_read_timeout_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            MCPStdioConnectionConfig(read_timeout=0)



def test_auth_headers_with_token() -> None:
    assert _mcp_auth_headers("my-token") == {"Authorization": "Bearer my-token"}


def test_auth_headers_none_returns_none() -> None:
    assert _mcp_auth_headers(None) is None


def test_auth_headers_empty_string_returns_none() -> None:
    assert _mcp_auth_headers("") is None



def test_matches_any_pattern_none_means_all_match() -> None:
    assert _matches_any_pattern("anything", None) is True


def test_matches_any_pattern_exact_hit() -> None:
    assert _matches_any_pattern("read_file", {"read_file"}) is True


def test_matches_any_pattern_exact_miss() -> None:
    assert _matches_any_pattern("write_file", {"read_file"}) is False


def test_matches_any_pattern_wildcard_hit() -> None:
    assert _matches_any_pattern("read_document", {"read_*"}) is True


def test_matches_any_pattern_wildcard_miss() -> None:
    assert _matches_any_pattern("write_document", {"read_*"}) is False


def test_matches_any_pattern_multi_set_hit() -> None:
    assert _matches_any_pattern("delete_doc", {"read_*", "delete_*"}) is True


def test_matches_any_pattern_empty_set_is_no_match() -> None:
    assert _matches_any_pattern("any_tool", set()) is False



def test_extract_from_string() -> None:
    name, annotations = _extract_tool_metadata("my_tool")
    assert name == "my_tool"
    assert annotations == {}


def test_extract_from_dict_with_annotations() -> None:
    name, annotations = _extract_tool_metadata({"name": "tool_a", "annotations": {"readOnly": True}})
    assert name == "tool_a"
    assert annotations == {"readOnly": True}


def test_extract_from_dict_no_annotations_key() -> None:
    name, annotations = _extract_tool_metadata({"name": "tool_b"})
    assert name == "tool_b"
    assert annotations == {}


def test_extract_from_dict_non_dict_annotations_ignored() -> None:
    name, annotations = _extract_tool_metadata({"name": "tool_c", "annotations": "bad"})
    assert name == "tool_c"
    assert annotations == {}


def test_extract_from_object_with_metadata() -> None:
    obj = _ToolDefObj("obj_tool", {"readOnlyHint": True})
    name, annotations = _extract_tool_metadata(obj)
    assert name == "obj_tool"
    assert annotations == {"readOnlyHint": True}


def test_extract_from_object_no_metadata_returns_empty() -> None:
    name, annotations = _extract_tool_metadata(_NoMetadataObj())
    assert name == "nameless"
    assert annotations == {}


def _mock_server_classes() -> tuple[MagicMock, MagicMock, MagicMock]:
    return (
        MagicMock(name="MCPServerSSE"),
        MagicMock(name="MCPServerStreamableHTTP"),
        MagicMock(name="MCPServerStdio"),
    )


class TestMCPServerFactory:
    def test_create_streamable_http(self) -> None:
        sse, http, stdio = _mock_server_classes()
        with patch("akgentic.tool.mcp.mcp._load_mcp_server_classes", return_value=(sse, http, stdio)):
            result = MCPServerFactory().create(MCPHTTPConnectionConfig(url="http://mcp.example.com/mcp"))

        http.assert_called_once()
        kw = http.call_args.kwargs
        assert kw["url"] == "http://mcp.example.com/mcp"
        assert kw["headers"] is None
        assert result is http.return_value

    def test_create_streamable_http_with_bearer_auth(self) -> None:
        sse, http, stdio = _mock_server_classes()
        with patch("akgentic.tool.mcp.mcp._load_mcp_server_classes", return_value=(sse, http, stdio)):
            MCPServerFactory().create(MCPHTTPConnectionConfig(url="http://x", bearer_token="tok"))

        kw = http.call_args.kwargs
        assert kw["headers"] == {"Authorization": "Bearer tok"}

    def test_create_sse(self) -> None:
        sse, http, stdio = _mock_server_classes()
        with patch("akgentic.tool.mcp.mcp._load_mcp_server_classes", return_value=(sse, http, stdio)):
            result = MCPServerFactory().create(
                MCPHTTPConnectionConfig(url="http://x", transport="sse")
            )

        sse.assert_called_once()
        assert result is sse.return_value

    def test_create_stdio(self) -> None:
        sse, http, stdio = _mock_server_classes()
        cfg = MCPStdioConnectionConfig(
            stdio_command="npx",
            stdio_args=["-y", "@modelcontextprotocol/server-filesystem"],
            stdio_env={"DEBUG": "1"},
            stdio_token_env_var="API_TOKEN",
            bearer_token="secret",
        )
        with patch("akgentic.tool.mcp.mcp._load_mcp_server_classes", return_value=(sse, http, stdio)):
            result = MCPServerFactory().create(cfg)

        stdio.assert_called_once()
        kw = stdio.call_args.kwargs
        assert kw["command"] == "npx"
        assert kw["args"] == ["-y", "@modelcontextprotocol/server-filesystem"]
        assert kw["env"]["DEBUG"] == "1"
        assert kw["env"]["API_TOKEN"] == "secret"
        assert result is stdio.return_value

    def test_create_stdio_injects_token_into_env_var(self) -> None:
        sse, http, stdio = _mock_server_classes()
        cfg = MCPStdioConnectionConfig(
            stdio_command="uvx",
            stdio_token_env_var="GITHUB_TOKEN",
            bearer_token="gh_secret",
        )
        with patch("akgentic.tool.mcp.mcp._load_mcp_server_classes", return_value=(sse, http, stdio)):
            MCPServerFactory().create(cfg)

        assert stdio.call_args.kwargs["env"]["GITHUB_TOKEN"] == "gh_secret"

    def test_create_stdio_missing_command_raises(self) -> None:
        with pytest.raises(ValueError, match="stdio_command is required"):
            MCPServerFactory().create(MCPStdioConnectionConfig())

    def test_create_forwards_timeouts(self) -> None:
        sse, http, stdio = _mock_server_classes()
        cfg = MCPHTTPConnectionConfig(url="http://x", timeout=3.0, read_timeout=90.0)
        with patch("akgentic.tool.mcp.mcp._load_mcp_server_classes", return_value=(sse, http, stdio)):
            MCPServerFactory().create(cfg)

        kw = http.call_args.kwargs
        assert kw["timeout"] == 3.0
        assert kw["read_timeout"] == 90.0

    def test_create_forwards_tool_prefix(self) -> None:
        sse, http, stdio = _mock_server_classes()
        cfg = MCPHTTPConnectionConfig(url="http://x", tool_prefix="my_prefix")
        with patch("akgentic.tool.mcp.mcp._load_mcp_server_classes", return_value=(sse, http, stdio)):
            MCPServerFactory().create(cfg)

        assert http.call_args.kwargs["tool_prefix"] == "my_prefix"

    async def test_process_tool_call_notifies_observer(self) -> None:
        obs = _RecordingObserver()
        factory = MCPServerFactory(get_observer=lambda: obs)
        call_next = AsyncMock(return_value="result")

        result = await factory._build_process_tool_call()(None, call_next, "my_tool", {"x": 1})

        assert result == "result"
        assert len(obs.events) == 1
        assert obs.events[0].tool_name == "my_tool"
        assert obs.events[0].kwargs == {"x": 1}
        call_next.assert_awaited_once_with("my_tool", {"x": 1})

    async def test_process_tool_call_without_observer(self) -> None:
        factory = MCPServerFactory()
        call_next = AsyncMock(return_value="ok")

        result = await factory._build_process_tool_call()(None, call_next, "tool_x", {})

        assert result == "ok"
        call_next.assert_awaited_once_with("tool_x", {})

    async def test_process_tool_call_observer_returning_none(self) -> None:
        factory = MCPServerFactory(get_observer=lambda: None)
        call_next = AsyncMock(return_value="data")

        result = await factory._build_process_tool_call()(None, call_next, "t", {})

        assert result == "data"


class TestApplyToolFilters:
    def test_no_filters_returns_server_unchanged(self) -> None:
        server = MagicMock()
        assert _apply_tool_filters(server, None, None, False) is server

    def test_allowed_tools_wraps_server(self) -> None:
        server = MagicMock()
        filtered_cls = MagicMock()
        with patch("akgentic.tool.mcp.mcp._load_filtered_toolset_class", return_value=filtered_cls):
            result = _apply_tool_filters(server, {"read_*"}, None, False)
        filtered_cls.assert_called_once()
        assert result is filtered_cls.return_value

    def test_blocked_tools_wraps_server(self) -> None:
        server = MagicMock()
        filtered_cls = MagicMock()
        with patch("akgentic.tool.mcp.mcp._load_filtered_toolset_class", return_value=filtered_cls):
            _apply_tool_filters(server, None, {"delete_*"}, False)
        filtered_cls.assert_called_once()

    def test_read_only_wraps_server(self) -> None:
        server = MagicMock()
        filtered_cls = MagicMock()
        with patch("akgentic.tool.mcp.mcp._load_filtered_toolset_class", return_value=filtered_cls):
            _apply_tool_filters(server, None, None, True)
        filtered_cls.assert_called_once()

    def _capture_filter_fn(
        self,
        allowed: set[str] | None = None,
        blocked: set[str] | None = None,
        read_only: bool = False,
        tool_prefix: str | None = None,
    ) -> Any:
        captured: list[Any] = []

        def fake_cls(srv: Any, fn: Any) -> MagicMock:
            captured.append(fn)
            return MagicMock()

        with patch("akgentic.tool.mcp.mcp._load_filtered_toolset_class", return_value=fake_cls):
            _apply_tool_filters(MagicMock(), allowed, blocked, read_only, tool_prefix)

        return captured[0]

    def test_filter_fn_allows_tool_in_allowlist(self) -> None:
        fn = self._capture_filter_fn(allowed={"read_file"})
        assert fn(None, "read_file") is True

    def test_filter_fn_blocks_tool_not_in_allowlist(self) -> None:
        fn = self._capture_filter_fn(allowed={"read_file"})
        assert fn(None, "write_file") is False

    def test_filter_fn_blocks_tool_in_blocklist(self) -> None:
        fn = self._capture_filter_fn(blocked={"delete_*"})
        assert fn(None, "delete_doc") is False

    def test_filter_fn_allows_tool_not_in_blocklist(self) -> None:
        fn = self._capture_filter_fn(blocked={"delete_*"})
        assert fn(None, "read_file") is True

    def test_filter_fn_read_only_blocks_non_readonly_tool(self) -> None:
        fn = self._capture_filter_fn(read_only=True)
        assert fn(None, {"name": "write_file", "annotations": {}}) is False

    def test_filter_fn_read_only_allows_readonly_hint_tool(self) -> None:
        fn = self._capture_filter_fn(read_only=True)
        assert fn(None, {"name": "list_files", "annotations": {"readOnlyHint": True}}) is True

    def test_filter_fn_read_only_allows_readonly_annotation(self) -> None:
        fn = self._capture_filter_fn(read_only=True)
        assert fn(None, {"name": "list_files", "annotations": {"readOnly": True}}) is True

    def test_filter_fn_prefix_strip_allows_normalized_name(self) -> None:
        fn = self._capture_filter_fn(allowed={"read_file"}, tool_prefix="fs")
        assert fn(None, "fs_read_file") is True

    def test_filter_fn_prefix_strip_blocks_normalized_name(self) -> None:
        fn = self._capture_filter_fn(allowed={"read_file"}, tool_prefix="fs")
        assert fn(None, "fs_write_file") is False

    def test_filter_fn_no_prefix_match_uses_full_name(self) -> None:
        fn = self._capture_filter_fn(allowed={"read_file"}, tool_prefix="fs")
        # "other_read_file" doesn't start with "fs_" so no stripping → no match
        assert fn(None, "other_read_file") is False


class TestMCPTool:
    def _http_tool(self, **kwargs: Any) -> MCPTool:
        return MCPTool(
            name="t",
            description="d",
            connection=MCPHTTPConnectionConfig(url="http://mcp.example.com"),
            **kwargs,
        )

    def test_get_tools_is_empty(self) -> None:
        assert self._http_tool().get_tools() == []

    def test_serializable_round_trip_http(self) -> None:
        tool = MCPTool(
            name="my-mcp",
            description="desc",
            connection=MCPHTTPConnectionConfig(url="http://x", bearer_token="tok"),
        )
        restored = MCPTool.model_validate(tool.model_dump())
        assert restored.name == "my-mcp"
        assert isinstance(restored.connection, MCPHTTPConnectionConfig)
        assert restored.connection.bearer_token == "tok"

    def test_serializable_round_trip_stdio(self) -> None:
        tool = MCPTool(
            name="stdio-mcp",
            description="desc",
            connection=MCPStdioConnectionConfig(stdio_command="npx"),
        )
        restored = MCPTool.model_validate(tool.model_dump())
        assert isinstance(restored.connection, MCPStdioConnectionConfig)
        assert restored.connection.stdio_command == "npx"

    def test_allowed_and_blocked_fields_stored(self) -> None:
        tool = self._http_tool(allowed_tools={"read_*"}, blocked_tools={"delete_*"})
        assert tool.allowed_tools == {"read_*"}
        assert tool.blocked_tools == {"delete_*"}

    def test_read_only_default_false(self) -> None:
        assert self._http_tool().read_only_tools is False

    def test_get_toolsets_returns_one_server(self) -> None:
        mock_server = MagicMock()
        with patch("akgentic.tool.mcp.mcp.MCPServerFactory.create", return_value=mock_server):
            toolsets = self._http_tool().get_toolsets()
        assert len(toolsets) == 1
        assert toolsets[0] is mock_server

    def test_get_toolsets_applies_no_filter_by_default(self) -> None:
        mock_server = MagicMock()
        with patch("akgentic.tool.mcp.mcp.MCPServerFactory.create", return_value=mock_server):
            toolsets = self._http_tool().get_toolsets()
        # With no allowed/blocked/read_only, _apply_tool_filters returns the server unchanged
        assert toolsets[0] is mock_server


class TestListMcpTools:
    async def test_returns_tool_names(self) -> None:
        mock_server = AsyncMock()
        mock_server.list_tools = AsyncMock(return_value=["tool_a", "tool_b"])

        with patch("akgentic.tool.mcp.mcp.MCPServerFactory.create", return_value=mock_server):
            names = await list_mcp_tools(MCPHTTPConnectionConfig(url="http://x"))

        assert names == ["tool_a", "tool_b"]

    async def test_empty_toolsets_raises(self) -> None:
        with patch.object(MCPTool, "get_toolsets", return_value=[]):
            with pytest.raises(ValueError, match="MCPTool.get_toolsets\\(\\) returned empty list"):
                await list_mcp_tools(MCPHTTPConnectionConfig(url="http://x"))

    async def test_tool_name_extracted_from_dict(self) -> None:
        mock_server = AsyncMock()
        mock_server.list_tools = AsyncMock(
            return_value=[{"name": "search_web", "annotations": {}}]
        )

        with patch("akgentic.tool.mcp.mcp.MCPServerFactory.create", return_value=mock_server):
            names = await list_mcp_tools(MCPHTTPConnectionConfig(url="http://x"))

        assert names == ["search_web"]


class TestProbeMcpConnection:
    async def test_probe_returns_summary_structure(self) -> None:
        tools = [f"tool_{i}" for i in range(25)]
        with patch(
            "akgentic.tool.mcp.mcp.list_mcp_tools",
            new_callable=AsyncMock,
            return_value=tools,
        ):
            summary = await probe_mcp_connection(
                MCPHTTPConnectionConfig(url="http://x"), max_tools_to_print=10
            )

        assert summary["tool_count"] == 25
        assert len(summary["tools"]) == 10
        assert len(summary["tool_annotations"]) == 10
        assert summary["feasible"] is True

    async def test_probe_feasible_false_when_no_tools(self) -> None:
        with patch(
            "akgentic.tool.mcp.mcp.list_mcp_tools",
            new_callable=AsyncMock,
            return_value=[],
        ):
            summary = await probe_mcp_connection(MCPHTTPConnectionConfig(url="http://x"))

        assert summary["tool_count"] == 0
        assert summary["tools"] == []
        assert summary["feasible"] is False

    async def test_probe_default_max_tools_is_20(self) -> None:
        tools = [f"t_{i}" for i in range(30)]
        with patch(
            "akgentic.tool.mcp.mcp.list_mcp_tools",
            new_callable=AsyncMock,
            return_value=tools,
        ):
            summary = await probe_mcp_connection(MCPHTTPConnectionConfig(url="http://x"))

        assert len(summary["tools"]) == 20

    async def test_probe_tool_annotations_contain_name(self) -> None:
        tools = ["alpha", "beta"]
        with patch(
            "akgentic.tool.mcp.mcp.list_mcp_tools",
            new_callable=AsyncMock,
            return_value=tools,
        ):
            summary = await probe_mcp_connection(MCPHTTPConnectionConfig(url="http://x"))

        names_in_annotations = [a["name"] for a in summary["tool_annotations"]]
        assert names_in_annotations == ["alpha", "beta"]
