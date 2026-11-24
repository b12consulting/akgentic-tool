"""Shared fixtures for the MCP tests."""

from __future__ import annotations

import pytest

from akgentic.tool.mcp.mcp import MCP_ENABLED_ENV


@pytest.fixture(autouse=True)
def mcp_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Switch MCP on for every test here; MCPTool is off by default on a worker.

    Tests of the switch itself undo this with `monkeypatch.delenv(MCP_ENABLED_ENV)`.
    """
    monkeypatch.setenv(MCP_ENABLED_ENV, "true")
