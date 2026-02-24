from __future__ import annotations

from typing import Any

from akgentic.tool.core import BaseTool, ToolCard, ToolFactory


class _DummyTool(BaseTool):
    def get_tools(self, observer: Any | None) -> list[Any]:
        return [lambda: "ok"]

    def get_toolset(self) -> Any:
        return {"kind": "dummy-toolset"}


def test_tool_factory_get_tools_and_toolsets() -> None:
    card = ToolCard(
        name="dummy",
        module=_DummyTool,
        description="dummy",
        params=[],
    )

    factory = ToolFactory(tool_cards=[card])

    tools = factory.get_tools()
    toolsets = factory.get_toolsets()

    assert len(tools) == 1
    assert tools[0]() == "ok"
    assert toolsets == [{"kind": "dummy-toolset"}]
