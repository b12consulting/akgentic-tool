from __future__ import annotations

from typing import Any, Callable

from akgentic.tool.core import BaseToolParam, ToolCard, ToolFactory, _resolve


class _DummyParam(BaseToolParam):
    value: str = "default"


class _DummyTool(ToolCard):
    name: str = "dummy"
    description: str = "dummy tool"
    cap: _DummyParam | bool = True

    def get_tools(self) -> list[Callable]:
        p = _resolve(self.cap, _DummyParam)
        if p and p.llm_tool:
            return [lambda: f"ok-{p.value}"]
        return []

    def get_toolsets(self) -> list[Any]:
        return [{"kind": "dummy-toolset"}]


def test_tool_factory_get_tools_and_toolsets() -> None:
    card = _DummyTool()
    factory = ToolFactory(tool_cards=[card])
    tools = factory.get_tools()
    toolsets = factory.get_toolsets()
    assert len(tools) == 1
    assert tools[0]() == "ok-default"
    assert toolsets == [{"kind": "dummy-toolset"}]


def test_tool_factory_custom_param() -> None:
    card = _DummyTool(cap=_DummyParam(value="custom"))
    factory = ToolFactory(tool_cards=[card])
    tools = factory.get_tools()
    assert len(tools) == 1
    assert tools[0]() == "ok-custom"


def test_tool_factory_disabled_capability() -> None:
    card = _DummyTool(cap=False)
    factory = ToolFactory(tool_cards=[card])
    tools = factory.get_tools()
    assert len(tools) == 0


def test_resolve_true() -> None:
    result = _resolve(True, _DummyParam)
    assert result is not None
    assert isinstance(result, _DummyParam)
    assert result.value == "default"


def test_resolve_false() -> None:
    result = _resolve(False, _DummyParam)
    assert result is None


def test_resolve_instance() -> None:
    param = _DummyParam(value="custom")
    result = _resolve(param, _DummyParam)
    assert result is param
    assert result.value == "custom"


def test_base_tool_param_defaults() -> None:
    p = BaseToolParam()
    assert p.instructions is None
    assert p.system_prompt is False
    assert p.llm_tool is True


def test_tool_factory_with_null_observer() -> None:
    """Test that ToolFactory handles None observer gracefully without calling card.observer()."""
    observer_called = False

    class _ObserverCheckTool(ToolCard):
        name: str = "observer_check"
        description: str = "checks observer calls"

        def observer(self, observer):
            nonlocal observer_called
            observer_called = True
            return super().observer(observer)

        def get_tools(self) -> list[Callable]:
            return []

    card = _ObserverCheckTool()
    factory = ToolFactory(tool_cards=[card], observer=None)

    # Observer should NOT be called when observer is None
    assert not observer_called
    assert factory.observer is None
