from __future__ import annotations

from typing import Any, Callable

from akgentic.tool.core import (
    COMMAND,
    TOOL_CALL,
    BaseToolParam,
    ToolCard,
    ToolFactory,
    _resolve,
)


class _DummyParam(BaseToolParam):
    value: str = "default"


class _DummyTool(ToolCard):
    name: str = "dummy"
    description: str = "dummy tool"
    cap: _DummyParam | bool = True

    def get_tools(self) -> list[Callable]:
        p = _resolve(self.cap, _DummyParam)
        if p and TOOL_CALL in p.expose:
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
    """BaseToolParam defaults: expose={TOOL_CALL}, instructions=None."""
    p = BaseToolParam()
    assert p.instructions is None
    assert p.expose == {TOOL_CALL}


def test_base_tool_param_subclass_defaults() -> None:
    """Subclass inherits expose default from BaseToolParam."""
    p = _DummyParam()
    assert p.instructions is None
    assert p.expose == {TOOL_CALL}


def test_tool_factory_get_commands() -> None:
    """ToolFactory.get_commands() aggregates from all tool cards."""

    class _CommandParam(BaseToolParam):
        expose: set[str] = {COMMAND}

    class _CommandTool(ToolCard):
        name: str = "cmd_tool"
        description: str = "command tool"
        cmd: _CommandParam | bool = True

        def get_tools(self) -> list[Callable]:
            return []

        def get_commands(self) -> dict[type[BaseToolParam], Callable]:
            p = _resolve(self.cmd, _CommandParam)
            if p and COMMAND in p.expose:

                def my_cmd() -> str:
                    return "cmd-result"

                return {_CommandParam: my_cmd}
            return {}

    card = _CommandTool()
    factory = ToolFactory(tool_cards=[card])
    commands = factory.get_commands()
    assert len(commands) == 1
    assert _CommandParam in commands
    assert commands[_CommandParam]() == "cmd-result"


def test_tool_factory_get_commands_empty() -> None:
    """ToolFactory.get_commands() returns empty dict when no cards have commands."""
    card = _DummyTool()
    factory = ToolFactory(tool_cards=[card])
    commands = factory.get_commands()
    assert commands == {}


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
