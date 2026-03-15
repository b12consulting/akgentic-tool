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


def test_tool_factory_with_observer_calls_card_observer() -> None:
    """ToolFactory with a non-None observer propagates it to each card via card.observer()."""
    from unittest.mock import MagicMock  # noqa: PLC0415

    mock_observer = MagicMock()
    observer_received: list[Any] = []

    class _ObservedTool(ToolCard):
        name: str = "observed"
        description: str = "observed tool"

        def observer(self, obs: Any) -> "_ObservedTool":  # type: ignore[override]
            observer_received.append(obs)
            return super().observer(obs)  # type: ignore[return-value]

        def get_tools(self) -> list[Callable]:
            return []

    card = _ObservedTool()
    ToolFactory(tool_cards=[card], observer=mock_observer)
    assert observer_received == [mock_observer]


def test_tool_card_get_system_prompts_default_returns_empty() -> None:
    """ToolCard.get_system_prompts() default implementation returns empty list."""

    class _MinimalTool(ToolCard):
        name: str = "minimal"
        description: str = "minimal tool"

        def get_tools(self) -> list[Callable]:
            return []

    card = _MinimalTool()
    assert card.get_system_prompts() == []


def test_tool_card_get_toolsets_default_returns_empty() -> None:
    """ToolCard.get_toolsets() default implementation returns empty list."""

    class _MinimalTool(ToolCard):
        name: str = "minimal2"
        description: str = "minimal tool 2"

        def get_tools(self) -> list[Callable]:
            return []

    card = _MinimalTool()
    assert card.get_toolsets() == []


def test_tool_factory_get_system_prompts_aggregates_from_cards() -> None:
    """ToolFactory.get_system_prompts() concatenates prompts from all cards."""

    def _prompt1() -> str:
        return "prompt-1"

    def _prompt2() -> str:
        return "prompt-2"

    class _PromptTool(ToolCard):
        name: str = "prompt_tool"
        description: str = "tool with prompts"
        _prompts: list[Callable] = []

        def get_tools(self) -> list[Callable]:
            return []

        def get_system_prompts(self) -> list[Callable]:
            return self._prompts

    card1 = _PromptTool()
    card1._prompts = [_prompt1]
    card2 = _PromptTool()
    card2._prompts = [_prompt2]
    factory = ToolFactory(tool_cards=[card1, card2])
    prompts = factory.get_system_prompts()
    assert len(prompts) == 2
    assert _prompt1 in prompts
    assert _prompt2 in prompts


def test_tool_factory_retry_exception_wraps_retriable_error() -> None:
    """ToolFactory wraps RetriableError in retry_exception when configured."""
    from akgentic.tool.errors import RetriableError  # noqa: PLC0415

    class _RetriableTool(ToolCard):
        name: str = "retriable"
        description: str = "raises retriable errors"

        def get_tools(self) -> list[Callable]:
            def _fail() -> str:
                raise RetriableError("transient")

            return [_fail]

    class _MyRetryError(Exception):
        pass

    card = _RetriableTool()
    factory = ToolFactory(tool_cards=[card], retry_exception=_MyRetryError)
    tools = factory.get_tools()
    assert len(tools) == 1
    import pytest as _pytest  # noqa: PLC0415

    with _pytest.raises(_MyRetryError):
        tools[0]()
