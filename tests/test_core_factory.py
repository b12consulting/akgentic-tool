from __future__ import annotations

from typing import Any, Callable, ClassVar

import pytest

from akgentic.tool.core import (
    COMMAND,
    TOOL_CALL,
    BaseToolParam,
    ToolCard,
    ToolFactory,
    _resolve,
    _topological_sort,
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


# ---------------------------------------------------------------------------
# depends_on / topological-sort tests (story 10-8)
# ---------------------------------------------------------------------------


class _NoDepsTool(ToolCard):
    """ToolCard subclass with no depends_on override (default empty)."""

    name: str = "no-deps"
    description: str = "no dependency"

    def get_tools(self) -> list[Callable]:
        return []


class _DependsOnXTool(ToolCard):
    """ToolCard subclass declaring depends_on = ['X']."""

    name: str = "depends-x"
    description: str = "declares a class-level dependency on X"
    depends_on: ClassVar[list[str]] = ["X"]

    def get_tools(self) -> list[Callable]:
        return []


def test_tool_card_depends_on_default_is_empty() -> None:
    """Default depends_on is [] and is not a Pydantic field."""
    assert _NoDepsTool.depends_on == []
    assert "depends_on" not in _NoDepsTool.model_fields
    assert "depends_on" not in ToolCard.model_fields


def test_tool_card_depends_on_is_not_a_pydantic_field() -> None:
    """depends_on must never leak into model_dump() output."""
    # Default-empty subclass.
    no_deps = _NoDepsTool()
    dump = no_deps.model_dump()
    assert "depends_on" not in dump

    # Subclass that overrides depends_on — still must not appear in the dump.
    depends = _DependsOnXTool()
    dump_with_deps = depends.model_dump()
    assert "depends_on" not in dump_with_deps


def test_tool_card_subclass_can_override_depends_on() -> None:
    """Overriding depends_on on one subclass does not contaminate siblings."""

    class _A(ToolCard):
        name: str = "a"
        description: str = "a"
        depends_on: ClassVar[list[str]] = ["B"]

        def get_tools(self) -> list[Callable]:
            return []

    class _C(ToolCard):
        name: str = "c"
        description: str = "c"

        def get_tools(self) -> list[Callable]:
            return []

    assert _A.depends_on == ["B"]
    assert _C.depends_on == []
    # Base class unaffected by either subclass.
    assert ToolCard.depends_on == []


# --- Stubs used by the factory-level tests below ----------------------------


class StubVectorStoreTool(ToolCard):
    """Stub prerequisite — no dependencies."""

    name: str = "stub-vector-store"
    description: str = "stub VS"

    def get_tools(self) -> list[Callable]:
        return []


class StubKnowledgeGraphTool(ToolCard):
    """Stub consumer — depends on StubVectorStoreTool."""

    name: str = "stub-kg"
    description: str = "stub KG"
    depends_on: ClassVar[list[str]] = ["StubVectorStoreTool"]

    def get_tools(self) -> list[Callable]:
        return []


class StubPlanningTool(ToolCard):
    """Stub consumer — depends on StubVectorStoreTool."""

    name: str = "stub-planning"
    description: str = "stub planning"
    depends_on: ClassVar[list[str]] = ["StubVectorStoreTool"]

    def get_tools(self) -> list[Callable]:
        return []


def test_tool_factory_topological_sort_happy_path() -> None:
    """VectorStore is wired before KG and Planning regardless of input order."""
    calls: list[str] = []

    class _RecordingVS(StubVectorStoreTool):
        def observer(self, obs):  # type: ignore[override,no-untyped-def]
            calls.append(type(self).__name__)
            return super().observer(obs)

    class _RecordingKG(StubKnowledgeGraphTool):
        depends_on: ClassVar[list[str]] = ["_RecordingVS"]

        def observer(self, obs):  # type: ignore[override,no-untyped-def]
            calls.append(type(self).__name__)
            return super().observer(obs)

    class _RecordingPlan(StubPlanningTool):
        depends_on: ClassVar[list[str]] = ["_RecordingVS"]

        def observer(self, obs):  # type: ignore[override,no-untyped-def]
            calls.append(type(self).__name__)
            return super().observer(obs)

    kg = _RecordingKG()
    plan = _RecordingPlan()
    vs = _RecordingVS()

    factory = ToolFactory(tool_cards=[kg, plan, vs], observer=object())
    # VS must be first; KG and Planning preserve their relative input order (kg before plan).
    assert calls == ["_RecordingVS", "_RecordingKG", "_RecordingPlan"]
    assert [type(c).__name__ for c in factory.tool_cards] == [
        "_RecordingVS",
        "_RecordingKG",
        "_RecordingPlan",
    ]


def test_tool_factory_deterministic_stable_sort() -> None:
    """Sorting the same input twice produces the same ordering (AC-3)."""
    kg = StubKnowledgeGraphTool()
    plan = StubPlanningTool()
    vs = StubVectorStoreTool()
    cards = [kg, plan, vs]

    f1 = ToolFactory(tool_cards=list(cards))
    f2 = ToolFactory(tool_cards=list(cards))
    names1 = [type(c).__name__ for c in f1.tool_cards]
    names2 = [type(c).__name__ for c in f2.tool_cards]
    assert names1 == names2
    assert names1 == ["StubVectorStoreTool", "StubKnowledgeGraphTool", "StubPlanningTool"]


def test_tool_factory_missing_dependency_raises_value_error() -> None:
    """Missing dependency raises ValueError with both class names in the message."""
    with pytest.raises(ValueError) as exc:
        ToolFactory(tool_cards=[StubKnowledgeGraphTool()])
    msg = str(exc.value)
    assert "StubKnowledgeGraphTool" in msg
    assert "StubVectorStoreTool" in msg


def test_tool_factory_cycle_detection_raises_value_error() -> None:
    """A 2-node cycle is detected and named in the error message."""

    class CyclicA(ToolCard):
        name: str = "a"
        description: str = "cycle a"
        depends_on: ClassVar[list[str]] = ["CyclicB"]

        def get_tools(self) -> list[Callable]:
            return []

    class CyclicB(ToolCard):
        name: str = "b"
        description: str = "cycle b"
        depends_on: ClassVar[list[str]] = ["CyclicA"]

        def get_tools(self) -> list[Callable]:
            return []

    with pytest.raises(ValueError) as exc:
        ToolFactory(tool_cards=[CyclicA(), CyclicB()])
    msg = str(exc.value)
    assert "cycle" in msg.lower()
    assert "CyclicA" in msg
    assert "CyclicB" in msg


def test_tool_factory_cycle_detection_three_node_cycle() -> None:
    """A 3-node cycle (A → B → C → A) is detected and named."""

    class CycA(ToolCard):
        name: str = "a3"
        description: str = "cycle a3"
        depends_on: ClassVar[list[str]] = ["CycB"]

        def get_tools(self) -> list[Callable]:
            return []

    class CycB(ToolCard):
        name: str = "b3"
        description: str = "cycle b3"
        depends_on: ClassVar[list[str]] = ["CycC"]

        def get_tools(self) -> list[Callable]:
            return []

    class CycC(ToolCard):
        name: str = "c3"
        description: str = "cycle c3"
        depends_on: ClassVar[list[str]] = ["CycA"]

        def get_tools(self) -> list[Callable]:
            return []

    with pytest.raises(ValueError) as exc:
        ToolFactory(tool_cards=[CycA(), CycB(), CycC()])
    msg = str(exc.value)
    assert "cycle" in msg.lower()
    for name in ("CycA", "CycB", "CycC"):
        assert name in msg


def test_tool_factory_no_dependencies_preserves_input_order() -> None:
    """With no declared dependencies, input order is preserved (AC-6)."""

    class CardAlpha(ToolCard):
        name: str = "alpha"
        description: str = "alpha"

        def get_tools(self) -> list[Callable]:
            return []

    class CardBeta(ToolCard):
        name: str = "beta"
        description: str = "beta"

        def get_tools(self) -> list[Callable]:
            return []

    class CardGamma(ToolCard):
        name: str = "gamma"
        description: str = "gamma"

        def get_tools(self) -> list[Callable]:
            return []

    a = CardAlpha()
    b = CardBeta()
    c = CardGamma()
    factory = ToolFactory(tool_cards=[a, b, c])
    assert [type(card).__name__ for card in factory.tool_cards] == [
        "CardAlpha",
        "CardBeta",
        "CardGamma",
    ]


def test_topological_sort_helper_directly() -> None:
    """Direct coverage for _topological_sort on a small 4-node DAG."""

    class NodeD(ToolCard):
        name: str = "d"
        description: str = "d"

        def get_tools(self) -> list[Callable]:
            return []

    class NodeB(ToolCard):
        name: str = "b"
        description: str = "b"
        depends_on: ClassVar[list[str]] = ["NodeD"]

        def get_tools(self) -> list[Callable]:
            return []

    class NodeC(ToolCard):
        name: str = "c"
        description: str = "c"
        depends_on: ClassVar[list[str]] = ["NodeD"]

        def get_tools(self) -> list[Callable]:
            return []

    class NodeA(ToolCard):
        name: str = "a"
        description: str = "a"
        depends_on: ClassVar[list[str]] = ["NodeB", "NodeC", "NodeD"]

        def get_tools(self) -> list[Callable]:
            return []

    a = NodeA()
    b = NodeB()
    c = NodeC()
    d = NodeD()
    # Input order is [A, B, C, D] — A has three unresolved deps, so it lands last.
    ordered = _topological_sort([a, b, c, d])
    names = [type(card).__name__ for card in ordered]
    # D must precede B, C, A; B must precede A; C must precede A.
    assert names.index("NodeD") < names.index("NodeB")
    assert names.index("NodeD") < names.index("NodeC")
    assert names.index("NodeD") < names.index("NodeA")
    assert names.index("NodeB") < names.index("NodeA")
    assert names.index("NodeC") < names.index("NodeA")
    # Determinism: B appears before C (input order for independent nodes).
    assert names.index("NodeB") < names.index("NodeC")
    # Exactly one instance per node returned.
    assert len(ordered) == 4
    assert set(id(x) for x in ordered) == {id(a), id(b), id(c), id(d)}


def test_topological_sort_duplicate_class_allowed() -> None:
    """Two instances of the same ToolCard subclass are permitted."""

    class DupTool(ToolCard):
        name: str = "dup"
        description: str = "dup"

        def get_tools(self) -> list[Callable]:
            return []

    d1 = DupTool()
    d2 = DupTool()
    ordered = _topological_sort([d1, d2])
    # Both instances must survive the sort, preserving input order.
    assert len(ordered) == 2
    assert ordered[0] is d1
    assert ordered[1] is d2
