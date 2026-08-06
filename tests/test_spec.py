from __future__ import annotations

from typing import Any, Callable

import pytest

from akgentic.tool import ToolCard, ToolCardSpec, ToolFactory
from akgentic.tool.core import TOOL_CALL, BaseToolParam, _resolve


class _SegmentParam(BaseToolParam):
    value: str = "default"


class _Observer:
    """Trivial weak-referenceable observer stand-in."""


class _SegmentCommand(_SegmentParam):
    pass


def _seg_prompt() -> str:
    return "seg-prompt"


def _seg_command() -> str:
    return "seg-command"


class _SegmentTool(ToolCard):
    """Stands in for an application-owned tool defined outside ``akgentic.*``."""

    cap: _SegmentParam | bool = True

    def get_tools(self) -> list[Callable]:
        p = _resolve(self.cap, _SegmentParam)
        if p and TOOL_CALL in p.expose:
            return [lambda: f"seg-{p.value}"]
        return []

    def get_system_prompts(self) -> list[Callable]:
        return [_seg_prompt]

    def get_commands(self) -> dict[type[BaseToolParam], Callable]:
        return {_SegmentCommand: _seg_command}

    def get_toolsets(self) -> list[Any]:
        return [{"kind": "seg-toolset"}]


_SEGMENT_TOOL_PATH = f"{__name__}._SegmentTool"


def test_spec_builds_concrete_tool_from_dotted_path() -> None:
    spec = ToolCardSpec(tool_class=_SEGMENT_TOOL_PATH, config={"cap": {"value": "custom"}})
    built = spec.build()
    assert isinstance(built, _SegmentTool)
    assert built.cap.value == "custom"


def test_spec_accepts_class_object() -> None:
    spec = ToolCardSpec(tool_class=_SegmentTool)
    assert spec.get_tool_class() is _SegmentTool


def test_spec_is_usable_as_toolcard_in_factory() -> None:
    spec = ToolCardSpec(tool_class=_SEGMENT_TOOL_PATH, config={"cap": {"value": "x"}})
    factory = ToolFactory(tool_cards=[spec])
    tools = factory.get_tools()
    assert len(tools) == 1
    assert tools[0]() == "seg-x"
    assert factory.get_toolsets() == [{"kind": "seg-toolset"}]


def test_spec_build_is_cached() -> None:
    spec = ToolCardSpec(tool_class=_SEGMENT_TOOL_PATH)
    assert spec.build() is spec.build()


def test_spec_delegates_prompts_commands_and_toolsets() -> None:
    spec = ToolCardSpec(tool_class=_SEGMENT_TOOL_PATH)
    assert [p() for p in spec.get_system_prompts()] == ["seg-prompt"]
    commands = spec.get_commands()
    assert commands == {_SegmentCommand: _seg_command}
    assert spec.get_toolsets() == [{"kind": "seg-toolset"}]


def test_spec_observer_is_attached_to_built_card() -> None:
    spec = ToolCardSpec(tool_class=_SEGMENT_TOOL_PATH)
    observer = _Observer()
    assert spec.observer(observer) is spec  # chaining preserved
    assert spec.build()._observer is observer


def test_spec_round_trips_through_serialization() -> None:
    spec = ToolCardSpec(tool_class=_SEGMENT_TOOL_PATH, config={"cap": {"value": "rt"}})
    reloaded = ToolCardSpec.model_validate(spec.model_dump())
    assert reloaded.tool_class == _SEGMENT_TOOL_PATH
    assert reloaded.build().cap.value == "rt"


def test_spec_rejects_non_toolcard_class() -> None:
    with pytest.raises(ValueError, match="not a ToolCard subclass"):
        ToolCardSpec(tool_class="builtins.dict").get_tool_class()


def test_spec_rejects_unqualified_path() -> None:
    with pytest.raises(ValueError, match="fully qualified dotted path"):
        ToolCardSpec(tool_class="NotDotted").get_tool_class()
