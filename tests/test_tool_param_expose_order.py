"""``BaseToolParam.expose`` dumps in one order, whatever order the set iterates in."""

from __future__ import annotations

from akgentic.tool.core import COMMAND, LLM_CONTEXT, TOOL_CALL, BaseToolParam


class _Param(BaseToolParam):
    expose: set[str] = {TOOL_CALL, COMMAND}


def test_expose_dumps_sorted() -> None:
    assert _Param().model_dump()["expose"] == sorted({TOOL_CALL, COMMAND})
    assert _Param(expose={LLM_CONTEXT, COMMAND, TOOL_CALL}).model_dump()["expose"] == sorted(
        {LLM_CONTEXT, COMMAND, TOOL_CALL}
    )


def test_a_round_trip_dumps_the_same_list() -> None:
    first = _Param(expose={COMMAND, TOOL_CALL})
    restored = _Param.model_validate(first.model_dump())
    assert restored.model_dump() == first.model_dump()
