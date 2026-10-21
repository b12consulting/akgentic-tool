"""``/search_planning`` takes ``name=value`` options first, then free text.

The command is its own closure, not the tool's: the tool takes ``query`` first, so
``/search_planning mailing list`` used to bind ``list`` to ``mode``. Every spec here
goes through ``registry.dispatch(...)`` or ``descriptors()`` — a call to the closure
alone would skip the tokenising that decides what an option is.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock

from akgentic.tool.core import CommandRegistry, ToolFactory
from akgentic.tool.planning.planning import PlanningTool
from akgentic.tool.planning.planning_actor import PlanActor, PlanConfig, Task
from akgentic.tool.vector_store.protocol import SEMANTIC_DISABLED
from tests.conftest import MockActorAddress


def _wire(proxy: Any) -> PlanningTool:
    """A ``PlanningTool`` whose planning proxy is *proxy*."""
    tool = PlanningTool()
    observer = MagicMock()
    observer.myAddress = MockActorAddress()
    tool._observer = observer
    tool._planning_proxy = proxy
    return tool


def _registry(proxy: Any) -> CommandRegistry:
    return ToolFactory(tool_cards=[_wire(proxy)]).get_command_registry()


def _recording() -> tuple[CommandRegistry, MagicMock]:
    proxy = MagicMock()
    proxy.search_planning.return_value = []
    return _registry(proxy), proxy


def _actor(*descriptions: str) -> PlanActor:
    """A bare ``PlanActor`` whose card declined a store, holding one pending task each."""
    actor = PlanActor()
    actor.config = PlanConfig(name="test-plan", role="ToolActor", vector_store=None)
    actor.on_start()
    for task_id, description in enumerate(descriptions, start=1):
        actor.state.task_list.append(
            Task(id=task_id, status="pending", description=description, owner="@A", creator="@B")
        )
    return actor


class TestTheCommandTakesOptionsThenText:
    def test_a_plain_first_word_is_the_query_not_the_mode(self) -> None:
        registry, proxy = _recording()

        registry.dispatch("/search_planning mailing list")

        proxy.search_planning.assert_called_once_with(
            status=None,
            owner=None,
            creator=None,
            query="mailing list",
            mode="hybrid",
            top_k=None,
            score_threshold=None,
        )

    def test_options_alone_list_with_no_query(self) -> None:
        registry, proxy = _recording()

        registry.dispatch("/search_planning status=pending")

        kwargs = proxy.search_planning.call_args.kwargs
        assert kwargs["status"] == "pending"
        assert kwargs["query"] is None

    def test_the_bare_command_lists_every_task(self) -> None:
        registry, proxy = _recording()

        registry.dispatch("/search_planning")

        proxy.search_planning.assert_called_once_with(
            status=None,
            owner=None,
            creator=None,
            query=None,
            mode="hybrid",
            top_k=None,
            score_threshold=None,
        )

    def test_options_are_coerced_and_the_rest_is_the_query(self) -> None:
        registry, proxy = _recording()

        registry.dispatch("/search_planning mode=keyword top_k=2 deploy")

        kwargs = proxy.search_planning.call_args.kwargs
        assert (kwargs["mode"], kwargs["top_k"], kwargs["query"]) == ("keyword", 2, "deploy")

    def test_a_quoted_owner_is_one_option_value(self) -> None:
        registry, proxy = _recording()

        registry.dispatch('/search_planning owner="Alice Smith" deploy')

        kwargs = proxy.search_planning.call_args.kwargs
        assert (kwargs["owner"], kwargs["query"]) == ("Alice Smith", "deploy")

    def test_a_bad_status_is_the_commands_failure_and_the_actor_is_not_asked(self) -> None:
        registry, proxy = _recording()

        result = registry.dispatch("/search_planning status=done deploy")

        assert result is not None
        assert result.startswith("Command 'search_planning' failed:")
        assert proxy.search_planning.call_count == 0


class TestTheResultIsRenderedForAHuman:
    def test_the_lines_are_joined_one_per_line(self) -> None:
        actor = _actor("deploy the service", "deploy the docs")

        result = _registry(actor).dispatch("/search_planning")

        assert result == "\n".join(actor.search_planning())
        assert result is not None
        assert len(result.splitlines()) == 2

    def test_an_empty_result_says_so(self) -> None:
        result = _registry(_actor("deploy")).dispatch("/search_planning status=abort")
        assert result == "No tasks found."

    def test_a_store_less_card_answers_vector_mode_with_the_sentence(self) -> None:
        result = _registry(_actor("deploy")).dispatch(
            "/search_planning mode=vector deploy"
        )
        assert result == SEMANTIC_DISABLED


class TestTheDescriptor:
    EXPECTED = [
        # name, keyword_only, rest_of_line, required, choices
        ("mode", True, False, False, ["hybrid", "vector", "keyword"]),
        ("status", True, False, False, ["pending", "started", "completed", "abort"]),
        ("owner", True, False, False, None),
        ("creator", True, False, False, None),
        ("top_k", True, False, False, None),
        ("score_threshold", True, False, False, None),
        ("query", False, True, False, None),
    ]

    def test_each_arg_says_what_it_is_and_which_values_it_takes(self) -> None:
        registry, _ = _recording()

        descriptor = next(d for d in registry.descriptors() if d.name == "search_planning")

        assert [
            (a.name, a.keyword_only, a.rest_of_line, a.required, a.choices)
            for a in descriptor.args
        ] == self.EXPECTED
        assert all(a.description for a in descriptor.args), descriptor.args
        assert "Example: /search_planning " in descriptor.description
        assert descriptor.tool_card == "PlanningTool"


class TestTheToolIsUnchanged:
    def test_the_tool_keeps_query_first_and_answers_a_list(self) -> None:
        proxy = MagicMock()
        proxy.search_planning.return_value = ["Task 1: x"]
        tool = _wire(proxy)

        search_tool = next(t for t in tool.get_tools() if t.__name__ == "search_planning")

        assert list(inspect.signature(search_tool).parameters) == [
            "query",
            "mode",
            "status",
            "owner",
            "creator",
            "top_k",
            "score_threshold",
        ]
        assert search_tool(query="x") == ["Task 1: x"]
