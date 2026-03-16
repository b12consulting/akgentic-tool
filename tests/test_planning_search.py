"""Tests for PlanActor.search_planning and PlanningTool._search_planning_factory.

Covers AC#2–#12 from Story 4.3: search_planning Filtered Search Capability.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from akgentic.tool.planning.planning_actor import (
    PlanActor,
    PlanConfig,
    Task,
)
from tests.conftest import MockActorAddress


def _make_actor(semantic_search: bool = True) -> PlanActor:
    """Construct a bare PlanActor with no Pykka runtime — calls on_start directly."""
    actor = PlanActor()
    actor.config = PlanConfig(
        name="test-plan",
        role="ToolActor",
        semantic_search=semantic_search,
    )
    actor.on_start()
    return actor


def _add_task(
    actor: PlanActor,
    task_id: int,
    description: str,
    status: str = "pending",
    owner: str = "@Alice",
    creator: str = "@Bob",
) -> Task:
    """Helper: add a task to actor state without triggering embedding.

    Directly appends to task_list so tests can control the lifecycle independently.
    """
    task = Task(
        id=task_id,
        status=status,  # type: ignore[arg-type]
        description=description,
        owner=owner,
        creator=creator,
    )
    actor.state.task_list.append(task)
    return task


# ---------------------------------------------------------------------------
# AC#8 — all params None → full list returned
# ---------------------------------------------------------------------------


class TestSearchPlanningAllNone:
    """AC8: When all parameters are None, the full task list is returned."""

    def test_returns_full_list_when_all_none(self) -> None:
        actor = _make_actor(semantic_search=False)
        _add_task(actor, 1, "Setup database", status="pending")
        _add_task(actor, 2, "Auth module", status="started")
        _add_task(actor, 3, "Deployment pipeline", status="completed")

        result = actor.search_planning()

        assert len(result) == 3
        assert {t.id for t in result} == {1, 2, 3}

    def test_returns_empty_list_when_no_tasks_and_all_none(self) -> None:
        actor = _make_actor(semantic_search=False)

        result = actor.search_planning()

        assert result == []


# ---------------------------------------------------------------------------
# AC#8 (edge case) — empty task list
# ---------------------------------------------------------------------------


class TestSearchPlanningEmptyList:
    """Empty task list → empty result for any filter."""

    def test_empty_list_with_status_filter(self) -> None:
        actor = _make_actor(semantic_search=False)

        result = actor.search_planning(status="pending")

        assert result == []

    def test_empty_list_with_owner_filter(self) -> None:
        actor = _make_actor(semantic_search=False)

        result = actor.search_planning(owner="@Alice")

        assert result == []

    def test_empty_list_with_query_filter(self) -> None:
        actor = _make_actor(semantic_search=False)

        result = actor.search_planning(query="auth")

        assert result == []


# ---------------------------------------------------------------------------
# AC#2 — status filter
# ---------------------------------------------------------------------------


class TestSearchPlanningStatusFilter:
    """AC2: status filter returns only tasks with exact matching status."""

    def test_status_pending_returns_only_pending(self) -> None:
        actor = _make_actor(semantic_search=False)
        _add_task(actor, 1, "Task A", status="pending")
        _add_task(actor, 2, "Task B", status="started")
        _add_task(actor, 3, "Task C", status="pending")
        _add_task(actor, 4, "Task D", status="completed")

        result = actor.search_planning(status="pending")

        assert len(result) == 2
        assert all(t.status == "pending" for t in result)
        assert {t.id for t in result} == {1, 3}

    def test_status_started_returns_only_started(self) -> None:
        actor = _make_actor(semantic_search=False)
        _add_task(actor, 1, "Task A", status="pending")
        _add_task(actor, 2, "Task B", status="started")
        _add_task(actor, 3, "Task C", status="started")

        result = actor.search_planning(status="started")

        assert len(result) == 2
        assert all(t.status == "started" for t in result)
        assert {t.id for t in result} == {2, 3}

    def test_status_filter_returns_empty_when_no_match(self) -> None:
        actor = _make_actor(semantic_search=False)
        _add_task(actor, 1, "Task A", status="pending")

        result = actor.search_planning(status="completed")

        assert result == []


# ---------------------------------------------------------------------------
# AC#3 — owner filter
# ---------------------------------------------------------------------------


class TestSearchPlanningOwnerFilter:
    """AC3: owner filter returns only tasks with exact owner match."""

    def test_owner_exact_match(self) -> None:
        actor = _make_actor(semantic_search=False)
        _add_task(actor, 1, "Task A", owner="@Alice")
        _add_task(actor, 2, "Task B", owner="@Bob")
        _add_task(actor, 3, "Task C", owner="@Alice")

        result = actor.search_planning(owner="@Alice")

        assert len(result) == 2
        assert all(t.owner == "@Alice" for t in result)
        assert {t.id for t in result} == {1, 3}

    def test_owner_empty_string_matches_unassigned(self) -> None:
        actor = _make_actor(semantic_search=False)
        _add_task(actor, 1, "Task A", owner="")
        _add_task(actor, 2, "Task B", owner="@Alice")
        _add_task(actor, 3, "Task C", owner="")

        result = actor.search_planning(owner="")

        assert len(result) == 2
        assert {t.id for t in result} == {1, 3}

    def test_owner_filter_returns_empty_when_no_match(self) -> None:
        actor = _make_actor(semantic_search=False)
        _add_task(actor, 1, "Task A", owner="@Alice")

        result = actor.search_planning(owner="@Charlie")

        assert result == []


# ---------------------------------------------------------------------------
# AC#4 — creator filter
# ---------------------------------------------------------------------------


class TestSearchPlanningCreatorFilter:
    """AC4: creator filter returns only tasks with exact creator match."""

    def test_creator_exact_match(self) -> None:
        actor = _make_actor(semantic_search=False)
        _add_task(actor, 1, "Task A", creator="@Bob")
        _add_task(actor, 2, "Task B", creator="@Charlie")
        _add_task(actor, 3, "Task C", creator="@Bob")

        result = actor.search_planning(creator="@Bob")

        assert len(result) == 2
        assert all(t.creator == "@Bob" for t in result)
        assert {t.id for t in result} == {1, 3}

    def test_creator_filter_returns_empty_when_no_match(self) -> None:
        actor = _make_actor(semantic_search=False)
        _add_task(actor, 1, "Task A", creator="@Bob")

        result = actor.search_planning(creator="@Alice")

        assert result == []


# ---------------------------------------------------------------------------
# AC#5 — query keyword filter (no vector deps)
# ---------------------------------------------------------------------------


class TestSearchPlanningQueryKeyword:
    """AC5: query keyword filter — case-insensitive substring on task.description."""

    def test_keyword_case_insensitive_match(self) -> None:
        actor = _make_actor(semantic_search=False)
        _add_task(actor, 1, "Authentication module setup")
        _add_task(actor, 2, "Database schema migration")
        _add_task(actor, 3, "OAuth authentication integration")

        result = actor.search_planning(query="authentication")

        assert len(result) == 2
        assert {t.id for t in result} == {1, 3}

    def test_keyword_uppercase_query_matches_lowercase_description(self) -> None:
        actor = _make_actor(semantic_search=False)
        _add_task(actor, 1, "auth flow setup")
        _add_task(actor, 2, "payment service")

        result = actor.search_planning(query="AUTH")

        assert len(result) == 1
        assert result[0].id == 1

    def test_keyword_no_match_returns_empty(self) -> None:
        actor = _make_actor(semantic_search=False)
        _add_task(actor, 1, "Database setup")

        result = actor.search_planning(query="authentication")

        assert result == []

    def test_keyword_with_semantic_search_disabled(self) -> None:
        """Keyword-only mode when semantic_search=False."""
        actor = _make_actor(semantic_search=False)
        _add_task(actor, 1, "auth flow")
        _add_task(actor, 2, "payment gateway")

        with patch.object(actor, "_get_or_create_embedding_svc") as mock_svc:
            result = actor.search_planning(query="auth")
            # _get_or_create_embedding_svc is still called but returns None for semantic_search=False
            # The actual behaviour is that it returns None from the real method

        assert len(result) == 1
        assert result[0].id == 1


# ---------------------------------------------------------------------------
# AC#6 — query semantic filter with vector deps
# ---------------------------------------------------------------------------


class TestSearchPlanningQuerySemantic:
    """AC6: query with mocked vector deps — union of keyword + semantic, dedup, cosine ≥ 0.5."""

    def test_semantic_union_with_keyword(self) -> None:
        """Semantic hits union with keyword hits; tasks with score ≥ 0.5 included."""
        actor = _make_actor(semantic_search=True)
        _add_task(actor, 1, "auth flow setup")  # keyword match ("auth flow" is substring)
        _add_task(actor, 2, "database schema")  # semantic match only
        _add_task(actor, 3, "deployment pipeline")  # no match

        # Mock embedding service
        mock_svc = MagicMock()
        mock_svc.embed.return_value = [[0.1, 0.2, 0.3]]

        # Mock vector index with non-zero size
        mock_index = MagicMock()
        mock_index.__len__ = MagicMock(return_value=3)
        # Task 2 has cosine score ≥ 0.5, task 3 has score < 0.5
        mock_index.search_cosine.return_value = [("2", 0.75), ("3", 0.3)]
        actor._vector_index = mock_index

        with patch.object(actor, "_get_or_create_embedding_svc", return_value=mock_svc):
            result = actor.search_planning(query="auth flow")

        # task 1 via keyword, task 2 via semantic (score 0.75 ≥ 0.5)
        # task 3 excluded (score 0.3 < 0.5)
        assert {t.id for t in result} == {1, 2}

    def test_deduplication_keyword_and_semantic_overlap(self) -> None:
        """Tasks matched by both keyword and semantic are deduplicated."""
        actor = _make_actor(semantic_search=True)
        _add_task(actor, 1, "auth module")

        mock_svc = MagicMock()
        mock_svc.embed.return_value = [[0.1, 0.2, 0.3]]

        mock_index = MagicMock()
        mock_index.__len__ = MagicMock(return_value=1)
        # Task 1 is both a keyword match AND semantic match
        mock_index.search_cosine.return_value = [("1", 0.9)]
        actor._vector_index = mock_index

        with patch.object(actor, "_get_or_create_embedding_svc", return_value=mock_svc):
            result = actor.search_planning(query="auth")

        # Deduplicated: only one entry for task 1
        assert len(result) == 1
        assert result[0].id == 1

    def test_cosine_threshold_exactly_05_included(self) -> None:
        """Tasks with cosine score exactly 0.5 are included."""
        actor = _make_actor(semantic_search=True)
        _add_task(actor, 1, "database task")

        mock_svc = MagicMock()
        mock_svc.embed.return_value = [[0.1, 0.2]]

        mock_index = MagicMock()
        mock_index.__len__ = MagicMock(return_value=1)
        mock_index.search_cosine.return_value = [("1", 0.5)]
        actor._vector_index = mock_index

        with patch.object(actor, "_get_or_create_embedding_svc", return_value=mock_svc):
            result = actor.search_planning(query="db")

        assert len(result) == 1
        assert result[0].id == 1

    def test_cosine_threshold_below_05_excluded(self) -> None:
        """Tasks with cosine score < 0.5 are excluded (if not keyword match)."""
        actor = _make_actor(semantic_search=True)
        _add_task(actor, 1, "database task")

        mock_svc = MagicMock()
        mock_svc.embed.return_value = [[0.1, 0.2]]

        mock_index = MagicMock()
        mock_index.__len__ = MagicMock(return_value=1)
        mock_index.search_cosine.return_value = [("1", 0.49)]
        actor._vector_index = mock_index

        with patch.object(actor, "_get_or_create_embedding_svc", return_value=mock_svc):
            result = actor.search_planning(query="completely different topic")

        # Not a keyword match AND cosine < 0.5 → excluded
        assert result == []

    def test_empty_vector_index_falls_back_to_keyword(self) -> None:
        """When vector index is empty (len=0), semantic phase is skipped."""
        actor = _make_actor(semantic_search=True)
        _add_task(actor, 1, "auth flow setup")

        mock_svc = MagicMock()

        mock_index = MagicMock()
        mock_index.__len__ = MagicMock(return_value=0)
        actor._vector_index = mock_index

        with patch.object(actor, "_get_or_create_embedding_svc", return_value=mock_svc):
            result = actor.search_planning(query="auth")

        # Keyword match only; search_cosine should not be called (index is empty)
        mock_index.search_cosine.assert_not_called()
        assert len(result) == 1
        assert result[0].id == 1


# ---------------------------------------------------------------------------
# AC#7 — combined filters (AND logic)
# ---------------------------------------------------------------------------


class TestSearchPlanningCombinedFilters:
    """AC7: multiple filters combined as AND conditions."""

    def test_status_and_owner_filter(self) -> None:
        actor = _make_actor(semantic_search=False)
        _add_task(actor, 1, "Task A", status="started", owner="@Alice")
        _add_task(actor, 2, "Task B", status="pending", owner="@Alice")
        _add_task(actor, 3, "Task C", status="started", owner="@Bob")
        _add_task(actor, 4, "Task D", status="started", owner="@Alice")

        result = actor.search_planning(status="started", owner="@Alice")

        assert len(result) == 2
        assert {t.id for t in result} == {1, 4}

    def test_status_owner_query_triple_filter(self) -> None:
        """AC7: status + owner + query together (AND logic)."""
        actor = _make_actor(semantic_search=False)
        _add_task(actor, 1, "database migration", status="started", owner="@Alice")
        _add_task(actor, 2, "database backup", status="pending", owner="@Alice")
        _add_task(actor, 3, "auth setup", status="started", owner="@Alice")
        _add_task(actor, 4, "database schema", status="started", owner="@Bob")

        result = actor.search_planning(status="started", owner="@Alice", query="database")

        # Only task 1 matches ALL three: started + @Alice + "database" in description
        assert len(result) == 1
        assert result[0].id == 1

    def test_creator_and_status_filter(self) -> None:
        actor = _make_actor(semantic_search=False)
        _add_task(actor, 1, "Task A", status="pending", creator="@Bob")
        _add_task(actor, 2, "Task B", status="completed", creator="@Bob")
        _add_task(actor, 3, "Task C", status="pending", creator="@Charlie")

        result = actor.search_planning(status="pending", creator="@Bob")

        assert len(result) == 1
        assert result[0].id == 1


# ---------------------------------------------------------------------------
# AC#5/AC#6 — vector fallback when svc unavailable
# ---------------------------------------------------------------------------


class TestSearchPlanningVectorFallback:
    """AC5/AC6: when _get_or_create_embedding_svc returns None, degrades to keyword-only."""

    def test_no_exception_when_svc_returns_none(self) -> None:
        """When embedding svc unavailable, search_planning works via keyword only."""
        actor = _make_actor(semantic_search=True)
        _add_task(actor, 1, "auth service")
        _add_task(actor, 2, "payment service")

        with patch.object(actor, "_get_or_create_embedding_svc", return_value=None):
            result = actor.search_planning(query="auth")

        # Falls back to keyword-only, no exception raised
        assert len(result) == 1
        assert result[0].id == 1

    def test_no_exception_when_vector_index_is_none(self) -> None:
        """When _vector_index is None (semantic_search=False), keyword fallback works."""
        actor = _make_actor(semantic_search=False)
        _add_task(actor, 1, "auth service")

        mock_svc = MagicMock()
        # Even if svc somehow returned non-None, _vector_index being None skips semantic
        with patch.object(actor, "_get_or_create_embedding_svc", return_value=mock_svc):
            # _vector_index is None because semantic_search=False
            result = actor.search_planning(query="auth")

        assert len(result) == 1
        assert result[0].id == 1

    def test_semantic_exception_falls_back_to_keyword(self) -> None:
        """When semantic search raises an exception, keyword results still returned."""
        actor = _make_actor(semantic_search=True)
        _add_task(actor, 1, "auth service")

        mock_svc = MagicMock()
        mock_svc.embed.side_effect = RuntimeError("embedding API error")

        mock_index = MagicMock()
        mock_index.__len__ = MagicMock(return_value=1)
        actor._vector_index = mock_index

        with patch.object(actor, "_get_or_create_embedding_svc", return_value=mock_svc):
            # Should not raise; falls back to keyword
            result = actor.search_planning(query="auth")

        assert len(result) == 1
        assert result[0].id == 1


# ---------------------------------------------------------------------------
# AC#11 — _search_planning_factory emits ToolCallEvent
# ---------------------------------------------------------------------------


class TestSearchPlanningToolCallEvent:
    """AC11: _search_planning_factory emits ToolCallEvent(tool_name='Search planning', ...)."""

    def test_notify_event_called_with_correct_kwargs(self) -> None:
        from akgentic.tool.event import ToolCallEvent
        from akgentic.tool.planning.planning import PlanningTool, SearchPlanning

        tool = PlanningTool()

        mock_observer = MagicMock()
        tool._observer = mock_observer

        mock_proxy = MagicMock()
        mock_proxy.search_planning.return_value = []
        tool._planning_proxy = mock_proxy

        fn = tool._search_planning_factory(SearchPlanning())
        fn(status="pending", owner="@Alice", creator="@Bob", query="auth")

        mock_observer.notify_event.assert_called_once()
        call_args = mock_observer.notify_event.call_args[0][0]
        assert isinstance(call_args, ToolCallEvent)
        assert call_args.tool_name == "Search planning"
        assert call_args.args == []
        assert call_args.kwargs == {
            "status": "pending",
            "owner": "@Alice",
            "creator": "@Bob",
            "query": "auth",
        }

    def test_notify_event_with_all_none_kwargs(self) -> None:
        from akgentic.tool.event import ToolCallEvent
        from akgentic.tool.planning.planning import PlanningTool, SearchPlanning

        tool = PlanningTool()

        mock_observer = MagicMock()
        tool._observer = mock_observer

        mock_proxy = MagicMock()
        mock_proxy.search_planning.return_value = []
        tool._planning_proxy = mock_proxy

        fn = tool._search_planning_factory(SearchPlanning())
        fn()

        call_args = mock_observer.notify_event.call_args[0][0]
        assert isinstance(call_args, ToolCallEvent)
        assert call_args.kwargs == {
            "status": None,
            "owner": None,
            "creator": None,
            "query": None,
        }

    def test_no_notify_when_observer_is_none(self) -> None:
        """When observer is None, no AttributeError is raised."""
        from akgentic.tool.planning.planning import PlanningTool, SearchPlanning

        tool = PlanningTool()
        tool._observer = None  # type: ignore[assignment]

        mock_proxy = MagicMock()
        mock_proxy.search_planning.return_value = []
        tool._planning_proxy = mock_proxy

        fn = tool._search_planning_factory(SearchPlanning())
        result = fn(status="pending")

        assert result == []


# ---------------------------------------------------------------------------
# AC#1 — SearchPlanning param class structure
# ---------------------------------------------------------------------------


class TestSearchPlanningParamClass:
    """AC1: SearchPlanning has correct expose and field types."""

    def test_expose_contains_tool_call_and_command(self) -> None:
        from akgentic.tool.core import COMMAND, TOOL_CALL
        from akgentic.tool.planning.planning import SearchPlanning

        sp = SearchPlanning()
        assert TOOL_CALL in sp.expose
        assert COMMAND in sp.expose

    def test_all_fields_default_to_none(self) -> None:
        from akgentic.tool.planning.planning import SearchPlanning

        sp = SearchPlanning()
        assert sp.status is None
        assert sp.owner is None
        assert sp.creator is None
        assert sp.query is None

    def test_fields_accept_values(self) -> None:
        from akgentic.tool.planning.planning import SearchPlanning

        sp = SearchPlanning(status="pending", owner="@Alice", creator="@Bob", query="auth")
        assert sp.status == "pending"
        assert sp.owner == "@Alice"
        assert sp.creator == "@Bob"
        assert sp.query == "auth"


# ---------------------------------------------------------------------------
# AC#9 / AC#10 — PlanningTool.get_tools() / get_commands() include search_planning
# ---------------------------------------------------------------------------


class TestSearchPlanningWiring:
    """AC9 / AC10: PlanningTool wires search_planning into get_tools and get_commands."""

    def _make_wired_tool(self) -> "PlanningTool":
        from akgentic.tool.planning.planning import PlanningTool

        tool = PlanningTool()
        mock_observer = MagicMock()
        mock_observer.myAddress = MockActorAddress()
        tool._observer = mock_observer
        tool._planning_proxy = MagicMock()
        return tool

    def test_get_tools_includes_search_planning(self) -> None:
        from akgentic.tool.planning.planning import PlanningTool

        tool = self._make_wired_tool()
        tools = tool.get_tools()
        tool_names = [fn.__name__ for fn in tools]
        assert "search_planning" in tool_names

    def test_get_commands_includes_search_planning(self) -> None:
        from akgentic.tool.planning.planning import PlanningTool, SearchPlanning

        tool = self._make_wired_tool()
        commands = tool.get_commands()
        assert SearchPlanning in commands

    def test_get_tools_excludes_search_planning_when_disabled(self) -> None:
        from akgentic.tool.planning.planning import PlanningTool

        tool = PlanningTool(search_planning=False)
        mock_observer = MagicMock()
        mock_observer.myAddress = MockActorAddress()
        tool._observer = mock_observer
        tool._planning_proxy = MagicMock()

        tools = tool.get_tools()
        tool_names = [fn.__name__ for fn in tools]
        assert "search_planning" not in tool_names

    def test_get_commands_excludes_search_planning_when_disabled(self) -> None:
        from akgentic.tool.planning.planning import PlanningTool, SearchPlanning

        tool = PlanningTool(search_planning=False)
        mock_observer = MagicMock()
        mock_observer.myAddress = MockActorAddress()
        tool._observer = mock_observer
        tool._planning_proxy = MagicMock()

        commands = tool.get_commands()
        assert SearchPlanning not in commands
