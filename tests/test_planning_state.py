"""Tests for ``PlanningState`` and its ``PlanningTool`` provider (ADR-037 §5)."""

from __future__ import annotations

import gc
from unittest.mock import MagicMock

from akgentic.tool.core import COMMAND, LLM_CONTEXT, SYSTEM_PROMPT, Channels
from akgentic.tool.planning import PlanningState, PlanningTool, TaskRow
from akgentic.tool.planning.planning import GetPlanning, _build_planning_state
from akgentic.tool.planning.planning_actor import Task

_NAV_HINT = (
    "Use get_planning_task(id) for exact ID lookup or search_planning(...) to filter tasks."
)


def make_task(
    *,
    id: int,
    status: str = "pending",
    description: str = "A task",
    owner: str = "",
    creator: str = "",
    output: str = "",
) -> Task:
    """Build a Task with explicit fields for concise test setup."""
    return Task(
        id=id,
        status=status,  # type: ignore[arg-type]
        description=description,
        owner=owner,
        creator=creator,
        output=output,
        dependencies=[],
    )


def make_state(
    tasks: list[Task],
    agent_name: str = "@DevAgent",
    filter_by_agent: bool = True,
) -> PlanningState:
    """Build a PlanningState the way the provider does."""
    return _build_planning_state(tasks, agent_name, filter_by_agent)


def make_tool(
    tasks: list[Task],
    agent_name: str = "@DevAgent",
) -> tuple[PlanningTool, MagicMock]:
    """Build a PlanningTool with mocked observer + planning proxy.

    Returns the tool and the observer mock — the tool only holds the observer
    weakly, so the caller must keep the returned mock alive.
    """
    tool = PlanningTool.model_construct()

    mock_observer = MagicMock()
    mock_observer.myAddress.name = agent_name
    tool._observer = mock_observer

    mock_proxy = MagicMock()
    mock_proxy.get_planning.return_value = tasks
    tool._planning_proxy = mock_proxy

    tool.get_planning = GetPlanning()
    return tool, mock_observer


# ── render_full: byte-identical port of the historical planning_summary ──────


def test_render_full_filtered_byte_identical() -> None:
    """Filtered render_full reproduces the planning_summary prompt byte for byte."""
    tasks = [
        make_task(id=1, description="Implement auth", owner="@DevAgent", creator="@DevAgent"),
        make_task(
            id=2,
            status="started",
            description="Review PR",
            owner="@ArchAgent",
            creator="@DevAgent",
            output="comments posted",
        ),
        make_task(id=3, description="Orphan", owner="", creator="@DevAgent"),
    ]

    assert make_state(tasks).render_full() == (
        "**Team planning:** 3 tasks total\n"
        "Owners: @ArchAgent: 1 | @DevAgent: 1 | unassigned: 1\n"
        "\n**Your tasks** (owner or creator: @DevAgent):\n"
        "- ID 1 [pending] Implement auth (Owner: @DevAgent, Creator: @DevAgent)\n"
        "- ID 2 [started] Review PR — Output: comments posted "
        "(Owner: @ArchAgent, Creator: @DevAgent)\n"
        "\nUse get_planning_task(id) for exact ID lookup or "
        "search_planning(...) to filter tasks."
    )


def test_render_full_unfiltered_byte_identical() -> None:
    """Unfiltered render_full reproduces the all-tasks prompt byte for byte."""
    tasks = [make_task(id=1, description="Implement auth", owner="@DevAgent", creator="@Boss")]

    assert make_state(tasks, filter_by_agent=False).render_full() == (
        "**Team planning:** 1 task total\n"
        "Owners: @DevAgent: 1\n"
        "\n**All tasks:**\n"
        "- ID 1 [pending] Implement auth (Owner: @DevAgent, Creator: @Boss)\n"
        "\nUse get_planning_task(id) for exact ID lookup or "
        "search_planning(...) to filter tasks."
    )


def test_render_full_empty_board_sentinel_both_modes() -> None:
    """An empty board renders the sentinel line — not '' — in both view modes."""
    assert make_state([]).render_full() == "No current team planning."
    assert make_state([], filter_by_agent=False).render_full() == "No current team planning."


def test_render_full_named_owners_sorted_alphabetically_unassigned_last() -> None:
    """Named owners sort alphabetically; unassigned comes last, only when non-zero."""
    tasks = [
        make_task(id=1, owner="@ZAgent"),
        make_task(id=2, owner="@AAgent"),
        make_task(id=3, owner=""),
    ]
    result = make_state(tasks, agent_name="@AAgent").render_full()

    assert "Owners: @AAgent: 1 | @ZAgent: 1 | unassigned: 1" in result


def test_render_full_no_unassigned_entry_when_zero() -> None:
    """The unassigned entry is absent from the breakdown when every task has an owner."""
    tasks = [make_task(id=1, owner="@DevAgent")]
    result = make_state(tasks).render_full()

    assert "unassigned" not in result.splitlines()[1]


def test_render_full_total_line_singular_and_plural() -> None:
    """1 task uses 'task'; several use 'tasks'."""
    one = make_state([make_task(id=1, owner="@DevAgent")]).render_full()
    two = make_state(
        [make_task(id=1, owner="@DevAgent"), make_task(id=2, owner="@DevAgent")]
    ).render_full()

    assert "**Team planning:** 1 task total" in one
    assert "1 tasks total" not in one
    assert "**Team planning:** 2 tasks total" in two


def test_filtered_state_excludes_other_agents_tasks() -> None:
    """Only tasks owned or (assigned and) created by the agent enter the rows."""
    tasks = [
        make_task(id=1, owner="@DevAgent", description="Dev task"),
        make_task(id=2, owner="@ArchAgent", description="Arch task"),
    ]
    state = make_state(tasks)

    assert [row.id for row in state.tasks] == [1]
    assert "Arch task" not in state.render_full()


def test_filtered_state_excludes_unassigned_created_task_but_counts_it() -> None:
    """An unassigned task created by the agent is out of the rows, in the counts."""
    tasks = [make_task(id=1, owner="", creator="@DevAgent", description="Orphan task")]
    state = make_state(tasks)
    result = state.render_full()

    assert state.tasks == []
    assert state.total == 1
    assert "Orphan task" not in result
    assert "unassigned: 1" in result


def test_render_full_no_tasks_message_when_filtered_view_is_empty() -> None:
    """No visible rows renders the no-tasks message, with summary and nav hint."""
    tasks = [make_task(id=1, owner="@ArchAgent")]
    result = make_state(tasks).render_full()

    assert "**Team planning:**" in result
    assert "\nNo tasks assigned to or created by @DevAgent yet." in result
    assert _NAV_HINT in result


def test_render_full_output_suffix_present_only_when_output_nonempty() -> None:
    """' — Output: ...' appears exactly for rows whose output text is non-empty."""
    tasks = [
        make_task(id=1, owner="@DevAgent", output="skeleton committed"),
        make_task(id=2, owner="@DevAgent", output=""),
    ]
    result = make_state(tasks).render_full()

    assert "— Output: skeleton committed" in result
    assert result.count("— Output:") == 1


def test_render_full_nav_hint_absent_on_empty_board() -> None:
    """The empty-board sentinel carries no navigation hint."""
    assert _NAV_HINT not in make_state([]).render_full()


# ── render_delta: names only what moved ──────────────────────────────────────


def test_delta_status_change_renders_status_movement_and_nothing_else() -> None:
    """A pure status change names the movement — no total, no re-listed rows."""
    previous = make_state(
        [make_task(id=1, owner="@DevAgent"), make_task(id=2, owner="@DevAgent")]
    )
    current = make_state(
        [
            make_task(id=1, status="completed", owner="@DevAgent"),
            make_task(id=2, owner="@DevAgent"),
        ]
    )

    delta = current.render_delta(previous)

    assert delta == "ID 1 [pending] → [completed] (Owner: @DevAgent)."


def test_delta_new_task_renders_as_new_with_total() -> None:
    """A new task renders as new; the moved total is restated."""
    previous = make_state([make_task(id=1, owner="@DevAgent")])
    current = make_state(
        [
            make_task(id=1, owner="@DevAgent"),
            make_task(
                id=9,
                description="Draft the migration plan",
                owner="@DevAgent",
                creator="@AgentAlice",
            ),
        ]
    )

    delta = current.render_delta(previous)

    assert delta is not None
    assert "2 tasks total." in delta
    assert (
        "New: ID 9 [pending] Draft the migration plan "
        "(Owner: @DevAgent, Creator: @AgentAlice)." in delta
    )
    assert "ID 1" not in delta


def test_delta_deleted_task_renders_as_removed_with_total() -> None:
    """A deleted task renders as removed; the moved total is restated."""
    previous = make_state(
        [
            make_task(id=1, owner="@DevAgent"),
            make_task(id=2, owner="@DevAgent", description="Old task"),
        ]
    )
    current = make_state([make_task(id=1, owner="@DevAgent")])

    delta = current.render_delta(previous)

    assert delta is not None
    assert "1 task total." in delta
    assert "Removed: ID 2 [pending] Old task." in delta


def test_delta_reassigned_away_is_removed_in_filtered_view() -> None:
    """A task reassigned away from the agent leaves the filtered view as removed."""
    previous = make_state([make_task(id=2, owner="@DevAgent", description="Handover")])
    current = make_state([make_task(id=2, owner="@ArchAgent", description="Handover")])

    delta = current.render_delta(previous)

    assert delta == "Removed: ID 2 [pending] Handover."


def test_delta_reassignment_is_owner_change_in_unfiltered_view() -> None:
    """The same reassignment renders as an owner change in an unfiltered state."""
    previous = make_state(
        [make_task(id=2, owner="@DevAgent", description="Handover")], filter_by_agent=False
    )
    current = make_state(
        [make_task(id=2, owner="@ArchAgent", description="Handover")], filter_by_agent=False
    )

    delta = current.render_delta(previous)

    assert delta == "ID 2 owner: @DevAgent → @ArchAgent."


def test_delta_output_appearing_renders_once() -> None:
    """Output appearing on a row renders exactly once, with no other noise."""
    previous = make_state([make_task(id=1, owner="@DevAgent")])
    current = make_state([make_task(id=1, owner="@DevAgent", output="artifact.md")])

    delta = current.render_delta(previous)

    assert delta == "ID 1 output: artifact.md."


def test_delta_description_change_renders_new_description() -> None:
    """A re-described task renders its new description and nothing else."""
    previous = make_state([make_task(id=1, owner="@DevAgent", description="Draft plan")])
    current = make_state([make_task(id=1, owner="@DevAgent", description="Ship plan")])

    delta = current.render_delta(previous)

    assert delta == "ID 1 description: Ship plan."


def test_delta_output_cleared_renders_once() -> None:
    """Output disappearing from a row renders as cleared, not as re-listed."""
    previous = make_state([make_task(id=1, owner="@DevAgent", output="draft.md")])
    current = make_state([make_task(id=1, owner="@DevAgent")])

    delta = current.render_delta(previous)

    assert delta == "ID 1 output cleared."


def test_delta_summary_only_movement_is_total_only() -> None:
    """Invisible-task churn under filtering yields a total-only delta, no rows."""
    previous = make_state(
        [make_task(id=1, owner="@DevAgent"), make_task(id=2, owner="@ArchAgent")]
    )
    current = make_state(
        [
            make_task(id=1, owner="@DevAgent"),
            make_task(id=2, owner="@ArchAgent"),
            make_task(id=3, owner="@ArchAgent"),
        ]
    )

    assert current.render_delta(previous) == "3 tasks total."


def test_delta_identical_states_return_none() -> None:
    """An unchanged board diffs to None."""
    tasks = [make_task(id=1, owner="@DevAgent", output="done")]

    assert make_state(tasks).render_delta(make_state(tasks)) is None


def test_delta_two_empty_states_return_none() -> None:
    """Two empty boards diff to None."""
    assert make_state([]).render_delta(make_state([])) is None


# ── serialization round-trip ─────────────────────────────────────────────────


def test_planning_state_round_trip() -> None:
    """PlanningState round-trips through model_dump / model_validate."""
    state = make_state(
        [
            make_task(id=1, owner="@DevAgent", creator="@Boss", output="done"),
            make_task(id=2, owner=""),
        ]
    )
    restored = PlanningState.model_validate(state.model_dump())

    assert restored == state
    assert restored.render_full() == state.render_full()


def test_task_row_round_trip() -> None:
    """TaskRow round-trips through model_dump / model_validate."""
    row = TaskRow(
        id=1, status="pending", description="A task", owner="@A", creator="@B", output="x"
    )

    assert TaskRow.model_validate(row.model_dump()) == row


# ── provider gating on PlanningTool.get_context_states() ─────────────────────


def test_default_tool_yields_one_named_provider() -> None:
    """The default PlanningTool exposes exactly one provider, by stable name."""
    tool, _observer = make_tool([])

    providers = tool.get_context_states()

    assert [p.__name__ for p in providers] == ["planning_state"]


def test_disabled_get_planning_yields_no_provider() -> None:
    """get_planning=False drops the provider."""
    tool, _observer = make_tool([])
    tool.get_planning = False

    assert tool.get_context_states() == []


def test_expose_without_llm_context_yields_no_provider() -> None:
    """A get_planning narrowed to COMMAND only must not surface a provider
    (silent-drop trap)."""
    tool, _observer = make_tool([])
    tool.get_planning = GetPlanning(expose={Channels.COMMAND})

    assert tool.get_context_states() == []


# ── provider behavior ────────────────────────────────────────────────────────


def test_provider_builds_agent_shaped_state() -> None:
    """The provider bakes agent name and filtering into the produced state."""
    tool, _observer = make_tool(
        [
            make_task(id=1, owner="@DevAgent", description="Mine"),
            make_task(id=2, owner="@ArchAgent", description="Theirs"),
        ]
    )
    provider = tool.get_context_states()[0]

    state = provider()

    assert isinstance(state, PlanningState)
    assert state.agent_name == "@DevAgent"
    assert state.filter_by_agent is True
    assert state.total == 2
    assert [row.id for row in state.tasks] == [1]


def test_provider_empty_board_is_a_state_not_none() -> None:
    """An empty board is a real state whose render_full is the sentinel line."""
    tool, _observer = make_tool([])
    provider = tool.get_context_states()[0]

    state = provider()

    assert isinstance(state, PlanningState)
    assert state.render_full() == "No current team planning."


def test_provider_returns_none_after_observer_collected() -> None:
    """A collected observer makes the provider return None — never raise."""
    tool, observer = make_tool([make_task(id=1, owner="@DevAgent")])
    provider = tool.get_context_states()[0]

    del observer
    gc.collect()

    assert tool._observer_or_none() is None
    assert provider() is None


def test_provider_returns_none_on_proxy_failure() -> None:
    """A raising planning proxy makes the provider return None — never raise."""
    tool, _observer = make_tool([])
    tool._planning_proxy.get_planning.side_effect = RuntimeError("actor gone")
    provider = tool.get_context_states()[0]

    assert provider() is None


# ── persisted-payload normalizer adoption (ADR-037 §4) ───────────────────────


def test_persisted_system_prompt_expose_revalidates_to_llm_context() -> None:
    """A payload with expose ['system_prompt', 'command'] resolves to LLM_CONTEXT
    and still yields the provider."""
    payload = PlanningTool().model_dump()
    payload["get_planning"] = {"expose": ["system_prompt", "command"]}

    restored = PlanningTool.model_validate(payload)

    assert isinstance(restored.get_planning, GetPlanning)
    assert restored.get_planning.expose == {LLM_CONTEXT, COMMAND}

    mock_observer = MagicMock()
    mock_observer.myAddress.name = "@DevAgent"
    restored._observer = mock_observer
    restored._planning_proxy = MagicMock()
    providers = restored.get_context_states()
    assert [p.__name__ for p in providers] == ["planning_state"]


def test_other_planning_params_keep_system_prompt_untouched() -> None:
    """GetPlanningTask / SearchPlanning payloads keep system_prompt in expose —
    the normalizer is per-param, never a global rewrite."""
    payload = PlanningTool().model_dump()
    payload["get_planning_task"] = {"expose": ["system_prompt", "command"]}
    payload["search_planning"] = {"expose": ["system_prompt", "tool_call"]}

    restored = PlanningTool.model_validate(payload)

    assert not isinstance(restored.get_planning_task, bool)
    assert SYSTEM_PROMPT in restored.get_planning_task.expose
    assert not isinstance(restored.search_planning, bool)
    assert SYSTEM_PROMPT in restored.search_planning.expose
