"""Tests for the team context states and their ``TeamTool`` providers (ADR-037 §5)."""

from __future__ import annotations

import uuid
from unittest.mock import Mock

from akgentic.core import ActorAddressProxy
from akgentic.core.agent_card import AgentCard
from akgentic.core.agent_config import BaseConfig
from akgentic.core.orchestrator import Orchestrator

from akgentic.tool.core import COMMAND, LLM_CONTEXT, Channels
from akgentic.tool.team import (
    GetRoleProfiles,
    GetTeamRoster,
    RoleCatalogState,
    RoleRow,
    TeamMemberRow,
    TeamRosterState,
    TeamTool,
)
from akgentic.tool.team.observer import TeamManagementToolObserver


def _address(name: str, role: str = "Agent") -> ActorAddressProxy:
    """Create a mock ActorAddress for testing."""
    return ActorAddressProxy(
        {
            "__actor_address__": True,
            "__actor_type__": "test.Agent",
            "agent_id": str(uuid.uuid4()),
            "name": name,
            "role": role,
            "team_id": str(uuid.uuid4()),
            "squad_id": str(uuid.uuid4()),
            "is_user_proxy": False,
        }
    )


def _mock_observer() -> Mock:
    """Create a mock TeamManagementToolObserver with an orchestrator proxy."""
    observer = Mock(spec=TeamManagementToolObserver)
    observer.orchestrator = _address("@Orchestrator", "Orchestrator")
    observer.myAddress = _address("@Manager", "Manager")

    orchestrator_mock = Mock(spec=Orchestrator)
    orchestrator_mock.get_team.return_value = []
    orchestrator_mock.get_agent_catalog.return_value = []
    observer.proxy_ask.return_value = orchestrator_mock

    return observer


def _agent_card(
    role: str, description: str, skills: list[str], can_be_hired: bool = False
) -> AgentCard:
    """Build a real AgentCard for catalog tests.

    Not ``Mock(spec=AgentCard)``: the row builder reads ``can_be_hired``, and a
    spec'd mock raises ``AttributeError`` for a pydantic field name — which the
    provider's catch-all would swallow into a bare ``None``.
    """
    return AgentCard(
        agent_class=Mock,
        description=description,
        skills=skills,
        config=BaseConfig(name=role, role=role),
        can_be_hired=can_be_hired,
    )


def _member(name: str, role: str, is_self: bool = False) -> TeamMemberRow:
    return TeamMemberRow(name=name, role=role, is_self=is_self)


def _role(
    role: str,
    description: str = "desc",
    skills: list[str] | None = None,
    can_be_hired: bool = False,
) -> RoleRow:
    return RoleRow(
        role=role, description=description, skills=skills or [], can_be_hired=can_be_hired
    )


# ── render_full: byte-identical ports of the historical prompts ──────────────


def test_roster_render_full() -> None:
    """Roster render_full reproduces the team_members prompt byte for byte."""
    state = TeamRosterState(
        members=[
            _member("@Manager", "Manager", is_self=True),
            _member("@Developer", "Developer"),
        ]
    )

    assert state.render_full() == (
        "**Here is the team member list by name (and role):**\n"
        "@Manager (role: Manager) - [you]\n"
        "@Developer (role: Developer)"
    )


def test_roster_render_full_empty() -> None:
    """Roster render_full returns empty string when there are no members."""
    assert TeamRosterState(members=[]).render_full() == ""


def test_catalog_render_full() -> None:
    """Catalog render_full lists every role, byte for byte, each marked either way."""
    state = RoleCatalogState(
        roles=[
            _role("Developer", "Writes code", ["python", "testing"], can_be_hired=True),
            _role("Intern", "Learns", []),
        ]
    )

    assert state.render_full() == (
        "**Here is the team role list:**\n"
        "Developer: Writes code (Skills: python, testing) [hireable]\n"
        "Intern: Learns (Skills: none) [not hireable]"
    )


def test_catalog_render_full_with_no_hireable_role_says_so() -> None:
    """Nothing hireable is a legitimate catalog: the whole list renders, plus the sentence."""
    state = RoleCatalogState(
        roles=[_role("Developer", "Writes code", ["python"]), _role("Intern", "Learns", [])]
    )

    rendered = state.render_full()

    assert rendered == (
        "**Here is the team role list:**\n"
        "Developer: Writes code (Skills: python) [not hireable]\n"
        "Intern: Learns (Skills: none) [not hireable]\n"
        "No role in this list can be hired."
    )


def test_catalog_render_full_omits_the_sentence_when_a_role_is_hireable() -> None:
    """One hireable role makes the blanket sentence false, so it must not appear."""
    state = RoleCatalogState(
        roles=[
            _role("Developer", "Writes code", ["python"], can_be_hired=True),
            _role("Intern", "Learns", []),
        ]
    )

    assert "No role in this list can be hired." not in state.render_full()


def test_catalog_render_full_empty() -> None:
    """Catalog render_full returns empty string when there are no roles."""
    assert RoleCatalogState(roles=[]).render_full() == ""


# ── render_delta: names only what moved ──────────────────────────────────────


def test_roster_delta_hire_shows_joined_only() -> None:
    """A hire renders as joined; the unchanged member is not re-listed."""
    previous = TeamRosterState(members=[_member("@Manager", "Manager", is_self=True)])
    current = TeamRosterState(
        members=[
            _member("@Manager", "Manager", is_self=True),
            _member("@AgentBob", "architect"),
        ]
    )

    delta = current.render_delta(previous)

    assert delta == "@AgentBob (role: architect) joined the team."
    assert "@Manager" not in delta


def test_roster_delta_fire_shows_left_only() -> None:
    """A fire renders as left; the unchanged member is not re-listed."""
    previous = TeamRosterState(
        members=[
            _member("@Manager", "Manager", is_self=True),
            _member("@AgentCarol", "tester"),
        ]
    )
    current = TeamRosterState(members=[_member("@Manager", "Manager", is_self=True)])

    delta = current.render_delta(previous)

    assert delta == "@AgentCarol (role: tester) left the team."
    assert "@Manager" not in delta


def test_roster_delta_simultaneous_hire_and_fire() -> None:
    """A simultaneous hire and fire produces both lines in one delta."""
    previous = TeamRosterState(members=[_member("@AgentCarol", "tester")])
    current = TeamRosterState(members=[_member("@AgentBob", "architect")])

    delta = current.render_delta(previous)

    assert delta is not None
    assert "@AgentBob (role: architect) joined the team." in delta
    assert "@AgentCarol (role: tester) left the team." in delta


def test_roster_delta_unchanged_is_none() -> None:
    """An unchanged roster diffs to None."""
    previous = TeamRosterState(members=[_member("@Manager", "Manager", is_self=True)])
    current = TeamRosterState(members=[_member("@Manager", "Manager", is_self=True)])

    assert current.render_delta(previous) is None


def test_roster_delta_rehire_under_new_role_is_left_plus_joined() -> None:
    """Keying on (name, role): a role change surfaces as left + joined."""
    previous = TeamRosterState(members=[_member("@AgentBob", "tester")])
    current = TeamRosterState(members=[_member("@AgentBob", "architect")])

    delta = current.render_delta(previous)

    assert delta is not None
    assert "@AgentBob (role: architect) joined the team." in delta
    assert "@AgentBob (role: tester) left the team." in delta


def test_catalog_delta_role_added() -> None:
    """A new role renders once as added; unchanged roles are not re-listed."""
    previous = RoleCatalogState(roles=[_role("Developer")])
    current = RoleCatalogState(roles=[_role("Developer"), _role("Tester", "Tests code")])

    delta = current.render_delta(previous)

    assert delta is not None
    assert "Tester" in delta
    assert delta.count("\n") == 0  # one line: nothing else moved
    assert "Developer" not in delta


def test_catalog_delta_added_line_carries_the_marker() -> None:
    """The *added* delta line states hireability — the delta is the path every turn uses."""
    previous = RoleCatalogState(roles=[])
    current = RoleCatalogState(
        roles=[
            _role("Developer", "Writes code", ["python"], can_be_hired=True),
            _role("Intern", "Learns", []),
        ]
    )

    delta = current.render_delta(previous)

    assert delta is not None
    assert "Role added — Developer: Writes code (Skills: python) [hireable]." in delta
    assert "Role added — Intern: Learns (Skills: none) [not hireable]." in delta


def test_catalog_delta_role_removed() -> None:
    """A removed role renders once as removed."""
    previous = RoleCatalogState(roles=[_role("Developer"), _role("Tester")])
    current = RoleCatalogState(roles=[_role("Developer")])

    delta = current.render_delta(previous)

    assert delta is not None
    assert "Tester" in delta
    assert "Developer" not in delta


def test_catalog_delta_role_redescribed() -> None:
    """A changed description or skills list renders once as re-described."""
    previous = RoleCatalogState(roles=[_role("Developer", "Writes code", ["python"])])
    current = RoleCatalogState(roles=[_role("Developer", "Writes code", ["python", "rust"])])

    delta = current.render_delta(previous)

    assert delta is not None
    assert "Developer" in delta
    assert delta.count("Developer") == 1


def test_catalog_delta_redescribed_line_carries_the_marker() -> None:
    """The *re-described* delta line states hireability too, through the same renderer."""
    previous = RoleCatalogState(roles=[_role("Developer", "Writes code", ["python"])])
    current = RoleCatalogState(
        roles=[_role("Developer", "Writes code", ["python", "rust"], can_be_hired=True)]
    )

    delta = current.render_delta(previous)

    assert delta is not None
    assert delta == "Role re-described — Developer: Writes code (Skills: python, rust) [hireable]."


def test_catalog_delta_reports_a_hireability_only_change() -> None:
    """A role that only becomes hireable is re-described, with the NEW marker.

    The diff compares whole rows, so this holds by construction today. Narrowing
    that comparison to ``(description, skills)`` would read as a harmless
    optimisation and would silently stop reporting hireability changes.
    """
    previous = RoleCatalogState(roles=[_role("Developer", "Writes code", ["python"])])
    current = RoleCatalogState(
        roles=[_role("Developer", "Writes code", ["python"], can_be_hired=True)]
    )

    delta = current.render_delta(previous)

    assert delta is not None
    assert "[hireable]" in delta
    assert "[not hireable]" not in delta
    assert delta == "Role re-described — Developer: Writes code (Skills: python) [hireable]."


def test_catalog_delta_unchanged_is_none() -> None:
    """An unchanged catalog diffs to None."""
    previous = RoleCatalogState(roles=[_role("Developer", "Writes code", ["python"])])
    current = RoleCatalogState(roles=[_role("Developer", "Writes code", ["python"])])

    assert current.render_delta(previous) is None


# ── serialization round-trip ─────────────────────────────────────────────────


def test_roster_state_round_trip() -> None:
    """TeamRosterState round-trips through model_dump / model_validate."""
    state = TeamRosterState(
        members=[_member("@Manager", "Manager", is_self=True), _member("@Dev", "Developer")]
    )
    restored = TeamRosterState.model_validate(state.model_dump())

    assert restored == state
    assert restored.render_full() == state.render_full()


def test_catalog_state_round_trip() -> None:
    """RoleCatalogState round-trips through model_dump / model_validate.

    One row of each kind. A flag lost in serialization raises nothing — it comes
    back at the ``False`` default and renders a whole team ``[not hireable]``,
    which is indistinguishable from a team nobody marked.
    """
    state = RoleCatalogState(
        roles=[
            _role("Developer", "Writes code", ["python"], can_be_hired=True),
            _role("Intern", "Learns", []),
        ]
    )
    restored = RoleCatalogState.model_validate(state.model_dump())

    assert restored == state
    assert restored.roles[0].can_be_hired is True
    assert restored.render_full() == state.render_full()
    assert "[hireable]" in restored.render_full()


def test_catalog_state_loads_a_payload_written_before_hireability() -> None:
    """A persisted row with no ``can_be_hired`` key loads, fail-closed.

    This is what the field's default buys: a ``RoleCatalogState`` written before
    the flag existed stays loadable, and reads as not hireable rather than
    silently permitting a hire the team never granted.
    """
    restored = RoleCatalogState.model_validate(
        {"roles": [{"role": "Developer", "description": "Writes code", "skills": ["python"]}]}
    )

    assert restored.roles[0].can_be_hired is False
    assert restored.render_full() == (
        "**Here is the team role list:**\n"
        "Developer: Writes code (Skills: python) [not hireable]\n"
        "No role in this list can be hired."
    )


# ── provider gating on TeamTool.get_context_states() ─────────────────────────


def test_default_tool_yields_two_named_providers() -> None:
    """The default TeamTool exposes exactly the two providers, by stable name."""
    tool = TeamTool()
    tool.observer(_mock_observer())

    providers = tool.get_context_states()

    assert [p.__name__ for p in providers] == ["team_roster_state", "role_catalog_state"]


def test_disabled_roster_yields_no_roster_provider() -> None:
    """get_team_roster=False drops the roster provider."""
    tool = TeamTool(get_team_roster=False)
    tool.observer(_mock_observer())

    providers = tool.get_context_states()

    assert [p.__name__ for p in providers] == ["role_catalog_state"]


def test_expose_without_llm_context_yields_no_roster_provider() -> None:
    """A roster narrowed to COMMAND only must not surface a provider (silent-drop trap)."""
    tool = TeamTool(get_team_roster=GetTeamRoster(expose={Channels.COMMAND}))
    tool.observer(_mock_observer())

    providers = tool.get_context_states()

    assert [p.__name__ for p in providers] == ["role_catalog_state"]


def test_get_system_prompts_returns_nothing() -> None:
    """The roster/profiles prompt entries are gone from the SYSTEM_PROMPT surface."""
    tool = TeamTool()
    tool.observer(_mock_observer())

    assert tool.get_system_prompts() == []


# ── provider behavior ────────────────────────────────────────────────────────


def test_roster_provider_shapes_rows_per_agent() -> None:
    """The roster provider excludes #-tool actors and bakes in is_self."""
    observer = _mock_observer()
    observer.proxy_ask.return_value.get_team.return_value = [
        _address("@Manager", "Manager"),
        _address("@Developer", "Developer"),
        _address("#PlanningTool", "ToolActor"),  # Should be excluded
    ]

    tool = TeamTool()
    tool.observer(observer)
    roster_provider = tool.get_context_states()[0]

    state = roster_provider()

    assert isinstance(state, TeamRosterState)
    assert state.members == [
        _member("@Manager", "Manager", is_self=True),
        _member("@Developer", "Developer"),
    ]


def test_roster_provider_empty_team_is_a_state_not_none() -> None:
    """An empty team is a valid state whose render_full is '', never None."""
    tool = TeamTool()
    tool.observer(_mock_observer())
    roster_provider = tool.get_context_states()[0]

    state = roster_provider()

    assert isinstance(state, TeamRosterState)
    assert state.render_full() == ""


def test_roster_provider_returns_none_on_proxy_failure() -> None:
    """A raising orchestrator proxy makes the roster provider return None."""
    observer = _mock_observer()
    observer.proxy_ask.return_value.get_team.side_effect = RuntimeError("actor gone")

    tool = TeamTool()
    tool.observer(observer)
    roster_provider = tool.get_context_states()[0]

    assert roster_provider() is None


def test_catalog_provider_reads_the_agent_catalog() -> None:
    """The provider snapshots the catalog, each row's flag matching its card one for one."""
    observer = _mock_observer()
    observer.proxy_ask.return_value.get_agent_catalog.return_value = [
        _agent_card("Developer", "Writes code", ["python", "testing"], can_be_hired=True),
        _agent_card("Intern", "Learns", [], can_be_hired=False),
    ]

    tool = TeamTool()
    tool.observer(observer)
    catalog_provider = tool.get_context_states()[1]

    state = catalog_provider()

    assert isinstance(state, RoleCatalogState)
    assert state.roles == [
        _role("Developer", "Writes code", ["python", "testing"], can_be_hired=True),
        _role("Intern", "Learns", [], can_be_hired=False),
    ]


def test_catalog_provider_empty_catalog_is_a_state_not_none() -> None:
    """An empty catalog is a valid state whose render_full is '', never None."""
    tool = TeamTool()
    tool.observer(_mock_observer())
    catalog_provider = tool.get_context_states()[1]

    state = catalog_provider()

    assert isinstance(state, RoleCatalogState)
    assert state.render_full() == ""


def test_catalog_provider_returns_none_on_proxy_failure() -> None:
    """A raising orchestrator proxy makes the catalog provider return None."""
    observer = _mock_observer()
    observer.proxy_ask.return_value.get_agent_catalog.side_effect = RuntimeError("actor gone")

    tool = TeamTool()
    tool.observer(observer)
    catalog_provider = tool.get_context_states()[1]

    assert catalog_provider() is None


# ── persisted-payload normalizer adoption (ADR-037 §4) ───────────────────────


def test_persisted_system_prompt_expose_revalidates_to_llm_context() -> None:
    """A payload with expose ['system_prompt', 'command'] resolves to LLM_CONTEXT."""
    payload = TeamTool().model_dump()
    payload["get_team_roster"] = {"expose": ["system_prompt", "command"]}
    payload["get_role_profiles"] = {"expose": ["system_prompt", "command"]}

    restored = TeamTool.model_validate(payload)

    assert isinstance(restored.get_team_roster, GetTeamRoster)
    assert restored.get_team_roster.expose == {LLM_CONTEXT, COMMAND}
    assert isinstance(restored.get_role_profiles, GetRoleProfiles)
    assert restored.get_role_profiles.expose == {LLM_CONTEXT, COMMAND}

    restored.observer(_mock_observer())
    providers = restored.get_context_states()
    assert [p.__name__ for p in providers] == ["team_roster_state", "role_catalog_state"]
