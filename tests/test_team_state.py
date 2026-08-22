"""Tests for the team context states and their ``TeamTool`` providers (ADR-037 §5)."""

from __future__ import annotations

import uuid
from unittest.mock import Mock

from akgentic.core import ActorAddressProxy
from akgentic.core.agent_card import AgentCard
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


def _agent_card(role: str, description: str, skills: list[str]) -> Mock:
    """Create a mock AgentCard for catalog tests."""
    card = Mock(spec=AgentCard)
    card.role = role
    card.description = description
    card.skills = skills
    return card


def _member(name: str, role: str, is_self: bool = False) -> TeamMemberRow:
    return TeamMemberRow(name=name, role=role, is_self=is_self)


def _role(role: str, description: str = "desc", skills: list[str] | None = None) -> RoleRow:
    return RoleRow(role=role, description=description, skills=skills or [])


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
    """Catalog render_full reproduces the team_roles prompt byte for byte."""
    state = RoleCatalogState(
        roles=[
            _role("Developer", "Writes code", ["python", "testing"]),
            _role("Intern", "Learns", []),
        ]
    )

    assert state.render_full() == (
        "**Here is the available team role list (for hiring):**\n"
        "Developer: Writes code (Skills: python, testing)\n"
        "Intern: Learns (Skills: none)"
    )


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
    """RoleCatalogState round-trips through model_dump / model_validate."""
    state = RoleCatalogState(roles=[_role("Developer", "Writes code", ["python"])])
    restored = RoleCatalogState.model_validate(state.model_dump())

    assert restored == state
    assert restored.render_full() == state.render_full()


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
    """The catalog provider snapshots the orchestrator's agent catalog."""
    observer = _mock_observer()
    observer.proxy_ask.return_value.get_agent_catalog.return_value = [
        _agent_card("Developer", "Writes code", ["python", "testing"]),
        _agent_card("Intern", "Learns", []),
    ]

    tool = TeamTool()
    tool.observer(observer)
    catalog_provider = tool.get_context_states()[1]

    state = catalog_provider()

    assert isinstance(state, RoleCatalogState)
    assert state.roles == [
        _role("Developer", "Writes code", ["python", "testing"]),
        _role("Intern", "Learns", []),
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
