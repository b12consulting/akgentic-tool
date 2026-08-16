"""Team management tool implementation.

Provides hire, fire, and team awareness capabilities as a reusable ToolCard
that can be attached to any agent.
"""

import logging
import random
from collections.abc import Callable
from typing import Any, cast

from pydantic import Field, PrivateAttr

from akgentic.core.actor_address import ActorAddress
from akgentic.core.agent import Akgent
from akgentic.core.agent_card import AgentCard
from akgentic.core.agent_config import BaseConfig
from akgentic.core.orchestrator import Orchestrator
from akgentic.tool.core import (
    COMMAND,
    SYSTEM_PROMPT,
    TOOL_CALL,
    BaseToolParam,
    Channels,
    ToolCard,
    _resolve,
)
from akgentic.tool.core.observer import ToolObserver
from akgentic.tool.errors import RetriableError
from akgentic.tool.team.activity import (
    TEAM_ACTIVITY_ACTOR_NAME,
    TEAM_ACTIVITY_ACTOR_ROLE,
    ActivitySnapshot,
    GetTeamActivity,
    SummaryBudget,
    TeamActivityActor,
    TeamActivityReport,
    apply_summaries,
    apply_truncations,
    build_snapshot,
)
from akgentic.tool.team.observer import TeamManagementToolObserver

logger = logging.getLogger(__name__)


class HireTeamMember(BaseToolParam):
    """Hire new team members by role."""

    expose: set[Channels] = {TOOL_CALL, COMMAND}


class FireTeamMember(BaseToolParam):
    """Fire existing team members by name."""

    expose: set[Channels] = {TOOL_CALL, COMMAND}


class GetTeamRoster(BaseToolParam):
    """Get current team roster as system prompt."""

    expose: set[Channels] = {SYSTEM_PROMPT, COMMAND}


class GetRoleProfiles(BaseToolParam):
    """Get available role profiles as system prompt."""

    expose: set[Channels] = {SYSTEM_PROMPT, COMMAND}


def _hire_single_member(
    orchestrator_proxy: Orchestrator,
    observer: TeamManagementToolObserver,
    role: str,
    name: str | None,
    existing_names: set[str],
    agent_catalog: list[AgentCard] | None = None,
) -> ActorAddress:
    """Core hire logic for a single member.

    Args:
        orchestrator_proxy: Proxy to the orchestrator actor.
        observer: TeamManagementToolObserver for actor creation and hooks.
        role: Role to hire.
        name: Optional specific name. If None, auto-generated.
        existing_names: Set of existing member names (for uniqueness).
        agent_catalog: Pre-fetched catalog. If None, fetched from orchestrator.

    Returns:
        ActorAddress: Address of the newly hired child actor.

    Raises:
        RetriableError: If no agent card found for the role.
        ValueError: If agent class is a string (configuration error).
    """
    if agent_catalog is None:
        agent_catalog = orchestrator_proxy.get_agent_catalog()
    agent_card = next((card for card in agent_catalog if card.role == role), None)
    if agent_card is None:
        available_roles = orchestrator_proxy.get_available_roles()
        raise RetriableError(
            f"Hire error - cannot find agent card for role '{role}'. "
            f"Available roles: {available_roles}"
        )

    actor_class = agent_card.get_agent_class()
    if isinstance(actor_class, str):
        raise ValueError(f"Hire error - agent class for role {role} is a string, not a type.")

    if name is None:
        role_prefix = f"@{role.replace(' ', '')}"
        suffix = random.randint(100, 999)
        child_name = f"{role_prefix}{suffix}"
        while child_name in existing_names:
            suffix += 1
            child_name = f"{role_prefix}{suffix}"
    else:
        child_name = name
        if not isinstance(child_name, str):
            raise RetriableError("Hire error - member name must be a string.")
        child_name = child_name.strip()
        if not child_name:
            raise RetriableError("Hire error - member name cannot be empty.")
        if child_name in existing_names:
            raise RetriableError(
                f"Hire error - member name '{child_name}' already exists. "
                "Please choose a unique name."
            )

    agent_card_config = agent_card.get_config_copy()
    agent_card_config.name = child_name
    agent_card_config.role = role

    child_address = observer.createActor(actor_class, config=agent_card_config)
    observer.on_hire(child_address)

    logger.info(f"Hired {role} agent: {child_name} at {child_address}")
    return child_address


def _fire_single_member(
    orchestrator_proxy: Orchestrator,
    observer: TeamManagementToolObserver,
    name: str,
) -> str:
    """Core fire logic for a single member.

    Args:
        orchestrator_proxy: Proxy to the orchestrator actor.
        observer: TeamManagementToolObserver for hooks.
        name: Name of the member to fire.

    Returns:
        The fired member's name.

    Raises:
        RetriableError: If member not found in team.
    """
    address = orchestrator_proxy.get_team_member(name)
    if address is None:
        team_members = [member.name for member in orchestrator_proxy.get_team()]
        raise RetriableError(
            f"Fire error - member '{name}' not part of the team. "
            f"Current team members: {team_members}"
        )

    observer.proxy_ask(address, Akgent).stop()
    observer.on_fire(address)
    logger.info(f"Fired team member: {name}")
    return name


class TeamTool(ToolCard):
    """Team management tool for hiring, firing, and team awareness.

    Provides:
    - hire_members(roles: list[str]) -> str: Hire team members
    - fire_members(names: list[str]) -> str: Fire team members
    - who_is_working() -> TeamActivityReport: Who is mid-handler (opt-in, off by default)
    - Team roster system prompt: Current team composition
    - Role profiles system prompt: Available roles and descriptions
    """

    hire_team_members: HireTeamMember | bool = Field(
        default=True, description="Enable hiring team members (default: True)"
    )
    fire_team_members: FireTeamMember | bool = Field(
        default=True, description="Enable firing team members (default: True)"
    )
    get_role_profiles: GetRoleProfiles | bool = Field(
        default=True, description="Include role profiles in system prompt (default: True)"
    )
    get_team_roster: GetTeamRoster | bool = Field(
        default=True, description="Include team roster in system prompt (default: True)"
    )
    get_team_activity: GetTeamActivity | bool = Field(
        default=False, description="Enable the who_is_working report (default: False)"
    )

    # Runtime handle: an actor proxy is not serializable and never a field.
    _activity_proxy: TeamActivityActor | None = PrivateAttr(default=None)

    def observer(self, observer: ToolObserver) -> "TeamTool":
        """Attach observer and set up the orchestrator proxy.

        Requires a TeamManagementToolObserver for actor system access. The
        parameter keeps the base ``ToolObserver`` type — ``ToolFactory`` attaches
        one observer to every card uniformly, so narrowing it would break
        substitutability; :meth:`_team_observer` applies the narrower type.

        Args:
            observer: Observer implementing TeamManagementToolObserver protocol

        Returns:
            Self, enabling method chaining

        Raises:
            ValueError: If observer.orchestrator is None
        """
        super().observer(observer)  # store the observer weakly via the base setter
        team_observer = self._team_observer()
        if team_observer.orchestrator is None:
            raise ValueError("TeamTool requires access to the orchestrator.")

        self._orchestrator_proxy = team_observer.proxy_ask(
            team_observer.orchestrator, Orchestrator
        )
        self._bind_activity_actor(team_observer)
        return self

    def _bind_activity_actor(self, observer: TeamManagementToolObserver) -> None:
        """Bind the ``#TeamActivity`` singleton — only when a summarizer is configured.

        Two independent gates, and conflating them is the defect this shape exists
        to prevent. ``get_team_activity`` alone exposes ``who_is_working`` as pure
        telemetry plus truncation; the actor exists *solely* to cache summaries, so
        with no summarizer there is nothing to cache and the team pays for no actor
        at all — which is what makes an opt-in capability safe on the most widely
        used card in the package.

        The cache actor is reached through an **ask** proxy and its TELL-shaped
        ``request`` is called on that same proxy. That is safe rather than a partial
        adoption of the mechanism: ``request`` adds to a set, spawns a worker and
        tells it the payload — all O(1) on the cache actor's thread, so the ask
        never waits on external work.

        Args:
            observer: The team observer, already validated by :meth:`observer`.
        """
        gta = _resolve(self.get_team_activity, GetTeamActivity)
        if gta is None or gta.summarizer is None:
            return

        activity_addr = self._orchestrator_proxy.getChildrenOrCreate(
            TeamActivityActor,
            config=BaseConfig(
                name=TEAM_ACTIVITY_ACTOR_NAME,
                role=TEAM_ACTIVITY_ACTOR_ROLE,
            ),
        )
        self._activity_proxy = observer.proxy_ask(activity_addr, TeamActivityActor)

    def _team_observer(self) -> TeamManagementToolObserver:
        """Live observer typed as the team protocol. Raises once the agent stops.

        Conformance is a documented precondition of :meth:`observer`, not a runtime
        gate — observers are duck-typed, so a non-conforming one fails at first use
        just as it did before.
        """
        return cast(TeamManagementToolObserver, self._observer)

    def _team_observer_or_none(self) -> TeamManagementToolObserver | None:
        """Live observer typed as the team protocol; ``None`` once the agent stops."""
        return cast(TeamManagementToolObserver | None, self._observer_or_none())

    def get_system_prompts(self) -> list[Callable[..., Any]]:
        """Get dynamic system prompts for team context.

        Returns:
            List of callable system prompts (roster and/or profiles)
        """
        prompts: list[Callable[..., Any]] = []

        gtr = _resolve(self.get_team_roster, GetTeamRoster)
        if gtr and SYSTEM_PROMPT in gtr.expose:
            prompts.append(self._team_roster_prompt_factory(gtr))

        grp = _resolve(self.get_role_profiles, GetRoleProfiles)
        if grp and SYSTEM_PROMPT in grp.expose:
            prompts.append(self._role_profiles_prompt_factory(grp))

        return prompts

    def get_tools(self) -> list[Callable[..., Any]]:
        """Get LLM-callable tools for team management.

        Returns:
            List of callable tools (hire_members and/or fire_members)
        """
        tools: list[Callable[..., Any]] = []

        htm = _resolve(self.hire_team_members, HireTeamMember)
        if htm and TOOL_CALL in htm.expose:
            tools.append(self._hire_members_factory(htm))

        ftm = _resolve(self.fire_team_members, FireTeamMember)
        if ftm and TOOL_CALL in ftm.expose:
            tools.append(self._fire_members_factory(ftm))

        gta = _resolve(self.get_team_activity, GetTeamActivity)
        if gta and TOOL_CALL in gta.expose:
            tools.append(self._who_is_working(gta))

        return tools

    def get_commands(self) -> dict[type[BaseToolParam], Callable[..., Any]]:
        """Get programmatic commands for inter-agent orchestration.

        Returns:
            Dict mapping param class to callable.
        """
        commands: dict[type[BaseToolParam], Callable[..., Any]] = {}

        htm = _resolve(self.hire_team_members, HireTeamMember)
        if htm and COMMAND in htm.expose:
            commands[HireTeamMember] = self._hire_member_command_factory(htm)

        ftm = _resolve(self.fire_team_members, FireTeamMember)
        if ftm and COMMAND in ftm.expose:
            commands[FireTeamMember] = self._fire_member_command_factory(ftm)

        gtr = _resolve(self.get_team_roster, GetTeamRoster)
        if gtr and COMMAND in gtr.expose:
            commands[GetTeamRoster] = self._team_roster_prompt_factory(gtr)

        grp = _resolve(self.get_role_profiles, GetRoleProfiles)
        if grp and COMMAND in grp.expose:
            commands[GetRoleProfiles] = self._role_profiles_prompt_factory(grp)

        gta = _resolve(self.get_team_activity, GetTeamActivity)
        if gta and COMMAND in gta.expose:
            commands[GetTeamActivity] = self._who_is_working(gta)

        return commands

    def _hire_members_factory(self, params: HireTeamMember) -> Callable[..., Any]:
        """Create hire_members tool callable.

        Args:
            params: Configuration for hire capability

        Returns:
            Callable that hires team members
        """
        orchestrator_proxy = self._orchestrator_proxy
        observer_or_none = self._team_observer_or_none  # bound method -> weak edge to agent

        def hire_members(roles: list[str]) -> str:
            """Hire multiple new team members with the given roles.

            Creates new agent actors with specified roles. Names are auto-generated
            as @<Role><RandomNumber>. Validates roles against available roles.

            Note: Should only be used when explicitly requested by user to prevent
            unnecessary agent proliferation.

            Args:
                roles: List of roles to hire (each must be in available_roles)

            Returns:
                Confirmation message with hired member names
                (e.g., 'Members hired: [@Developer123, @Tester456]')
            """
            observer = observer_or_none()
            if observer is None:
                raise RetriableError("Team is shutting down; cannot hire.")
            if not roles:
                raise RetriableError("No roles provided. Specify at least one role to hire.")

            hired_members = []
            errors = []
            existing_names = {member.name for member in orchestrator_proxy.get_team()}
            agent_catalog = orchestrator_proxy.get_agent_catalog()

            for role in roles:
                try:
                    child_address = _hire_single_member(
                        orchestrator_proxy,
                        observer,
                        role,
                        None,
                        existing_names,
                        agent_catalog=agent_catalog,
                    )
                    existing_names.add(child_address.name)
                    hired_members.append(child_address.name)
                except RetriableError:
                    errors.append(role)

            if errors:
                available_roles = orchestrator_proxy.get_available_roles()
                error_details = "; ".join([f"role '{e}'" for e in errors])
                error_message = f"Hire errors - cannot find agent card(s) for {error_details}. "
                error_message += f"Available roles: {available_roles}"
                if hired_members:
                    error_message = (
                        f"Partial success - Members hired: {hired_members}. " + error_message
                    )
                raise RetriableError(error_message)

            return f"Members hired: {hired_members}"

        hire_members.__doc__ = params.format_docstring(hire_members.__doc__)
        return hire_members

    def _hire_member_command_factory(self, params: HireTeamMember) -> Callable[..., Any]:
        """Create hire_member command callable.

        Args:
            params: Configuration for hire capability

        Returns:
            Callable that hires a single team member
        """
        orchestrator_proxy = self._orchestrator_proxy
        observer_or_none = self._team_observer_or_none  # bound method -> weak edge to agent

        def hire_member(role: str, name: str | None = None) -> ActorAddress:
            """Hire a single new team member with the given role.

            Creates a new agent actor with the specified role. If no name is
            provided, one is auto-generated as @<Role><RandomNumber>.

            Args:
                role: Role to hire (must be in available_roles)
                name: Optional specific name for the member

            Returns:
                Address of the newly hired member.
            """
            observer = observer_or_none()
            if observer is None:
                raise RetriableError("Team is shutting down; cannot hire.")
            existing_names = {member.name for member in orchestrator_proxy.get_team()}
            return _hire_single_member(orchestrator_proxy, observer, role, name, existing_names)

        return hire_member

    def _fire_members_factory(self, params: FireTeamMember) -> Callable[..., Any]:
        """Create fire_members tool callable.

        Args:
            params: Configuration for fire capability

        Returns:
            Callable that fires team members
        """
        orchestrator_proxy = self._orchestrator_proxy
        observer_or_none = self._team_observer_or_none  # bound method -> weak edge to agent

        def fire_members(names: list[str]) -> str:
            """Fire multiple team members with the given names.

            Stops member actors and removes them from team. Member names typically
            start with '@' (e.g., '@Developer123').

            Note: Should only be used when explicitly requested by user to prevent
            accidental team disruption.

            Args:
                names: List of member names to fire (e.g., ['@Developer123', '@Tester456'])

            Returns:
                Combined confirmation messages (e.g., "Members fired: @Developer123, @Tester456")
            """
            observer = observer_or_none()
            if observer is None:
                raise RetriableError("Team is shutting down; cannot fire.")
            if not names:
                raise RetriableError("No names provided. Specify at least one member name to fire.")

            fired_members = []
            errors = []
            for name in names:
                try:
                    _fire_single_member(orchestrator_proxy, observer, name)
                    fired_members.append(name)
                except RetriableError:
                    errors.append(name)
                    logger.error(f"Fire error, team member not part of the team: {name}")

            if errors:
                team_members = [member.name for member in orchestrator_proxy.get_team()]
                error_details = "; ".join([f"member '{e}'" for e in errors])
                error_message = f"Fire errors - {error_details} not part of the team. "
                error_message += f"Current team members: {team_members}"
                if fired_members:
                    error_message = (
                        f"Partial success - Members fired: {fired_members}. " + error_message
                    )
                raise RetriableError(error_message)

            return f"Members fired: {', '.join(fired_members)}"

        fire_members.__doc__ = params.format_docstring(fire_members.__doc__)
        return fire_members

    def _fire_member_command_factory(self, params: FireTeamMember) -> Callable[..., Any]:
        """Create fire_member command callable.

        Args:
            params: Configuration for fire capability

        Returns:
            Callable that fires a single team member
        """
        orchestrator_proxy = self._orchestrator_proxy
        observer_or_none = self._team_observer_or_none  # bound method -> weak edge to agent

        def fire_member(name: str) -> str:
            """Fire a team member with the given name.

            Stops the member actor and removes them from the team.

            Args:
                name: Member name to fire (e.g., '@Developer123')

            Returns:
                Confirmation message (e.g., "Member @Developer123 has been fired.")
            """
            observer = observer_or_none()
            if observer is None:
                raise RetriableError("Team is shutting down; cannot fire.")
            _fire_single_member(orchestrator_proxy, observer, name)
            return f"Member {name} has been fired."

        return fire_member

    def _team_roster_prompt_factory(self, params: GetTeamRoster) -> Callable[..., Any]:
        """Create team roster system prompt callable.

        Args:
            params: Configuration for roster prompt

        Returns:
            Callable that generates team roster prompt
        """
        orchestrator_proxy = self._orchestrator_proxy
        observer_or_none = self._team_observer_or_none  # bound method -> weak edge to agent

        def team_members() -> str:
            """Get current team composition as context.

            Returns formatted list of team members with their roles, marking the
            current agent with '[you]'. Excludes tool actors (names starting with '#').

            Returns:
                Formatted team roster or empty string if no members
            """
            try:
                observer = observer_or_none()
                if observer is None:
                    return ""  # agent gone -> no roster context
                team_members = orchestrator_proxy.get_team()
                if not team_members:
                    return ""

                team_members_names = [
                    f"{member.name} (role: {member.role})"
                    + (" - [you]" if member.name == observer.myAddress.name else "")
                    for member in team_members
                    if not member.name.startswith("#")  # Exclude tool actors
                ]

                if not team_members_names:
                    return ""

                return "**Here is the team member list by name (and role):**\n" + "\n".join(
                    team_members_names
                )
            except Exception:
                logger.error("Failed to get team roster", exc_info=True)
                return "Cannot get team roster..."

        return team_members

    def _role_profiles_prompt_factory(self, params: GetRoleProfiles) -> Callable[..., Any]:
        """Create role profiles system prompt callable.

        Args:
            params: Configuration for profiles prompt

        Returns:
            Callable that generates role profiles prompt
        """
        orchestrator_proxy = self._orchestrator_proxy

        def team_roles() -> str:
            """Get available team roles and their descriptions.

            Returns formatted list of roles with descriptions and skills from the
            agent catalog.

            Returns:
                Formatted role profiles or empty string if no roles
            """
            try:
                agent_catalog = orchestrator_proxy.get_agent_catalog()
                if not agent_catalog:
                    return ""

                profiles = []
                for card in agent_catalog:
                    skills_str = ", ".join(card.skills) if card.skills else "none"
                    profiles.append(f"{card.role}: {card.description} (Skills: {skills_str})")

                if not profiles:
                    return ""

                return "**Here is the available team role list (for hiring):**\n" + "\n".join(
                    profiles
                )
            except Exception:
                logger.error("Failed to get role profiles", exc_info=True)
                return "Cannot get role profiles..."

        return team_roles

    def _who_is_working(self, params: GetTeamActivity) -> Callable[..., Any]:
        """Build the ``who_is_working`` variant this configuration calls for.

        The signature follows the configuration rather than being fixed: without a
        summarizer the callable takes no ``summarize_over`` parameter at all, so
        the model cannot request a summary that nothing could produce. Two factories
        rather than two branches inside one, because two inner ``def``s of the same
        name in one function body is a redefinition mypy rejects.
        """
        if params.summarizer is not None:
            return self._who_is_working_summarizing_factory(params)
        return self._who_is_working_factory(params)

    def _activity_collector(self, params: GetTeamActivity) -> Callable[[], ActivitySnapshot]:
        """Bind the telemetry derivation shared by both ``who_is_working`` variants.

        Args:
            params: Configuration for the team-activity capability.

        Returns:
            A zero-argument callable producing one snapshot per invocation.
        """
        orchestrator_proxy = self._orchestrator_proxy
        observer_or_none = self._team_observer_or_none  # bound method -> weak edge to agent
        stale_after_seconds = params.stale_after_seconds

        def collect() -> ActivitySnapshot:
            observer = observer_or_none()
            if observer is None:
                raise RetriableError("Team is shutting down; cannot report team activity.")
            return build_snapshot(
                orchestrator_proxy, observer.myAddress.agent_id, stale_after_seconds
            )

        return collect

    def _who_is_working_factory(self, params: GetTeamActivity) -> Callable[..., Any]:
        """Create the truncate-only ``who_is_working``: no cache proxy in scope.

        Args:
            params: Configuration for the team-activity capability.

        Returns:
            Callable reporting who is mid-handler, at no cost.
        """
        collect = self._activity_collector(params)
        max_task_chars = params.max_task_chars

        def who_is_working() -> TeamActivityReport:
            """Report which teammates are currently handling a message, and on what.

            Derived from the team's own telemetry, so it costs nothing at all. Task
            text longer than the report budget is truncated.

            Returns:
                A ``TeamActivityReport``. Members that are idle, are you, are tool
                actors, or are the user proxy never appear.
            """
            snapshot = collect()
            return TeamActivityReport(
                generated_at=snapshot.generated_at,
                members=apply_truncations(snapshot.rows, snapshot.texts, max_task_chars),
                pending_summaries=0,
            )

        who_is_working.__doc__ = params.format_docstring(who_is_working.__doc__)
        return who_is_working

    def _who_is_working_summarizing_factory(self, params: GetTeamActivity) -> Callable[..., Any]:
        """Create the summarizing ``who_is_working``: the threshold is the consent.

        Args:
            params: Configuration for the team-activity capability, with a
                summarizer set.

        Returns:
            Callable reporting who is mid-handler, summarizing on request.
        """
        activity_proxy = self._require_activity_proxy()
        collect = self._activity_collector(params)
        budget = SummaryBudget.from_params(params)

        def who_is_working(summarize_over: int | None = None) -> TeamActivityReport:
            """Report which teammates are currently handling a message, and on what.

            Derived from team telemetry, so it costs no model call by default: task
            text longer than the report budget is simply truncated.

            Args:
                summarize_over: Omit (the default) for zero-cost truncation. Pass a
                    character count to summarize only tasks longer than it; the
                    summary is cached per task, so asking again is free. A summary
                    that has not arrived in time comes back truncated and is counted
                    in ``pending_summaries``.

            Returns:
                A ``TeamActivityReport``. Members that are idle, are you, are tool
                actors, or are the user proxy never appear.
            """
            snapshot = collect()
            members, pending_summaries = apply_summaries(
                snapshot.rows, snapshot.texts, activity_proxy, budget, summarize_over
            )
            return TeamActivityReport(
                generated_at=snapshot.generated_at,
                members=members,
                pending_summaries=pending_summaries,
            )

        who_is_working.__doc__ = params.format_docstring(who_is_working.__doc__)
        return who_is_working

    def _require_activity_proxy(self) -> TeamActivityActor:
        """Return the bound cache proxy, or fail loudly if :meth:`observer` never ran."""
        if self._activity_proxy is None:
            raise ValueError(
                "TeamTool.observer() must run before the summarizing who_is_working is built."
            )
        return self._activity_proxy
