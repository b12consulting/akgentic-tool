"""Team management tool implementation.

Provides hire, fire, and team awareness capabilities as a reusable ToolCard
that can be attached to any agent.
"""

import logging
import random
from typing import Callable

from pydantic import Field

from akgentic.core.orchestrator import Orchestrator
from akgentic.tool.core import BaseToolParam, ToolCard, _resolve
from akgentic.tool.errors import RetriableError
from akgentic.tool.event import TeamManagementToolObserver, ToolCallEvent

logger = logging.getLogger(__name__)


class HireTeamMembers(BaseToolParam):
    """Hire new team members by role."""

    pass


class FireTeamMembers(BaseToolParam):
    """Fire existing team members by name."""

    pass


class GetTeamRoster(BaseToolParam):
    """Get current team roster as system prompt."""

    system_prompt: bool = True


class GetRoleProfiles(BaseToolParam):
    """Get available role profiles as system prompt."""

    system_prompt: bool = True


class TeamTool(ToolCard):
    """Team management tool for hiring, firing, and team awareness.

    Provides:
    - hire_members(roles: list[str]) -> str: Hire team members
    - fire_members(names: list[str]) -> str: Fire team members
    - Team roster system prompt: Current team composition
    - Role profiles system prompt: Available roles and descriptions
    """

    name: str = "Team"
    description: str = "Team management tool for hiring, firing, and team awareness"

    hire_team_members: HireTeamMembers | bool = Field(
        default=True, description="Enable hiring team members (default: True)"
    )
    fire_team_members: FireTeamMembers | bool = Field(
        default=True, description="Enable firing team members (default: True)"
    )
    get_role_profiles: GetRoleProfiles | bool = Field(
        default=True, description="Include role profiles in system prompt (default: True)"
    )
    get_team_roster: GetTeamRoster | bool = Field(
        default=True, description="Include team roster in system prompt (default: True)"
    )

    def observer(self, observer: TeamManagementToolObserver) -> "TeamTool":
        """Attach observer and set up the orchestrator proxy.

        Requires a TeamManagementToolObserver for actor system access.

        Args:
            observer: Observer implementing TeamManagementToolObserver protocol

        Returns:
            Self, enabling method chaining

        Raises:
            ValueError: If observer.orchestrator is None
        """
        self._observer = observer
        if observer.orchestrator is None:
            raise ValueError("TeamTool requires access to the orchestrator.")

        self._orchestrator_proxy = observer.proxy_ask(observer.orchestrator, Orchestrator)
        return self

    def get_system_prompts(self) -> list[Callable]:
        """Get dynamic system prompts for team context.

        Returns:
            List of callable system prompts (roster and/or profiles)
        """
        prompts: list[Callable] = []

        gtr = _resolve(self.get_team_roster, GetTeamRoster)
        if gtr and gtr.system_prompt:
            prompts.append(self._team_roster_prompt_factory(gtr))

        grp = _resolve(self.get_role_profiles, GetRoleProfiles)
        if grp and grp.system_prompt:
            prompts.append(self._role_profiles_prompt_factory(grp))

        return prompts

    def get_tools(self) -> list[Callable]:
        """Get LLM-callable tools for team management.

        Returns:
            List of callable tools (hire_members and/or fire_members)
        """
        tools: list[Callable] = []

        htm = _resolve(self.hire_team_members, HireTeamMembers)
        if htm and htm.llm_tool:
            tools.append(self._hire_members_factory(htm))

        ftm = _resolve(self.fire_team_members, FireTeamMembers)
        if ftm and ftm.llm_tool:
            tools.append(self._fire_members_factory(ftm))

        return tools

    def _hire_members_factory(self, params: HireTeamMembers) -> Callable:
        """Create hire_members tool callable.

        Args:
            params: Configuration for hire capability

        Returns:
            Callable that hires team members
        """
        orchestrator_proxy = self._orchestrator_proxy
        observer = self._observer

        def hire_members(roles: list[str]) -> str:
            """Hire multiple new team members with the given roles.

            Creates new agent actors with specified roles. Names are auto-generated
            as @<Role><RandomNumber>. Validates roles against available roles.

            Note: Should only be used when explicitly requested by user to prevent
            unnecessary agent proliferation.

            Args:
                roles: List of roles to hire (each must be in available_roles)

            Returns:
                Confirmation message with hired member names (e.g., 'Members hired: [@Developer123, @Tester456]')
            """
            if not roles:
                raise RetriableError("No roles provided. Specify at least one role to hire.")

            observer.notify_event(
                ToolCallEvent(tool_name="hire_members", args=[], kwargs={"roles": roles})
            )

            hired_members = []
            errors = []
            agent_catalog = orchestrator_proxy.get_agent_catalog()
            existing_names = {member.name for member in orchestrator_proxy.get_team()}

            for role in roles:
                # Get agent card for role from catalog
                agent_card = next((card for card in agent_catalog if card.role == role), None)
                if agent_card is None:
                    errors.append(role)
                    continue

                # Create child agent using agent primitive
                # agent_class can be str | type, but in production it should be a type
                actor_class = agent_card.get_agent_class()
                if isinstance(actor_class, str):
                    ## The llm cannot retry, so raise a ValueError
                    raise ValueError(
                        f"Hire error - agent class for role {role} is a string, not a type. "
                    )

                # Generate unique name: random, increment on collision
                role_prefix = f"@{role.replace(' ', '')}"
                suffix = random.randint(100, 999)
                child_name = f"{role_prefix}{suffix}"
                while child_name in existing_names:
                    suffix += 1
                    child_name = f"{role_prefix}{suffix}"

                # Create config
                agent_card_config = agent_card.get_config_copy()
                agent_card_config.name = child_name
                agent_card_config.role = role

                child_address = observer.createActor(actor_class, config=agent_card_config)

                # Call agent hook
                observer.on_hire(child_name, child_address)

                existing_names.add(child_name)
                hired_members.append(child_name)
                logger.info(f"Hired {role} agent: {child_name} at {child_address}")

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

    def _fire_members_factory(self, params: FireTeamMembers) -> Callable:
        """Create fire_members tool callable.

        Args:
            params: Configuration for fire capability

        Returns:
            Callable that fires team members
        """
        orchestrator_proxy = self._orchestrator_proxy
        observer = self._observer

        def fire_members(names: list[str]) -> str:
            """Fire multiple team members with the given names.

            Stops member actors and removes them from team. Member names typically
            start with '@' (e.g., '@Developer123').

            Note: Should only be used when explicitly requested by user to prevent
            accidental team disruption.

            Args:
                names: List of member names to fire (e.g., ['@Developer123', '@Tester456'])

            Returns:
                Combined confirmation messages (e.g., "Member @Developer123 has been fired. - ...")
            """
            if not names:
                raise RetriableError("No names provided. Specify at least one member name to fire.")

            observer.notify_event(
                ToolCallEvent(tool_name="fire_members", args=[], kwargs={"names": names})
            )

            fired_members = []
            errors = []
            for name in names:
                # Lookup member
                address = orchestrator_proxy.get_team_member(name)
                if address is None:
                    errors.append(name)
                    logger.error(f"Fire error, team member not part of the team: {name}")
                    continue

                # Stop actor using agent primitive
                address.stop()

                # Call agent hook
                observer.on_fire(name, address)

                fired_members.append(name)
                logger.info(f"Fired team member: {name}")

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

    def _team_roster_prompt_factory(self, params: GetTeamRoster) -> Callable:
        """Create team roster system prompt callable.

        Args:
            params: Configuration for roster prompt

        Returns:
            Callable that generates team roster prompt
        """
        orchestrator_proxy = self._orchestrator_proxy
        observer = self._observer

        def team_roster_prompt() -> str:
            """Get current team composition as context.

            Returns formatted list of team members with their roles, marking the
            current agent with '[you]'. Excludes tool actors (names starting with '#').

            Returns:
                Formatted team roster or empty string if no members
            """
            try:
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

                return "Team members:\n" + "\n".join(team_members_names)
            except Exception:
                logger.error("Failed to get team roster", exc_info=True)
                return "Cannot get team roster..."

        return team_roster_prompt

    def _role_profiles_prompt_factory(self, params: GetRoleProfiles) -> Callable:
        """Create role profiles system prompt callable.

        Args:
            params: Configuration for profiles prompt

        Returns:
            Callable that generates role profiles prompt
        """
        orchestrator_proxy = self._orchestrator_proxy

        def role_profiles_prompt() -> str:
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
                    profiles.append(f"**{card.role}**: {card.description} (Skills: {skills_str})")

                if not profiles:
                    return ""

                return "Available team roles:\n" + "\n".join(profiles)
            except Exception:
                logger.error("Failed to get role profiles", exc_info=True)
                return "Cannot get role profiles..."

        return role_profiles_prompt
