"""Observer protocol for the team-management tool.

``TeamManagementToolObserver`` is one tool's contract, not the package's: ``TeamTool``
is its only consumer. It lives beside that tool rather than in ``core/`` so the global
surface stays limited to what more than one domain actually needs.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from akgentic.core.actor_address import ActorAddress
from akgentic.core.agent import AkgentType
from akgentic.tool.core.observer import ActorToolObserver


@runtime_checkable
class TeamManagementToolObserver(ActorToolObserver, Protocol):
    """Observer protocol for team management tools.

    Extends ActorToolObserver with team-specific capabilities needed by
    TeamTool for hiring, firing, and managing team members within the
    actor system.
    """

    def createActor(  # noqa: N802
        self,
        actor_class: type[AkgentType],
        *,
        config: object,
    ) -> ActorAddress:
        """Create a child actor with the given config.

        Args:
            actor_class: The actor class to instantiate
            config: Configuration object for the actor

        Returns:
            Address of the newly created actor
        """
        ...

    def on_hire(self, address: ActorAddress) -> None:
        """Hook called after hiring a team member.

        Handles agent-specific concerns such as:
        - Tracking child in agent's children list
        - Updating local caches
        - Any agent-specific bookkeeping

        Args:
            address: ActorAddress of hired agent
        """
        ...

    def on_fire(self, address: ActorAddress) -> None:
        """Hook called after firing a team member.

        Handles agent-specific concerns such as:
        - Removing from children tracking
        - Clearing from local caches
        - Any agent-specific cleanup

        Args:
            address: ActorAddress of fired agent
        """
        ...
