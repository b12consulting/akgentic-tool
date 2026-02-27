from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from akgentic.core.actor_address import ActorAddress
from akgentic.core.agent import AkgentType


@dataclass(frozen=True)
class ToolCallEvent:
    """Event emitted when the llm make a tool call."""

    tool_name: str
    args: list
    kwargs: dict


@runtime_checkable
class ToolObserver(Protocol):
    """Basic observer protocol for tool interactions.

    This protocol defines the minimal interface required for tools that only
    need to emit events. Tools requiring actor-aware features should use
    ActorToolObserver instead.
    """

    def notify_event(self, event: object) -> None:
        """Called when a tool domain event is emitted.

        Args:
            event: Domain event object
        """
        ...


@runtime_checkable
class ActorToolObserver(ToolObserver, Protocol):
    """Actor-aware observer protocol for tool interactions.

    Extends ToolObserver with actor-specific capabilities needed by tools
    that interact with the actor system (e.g., PlanningTool).
    """

    @property
    def myAddress(self) -> ActorAddress:  # noqa: N802
        """Get the current actor's address."""
        ...

    @property
    def orchestrator(self) -> ActorAddress | None:
        """Get the orchestrator address."""
        ...

    def proxy_ask(
        self,
        actor: ActorAddress,
        actor_type: type[AkgentType] | None = None,
        timeout: int | None = None,
    ) -> AkgentType:
        """Get a proxy to another actor.

        Args:
            actor: Address of the target actor
            actor_type: Optional expected type of the target actor for better type checking
            timeout: Optional timeout for the proxy ask

        Returns:
            Proxy object to interact with the target actor
        """
        ...


@runtime_checkable
class TeamManagementToolObserver(ActorToolObserver, Protocol):
    """Observer protocol for team management tools.

    Extends ActorToolObserver with team-specific capabilities needed by
    TeamTool for hiring, firing, and managing team members within the
    actor system.
    """

    def createActor(
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

    def send(self, target: ActorAddress, message: object) -> None:
        """Send a message to another actor.

        Args:
            target: Address of the target actor
            message: Message to send
        """
        ...

    def stop(self, target: ActorAddress) -> None:
        """Stop an actor and its children recursively.

        Args:
            target: Address of the actor to stop
        """
        ...

    def on_hire(self, name: str, address: ActorAddress) -> None:
        """Hook called after hiring a team member.

        Handles agent-specific concerns such as:
        - Tracking child in agent's children list
        - Updating local caches
        - Any agent-specific bookkeeping

        Args:
            name: Name of hired agent (e.g., '@Developer123')
            address: ActorAddress of hired agent
        """
        ...

    def on_fire(self, name: str, address: ActorAddress) -> None:
        """Hook called after firing a team member.

        Handles agent-specific concerns such as:
        - Removing from children tracking
        - Clearing from local caches
        - Any agent-specific cleanup

        Args:
            name: Name of fired agent (e.g., '@Developer123')
            address: ActorAddress of fired agent
        """
        ...
