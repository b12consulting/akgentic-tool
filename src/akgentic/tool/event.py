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
    """Observer protocol for tool interactions."""

    def notify_event(self, event: object) -> None:
        """Called when a tool domain event is emitted.

        Args:
            event: Domain event object
        """
        ...

    @property
    def myAddress(self) -> ActorAddress:
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
