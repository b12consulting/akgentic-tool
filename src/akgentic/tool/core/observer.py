"""Observer protocols every tool may be handed, regardless of domain.

``ToolObserver`` and ``ActorToolObserver`` are package-global contracts: ``card.py``
and ``factory.py`` both name them, and tools across every domain accept them. They
live in ``core/`` for that reason alone — audience, not size.

A domain-specific observer belongs to its domain instead; ``TeamManagementToolObserver``
sits in ``akgentic.tool.team.observer`` because exactly one tool needs it.
"""

from __future__ import annotations

import uuid
from typing import Protocol, runtime_checkable

from akgentic.core.actor_address import ActorAddress
from akgentic.core.agent import AkgentType


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

    @property
    def team_id(self) -> uuid.UUID:
        """Get the team id."""
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

    def proxy_tell(
        self,
        actor: ActorAddress,
        actor_type: type[AkgentType] | None = None,
    ) -> AkgentType:
        """Get a fire-and-forget proxy to another actor.

        A tool that sends an actor something it needs no answer to must send it
        this way. An ask has no default timeout, so a merely *slow* target stalls
        the caller indefinitely — and a fail-open ``except`` around the call
        catches a raising target and a dead one, never a hung one. A tell cannot
        stall the sender at all, which makes "this call never blocks the caller"
        a property of the mechanism rather than of a tuned timeout.

        Every real observer is an ``Akgent`` and already has this; the protocol
        is widened here so a tool can name it.

        Args:
            actor: Address of the target actor
            actor_type: Optional expected type of the target actor for better type checking

        Returns:
            Proxy object whose calls return immediately, discarding any result
        """
        ...
