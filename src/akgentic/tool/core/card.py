"""The ``ToolCard`` abstract base: tool configuration + callable factory in one class."""

import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from pydantic import PrivateAttr

from akgentic.core.utils import SerializableBaseModel
from akgentic.tool.errors import ToolObserverGone
from akgentic.tool.event import ToolObserver

from .params import BaseToolParam


class ToolCard(SerializableBaseModel, ABC):
    """Abstract base: tool configuration + callable factory in one class.

    Subclasses define typed fields for their capabilities and implement
    the factory methods that produce LLM-callable functions. Identity and
    human-readable description live on the catalog ``Entry`` envelope, not
    on the card payload.
    """

    # Runtime-only, WEAK: a tool/closure/registry must never pin its agent.
    _observer_ref: "weakref.ref[ToolObserver] | None" = PrivateAttr(default=None)

    @property
    def depends_on(self) -> list[str]:
        """Class-name list of ToolCards that MUST be wired before this one.

        Default: no dependencies. Subclasses may override as a property
        whose return value depends on instance fields (e.g. the value of
        a ``vector_store`` field on consumer tools). The string is matched
        against ``type(card).__name__`` by ``ToolFactory``'s topological
        sort. Not a Pydantic field — does not appear in ``model_dump`` and
        cannot be set via ``model_validate``.
        """
        return []

    def observer(self, observer: ToolObserver) -> "ToolCard":
        """Attach an observer (held weakly) and perform runtime setup.

        Follows the same pattern as ``BaseState.observer()``.
        Override for setup that requires the observer (e.g., actor proxies).
        All methods can then access the observer via ``self._observer``.

        The observer is stored through a ``weakref`` so a tool, its closures, and
        its command registry can never pin a stopped owning agent in memory.

        Subclasses requiring a richer observer (``ActorToolObserver``,
        ``TeamManagementToolObserver``) keep this parameter type and narrow the
        stored observer in their own accessor — ``ToolFactory`` attaches one
        observer to every card uniformly, so narrowing the parameter here would
        violate the Liskov substitution principle.

        Args:
            observer: Optional observer for tool call events.

        Returns:
            Self, enabling method chaining.
        """
        self._observer = observer
        return self

    @property
    def _observer(self) -> "ToolObserver":
        """Live observer for synchronous, in-life use. Raises if the agent has stopped."""
        obs = self._observer_or_none()
        if obs is None:
            raise ToolObserverGone("tool used after its owning agent was stopped")
        return obs

    @_observer.setter
    def _observer(self, observer: ToolObserver) -> None:
        """Store the observer weakly (backward-compatible ``self._observer = observer``)."""
        self._observer_ref = weakref.ref(observer)

    def _observer_or_none(self) -> "ToolObserver | None":
        """Return the live observer, or ``None`` if unset or already collected."""
        return self._observer_ref() if self._observer_ref is not None else None


    @abstractmethod
    def get_tools(self) -> list[Callable[..., Any]]:
        """Return callable tool functions for LLM agents.

        Use ``self._observer`` when tool callables need to emit events.
        """
        ...

    def get_system_prompts(self) -> list[Callable[..., Any]]:
        """Return system prompt callables injected into LLM context.

        Use ``self._observer`` when prompts need runtime data.
        """
        return []

    def get_commands(self) -> dict[type["BaseToolParam"], Callable[..., Any]]:
        """Return callable commands for programmatic invocation.

        Commands are methods exposed for inter-agent orchestration
        (e.g., ``hire_member``, ``fire_member``). Unlike tools (invoked by
        the LLM), commands are called programmatically by other agents
        or system components via ``proxy_call`` or similar mechanisms.

        Returns:
            Dict mapping param class (e.g., ``HireTeamMember``) to callable.
        """
        return {}

    def get_toolsets(self) -> list[Any]:
        """Return runtime pydantic-ai toolset objects (e.g., an ``MCPToolset``)."""
        return []
