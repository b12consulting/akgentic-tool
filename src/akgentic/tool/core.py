"""Tool abstractions and factory for the akgentic tool package.

Defines the core contracts:
- ``BaseToolParam``: base for capability parameter models.
- ``ToolCard``: abstract base — tool configuration + callable factory in one class.
- ``ToolFactory``: resolves ``ToolCard`` instances into callable tools, prompts, and toolsets.
"""

import functools
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Callable, TypeVar

from akgentic.core.utils import SerializableBaseModel
from akgentic.tool.errors import RetriableError
from akgentic.tool.event import ToolObserver

T = TypeVar("T", bound="BaseToolParam")


def _resolve(value: "T | bool", cls: "type[T]") -> "T | None":
    """Resolve a ``ParamModel | bool`` field to a ``ParamModel`` or ``None``.

    Args:
        value: ``True`` (enable with defaults), ``False`` (disable), or a
            ``BaseToolParam`` instance (enable with custom parameters).
        cls: The param model class to instantiate when *value* is ``True``.

    Returns:
        A param model instance, or ``None`` if the capability is disabled.
    """
    if value is True:
        return cls()
    if value is False:
        return None
    return value  # already a ParamModel instance


class Channels(StrEnum):
    """Valid channel names for capability exposure."""

    SYSTEM_PROMPT = "system_prompt"
    """Expose as a system prompt injected into the LLM context."""

    TOOL_CALL = "tool_call"
    """Expose as a callable tool for the LLM."""

    COMMAND = "command"
    """Expose as a programmatic command for inter-agent orchestration."""


# Backward-compatible module-level aliases
SYSTEM_PROMPT = Channels.SYSTEM_PROMPT
TOOL_CALL = Channels.TOOL_CALL
COMMAND = Channels.COMMAND


class BaseToolParam(SerializableBaseModel):
    """Base for capability parameter models.

    Provides common fields that control how a capability is exposed
    and how its description can be customized.

    Each subclass can override the default ``expose`` set to declare the channels
    it participates in. Use the module-level channel constants:

    - ``TOOL_CALL``: callable tool invoked by the LLM (default).
    - ``SYSTEM_PROMPT``: prompt injected into the LLM context.
    - ``COMMAND``: programmatic call for inter-agent orchestration.
    """

    instructions: str | None = None
    """Additional instructions appended to the default tool docstring.

    When set, the factory appends these instructions to the built-in docstring
    under a structured header. When ``None``, only the default docstring is used.
    """

    expose: set[Channels] = {TOOL_CALL}
    """Set of channels this capability is exposed through.

    Defaults to ``{TOOL_CALL}``. Override in subclasses or at instantiation.
    Use ``Channels`` enum members or module-level aliases: ``TOOL_CALL``, ``SYSTEM_PROMPT``,
    ``COMMAND``.
    """

    def format_docstring(self, original: str | None) -> str | None:
        """Format the tool docstring with optional additional instructions.

        Args:
            original: The original docstring from the tool callable.

        Returns:
            The formatted docstring, or the original if no instructions are set.
        """
        if not self.instructions:
            return original

        base_doc = original or ""
        return f"{base_doc}\n\nAdditional Instructions:\n{self.instructions}"


class ToolCard(SerializableBaseModel, ABC):
    """Abstract base: tool configuration + callable factory in one class.

    Subclasses define typed fields for their capabilities and implement
    the factory methods that produce LLM-callable functions.

    Attributes:
        name: Human-readable tool provider name.
        description: Natural-language description of tool capabilities.
    """

    name: str
    description: str

    def observer(self, observer: ToolObserver) -> "ToolCard":
        """Attach an observer and perform runtime setup.

        Follows the same pattern as ``BaseState.observer()``.
        Override for setup that requires the observer (e.g., actor proxies).
        All methods can then access the observer via ``self._observer``.

        Args:
            observer: Optional observer for tool call events.

        Returns:
            Self, enabling method chaining.
        """
        self._observer = observer
        return self

    @abstractmethod
    def get_tools(self) -> list[Callable]:
        """Return callable tool functions for LLM agents.

        Use ``self._observer`` when tool callables need to emit events.
        """
        ...

    def get_system_prompts(self) -> list[Callable]:
        """Return system prompt callables injected into LLM context.

        Use ``self._observer`` when prompts need runtime data.
        """
        return []

    def get_commands(self) -> dict[type["BaseToolParam"], Callable]:
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
        """Return runtime toolset objects (e.g., MCP servers)."""
        return []


class ToolFactory:
    """Resolves ``ToolCard`` instances into callable tools, prompts, and toolsets."""

    def __init__(
        self,
        tool_cards: list[ToolCard],
        observer: ToolObserver | None = None,
        retry_exception: type[Exception] | None = None,
    ) -> None:
        """Create a factory for one or more tool cards.

        Attaches the observer to every card (triggers runtime setup in
        ``ToolCard.observer()``).

        Args:
            tool_cards: Tool cards to resolve into callable tools.
            observer: Optional observer notified by tool implementations during
                tool calls.
            retry_exception: Optional exception class to raise when a tool raises
                ``RetriableError``. Injected by the integration layer (e.g., ModelRetry
                from pydantic-ai) to keep the tool module framework-agnostic.
        """
        self.tool_cards = tool_cards
        self.observer = observer
        self._retry_exception = retry_exception

        if self.observer is not None:
            for card in self.tool_cards:
                card.observer(self.observer)

    def _wrap_with_retry(self, fn: Callable) -> Callable:
        """Wrap a tool callable to convert ``RetriableError`` into retry_exception."""
        assert self._retry_exception is not None
        retry_exc = self._retry_exception

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except RetriableError as e:
                raise retry_exc(str(e)) from e

        return wrapper

    def get_tools(self) -> list[Callable]:
        """Return tool callables aggregated from all tool cards."""
        tools = [t for card in self.tool_cards for t in card.get_tools()]
        if self._retry_exception is not None:
            tools = [self._wrap_with_retry(t) for t in tools]
        return tools

    def get_system_prompts(self) -> list[Callable]:
        """Return system prompt callables aggregated from all tool cards."""
        return [p for card in self.tool_cards for p in card.get_system_prompts()]

    def get_commands(self) -> dict[type[BaseToolParam], Callable]:
        """Return command callables aggregated from all tool cards.

        Returns:
            Dict mapping param class to callable, merged from all tool cards.
        """
        commands: dict[type[BaseToolParam], Callable] = {}
        for card in self.tool_cards:
            commands.update(card.get_commands())

        if self._retry_exception is not None:
            commands = {k: self._wrap_with_retry(v) for k, v in commands.items()}
        return commands

    def get_toolsets(self) -> list[Any]:
        """Return toolset instances aggregated from all tool cards."""
        return [ts for card in self.tool_cards for ts in card.get_toolsets()]
