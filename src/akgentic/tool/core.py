"""Tool abstractions and factory for the akgentic tool package.

Defines the core contracts:
- ``BaseToolParam``: base for capability parameter models.
- ``ToolCard``: abstract base — tool configuration + callable factory in one class.
- ``ToolFactory``: resolves ``ToolCard`` instances into callable tools, prompts, and toolsets.
"""

import functools
from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

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


class BaseToolParam(BaseModel):
    """Base for capability parameter models.

    Provides common fields that control how a capability is exposed
    to the LLM and how its description can be customized.
    """

    instructions: str | None = None
    """Additional instructions appended to the default tool docstring.

    When set, the factory appends these instructions to the built-in docstring
    under a structured header. When ``None``, only the default docstring is used.
    """

    system_prompt: bool = False
    """Whether this capability injects a system prompt into the LLM context."""

    llm_tool: bool = True
    """Whether this capability is exposed as a callable tool for the LLM."""

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


class ToolCard(BaseModel, ABC):
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
                ``ToolError``. Injected by the integration layer (e.g., ModelRetry
                from pydantic-ai) to keep the tool module framework-agnostic.
        """
        self.tool_cards = tool_cards
        self.observer = observer
        self._retry_exception = retry_exception

        if self.observer is not None:
            for card in self.tool_cards:
                card.observer(self.observer)

    def _wrap_with_retry(self, fn: Callable) -> Callable:
        """Wrap a tool callable to convert ``ToolError`` into retry_exception."""
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

    def get_toolsets(self) -> list[Any]:
        """Return toolset instances aggregated from all tool cards."""
        return [ts for card in self.tool_cards for ts in card.get_toolsets()]
