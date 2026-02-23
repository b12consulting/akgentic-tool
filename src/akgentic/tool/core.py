"""Tool abstractions and factory for dynamic tool instantiation.

Defines the core contracts used by the tool package:
- ``ToolCard``: metadata/configuration for a tool provider.
- ``BaseTool``: abstract interface implemented by concrete tool modules.
- ``ToolFactory``: resolver that loads tool classes and materializes callable tools.
"""

from abc import ABC
from typing import Any, Callable

from pydantic import BaseModel

from akgentic.tool.event import ToolObserver
from akgentic.tool.utils import import_class


class ToolCard(BaseModel):
    """Metadata and configuration for a tool provider.

    A ``ToolCard`` describes how a tool implementation should be resolved and
    what parameter presets should be passed to it.

    Attributes:
        name: Human-readable tool provider name.
        module: Fully qualified class path (``str``) or class type implementing
            ``BaseTool``.
        description: Natural-language description of tool capabilities.
        params: Optional list of parameter model instances used by the concrete
            tool implementation to build callable tools.
    """

    name: str
    module: str | type
    description: str
    params: list[Any] | None = None


class BaseTool(ABC):
    """Abstract base class for tool providers.

    Concrete subclasses wrap one logical tool module and expose one or more
    executable callables via ``get_tools``.
    """

    def __init__(self, tool_card: ToolCard) -> None:
        """Initialize the tool provider.

        Args:
            tool_card: Tool metadata and parameter presets for this provider.
        """
        self.tool_card = tool_card

    def get_tools(self, observer: ToolObserver | None) -> list[Callable]:
        """Return a list of callables that implement the tools defined in the tool card."""
        raise NotImplementedError("Must be implemented by subclasses")


class ToolFactory:
    """Factory responsible for resolving and instantiating tool callables.

    Supports two usage patterns:
    - Build callables for a specific ``ToolCard``.
    - Build callables for all cards provided at factory construction.
    """

    def __init__(
        self,
        tool_cards: list[ToolCard],
        observer: ToolObserver | None = None,
    ) -> None:
        """Create a factory for one or more tool cards.

        Args:
            tool_cards: Tool cards to resolve into callable tools.
            observer: Optional observer notified by tool implementations during
                tool calls.
        """

        self.tool_cards = tool_cards
        self.observer = observer

    def get_tools(self, tool_card: ToolCard | None = None) -> list[Callable]:
        """Resolve and return tool callables.

        Args:
            tool_card: Optional single tool card to resolve. When ``None``, all
                cards from ``self.tool_cards`` are resolved and flattened.

        Returns:
            List of executable callables produced by the resolved tool classes.
        """
        if tool_card is not None:
            tool_class = (
                import_class(tool_card.module)
                if isinstance(tool_card.module, str)
                else tool_card.module
            )
            tool: BaseTool = tool_class(tool_card)
            return tool.get_tools(self.observer)

        tools = []
        for card in self.tool_cards:
            tools.extend(self.get_tools(tool_card=card))
        return tools
