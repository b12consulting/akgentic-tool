from abc import ABC
from typing import Any, Callable

from pydantic import BaseModel

from akgentic.tool.utils import import_class


class ToolCard(BaseModel):
    name: str
    module: str | type
    description: str
    params: list[Any] | None = None


class BaseTool(ABC):
    def __init__(self, tool_card: ToolCard) -> None:
        self.tool_card = tool_card

    def get_tools(self) -> list[Callable]:
        """Return a list of callables that implement the tools defined in the tool card."""
        raise NotImplementedError("Must be implemented by subclasses")


class ToolFactory:
    def __init__(self, tool_cards: list[ToolCard]) -> None:
        self.tool_cards = tool_cards

    def get_tools(self, tool_card: ToolCard | None = None) -> list[Callable]:
        if tool_card is not None:
            tool_class = (
                import_class(tool_card.module)
                if isinstance(tool_card.module, str)
                else tool_card.module
            )
            tool: BaseTool = tool_class(tool_card)
            return tool.get_tools()

        tools = []
        for card in self.tool_cards:
            tools.extend(self.get_tools(tool_card=card))
        return tools
