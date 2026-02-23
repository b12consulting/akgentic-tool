from dataclasses import dataclass
from typing import Protocol, runtime_checkable


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
