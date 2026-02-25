from dataclasses import dataclass


@dataclass(frozen=True)
class PlanningEvent:
    """Event emitted when the llm make a tool call."""

    tool_name: str
    args: list
    kwargs: dict
