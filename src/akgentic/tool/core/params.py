"""Base capability parameter model and the ``ParamModel | bool`` resolver."""

from typing import TypeVar

from akgentic.core.utils import SerializableBaseModel

from .channels import TOOL_CALL, Channels

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
