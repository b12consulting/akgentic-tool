"""Catalog envelope for application-owned ``ToolCard`` subclasses.

``Entry.model_type`` is allowlisted to ``akgentic.*``, and a concrete tool
class *is* the ``model_type`` — so a ``ToolCard`` defined outside ``akgentic.*``
cannot be named directly in a v2 catalog entry. ``ToolCardSpec`` is the tool
equivalent of ``AgentCard.agent_class``: a framework-owned (hence allowlisted)
envelope that carries the application's concrete tool class as payload *data*.

Authoring becomes symmetric with agents::

    kind: tool
    model_type: akgentic.tool.ToolCardSpec   # framework-owned, allowlisted
    payload:
      tool_class: myapp.tools.MyToolCard     # application class, as DATA
      config:
        included_tools: [get_ticket, list_tickets]

``ToolCardSpec`` is itself a ``ToolCard`` that transparently delegates every
capability method to the built concrete card, so a resolved ``__ref__`` in a
``list[ToolCard]`` position is usable without callers special-casing envelope
vs. native entries.
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import Field, PrivateAttr

from akgentic.core.utils import import_class
from akgentic.tool.core import BaseToolParam, ToolCard
from akgentic.tool.event import ToolObserver


def _resolve_tool_class(value: str | type) -> type[ToolCard]:
    """Resolve ``tool_class`` (str FQCN or type) to a concrete ``ToolCard`` subclass.

    Mirrors ``AgentCard``'s ``_resolve_agent_class``: accepts a class object
    directly or a fully qualified dotted path, and validates the result is a
    ``ToolCard`` subclass so a misconfigured envelope fails loudly.

    Raises:
        ValueError: If *value* is an empty/unqualified string or does not
            resolve to a ``ToolCard`` subclass.
        ImportError / AttributeError: If the dotted path cannot be imported.
    """
    if isinstance(value, type):
        cls: type = value
    else:
        if not value or "." not in value:
            raise ValueError(
                f"tool_class must be a fully qualified dotted path "
                f"(e.g. 'mypackage.tools.MyToolCard'), got: {value!r}"
            )
        cls = import_class(value)

    if not (isinstance(cls, type) and issubclass(cls, ToolCard)):
        raise ValueError(f"tool_class {cls!r} is not a ToolCard subclass")
    return cls


class ToolCardSpec(ToolCard):
    """Catalog envelope for an application-owned ``ToolCard``.

    Mirrors ``AgentCard.agent_class``: the concrete class is payload data, so
    the allowlisted ``model_type`` stays inside ``akgentic.*``. The spec is
    itself a ``ToolCard`` that delegates every capability method to the built
    concrete card, so ``__ref__`` resolution yields a usable instance in
    ``list[ToolCard]`` positions and callers never special-case envelope vs.
    native entries.
    """

    tool_class: str | type
    config: dict[str, Any] = Field(default_factory=dict)

    # Runtime-only: the concrete card built from ``tool_class`` + ``config``.
    _built: ToolCard | None = PrivateAttr(default=None)

    def get_tool_class(self) -> type[ToolCard]:
        """Resolve ``tool_class`` to the concrete ``ToolCard`` subclass."""
        return _resolve_tool_class(self.tool_class)

    def build(self) -> ToolCard:
        """Build (once) and return the concrete ``ToolCard`` from ``config``.

        The built card is cached so a single instance receives the observer and
        serves tools/prompts/commands across the delegating methods below.
        """
        if self._built is None:
            self._built = self.get_tool_class().model_validate(self.config)
        return self._built

    @property
    def depends_on(self) -> list[str]:
        """Delegate dependency declaration to the built concrete card."""
        return self.build().depends_on

    @property
    def dependency_name(self) -> str:
        """Key this envelope by the wrapped concrete tool's class name.

        Without this override every ``ToolCardSpec`` would collapse into a
        single ``"ToolCardSpec"`` node in the dependency graph, so wrapped tools
        could not be ordered relative to one another and ``depends_on`` values
        (which name the concrete class) would never resolve.
        """
        return self.get_tool_class().__name__

    def observer(self, observer: ToolObserver) -> ToolCard:
        """Attach the observer to the built concrete card and return self."""
        self.build().observer(observer)
        return self

    def get_tools(self) -> list[Callable[..., Any]]:
        """Delegate tool callables to the built concrete card."""
        return self.build().get_tools()

    def get_system_prompts(self) -> list[Callable[..., Any]]:
        """Delegate system prompts to the built concrete card."""
        return self.build().get_system_prompts()

    def get_commands(self) -> dict[type[BaseToolParam], Callable[..., Any]]:
        """Delegate commands to the built concrete card."""
        return self.build().get_commands()

    def get_toolsets(self) -> list[Any]:
        """Delegate toolsets to the built concrete card."""
        return self.build().get_toolsets()
