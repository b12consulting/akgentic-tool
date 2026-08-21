"""The ``ToolFactory``: resolves tool cards into callable tools, prompts, and toolsets."""

import functools
import warnings
from collections.abc import Callable
from typing import Any

from akgentic.tool.core.observer import ToolObserver
from akgentic.tool.errors import RetriableError

from .card import ToolCard
from .commands import CommandRegistry, _build_command_entry, _CommandEntry
from .context_state import ContextState
from .dependencies import _topological_sort
from .params import BaseToolParam


class ToolFactory:
    """Resolves ``ToolCard`` instances into callable tools, prompts, and toolsets."""

    def __init__(
        self,
        tool_cards: list[ToolCard],
        observer: ToolObserver | None = None,
        retry_exception: type[Exception] | None = None,
    ) -> None:
        """Create a factory for one or more tool cards.

        Topologically sorts ``tool_cards`` by their ``depends_on`` class
        attribute, then attaches the observer to every card in dependency order
        (triggers runtime setup in ``ToolCard.observer()``). Prerequisites are
        wired before dependents, so a consumer card's ``observer()`` can safely
        look up actors or resources created by its prerequisites.

        Args:
            tool_cards: Tool cards to resolve into callable tools. The caller's
                list is not mutated; a new dependency-ordered list is stored on
                ``self.tool_cards``. Aggregators (``get_tools``,
                ``get_system_prompts``, ``get_commands``, ``get_toolsets``)
                iterate in this dependency order.
            observer: Optional observer notified by tool implementations during
                tool calls.
            retry_exception: Optional exception class to raise when a tool raises
                ``RetriableError``. Injected by the integration layer (e.g., ModelRetry
                from pydantic-ai) to keep the tool module framework-agnostic.

        Raises:
            ValueError: If the dependency graph is invalid — either a card
                declares ``depends_on`` for a class not present in
                ``tool_cards``, or a cycle exists. Raised before any observer
                is attached (fail fast at team creation).
        """
        self.tool_cards = _topological_sort(tool_cards)
        self.observer = observer
        self._retry_exception = retry_exception

        if self.observer is not None:
            for card in self.tool_cards:
                card.observer(self.observer)

    def _wrap_with_retry(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a tool callable to convert ``RetriableError`` into retry_exception."""
        assert self._retry_exception is not None
        retry_exc = self._retry_exception

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except RetriableError as e:
                raise retry_exc(str(e)) from e

        return wrapper

    def get_tools(self) -> list[Callable[..., Any]]:
        """Return tool callables aggregated from all tool cards."""
        tools = [t for card in self.tool_cards for t in card.get_tools()]
        if self._retry_exception is not None:
            tools = [self._wrap_with_retry(t) for t in tools]
        return tools

    def get_system_prompts(self) -> list[Callable[..., Any]]:
        """Return system prompt callables aggregated from all tool cards."""
        return [p for card in self.tool_cards for p in card.get_system_prompts()]

    def get_context_states(self) -> list[Callable[[], ContextState | None]]:
        """Return context-state providers aggregated from all tool cards.

        Iterates ``self.tool_cards`` in dependency order (same as
        :meth:`get_system_prompts`). A provider's key is its callable
        ``__name__`` — the same convention :meth:`get_command_registry` uses.
        This method runs at agent wiring/``on_start``, never mid-turn, so a
        collision fails at team-creation time.

        Raises:
            ValueError: If two providers share a ``__name__`` — never silent
                shadowing. The message names both owning card classes.
        """
        providers: list[Callable[[], ContextState | None]] = []
        owners: dict[str, str] = {}
        for card in self.tool_cards:
            card_name = type(card).__name__
            for provider in card.get_context_states():
                name = provider.__name__
                if name in owners:
                    raise ValueError(
                        f"Context-state provider name collision: '{name}' is exposed "
                        f"by both '{owners[name]}' and '{card_name}'."
                    )
                owners[name] = card_name
                providers.append(provider)
        return providers

    def get_commands(self) -> dict[type[BaseToolParam], Callable[..., Any]]:
        """Return command callables aggregated from all tool cards.

        Deprecated:
            Use :meth:`get_command_registry` instead. This param-class-keyed dict
            is retained for one migration cycle. The registry keys by canonical
            command name and adds signature-derived dispatch + discovery metadata.

        Returns:
            Dict mapping param class to callable, merged from all tool cards.
        """
        warnings.warn(
            "ToolFactory.get_commands() is deprecated; use get_command_registry() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        commands: dict[type[BaseToolParam], Callable[..., Any]] = {}
        for card in self.tool_cards:
            commands.update(card.get_commands())

        if self._retry_exception is not None:
            commands = {k: self._wrap_with_retry(v) for k, v in commands.items()}
        return commands

    def get_command_registry(
        self, extra_commands: list[Callable[..., Any]] | None = None
    ) -> CommandRegistry:
        """Build a name-keyed :class:`CommandRegistry` from every wired tool card.

        Iterates ``self.tool_cards`` in dependency order, calls each card's
        ``get_commands()``, and registers every callable under its ``__name__``.
        Each command's arg schema is derived from its signature at this point, so
        an un-derivable signature (``*args``/``**kwargs``/un-annotated param) fails
        loudly here. When ``retry_exception`` is configured, each command is
        wrapped via :meth:`_wrap_with_retry` (``functools.wraps`` preserves
        ``__name__``/``__doc__``), matching :meth:`get_commands` behavior.

        Args:
            extra_commands: Optional agent-owned plain callables (e.g. ``/compact``,
                ``/clear``) that join the same collision-checked, signature-derived
                table under the ``"BaseAgent"`` owner label. ``None`` (the default)
                ⇒ tool-card commands only, byte-identical to the no-arg call.

        Raises:
            ValueError: If two commands resolve to the same canonical name —
                whether two tool cards, a tool card and an extra callable, or two
                extra callables (collision is a wiring-time error, never a silent
                overwrite) — or if a command has an un-derivable signature. The
                message names the offending command.
        """
        entries: dict[str, _CommandEntry] = {}
        for card in self.tool_cards:
            tool_card_name = type(card).__name__
            for fn in card.get_commands().values():
                wrapped = self._wrap_with_retry(fn) if self._retry_exception is not None else fn
                name = wrapped.__name__
                if name in entries:
                    raise ValueError(
                        f"Command name collision: '{name}' is exposed by both "
                        f"'{entries[name].tool_card}' and '{tool_card_name}'."
                    )
                entries[name] = _build_command_entry(wrapped, tool_card_name)
        self._register_extra_commands(entries, extra_commands)
        return CommandRegistry(entries)

    def _register_extra_commands(
        self,
        entries: dict[str, _CommandEntry],
        extra_commands: list[Callable[..., Any]] | None,
    ) -> None:
        """Fold agent-owned ``extra_commands`` into *entries* in place.

        Each callable joins the same collision-checked, signature-derived table as
        tool-card commands, under the ``"BaseAgent"`` owner label. When a
        ``retry_exception`` is configured, the callable is retry-wrapped first;
        ``functools.wraps`` preserves ``__name__``/``__wrapped__`` so a wrapped
        ``compact`` still registers as ``/compact`` with its original signature.

        Raises:
            ValueError: If an extra callable's name collides with an already
                registered command (a tool-card command OR an earlier extra
                callable). The message names the colliding command.
        """
        for fn in extra_commands or []:
            wrapped = self._wrap_with_retry(fn) if self._retry_exception is not None else fn
            name = wrapped.__name__
            if name in entries:
                raise ValueError(f"Command name collision: '{name}'.")
            entries[name] = _build_command_entry(wrapped, "BaseAgent")

    def get_toolsets(self) -> list[Any]:
        """Return toolset instances aggregated from all tool cards."""
        return [ts for card in self.tool_cards for ts in card.get_toolsets()]
