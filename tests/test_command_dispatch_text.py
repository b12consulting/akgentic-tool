"""How ``CommandRegistry.dispatch`` turns the human's text into arguments.

The name is the text's first whitespace token. What follows is tokenised with
``shlex`` inside the failure contract, so a quote that does not close is the
named command's failure — a result string — and never a crash out of dispatch.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from akgentic.tool.core import COMMAND, BaseToolParam, CommandRegistry, ToolCard, ToolFactory
from akgentic.tool.errors import CommandNotRecognized


class _DeleteParam(BaseToolParam):
    expose: set[str] = {COMMAND}


class _DeleteCard(ToolCard):
    """Card exposing a ``remove(filepath)`` command that records what it was given."""

    name: str = "delete-card"
    description: str = "delete card"

    def get_tools(self) -> list[Callable[..., Any]]:
        return []

    def get_commands(self) -> dict[type[BaseToolParam], Callable[..., Any]]:
        def remove(filepath: str) -> str:
            """Remove one file.

            Example: /remove notes.md
            """
            return f"Deleted: {filepath}"

        return {_DeleteParam: remove}


def _registry() -> CommandRegistry:
    return ToolFactory(tool_cards=[_DeleteCard()]).get_command_registry()


class TestABrokenQuoteIsAResult:
    def test_a_known_command_with_an_unclosed_quote_returns_its_failure(self) -> None:
        result = _registry().dispatch('/remove "a.md')
        assert result == "Command 'remove' failed: No closing quotation"

    def test_an_unknown_command_with_an_unclosed_quote_still_falls_back(self) -> None:
        with pytest.raises(CommandNotRecognized):
            _registry().dispatch('/frobnicate "a.md')

    def test_the_name_is_the_first_whitespace_token(self) -> None:
        registry = _registry()
        assert registry.dispatch('/remove   "a b.md"') == "Deleted: a b.md"
        assert registry.dispatch("/remove\tnotes.md") == "Deleted: notes.md"
