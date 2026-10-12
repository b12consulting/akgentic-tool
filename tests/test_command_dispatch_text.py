"""How ``CommandRegistry.dispatch`` turns the human's text into arguments.

The name is the text's first whitespace token. What follows is tokenised with
``shlex`` inside the failure contract, so a quote that does not close is the
named command's failure — a result string — and never a crash out of dispatch.
A ``RestOfLine`` last parameter receives the raw remainder instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from akgentic.tool.core import (
    COMMAND,
    BaseToolParam,
    CommandRegistry,
    RestOfLine,
    ToolCard,
    ToolFactory,
)
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


# ---------------------------------------------------------------------------
# RestOfLine — the last parameter receives the raw rest of the line
# ---------------------------------------------------------------------------


class _WriteParam(BaseToolParam):
    expose: set[str] = {COMMAND}


class _RunParam(BaseToolParam):
    expose: set[str] = {COMMAND}


class _RestCard(ToolCard):
    """Card exposing ``write(filepath, content: RestOfLine)`` and ``run(cmd: RestOfLine)``."""

    name: str = "rest-card"
    description: str = "rest card"

    def get_tools(self) -> list[Callable[..., Any]]:
        return []

    def get_commands(self) -> dict[type[BaseToolParam], Callable[..., Any]]:
        def write(filepath: str, content: RestOfLine) -> str:
            """Write content."""
            return f"{filepath}|{content}"

        def run(cmd: RestOfLine) -> str:
            """Run a line."""
            return cmd

        return {_WriteParam: write, _RunParam: run}


def _rest_registry() -> CommandRegistry:
    return ToolFactory(tool_cards=[_RestCard()]).get_command_registry()


class TestRestOfLine:
    def test_quotes_and_apostrophes_are_kept_verbatim(self) -> None:
        result = _rest_registry().dispatch("""/write notes.md it's a "quoted" word""")
        assert result == """notes.md|it's a "quoted" word"""

    def test_newlines_inner_and_trailing_whitespace_are_kept_leading_is_stripped(self) -> None:
        result = _rest_registry().dispatch("/write notes.md   line one\n  line  two\n\n")
        assert result == "notes.md|line one\n  line  two\n\n"

    def test_a_quoted_leading_argument_is_still_tokenised(self) -> None:
        result = _rest_registry().dispatch('/write "my notes.md" body')
        assert result == "my notes.md|body"

    def test_a_leading_argument_may_still_be_named(self) -> None:
        assert _rest_registry().dispatch("/write filepath=a.md body") == "a.md|body"

    def test_name_equals_value_is_literal_text_in_the_rest(self) -> None:
        assert _rest_registry().dispatch("/write notes.md content=x") == "notes.md|content=x"

    def test_an_empty_remainder_is_the_missing_argument_result(self) -> None:
        result = _rest_registry().dispatch("/write notes.md   ")
        assert result == "Command 'write' failed: missing required argument(s): content"

    def test_a_sole_rest_of_line_parameter_takes_the_whole_line(self) -> None:
        assert _rest_registry().dispatch("/run ls -la | wc -l") == "ls -la | wc -l"

    def test_a_broken_quote_in_a_leading_argument_is_a_result(self) -> None:
        result = _rest_registry().dispatch('/write "notes.md body')
        assert result == "Command 'write' failed: No closing quotation"

    def test_a_broken_quote_inside_the_rest_is_content(self) -> None:
        assert _rest_registry().dispatch('/write notes.md a "b') == 'notes.md|a "b'

    def test_the_descriptor_lists_the_argument_names(self) -> None:
        descriptors = {d.name: d for d in _rest_registry().descriptors()}
        write = descriptors["write"]
        assert [(arg.name, arg.type, arg.required) for arg in write.args] == [
            ("filepath", "string", True),
            ("content", "string", True),
        ]


class TestRestOfLineIsOnlyLastAndRequired:
    def _build(self, fn: Callable[..., Any]) -> None:
        class _Card(ToolCard):
            name: str = "bad"
            description: str = "bad"

            def get_tools(self) -> list[Callable[..., Any]]:
                return []

            def get_commands(self) -> dict[type[BaseToolParam], Callable[..., Any]]:
                return {_WriteParam: fn}

        ToolFactory(tool_cards=[_Card()]).get_command_registry()

    def test_not_last_is_refused_naming_the_command(self) -> None:
        def misplaced(content: RestOfLine, filepath: str) -> str:
            """Bad."""
            return content

        with pytest.raises(ValueError, match="'misplaced'.*must be the last"):
            self._build(misplaced)

    def test_optional_is_refused_naming_the_command(self) -> None:
        def optional(filepath: str, content: RestOfLine = "") -> str:
            """Bad."""
            return content

        with pytest.raises(ValueError, match="'optional'.*must be required"):
            self._build(optional)
