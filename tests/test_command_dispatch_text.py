"""How ``CommandRegistry.dispatch`` turns the human's text into arguments.

The name is the text's first whitespace token. What follows is tokenised with
``shlex`` inside the failure contract, so a quote that does not close is the
named command's failure — a result string — and never a crash out of dispatch.
A ``RestOfLine`` last parameter receives the raw remainder instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

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


class TestRestOfLineIsOnlyLast:
    def _build(self, fn: Callable[..., Any]) -> CommandRegistry:
        class _Card(ToolCard):
            name: str = "bad"
            description: str = "bad"

            def get_tools(self) -> list[Callable[..., Any]]:
                return []

            def get_commands(self) -> dict[type[BaseToolParam], Callable[..., Any]]:
                return {_WriteParam: fn}

        return ToolFactory(tool_cards=[_Card()]).get_command_registry()

    def test_not_last_is_refused_naming_the_command(self) -> None:
        def misplaced(content: RestOfLine, filepath: str) -> str:
            """Bad."""
            return content

        with pytest.raises(ValueError, match="'misplaced'.*must be the last"):
            self._build(misplaced)

    def test_optional_registers_and_an_empty_remainder_keeps_its_default(self) -> None:
        def optional(filepath: str, content: RestOfLine = "<default>") -> str:
            """Optional text."""
            return content

        registry = self._build(optional)

        assert registry.dispatch("/optional a.md") == "<default>"
        assert registry.dispatch("/optional a.md hi") == "hi"

    def test_a_positional_after_an_option_is_refused_naming_the_command_and_parameter(
        self,
    ) -> None:
        def bad(mode: str = "a", *, path: str, text: RestOfLine) -> str:
            """Bad."""
            return text

        with pytest.raises(ValueError, match="'bad'.*'path'"):
            self._build(bad)

    def test_the_same_shape_without_rest_of_line_still_registers(self) -> None:
        """The refusal is a rest-of-line rule; a plain command keeps the 21.4 grammar."""

        def fine(mode: str = "a", *, path: str) -> str:
            """Fine."""
            return f"{mode}|{path}"

        assert self._build(fine).dispatch("/fine path=x") == "a|x"


# ---------------------------------------------------------------------------
# Options, then text — a defaulted parameter before RestOfLine is name=value only
# ---------------------------------------------------------------------------


class _FindParam(BaseToolParam):
    expose: set[str] = {COMMAND}


class _ListParam(BaseToolParam):
    expose: set[str] = {COMMAND}


class _PutParam(BaseToolParam):
    expose: set[str] = {COMMAND}


class _OptionsCard(ToolCard):
    """Card exposing options-then-text commands that echo what they were given."""

    name: str = "options-card"
    description: str = "options card"

    def get_tools(self) -> list[Callable[..., Any]]:
        return []

    def get_commands(self) -> dict[type[BaseToolParam], Callable[..., Any]]:
        def find(
            mode: Literal["hybrid", "vector", "keyword"] = "hybrid",
            top_k: int | None = None,
            include_edges: bool = False,
            *,
            text: RestOfLine,
        ) -> str:
            """Find with a required text."""
            return repr((mode, top_k, include_edges, text))

        def listing(
            status: Literal["pending", "started"] | None = None,
            owner: str | None = None,
            text: RestOfLine = "",
        ) -> str:
            """List with an optional text."""
            return repr((status, owner, text))

        def put(
            filepath: str, mode: Literal["a", "b"] = "a", *, text: RestOfLine
        ) -> str:
            """A positional, an option, then text."""
            return repr((filepath, mode, text))

        return {_FindParam: find, _ListParam: listing, _PutParam: put}


def _options_registry() -> CommandRegistry:
    return ToolFactory(tool_cards=[_OptionsCard()]).get_command_registry()


class TestAnOptionIsNeverBoundPositionally:
    def test_a_plain_first_word_is_text_not_the_first_option(self) -> None:
        result = _options_registry().dispatch("/find mailing list")
        assert result == repr(("hybrid", None, False, "mailing list"))

    def test_options_then_text(self) -> None:
        result = _options_registry().dispatch("/find mode=keyword top_k=2 mailing list")
        assert result == repr(("keyword", 2, False, "mailing list"))

    def test_a_positional_is_still_bound_before_the_options(self) -> None:
        result = _options_registry().dispatch("/put notes.md mode=b hello there")
        assert result == repr(("notes.md", "b", "hello there"))

    def test_a_positional_may_still_be_named(self) -> None:
        result = _options_registry().dispatch("/put filepath=notes.md hello")
        assert result == repr(("notes.md", "a", "hello"))

    def test_a_line_ending_before_the_positionals_is_the_missing_argument_result(self) -> None:
        result = _options_registry().dispatch("/put")
        assert result == "Command 'put' failed: missing required argument(s): filepath, text"

    def test_an_option_before_a_positional_is_the_ordering_failure(self) -> None:
        result = _options_registry().dispatch("/put mode=b notes.md hello")
        assert result == (
            "Command 'put' failed: positional argument 'notes.md' cannot follow a keyword argument"
        )


class TestTheTextStartsAtTheFirstNonOption:
    def test_a_quoted_first_word_is_kept_verbatim(self) -> None:
        result = _options_registry().dispatch('/find mode=keyword "exact" phrase')
        assert result == repr(("keyword", None, False, '"exact" phrase'))

    def test_option_like_text_after_the_first_text_word_is_text(self) -> None:
        result = _options_registry().dispatch("/find mailing mode=vector")
        assert result == repr(("hybrid", None, False, "mailing mode=vector"))

    def test_name_value_naming_no_option_is_text(self) -> None:
        assert _options_registry().dispatch("/find a=b c") == repr(
            ("hybrid", None, False, "a=b c")
        )

    def test_inner_whitespace_and_newlines_are_kept_leading_is_stripped(self) -> None:
        result = _options_registry().dispatch("/find mode=keyword   two  spaces\n next line")
        assert result == repr(("keyword", None, False, "two  spaces\n next line"))

    def test_option_values_are_shlex_tokenised(self) -> None:
        result = _options_registry().dispatch('/listing owner="Alice Smith" deploy')
        assert result == repr((None, "Alice Smith", "deploy"))

    def test_a_broken_quote_in_the_text_is_text(self) -> None:
        result = _options_registry().dispatch('/find "mailing list')
        assert result == repr(("hybrid", None, False, '"mailing list'))

    def test_a_broken_quote_after_an_option_is_text(self) -> None:
        result = _options_registry().dispatch('/find mode=vector "mailing list')
        assert result == repr(("vector", None, False, '"mailing list'))

    def test_a_broken_quote_before_the_positionals_are_bound_is_a_result(self) -> None:
        result = _options_registry().dispatch('/put "notes.md hello')
        assert result == "Command 'put' failed: No closing quotation"


class TestOptionValuesAreCoerced:
    def test_a_bad_option_value_is_the_commands_failure(self) -> None:
        result = _options_registry().dispatch("/find mode=x mailing")
        assert result is not None
        assert result.startswith("Command 'find' failed:")

    def test_int_and_bool_options_are_coerced(self) -> None:
        result = _options_registry().dispatch("/find top_k=3 include_edges=true foo")
        assert result == repr(("hybrid", 3, True, "foo"))

    def test_a_repeated_option_is_a_duplicate_binding_failure(self) -> None:
        result = _options_registry().dispatch("/find mode=keyword mode=vector foo")
        assert result == "Command 'find' failed: got multiple values for argument 'mode'"


class TestADefaultedTextMayBeEmpty:
    def test_options_alone_leave_the_text_at_its_default(self) -> None:
        assert _options_registry().dispatch("/listing status=pending") == repr(
            ("pending", None, "")
        )

    def test_the_bare_command_applies_every_default(self) -> None:
        assert _options_registry().dispatch("/listing") == repr((None, None, ""))

    def test_a_required_text_left_empty_is_the_missing_argument_result(self) -> None:
        result = _options_registry().dispatch("/find mode=keyword   ")
        assert result == "Command 'find' failed: missing required argument(s): text"
