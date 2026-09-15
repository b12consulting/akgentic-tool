"""One definition of a line, and the two functions over it that must agree.

The package used to hold two. ``rag/splitter.py`` counted lines with
``markdown-it``'s own break pattern, because the token maps it indexes are the
parser's; ``read/_paginate`` counted them with :meth:`str.splitlines`, which
breaks on eight further characters. ``python-pptx`` renders a soft line break as
a vertical tab, so an extracted deck carries several — and on such a body the
splitter and the read path numbered the same document differently. A line number
minted by one half of the workspace then named a different region to the other.

What is guarded here is the agreement itself, and it is asserted **on the two
functions**: two calls on one string. Routing it through a chunk lookup or a
retrieval search would be green for reasons that have nothing to do with line
numbering — the fixture could hold one chunk, or the chunk could land before the
first vertical tab — which is this epic's recurring failure shape, a check
narrowed until it agrees.

The invariant every spec below is a face of:

    len(line_starts(t)) == len(split_lines(t)) + 1      for every t
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from akgentic.tool.workspace import lines
from akgentic.tool.workspace.lines import line_starts, split_lines
from akgentic.tool.workspace.read import _paginate
from tests.workspace.test_splitter import LONE_PYTHON_BREAKS, SLIDES

_PERMITTED_IMPORTS = {"__future__", "re"}
"""What the spine may name. Both ship with CPython."""

TRAILING_NEWLINE_TABLE: list[tuple[str, list[str]]] = [
    ("", []),
    ("a", ["a"]),
    ("only\n", ["only"]),
    ("a\nb\nc", ["a", "b", "c"]),
    ("a\nb\nc\n", ["a", "b", "c"]),
    ("a\n\n", ["a", ""]),
    ("a\r\nb", ["a", "b"]),
    ("a\r", ["a"]),
]
"""What :func:`split_lines` returns, pinned as a table rather than as a rule.

Two rows carry the whole of the trailing-newline decision. ``"a\\n\\n"`` is the
row a "strip every trailing empty" implementation fails — it must stay **two**
lines, an ``"a"`` and an empty one. The ``"a\\nb\\nc"`` / ``"a\\nb\\nc\\n"`` pair
is the off-by-one: both are three lines and both give four starts, which is only
true because exactly one trailing empty element is dropped and only when it is
empty.
"""

MID_LINE_BODIES = [f"alpha{character}beta" for character in LONE_PYTHON_BREAKS]
"""One body per character :meth:`str.splitlines` breaks on and the parser does not."""

AGREEMENT_BODIES: list[str] = [
    *(body for body, _ in TRAILING_NEWLINE_TABLE),
    *MID_LINE_BODIES,
    SLIDES,
    *(SLIDES.replace("\v", character) for character in LONE_PYTHON_BREAKS),
]
"""One fixture list shared by the agreement invariant and the slice invariant.

The awkward cases are all here on purpose: the empty body, a body with no
trailing newline, a body ending in two breaks, a lone ``\\r``, a ``\\r\\n`` pair,
each of the eight lone Python breaks mid-line, and the deck-shaped body those
breaks actually arrive in.
"""


def _lines_source() -> str:
    """The spine's own source, read from the module rather than from a path guess."""
    return Path(lines.__file__).read_text(encoding="utf-8")


def _imported_modules(source: str) -> set[str]:
    """Every module named by an ``import`` in *source*, including under ``TYPE_CHECKING``.

    Parsed rather than matched, so a conditional or indented import cannot hide
    from this the way a regular expression over the text would let it.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            found.add(node.module)
    return found


class TestTheSpineImportsOnlyTheStandardLibrary:
    """AC 1 — the property that lets both capabilities stand on this module.

    ``read/`` may not import ``rag/``'s private helper and ``rag/`` may not import
    ``read/``'s: a capability reaching into a peer is exactly the edge ADR-053
    forbids, and it is the edge this module exists to remove. A module that names
    nothing but ``re`` is importable by either without creating one.
    """

    def test_it_names_nothing_outside_the_standard_library(self) -> None:
        named = _imported_modules(_lines_source())

        assert named <= _PERMITTED_IMPORTS, (
            f"the spine imports {sorted(named - _PERMITTED_IMPORTS)} — a module every "
            f"capability stands on may name only the standard library"
        )

    def test_the_parse_found_the_import_it_must_have(self) -> None:
        """Non-vacuity: a subset assertion over an empty set passes for ever."""
        assert "re" in _imported_modules(_lines_source())


class TestSplitLinesBreaksWhereTheParserDoes:
    """AC 3 — eight characters that start a line for Python and for nothing else."""

    @pytest.mark.parametrize("character", LONE_PYTHON_BREAKS)
    def test_a_lone_python_break_mid_line_is_not_a_line_break(self, character: str) -> None:
        """Pinned per character, not on the vertical tab alone.

        The vertical tab is the one that arrives in practice — ``python-pptx``
        renders ``<a:br>`` as one — but the other seven fail identically and cost
        nothing to pin. A spec that covered only ``\\v`` would go green for an
        implementation that special-cased it.
        """
        body = f"alpha{character}beta"

        assert split_lines(body) == [body]

    @pytest.mark.parametrize("character", LONE_PYTHON_BREAKS)
    def test_python_disagrees_with_us_about_that_body(self, character: str) -> None:
        """Non-vacuity, and the whole reason the module exists.

        Every row above would pass against ``str.splitlines`` if these characters
        were not break characters for it. They are, which is what makes the two
        definitions disagree and this module necessary.
        """
        body = f"alpha{character}beta"

        assert len(body.splitlines()) == 2
        assert len(split_lines(body)) == 1


class TestSplitLinesDropsOneTrailingEmptyElement:
    """AC 4 — and only when it is empty, which is the whole rule."""

    @pytest.mark.parametrize(("body", "expected"), TRAILING_NEWLINE_TABLE)
    def test_the_table(self, body: str, expected: list[str]) -> None:
        assert split_lines(body) == expected

    def test_a_blank_last_line_survives(self) -> None:
        """The row a ``while`` loop or an ``rstrip`` flattens.

        ``"a\\n\\n"`` is an ``"a"`` and an empty line, not one line — and
        :func:`line_starts` agrees, giving three starts for two lines. A "strip
        every trailing empty" implementation returns one line and breaks the
        invariant on a body that is entirely ordinary in a document.
        """
        assert split_lines("a\n\n") == ["a", ""]
        assert len(line_starts("a\n\n")) == 3


class TestLineStartsKeepsItsContract:
    """AC 5 — first element ``0``, terminating ``len(text)``, once, and no clamp."""

    @pytest.mark.parametrize("body", AGREEMENT_BODIES)
    def test_it_starts_at_zero(self, body: str) -> None:
        assert line_starts(body)[0] == 0

    @pytest.mark.parametrize("body", AGREEMENT_BODIES)
    def test_it_terminates_at_the_length_exactly_once(self, body: str) -> None:
        """A token's exclusive ``map[1]`` must always index in range.

        Appended **only when it is not already the last element**, which is what
        keeps the count right for a body that does end in a break: ``"a\\nb\\nc"``
        and ``"a\\nb\\nc\\n"`` are both three lines and both give four starts.
        """
        starts = line_starts(body)

        assert starts[-1] == len(body)
        assert starts.count(len(body)) == 1

    @pytest.mark.parametrize("body", AGREEMENT_BODIES)
    def test_it_is_strictly_increasing_and_never_clamped(self, body: str) -> None:
        """No offset is invented and none is trimmed to fit.

        A clamp would hide a real indexing bug behind an in-range answer, which
        is the reason the original helper refused one.
        """
        starts = line_starts(body)

        assert starts == sorted(set(starts))
        assert all(0 <= start <= len(body) for start in starts)

    def test_a_body_with_no_trailing_newline_gets_its_terminator(self) -> None:
        assert line_starts("a\nb\nc") == [0, 2, 4, 5]

    def test_a_body_that_ends_on_a_break_does_not_get_a_second_one(self) -> None:
        assert line_starts("a\nb\nc\n") == [0, 2, 4, 6]


class TestTheTwoFunctionsAgree:
    """AC 6 and AC 7 — asserted on the two functions, never through a search."""

    @pytest.mark.parametrize("body", AGREEMENT_BODIES)
    def test_there_is_one_more_start_than_there_are_lines(self, body: str) -> None:
        """The agreement guard the epic asks for, in its whole form.

        Both halves of the workspace derive their numbering from one of these two
        lists. While this holds, a line number minted against either names the
        same region to the other; the moment it stops, every number past the
        first divergence is silently wrong.
        """
        assert len(line_starts(body)) == len(split_lines(body)) + 1

    @pytest.mark.parametrize("body", AGREEMENT_BODIES)
    def test_the_slice_between_two_starts_is_that_line(self, body: str) -> None:
        """The stronger claim: the starts index the lines the split produced.

        The counts agreeing is not enough on its own — two lists of the right
        length can still describe different regions. Every slice must be its own
        line, carrying at most the one break that ended it.
        """
        starts = line_starts(body)
        split = split_lines(body)

        for index, line in enumerate(split):
            sliced = body[starts[index] : starts[index + 1]]
            assert sliced in (line, f"{line}\n", f"{line}\r", f"{line}\r\n"), (
                f"line {index} slices to {sliced!r}, which is not {line!r} plus a break"
            )


class TestTheDeckNumbersTheSameEitherWay:
    """AC 8 — the ``\\v`` case, on the body those vertical tabs actually arrive in.

    A ``python-pptx`` extraction with a wrapped title, which is where this defect
    was found rather than a minimal repro of it. The assertion is still on the two
    functions plus ``_paginate``'s own reported total: no chunk lookup, no
    retrieval search, nothing that could be green because the fixture happened to
    produce one chunk.
    """

    @pytest.mark.parametrize("character", LONE_PYTHON_BREAKS)
    def test_the_splitter_and_the_read_path_count_the_same_lines(self, character: str) -> None:
        deck = SLIDES.replace("\v", character)
        expected = len(split_lines(deck))

        numbered, full = _paginate(deck, offset=1, limit=1_000_000)

        assert len(line_starts(deck)) - 1 == expected
        assert full is True
        assert len(numbered.split("\n")) == expected

    @pytest.mark.parametrize("character", LONE_PYTHON_BREAKS)
    def test_the_gutter_stops_where_the_split_does(self, character: str) -> None:
        """``_paginate``'s ``total`` is the split's count, read off its own notice.

        A window one line short reports how many lines it believes there are, so
        the number the agent is told matches the number the splitter minted
        against.
        """
        deck = SLIDES.replace("\v", character)
        expected = len(split_lines(deck))

        numbered, full = _paginate(deck, offset=1, limit=expected - 1)

        assert full is False
        assert f"truncated: {expected} lines total" in numbered

    def test_the_deck_is_a_body_the_two_definitions_disagree_about(self) -> None:
        """Non-vacuity. Against a deck with no vertical tab all of this is trivial."""
        assert len(SLIDES.splitlines()) > len(split_lines(SLIDES))
