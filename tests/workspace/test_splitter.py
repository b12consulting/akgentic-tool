"""The block splitter: offsets that are real, and four packing rules.

The headline property is **offset fidelity**: for every span the splitter emits,
``markdown[span.start:span.end]`` is a non-empty, verbatim, positionally-correct
region of the document. That is the property the rejected third-party splitter
could not provide and the entire reason this one is ours, so it is asserted
positionally — ``markdown.find(slice, start) == start`` — and against expected
literals, never with a tautology like ``markdown[a:b] == markdown[a:b]``.

Nothing here reaches an actor, a message handler or the filesystem: the module
under test is two pure functions, a Protocol and the class that composes them.
"""

from __future__ import annotations

import pytest

from akgentic.tool.core import COMMAND, TOOL_CALL
from akgentic.tool.workspace import BlockSplitter, Span, TextSplitter, WorkspaceRagIndex
from akgentic.tool.workspace.documents.splitter import pack_blocks, parse_blocks

# --------------------------------------------------------------------------- #
# Fixtures — module-level constants, each small enough to read in a failure.
# --------------------------------------------------------------------------- #

NESTED = (
    "# A\n"
    "\n"
    "Alpha paragraph under A.\n"
    "\n"
    "## B\n"
    "\n"
    "Beta paragraph under B.\n"
    "\n"
    "### C\n"
    "\n"
    "Gamma paragraph under C.\n"
    "\n"
    "# D\n"
    "\n"
    "Delta paragraph under D.\n"
)

NESTED_SLICES = [
    "Alpha paragraph under A.",
    "Beta paragraph under B.",
    "Gamma paragraph under C.",
    "Delta paragraph under D.",
]

NESTED_PATHS = [["A"], ["A", "B"], ["A", "B", "C"], ["D"]]

GFM_HEADER_ROW = "| Item      | Qty | Price |"

GFM = (
    "## Payment terms\n"
    "\n"
    f"{GFM_HEADER_ROW}\n"
    "|-----------|-----|-------|\n"
    "| Widget    |   2 | 10.00 |\n"
    "| Gadget    |   1 | 25.50 |\n"
    "| Doodad    |   7 |  3.25 |\n"
    "| Gizmo     |   4 | 12.75 |\n"
    "| Whatsit   |   3 |  8.10 |\n"
)

TABLE_THEN_CODE = (
    "## Terms\n"
    "\n"
    "| Item   | Qty |\n"
    "|--------|-----|\n"
    "| Widget |   2 |\n"
    "    an indented line, which a table cannot absorb\n"
)

FENCED = (
    "## Code\n"
    "\n"
    "```python\n"
    "# not a heading\n"
    "\n"
    "def compute(value):\n"
    "    total = value * 2\n"
    "    other = total + 1\n"
    "    return other + value\n"
    "    # one more line, to clear the target\n"
    "```\n"
)

LIST = "## Items\n\n- first item\n- second item\n- third item\n"

JUMPED = "# A\n\nAlpha.\n\n### C\n\nGamma.\n\n## B\n\nBeta.\n"

PREAMBLE = "Preamble before any heading.\n\n# A\n\nAlpha.\n"

REPEATED = "# A\n\nOne.\n\n## A\n\nTwo.\n"

SETEXT = "Title\n=====\n\nBody paragraph.\n"

MISC = (
    "## Misc\n"
    "\n"
    "> A quoted line.\n"
    "\n"
    "---\n"
    "\n"
    "    indented code line one\n"
    "    indented code line two\n"
    "\n"
    "<div>\n"
    "  <p>html block</p>\n"
    "</div>\n"
)

WIDE_TABLE_HEADER_ROW = "| Item              | Qty | Price |"

WIDE_TABLE = (
    "## Invoice\n"
    "\n"
    f"{WIDE_TABLE_HEADER_ROW}\n"
    "|-------------------|-----|-------|\n"
    + "".join(f"| Widget number {n:02d} | {n:3d} | {n * 3:5d} |\n" for n in range(1, 21))
)

LONG_PARA = "## Notes\n\n" + " ".join(
    f"Sentence number {n} explains one distinct point in adequate detail." for n in range(1, 13)
)

MANY = (
    "## Notes\n\n"
    + "\n\n".join(f"Paragraph number {n} with a little bit of text." for n in range(1, 8))
    + "\n"
)

LONG_SENTENCE = "## Notes\n\n" + " ".join(f"word{n:03d}" for n in range(1, 60)) + "\n"

WIDE_INDENTED = "## Code\n\n" + "".join(
    f"    indented code line number {n:02d}\n" for n in range(1, 12)
)

TINY_SAME = "## S\n\nTiny.\n\nA slightly longer paragraph.\n"

TINY_OTHER = "## S\n\nTiny.\n\n## T\n\nA slightly longer paragraph.\n"

# Two sections whose heading text is identical, which makes their heading paths
# identical too. A path is not an identity, so path equality alone would pack
# them together and the chunk would swallow the second heading line.
SIBLINGS = "# A\n\nOne paragraph here.\n\n# A\n\nTwo paragraph here.\n"

TINY_TWINS = "## S\n\nTiny.\n\n## S\n\nAlso tiny.\n"

# A document whose top level is ``##``, which is what an extractor routinely
# produces. The three sections are siblings; none is inside another.
DEEP_ROOT = "## A\n\nAlpha.\n\n## B\n\nBeta.\n\n## C\n\nGamma.\n"

# Two ``###`` siblings after a skipped level. The second must replace the first
# in the path, not stack on top of it.
JUMPED_SIBLINGS = "# A\n\nAlpha.\n\n### C\n\nGamma.\n\n### D\n\nDelta.\n"

# A slide-deck extraction in the shape MarkItDown produces from a ``.pptx``: one
# HTML comment per slide, a title, a body, and a ``### Notes:`` heading whose
# text repeats on every slide. The vertical tabs are the point — ``python-pptx``
# renders a soft line break (``<a:br>``) as one, so a deck with a wrapped title
# carries several, and ``str.splitlines`` counts each as a line that
# ``markdown-it`` does not.
SLIDES = (
    "<!-- Slide number: 1 -->\n"
    "\n"
    "# The Acme Platform\vAn introduction\n"
    "\n"
    "### Notes:\n"
    "\n"
    "Acme builds tooling for contoso teams.\n"
    "\n"
    "<!-- Slide number: 2 -->\n"
    "\n"
    "## Capabilities\vand what they cost\n"
    "\n"
    "| Capability             | Status |\n"
    "|------------------------|--------|\n"
    "| Multi party routing    | Ready  |\n"
    "| Workflow orchestration | Ready  |\n"
    "\n"
    "### Notes:\n"
    "\n"
    "Every capability ships behind the same address.\n"
    "\n"
    "<!-- Slide number: 3 -->\n"
    "\n"
    "## Roadmap\n"
    "\n"
    "![Roadmap diagram](roadmap.png)\n"
    "\n"
    "### Notes:\n"
    "\n"
    "The second half of the year is reserved for retrieval.\n"
)

# Every character ``str.splitlines`` breaks a line on that ``markdown-it`` does
# not. The vertical tab is the one that arrives in practice; the rest fail the
# same way and cost nothing to pin.
LONE_PYTHON_BREAKS = ["\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"]

EMPTY = ""

BLANK = "\n\n   \n"

FIXTURES: dict[str, str] = {
    "NESTED": NESTED,
    "GFM": GFM,
    "TABLE_THEN_CODE": TABLE_THEN_CODE,
    "FENCED": FENCED,
    "LIST": LIST,
    "JUMPED": JUMPED,
    "PREAMBLE": PREAMBLE,
    "REPEATED": REPEATED,
    "SETEXT": SETEXT,
    "MISC": MISC,
    "WIDE_TABLE": WIDE_TABLE,
    "LONG_PARA": LONG_PARA,
    "LONG_SENTENCE": LONG_SENTENCE,
    "WIDE_INDENTED": WIDE_INDENTED,
    "MANY": MANY,
    "TINY_SAME": TINY_SAME,
    "TINY_OTHER": TINY_OTHER,
    "SIBLINGS": SIBLINGS,
    "TINY_TWINS": TINY_TWINS,
    "DEEP_ROOT": DEEP_ROOT,
    "JUMPED_SIBLINGS": JUMPED_SIBLINGS,
    "SLIDES": SLIDES,
    "EMPTY": EMPTY,
    "BLANK": BLANK,
}

# Small values so a fixture stays readable in a failure. The defaults are
# asserted separately, once, in TestDefaults.
SMALL = WorkspaceRagIndex(
    chunk_chars=120, max_chunk_chars=200, min_chunk_chars=1, chunk_overlap_chars=0
)
OVERLAPPING = SMALL.model_copy(update={"chunk_overlap_chars": 60})
CONVERGING = SMALL.model_copy(update={"chunk_overlap_chars": SMALL.chunk_chars - 1})
MERGING = WorkspaceRagIndex(
    chunk_chars=30, max_chunk_chars=200, min_chunk_chars=20, chunk_overlap_chars=0
)
MERGING_AT_THE_CEILING = MERGING.model_copy(update={"max_chunk_chars": 31})

ALL_PARAMS = {"SMALL": SMALL, "OVERLAPPING": OVERLAPPING, "CONVERGING": CONVERGING}


def _slice(markdown: str, span: Span) -> str:
    """The document text a span names."""
    return markdown[span.start : span.end]


def _composed(markdown: str, span: Span) -> str:
    """What the embedder would send: the repeated header row, then the slice."""
    if span.header_start is None or span.header_end is None:
        return _slice(markdown, span)
    return f"{markdown[span.header_start : span.header_end]}\n{_slice(markdown, span)}"


def _covering(chunks: list[Span], offset: int) -> bool:
    """Whether any chunk covers *offset*."""
    return any(chunk.start <= offset < chunk.end for chunk in chunks)


def _outside_a_heading_line(markdown: str) -> list[int]:
    """Every non-whitespace offset of *markdown* that a heading line does not claim.

    The whole of the source a chunk is obliged to carry. It is computed here by
    scanning for ``\\n`` and nothing else, deliberately: reusing the module's own
    line arithmetic would make the expectation drift in lockstep with the bug
    this guards against, and the test would pass while the data was lost.

    Only sound for a fixture with no thematic break and no setext underline,
    which are the other two things no block covers.
    """
    offsets: list[int] = []
    position = 0
    for line in markdown.split("\n"):
        if not line.lstrip().startswith("#"):
            offsets.extend(
                i for i, character in enumerate(line, position) if not character.isspace()
            )
        position += len(line) + 1
    return offsets


def _touching(blocks: list[Span], chunk: Span) -> list[Span]:
    """The blocks whose range intersects *chunk*'s."""
    return [b for b in blocks if b.start < chunk.end and chunk.start < b.end]


class TestOffsetFidelity:
    """The headline property: a span names a real, verbatim region of the source."""

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    def test_every_block_slice_is_verbatim_at_its_own_offset(self, name: str) -> None:
        """``find`` from the span's own start must land on the span's own start."""
        markdown = FIXTURES[name]
        for span in parse_blocks(markdown):
            text = _slice(markdown, span)
            assert text, f"{name}: empty slice at {span.start}:{span.end}"
            assert markdown.find(text, span.start) == span.start
            assert markdown.count(text) >= 1

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    @pytest.mark.parametrize("params_name", sorted(ALL_PARAMS))
    def test_every_chunk_slice_is_verbatim_at_its_own_offset(
        self, name: str, params_name: str
    ) -> None:
        """The same guard on the packed output, not only on the parsed blocks."""
        markdown = FIXTURES[name]
        for span in BlockSplitter().split(markdown, ALL_PARAMS[params_name]):
            text = _slice(markdown, span)
            assert text
            assert markdown.find(text, span.start) == span.start

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    def test_block_offsets_are_ordered_and_inside_the_document(self, name: str) -> None:
        """``0 <= start < end <= len(markdown)`` on every span."""
        markdown = FIXTURES[name]
        for span in parse_blocks(markdown):
            assert 0 <= span.start < span.end <= len(markdown)

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    def test_chunk_offsets_are_ordered_and_inside_the_document(self, name: str) -> None:
        """The same bound on ``split``'s output."""
        markdown = FIXTURES[name]
        for span in BlockSplitter().split(markdown, SMALL):
            assert 0 <= span.start < span.end <= len(markdown)

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    def test_no_block_slice_carries_surrounding_whitespace(self, name: str) -> None:
        """A block's raw slice ends in its own newline; the trim is what removes it."""
        markdown = FIXTURES[name]
        for span in parse_blocks(markdown):
            text = _slice(markdown, span)
            assert text == text.strip(), f"{name}: {text!r} is not stripped"

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    def test_no_chunk_slice_carries_surrounding_whitespace(self, name: str) -> None:
        """The same on the packed output, where a cut could reintroduce it."""
        markdown = FIXTURES[name]
        for span in BlockSplitter().split(markdown, SMALL):
            text = _slice(markdown, span)
            assert text == text.strip(), f"{name}: {text!r} is not stripped"

    def test_the_nested_document_yields_exactly_the_expected_literals(self) -> None:
        """Named literals, so a shifted offset cannot pass by being plausible."""
        assert [_slice(NESTED, span) for span in parse_blocks(NESTED)] == NESTED_SLICES

    def test_the_list_slice_is_the_whole_list_and_nothing_after_it(self) -> None:
        """The blank line after the last item is inside the token map, not the span."""
        (block,) = parse_blocks(LIST)
        assert _slice(LIST, block) == "- first item\n- second item\n- third item"

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    def test_no_chunk_begins_or_ends_in_the_middle_of_a_word(self, name: str) -> None:
        """Every cut lands on a boundary the document already had."""
        markdown = FIXTURES[name]
        for span in BlockSplitter().split(markdown, SMALL):
            assert span.start == 0 or markdown[span.start - 1].isspace()
            assert span.end == len(markdown) or markdown[span.end].isspace()


class TestCoverage:
    """Blocks are never dropped: the chunks cover every block, as a union."""

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    @pytest.mark.parametrize("params_name", sorted(ALL_PARAMS))
    def test_every_block_offset_is_covered_by_the_union_of_the_chunks(
        self, name: str, params_name: str
    ) -> None:
        """Union, not containment.

        A paragraph that subdivides is spread across several chunks, so no single
        chunk contains it and a containment assertion would be unsatisfiable.

        Whitespace between two adjacent pieces is exempt, and necessarily so: a
        cut lands on a separator, and both pieces are then trimmed off it. The
        two halves of the offset-fidelity criterion — full coverage and no
        surrounding whitespace — are only simultaneously satisfiable over the
        non-whitespace offsets.
        """
        markdown = FIXTURES[name]
        chunks = BlockSplitter().split(markdown, ALL_PARAMS[params_name])
        for block in parse_blocks(markdown):
            for offset in range(block.start, block.end):
                if markdown[offset].isspace():
                    continue
                assert _covering(chunks, offset), f"{name}: offset {offset} is in no chunk"

    @pytest.mark.parametrize("character", LONE_PYTHON_BREAKS)
    @pytest.mark.parametrize("params_name", sorted(ALL_PARAMS))
    def test_a_break_only_python_sees_costs_the_document_nothing(
        self, character: str, params_name: str
    ) -> None:
        """Coverage measured against the **source**, not against the blocks.

        The guard above compares the chunks to ``parse_blocks``' own output, so
        it is blind by construction to anything ``parse_blocks`` never emitted in
        the first place — which is where a whole class of loss lives. This one
        starts from the document.

        ``markdown-it`` starts a line at ``\\r\\n``, ``\\r`` or ``\\n`` and at
        nothing else; ``str.splitlines`` starts one at eight further characters.
        One of those in the document and the line counts diverge, so every token
        map past it names the wrong line and every offset after it is wrong —
        chunks land on heading lines, begin mid-word, and whole sections fall
        into the gaps between them. A real deck lost 47% of itself this way.
        """
        markdown = SLIDES.replace("\v", character)
        chunks = BlockSplitter().split(markdown, ALL_PARAMS[params_name])
        for offset in _outside_a_heading_line(markdown):
            assert _covering(chunks, offset), (
                f"{character!r}: offset {offset} ({markdown[offset]!r}) is in no chunk"
            )

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    @pytest.mark.parametrize("params_name", sorted(ALL_PARAMS))
    def test_no_chunk_reaches_beyond_the_blocks_it_packs(self, name: str, params_name: str) -> None:
        """The converse, and the general form of "a chunk carries no heading".

        A block covers every piece of level-0 content there is, so the only
        source text no block covers is whitespace, a heading line and a thematic
        break. A chunk holding a non-whitespace offset that no block holds is
        therefore a chunk that has reached across a section boundary and
        swallowed the marker — which is exactly what rule 2 forbids, and what
        comparing heading *paths* alone does not prevent when two sibling
        sections happen to be named the same.
        """
        markdown = FIXTURES[name]
        blocks = parse_blocks(markdown)
        for chunk in BlockSplitter().split(markdown, ALL_PARAMS[params_name]):
            for offset in range(chunk.start, chunk.end):
                if markdown[offset].isspace():
                    continue
                assert _covering(blocks, offset), (
                    f"{name}: chunk {chunk.start}:{chunk.end} covers offset "
                    f"{offset} ({markdown[offset]!r}), which is in no block"
                )


class TestBlocks:
    """What counts as a block, and what deliberately does not."""

    def test_a_bullet_list_is_one_block_and_not_one_per_item(self) -> None:
        """The nested emission — the item paragraphs at level 2 — is skipped."""
        assert len(parse_blocks(LIST)) == 1

    def test_a_horizontal_rule_yields_no_block(self) -> None:
        """A chunk of ``---`` is noise, so ``hr`` is excluded by name."""
        assert all("---" not in _slice(MISC, span) for span in parse_blocks(MISC))

    def test_blockquote_indented_code_and_html_each_yield_one_block(self) -> None:
        """Not an allow-list of type names: these three are level-0 blocks too."""
        slices = [_slice(MISC, span) for span in parse_blocks(MISC)]
        assert len(slices) == 3
        assert slices[0] == "> A quoted line."
        assert slices[1] == "indented code line one\n    indented code line two"
        assert slices[2] == "<div>\n  <p>html block</p>\n</div>"

    @pytest.mark.parametrize("markdown", [EMPTY, BLANK])
    def test_an_empty_or_whitespace_only_document_yields_no_blocks(self, markdown: str) -> None:
        """No block tokens at all, so no special case is needed."""
        assert parse_blocks(markdown) == []

    @pytest.mark.parametrize("markdown", [EMPTY, BLANK])
    def test_an_empty_or_whitespace_only_document_yields_no_chunks(self, markdown: str) -> None:
        """The packer must survive the empty list rather than index into it."""
        assert BlockSplitter().split(markdown, SMALL) == []

    def test_no_block_covers_a_heading_line(self) -> None:
        """A heading drives the stack and is not itself content."""
        for span in parse_blocks(NESTED):
            assert "#" not in _slice(NESTED, span)

    def test_no_chunk_covers_a_heading_line(self) -> None:
        """The same, after packing — a chunk must not carry its own heading text."""
        for span in BlockSplitter().split(NESTED, SMALL):
            assert "#" not in _slice(NESTED, span)


class TestGfmTables:
    """GFM tables are parsed at all, and arrive as one atomic block."""

    def test_a_gfm_table_is_a_single_block(self) -> None:
        """One block for the whole table — the property no recursive splitter has."""
        assert len(parse_blocks(GFM)) == 1

    def test_an_indented_line_ends_a_table_instead_of_continuing_a_paragraph(self) -> None:
        """The one parse-level guard that only the table rule can satisfy.

        A table absorbs any non-blank line as a row, but breaks on an indented
        one — so with the rule on this document is a table and a code block, and
        with it off the four lines are a single lazily-continued paragraph.
        """
        blocks = parse_blocks(TABLE_THEN_CODE)
        assert len(blocks) == 2
        assert _slice(TABLE_THEN_CODE, blocks[0]).endswith("| Widget |   2 |")
        assert _slice(TABLE_THEN_CODE, blocks[1]) == "an indented line, which a table cannot absorb"

    def test_the_table_block_carries_every_row_of_the_source(self) -> None:
        """Header, delimiter and all five body rows, verbatim."""
        (block,) = parse_blocks(GFM)
        text = _slice(GFM, block)
        assert text.startswith(GFM_HEADER_ROW)
        for name in ("Widget", "Gadget", "Doodad", "Gizmo", "Whatsit"):
            assert name in text


class TestAtomicity:
    """Rule 1 — an atomic block is never split below ``max_chunk_chars``."""

    def test_a_table_below_the_ceiling_is_exactly_one_chunk(self) -> None:
        """It is over the *target* and under the *ceiling*, so it must survive whole."""
        (block,) = parse_blocks(GFM)
        assert SMALL.chunk_chars < block.end - block.start <= SMALL.max_chunk_chars
        chunks = BlockSplitter().split(GFM, SMALL)
        assert len(chunks) == 1
        for name in ("Widget", "Gadget", "Doodad", "Gizmo", "Whatsit"):
            assert name in _slice(GFM, chunks[0])

    def test_a_fenced_block_below_the_ceiling_is_exactly_one_chunk(self) -> None:
        """Both fence delimiters inside one chunk, and the blank line between them."""
        (block,) = parse_blocks(FENCED)
        assert SMALL.chunk_chars < block.end - block.start <= SMALL.max_chunk_chars
        chunks = BlockSplitter().split(FENCED, SMALL)
        assert len(chunks) == 1
        text = _slice(FENCED, chunks[0])
        assert text.startswith("```python")
        assert text.endswith("```")
        assert "# not a heading" in text


class TestHeadingBoundary:
    """Rule 2 — a chunk belongs to exactly one heading path."""

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    @pytest.mark.parametrize("params_name", sorted(ALL_PARAMS))
    def test_no_chunk_reaches_a_block_under_a_different_heading(
        self, name: str, params_name: str
    ) -> None:
        """Stated geometrically: every block the chunk touches shares its path."""
        markdown = FIXTURES[name]
        blocks = parse_blocks(markdown)
        for chunk in BlockSplitter().split(markdown, ALL_PARAMS[params_name]):
            for block in _touching(blocks, chunk):
                assert block.heading_path == chunk.heading_path, (
                    f"{name}: chunk {chunk.start}:{chunk.end} under "
                    f"{chunk.heading_path} reaches a block under {block.heading_path}"
                )

    def test_two_sections_with_the_same_heading_text_are_not_fused(self) -> None:
        """A heading path is not a section identity.

        Both paragraphs sit under ``["A"]`` because both headings read ``A``, so
        a packer comparing paths alone emits one chunk spanning the second
        heading. They are two sections and must be two chunks.
        """
        assert [span.heading_path for span in parse_blocks(SIBLINGS)] == [["A"], ["A"]]
        chunks = BlockSplitter().split(SIBLINGS, SMALL)
        assert len(chunks) == 2
        assert _slice(SIBLINGS, chunks[0]) == "One paragraph here."
        assert _slice(SIBLINGS, chunks[1]) == "Two paragraph here."

    def test_a_thematic_break_also_ends_a_chunk(self) -> None:
        """It falls out of the same test, and is right for a thematic break.

        ``MISC``'s ``hr`` is covered by no block, so a chunk spanning it would
        carry text nothing packed — the blockquote before it and the code after
        it belong to separate chunks.
        """
        for chunk in BlockSplitter().split(MISC, SMALL):
            assert "---" not in _slice(MISC, chunk)


class TestOverlap:
    """Rule 3 — the overlap is carried in whole blocks, never in characters."""

    def test_a_chunk_always_begins_where_some_block_begins(self) -> None:
        """A character-carried overlap would start a chunk mid-sentence."""
        starts = {block.start for block in parse_blocks(MANY)}
        chunks = BlockSplitter().split(MANY, OVERLAPPING)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.start in starts

    def test_the_overlap_actually_carries_something(self) -> None:
        """A budget of 60 over 48-character paragraphs must repeat one of them."""
        chunks = BlockSplitter().split(MANY, OVERLAPPING)
        assert any(later.start < earlier.end for earlier, later in zip(chunks, chunks[1:]))

    def test_a_zero_budget_produces_no_overlap_at_all(self) -> None:
        """``0`` disables it: consecutive chunks are disjoint."""
        chunks = BlockSplitter().split(MANY, SMALL)
        assert len(chunks) > 1
        for earlier, later in zip(chunks, chunks[1:]):
            assert earlier.end <= later.start


class TestTableContinuation:
    """Rule 4 — a table cut at the ceiling repeats its header row in every piece."""

    def test_a_wide_table_is_cut_only_at_row_starts(self) -> None:
        """Never inside a row: every piece opens on a pipe at a line start."""
        (block,) = parse_blocks(WIDE_TABLE)
        assert block.end - block.start > SMALL.max_chunk_chars
        chunks = BlockSplitter().split(WIDE_TABLE, SMALL)
        assert len(chunks) > 1
        for chunk in chunks:
            assert _slice(WIDE_TABLE, chunk).startswith("|")
            assert WIDE_TABLE[chunk.start - 1] == "\n"

    def test_every_piece_composes_with_the_header_row(self) -> None:
        """The first piece contains it; every later piece carries its offsets."""
        chunks = BlockSplitter().split(WIDE_TABLE, SMALL)
        assert WIDE_TABLE_HEADER_ROW in _slice(WIDE_TABLE, chunks[0])
        for chunk in chunks[1:]:
            assert chunk.header_start is not None
            assert chunk.header_end is not None
            assert WIDE_TABLE[chunk.header_start : chunk.header_end] == WIDE_TABLE_HEADER_ROW
        for chunk in chunks:
            assert WIDE_TABLE_HEADER_ROW in _composed(WIDE_TABLE, chunk)

    def test_the_header_offsets_are_none_when_no_table_was_cut(self) -> None:
        """Every span but a table continuation leaves both fields unset."""
        for name in ("NESTED", "GFM", "FENCED", "LIST", "MANY"):
            for chunk in BlockSplitter().split(FIXTURES[name], SMALL):
                assert chunk.header_start is None
                assert chunk.header_end is None


class TestParagraphSubdivision:
    """Rule 1's exception — only a paragraph subdivides, and at good boundaries."""

    def test_a_long_paragraph_subdivides_at_sentence_boundaries(self) -> None:
        """Every piece but the last ends on the punctuation that ended a sentence."""
        (block,) = parse_blocks(LONG_PARA)
        assert block.end - block.start > SMALL.max_chunk_chars
        chunks = BlockSplitter().split(LONG_PARA, SMALL)
        assert len(chunks) > 1
        for chunk in chunks:
            assert _slice(LONG_PARA, chunk).endswith(".")

    def test_a_long_paragraph_keeps_its_heading_path_on_every_piece(self) -> None:
        """A subdivided paragraph inherits the paragraph's path unchanged."""
        for chunk in BlockSplitter().split(LONG_PARA, SMALL):
            assert chunk.heading_path == ["Notes"]

    def test_a_sentence_longer_than_the_ceiling_falls_back_to_whitespace(self) -> None:
        """No terminal punctuation anywhere, so the second boundary set is used."""
        (block,) = parse_blocks(LONG_SENTENCE)
        assert block.end - block.start > SMALL.max_chunk_chars
        chunks = BlockSplitter().split(LONG_SENTENCE, SMALL)
        assert len(chunks) > 1
        for chunk in chunks:
            text = _slice(LONG_SENTENCE, chunk)
            assert text.startswith("word")
            assert text.endswith(tuple("0123456789"))

    def test_an_oversized_indented_code_block_is_emitted_whole(self) -> None:
        """It is re-parsed from its own line start, so it is not mistaken for prose."""
        (block,) = parse_blocks(WIDE_INDENTED)
        assert block.end - block.start > SMALL.max_chunk_chars
        chunks = BlockSplitter().split(WIDE_INDENTED, SMALL)
        assert len(chunks) == 1
        text = _slice(WIDE_INDENTED, chunks[0])
        assert text.startswith("indented code line number 01")
        assert text.endswith("indented code line number 11")
        assert text.count("    indented code line number") == 10

    def test_an_indivisible_block_over_the_ceiling_is_emitted_whole(self) -> None:
        """A fence cut mid-body is worse than a chunk over the ceiling."""
        tight = SMALL.model_copy(update={"chunk_chars": 30, "max_chunk_chars": 40})
        chunks = BlockSplitter().split(FENCED, tight)
        assert len(chunks) == 1
        assert _slice(FENCED, chunks[0]).endswith("```")


class TestShortChunkMerging:
    """``min_chunk_chars`` merges forward, and only when both guards allow it."""

    def test_a_short_chunk_merges_forward_under_the_same_heading(self) -> None:
        """Two blocks under one heading come back as one chunk."""
        assert len(parse_blocks(TINY_SAME)) == 2
        chunks = BlockSplitter().split(TINY_SAME, MERGING)
        assert len(chunks) == 1
        assert _slice(TINY_SAME, chunks[0]).startswith("Tiny.")
        assert _slice(TINY_SAME, chunks[0]).endswith("paragraph.")

    def test_a_short_chunk_before_a_different_heading_stays_short(self) -> None:
        """Rule 2 outranks the minimum — the merge is not unconditional."""
        chunks = BlockSplitter().split(TINY_OTHER, MERGING)
        assert len(chunks) == 2
        assert _slice(TINY_OTHER, chunks[0]) == "Tiny."
        assert chunks[0].end - chunks[0].start < MERGING.min_chunk_chars

    def test_a_merge_that_would_breach_the_ceiling_does_not_happen(self) -> None:
        """The ceiling is hard; the short chunk is emitted as it is."""
        chunks = BlockSplitter().split(TINY_SAME, MERGING_AT_THE_CEILING)
        assert len(chunks) == 2
        assert _slice(TINY_SAME, chunks[0]) == "Tiny."

    def test_a_short_chunk_does_not_merge_across_an_identically_named_heading(self) -> None:
        """The merge is the second door two same-named sections could fuse through.

        Both chunks are under the minimum and both carry ``["S"]``, so the path
        test and the ceiling test both pass — only the adjacency test stops the
        merge from spanning the second ``## S``.
        """
        chunks = BlockSplitter().split(TINY_TWINS, MERGING)
        assert len(chunks) == 2
        assert _slice(TINY_TWINS, chunks[0]) == "Tiny."
        assert _slice(TINY_TWINS, chunks[1]) == "Also tiny."


class TestHeadingPaths:
    """The stack, at every depth and across a skipped level."""

    def test_three_level_nesting_gives_the_expected_paths(self) -> None:
        """Outermost first, one entry per enclosing heading."""
        assert [span.heading_path for span in parse_blocks(NESTED)] == NESTED_PATHS

    def test_content_before_the_first_heading_has_an_empty_path(self) -> None:
        """Legal, and never special-cased into a placeholder or a None."""
        assert [span.heading_path for span in parse_blocks(PREAMBLE)] == [[], ["A"]]

    def test_a_skipped_level_is_not_padded_with_a_placeholder(self) -> None:
        """``#`` then ``###`` gives a path of length two, and ``##`` truncates it."""
        assert [span.heading_path for span in parse_blocks(JUMPED)] == [
            ["A"],
            ["A", "C"],
            ["A", "B"],
        ]

    def test_identical_heading_texts_are_not_deduplicated(self) -> None:
        """Two headings reading ``A`` at two depths give ``["A", "A"]``."""
        assert [span.heading_path for span in parse_blocks(REPEATED)] == [["A"], ["A", "A"]]

    def test_a_setext_heading_behaves_like_any_other(self) -> None:
        """``Title`` over ``=====`` is an ordinary ``heading_open``."""
        blocks = parse_blocks(SETEXT)
        assert [span.heading_path for span in blocks] == [["Title"]]
        assert _slice(SETEXT, blocks[0]) == "Body paragraph."

    def test_a_document_that_starts_below_h1_does_not_deepen_with_every_section(self) -> None:
        """Three ``##`` siblings are three paths of one, not one path of three.

        An extractor routinely produces a document whose top level is ``##``. A
        stack keyed by list position rather than by heading depth leaves the
        first entry in place and yields ``["A"]``, ``["A", "B"]``,
        ``["A", "B", "C"]`` — a hierarchy the document does not have, embedded
        ahead of every chunk by ``prepend_heading_path``.
        """
        assert [span.heading_path for span in parse_blocks(DEEP_ROOT)] == [["A"], ["B"], ["C"]]

    def test_siblings_after_a_skipped_level_replace_rather_than_accumulate(self) -> None:
        """``#`` then ``###`` then ``###``: the second ``###`` replaces the first.

        After the skip, the entry holding ``C`` was pushed at depth three while
        sitting at index one, so only a depth-keyed stack knows that ``D`` closes
        it. The skipped level is still not padded — both paths are length two.
        """
        assert [span.heading_path for span in parse_blocks(JUMPED_SIBLINGS)] == [
            ["A"],
            ["A", "C"],
            ["A", "D"],
        ]

    def test_the_heading_text_comes_from_the_token_and_not_from_a_slice(self) -> None:
        """Slicing the source would keep the ``#`` markers in the path."""
        for span in parse_blocks(NESTED):
            assert all(not part.startswith("#") for part in span.heading_path)


class TestConvergence:
    """Non-termination is impossible by construction, not by a timeout."""

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    @pytest.mark.parametrize("params_name", sorted(ALL_PARAMS))
    def test_chunk_starts_are_strictly_increasing(self, name: str, params_name: str) -> None:
        """The invariant that makes progress a property rather than a hope."""
        chunks = BlockSplitter().split(FIXTURES[name], ALL_PARAMS[params_name])
        starts = [chunk.start for chunk in chunks]
        assert starts == sorted(set(starts))

    def test_the_largest_legal_overlap_still_returns(self) -> None:
        """``chunk_chars - 1`` is accepted, and a multi-block document splits."""
        chunks = BlockSplitter().split(MANY, CONVERGING)
        assert len(chunks) > 1


class TestValidator:
    """The configuration is rejected where the operator can read the message."""

    def test_an_overlap_equal_to_the_target_is_rejected(self) -> None:
        """At the target the overlap consumes the whole previous chunk."""
        with pytest.raises(ValueError, match="chunk_overlap_chars"):
            WorkspaceRagIndex(chunk_chars=1200, chunk_overlap_chars=1200)

    def test_an_overlap_above_the_target_is_rejected(self) -> None:
        """Above it, worse still."""
        with pytest.raises(ValueError, match="1500"):
            WorkspaceRagIndex(chunk_chars=1200, chunk_overlap_chars=1500)

    def test_an_overlap_one_below_the_target_is_accepted(self) -> None:
        """The largest legal value, and it must remain legal."""
        assert WorkspaceRagIndex(chunk_chars=1200, chunk_overlap_chars=1199).chunk_overlap_chars

    def test_a_minimum_above_the_target_is_rejected(self) -> None:
        """A target outside its own bounds has no meaning."""
        with pytest.raises(ValueError, match="min_chunk_chars"):
            WorkspaceRagIndex(chunk_chars=1200, min_chunk_chars=1300)

    def test_a_target_above_the_ceiling_is_rejected(self) -> None:
        """The same, at the other end."""
        with pytest.raises(ValueError, match="max_chunk_chars"):
            WorkspaceRagIndex(chunk_chars=5000, max_chunk_chars=4000)


class TestDefaults:
    """The parameter class carries the ADR's numbers and round-trips."""

    def test_the_defaults_are_the_documented_ones(self) -> None:
        """1200 / 150 / 4000 / 200 / True."""
        params = WorkspaceRagIndex()
        assert params.chunk_chars == 1200
        assert params.chunk_overlap_chars == 150
        assert params.max_chunk_chars == 4000
        assert params.min_chunk_chars == 200
        assert params.prepend_heading_path is True

    def test_it_is_exposed_on_the_tool_call_and_command_channels(self) -> None:
        """What the ADR specifies for ``workspace_rag_index``; inert until 45-7."""
        assert WorkspaceRagIndex().expose == {TOOL_CALL, COMMAND}

    def test_it_round_trips_through_dump_and_validate(self) -> None:
        """Fully serialisable: primitives only, no runtime state.

        Compared as **models**, never as two dumps. ``expose`` is a ``set``, and
        ``serialize()`` renders a set as a *list* in whichever order that set
        happens to iterate — an order Python's per-process hash randomisation
        can flip when the two members collide in the table. Comparing the dumps
        therefore fails on roughly one interpreter seed in twelve
        (``PYTHONHASHSEED=2`` reproduces it), for no behavioural reason. The
        model comparison is both stable and stronger: it checks what came back,
        not how it was written down.
        """
        params = WorkspaceRagIndex(chunk_chars=800, chunk_overlap_chars=100)
        again = WorkspaceRagIndex.model_validate(params.model_dump())
        assert again == params

    def test_the_splitter_never_reads_prepend_heading_path(self) -> None:
        """It is the embedder's flag; flipping it must change nothing here."""
        off = SMALL.model_copy(update={"prepend_heading_path": False})
        assert BlockSplitter().split(NESTED, off) == BlockSplitter().split(NESTED, SMALL)


class TestSpanModel:
    """``Span`` is a model, not a tuple, and it serialises unaided."""

    def test_a_span_round_trips_through_dump_and_validate(self) -> None:
        """Every field is a primitive or a list of primitives."""
        span = Span(start=3, end=9, heading_path=["A", "B"], header_start=0, header_end=2)
        again = Span.model_validate(span.model_dump())
        assert again.model_dump() == span.model_dump()

    def test_the_header_offsets_default_to_none(self) -> None:
        """Unset on every span but a table continuation."""
        span = Span(start=0, end=1)
        assert span.header_start is None
        assert span.header_end is None
        assert span.heading_path == []


class _WholeDocumentSplitter:
    """A second implementation, inheriting nothing and defined right here.

    This is the whole point of a structural Protocol: it satisfies
    :class:`TextSplitter` because its ``split`` matches, not because it was
    declared to. The only thing it imports from the module under test is the
    return type its signature is obliged to name.
    """

    def split(self, markdown: str, params: WorkspaceRagIndex) -> list[Span]:
        """Return the whole document as one chunk, or nothing for a blank one."""
        return [Span(start=0, end=len(markdown))] if markdown.strip() else []


# The structural check mypy performs, which ``@runtime_checkable`` does not: an
# assignment that fails type checking if either class's ``split`` drifts.
_CHECKED: TextSplitter = BlockSplitter()
_ALSO_CHECKED: TextSplitter = _WholeDocumentSplitter()


class TestProtocolConformance:
    """Checked twice, because ``@runtime_checkable`` alone is inert."""

    def test_the_block_splitter_satisfies_the_protocol_at_runtime(self) -> None:
        """Member presence — all ``isinstance`` against a Protocol can see."""
        assert isinstance(BlockSplitter(), TextSplitter)

    def test_a_foreign_implementation_satisfies_it_too(self) -> None:
        """No base class, no registration: the extension point works."""
        assert isinstance(_WholeDocumentSplitter(), TextSplitter)

    def test_an_object_without_split_does_not_satisfy_it(self) -> None:
        """The check is not vacuous."""
        assert not isinstance(object(), TextSplitter)

    def test_the_module_level_annotations_bind_the_two_implementations(self) -> None:
        """mypy is the real signature guard; these two names are what it checks."""
        assert _CHECKED.split(NESTED, SMALL)
        assert _ALSO_CHECKED.split(NESTED, SMALL)


class TestPhasesCompose:
    """``split`` is exactly ``pack_blocks(parse_blocks(...))`` and nothing more."""

    @pytest.mark.parametrize("name", sorted(FIXTURES))
    def test_the_splitter_is_the_composition_of_its_two_phases(self, name: str) -> None:
        """The two module-public phases are reachable and are what ``split`` runs."""
        markdown = FIXTURES[name]
        assert BlockSplitter().split(markdown, SMALL) == pack_blocks(
            parse_blocks(markdown), markdown, SMALL
        )
