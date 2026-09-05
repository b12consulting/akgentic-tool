"""Markdown to embeddable chunks, in two phases: parse to blocks, pack blocks.

**Phase 1 — :func:`parse_blocks`.** ``markdown-it-py`` gives every block token a
``.map`` of ``[start_line, end_line]``; a line-start index turns that into
character offsets. What comes back is one :class:`Span` per structural block,
with a table and a fenced block each arriving as **one atomic block** — the
property no character-recursive splitter can have.

**Phase 2 — :func:`pack_blocks`.** Blocks are packed into chunks under four
rules, none of them configurable (ADR-045 §6): an atomic block is never split
below ``max_chunk_chars``; a chunk never crosses a heading boundary; overlap is
carried in whole blocks, so a chunk never starts mid-sentence; and a table cut at
the ceiling repeats its header row in every piece.

**Why the offsets are the point.** A chunk is a pair of offsets into the
extraction, never a copy of it. That is what lets a stored chunk stay small, stay
in sync with the body it describes, and be re-read verbatim from the document at
retrieval time. It is also why this splitter is ours: the obvious third-party
choice was probed and rejected because its stage-1 output is **not verbatim** —
``MarkdownHeaderTextSplitter`` joins lines with a Markdown hard break, strips
indentation and collapses blank lines, so the text it hands back cannot be found
in the source at all, and its ``add_start_index`` indexes the string handed in
rather than the document (ADR-045 *Rejected alternatives*).

Nothing here touches an actor, a message or the filesystem: two pure functions, a
Protocol, and the class that composes them.

**The import edge runs one way.**
:class:`~akgentic.tool.workspace.card.params.WorkspaceRagIndex` is imported under
``TYPE_CHECKING`` only, and that is load-bearing rather than stylistic:
``card/__init__.py`` imports :mod:`akgentic.tool.workspace.documents.models`,
which executes this package's ``__init__`` first — so a runtime import of
``card.params`` from here would close a cycle whose winner depends on which
package is imported first. The annotation is a string, the runtime edge does not
exist, and the packer reads ``params.chunk_chars`` without needing the class.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from markdown_it import MarkdownIt

from akgentic.core.utils.serializer import SerializableBaseModel

if TYPE_CHECKING:
    from markdown_it.token import Token

    from akgentic.tool.workspace.card.params import WorkspaceRagIndex

__all__ = ["BlockSplitter", "Span", "TextSplitter"]

_SENTENCE_END = re.compile(r"[.!?…][\"')\]]*\s")
"""A sentence boundary, deliberately naive: terminal punctuation, optional
closing quotes or brackets, then whitespace. No abbreviation handling and no
dependency to get it — a missed boundary costs one slightly worse cut, because
the whitespace fallback catches everything this misses."""

_WHITESPACE = re.compile(r"\s+")

_NON_BLOCK_TYPES = frozenset({"heading_open", "hr"})
"""Level-0 mapped tokens that are **not** blocks.

``heading_open`` drives the heading stack and its text is deliberately covered by
no span — ``prepend_heading_path`` re-supplies it at embed time. ``hr``'s whole
slice is ``---``, and a chunk of a horizontal rule is noise.
"""


class Span(SerializableBaseModel):
    """A region of one Markdown document, with the heading context around it.

    Every field is an ``int``, a ``str`` or a list of them, so the model
    round-trips through Pydantic unaided: no ``arbitrary_types_allowed`` of its
    own and no ``PrivateAttr``.

    A ``Span`` has no identity, no ordinal and no status — it is a coordinate
    pair with context. The persisted, identified form is 45-7's ``RagChunk``.

    Attributes:
        start: Character offset into the Markdown string handed to
            :meth:`TextSplitter.split` — never a line number and never a byte
            offset.
        end: Exclusive end offset. ``markdown[start:end]`` is a non-empty,
            verbatim region of that same string, carrying no surrounding
            whitespace.
        heading_path: The enclosing heading texts, outermost first —
            ``["Invoice", "Payment terms", "Late fees"]``. Empty for content
            before the first heading, which is a legal path and not a missing
            one. A skipped level is not padded: ``#`` then ``###`` gives a path
            of length two.
        header_start: Offset of a table's own header row, for a **continuation
            piece** of a table that rule 4 cut at the ceiling. ``None`` on every
            other span.

            Offsets rather than text, so "a chunk is offsets, never a copy"
            survives rule 4 intact: the header rides the same embed-time
            composition ``prepend_heading_path`` already uses. It is the one
            thing in a composed chunk that is not contiguous with ``[start:end)``.
        header_end: Exclusive end of that header row. Set exactly when
            ``header_start`` is.
    """

    start: int
    end: int
    heading_path: list[str] = []
    header_start: int | None = None
    header_end: int | None = None


@runtime_checkable
class TextSplitter(Protocol):
    """Turn one document's Markdown into chunks, as offsets into that Markdown.

    Structural: any object with a matching ``split`` satisfies it, with no base
    class to inherit and no import of this package from an implementation.
    Replacing :class:`BlockSplitter` is one class.

    ``@runtime_checkable`` is the package convention, and it buys less than it
    looks like it does: ``isinstance`` against a Protocol checks **member
    presence, not signatures**, so an object with a ``split`` taking the wrong
    arguments passes it. The real conformance guard is mypy, and a spec should
    exercise both.
    """

    def split(self, markdown: str, params: WorkspaceRagIndex) -> list[Span]:
        """Return the chunks of *markdown*, as spans into *markdown* itself."""
        ...


def _parser() -> MarkdownIt:
    """A CommonMark parser with GFM tables enabled.

    ``.enable("table")`` is required rather than decorative: without it no
    ``table_open`` token is produced at all and a GFM table degrades into
    paragraphs, which would make the atomic-table rule silently inert. MarkItDown
    produces our input, so the dialect is ours to fix rather than guess.
    """
    return MarkdownIt("commonmark").enable("table")


def _line_starts(text: str) -> list[int]:
    """Character offset of the start of every line, plus ``len(text)``.

    ``keepends=True`` is the whole of it: without it every offset past the first
    line is short by the number of preceding newlines, which is the classic
    offset bug this module exists to avoid. The result has ``n_lines + 1``
    entries and ends at ``len(text)``, so a token's exclusive ``map[1]`` always
    indexes in range — including for a document with no trailing newline. No
    clamp, which would hide a real indexing bug.
    """
    starts = [0]
    for line in text.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    return starts


def _trimmed(start: int, end: int, markdown: str) -> tuple[int, int] | None:
    """Shrink ``[start, end)`` off its surrounding whitespace, or ``None``.

    ``.map``'s end line is exclusive, so a block's raw slice carries its own
    trailing newline and — for the last item of a list — the blank line after it.
    An indented code block symmetrically opens on its indentation. The trimmed
    region is still an exact, verbatim, contiguous region of *markdown*; it
    simply no longer begins or ends in whitespace.

    Returns:
        The trimmed bounds, or ``None`` when nothing but whitespace is left.
    """
    raw = markdown[start:end]
    lead = len(raw) - len(raw.lstrip())
    tail = len(raw) - len(raw.rstrip())
    if lead + tail >= len(raw):
        return None
    return start + lead, end - tail


def _piece(span: Span, start: int, end: int, markdown: str) -> Span | None:
    """Derive a sub-span of *span* over ``[start, end)``, trimmed.

    Golden Rule #12: copy and override the two fields that change, so the
    heading path and any header offsets survive by construction rather than by
    being listed.
    """
    bounds = _trimmed(start, end, markdown)
    if bounds is None:
        return None
    return span.model_copy(update={"start": bounds[0], "end": bounds[1]})


def _block_map(token: Token) -> list[int] | None:
    """The line map of *token* when it is a structural block, else ``None``.

    One narrowing point for ``Token.map``'s ``list[int] | None``, so the caller
    indexes a map it was handed rather than re-checking one it was not — no
    ``assert`` and no ``# type: ignore`` anywhere on the path.

    ``level == 0`` is what skips the **nested emission**: the paragraphs inside a
    ``bullet_list`` arrive at level 2 and would otherwise be counted a second
    time, once as part of the list and once on their own. It is deliberately not
    an allow-list of type names — ``blockquote_open``, indented ``code_block``
    and ``html_block`` are all real level-0 blocks and all appear in extracted
    documents, and an allow-list would drop them silently.
    """
    if token.map is None or token.level != 0 or token.nesting == -1:
        return None
    if token.type in _NON_BLOCK_TYPES:
        return None
    return token.map


def parse_blocks(markdown: str) -> list[Span]:
    """Phase 1 — parse *markdown* into structural blocks with exact offsets.

    Module-public but deliberately **not** re-exported from the package façade:
    it is reached by its full module path, the precedent ``evict_document_bodies``
    set. :class:`BlockSplitter` is what a caller outside this package names.

    The heading stack is maintained over ``heading_open`` and the ``inline``
    token that follows it. The text comes from ``inline.content`` rather than
    from slicing the source, which would keep the ``#`` markers.
    ``del path[depth - 1:]`` truncates to the new depth, which is why a skipped
    level yields a shorter path rather than a padded one.

    Args:
        markdown: The document body. Empty and whitespace-only inputs produce no
            block tokens at all, so they return ``[]`` with no special case.

    Returns:
        One span per block, in document order. Heading lines are covered by no
        span, and neither are the blank lines between blocks.
    """
    starts = _line_starts(markdown)
    blocks: list[Span] = []
    path: list[str] = []
    depth: int | None = None
    for token in _parser().parse(markdown):
        if token.type == "heading_open":
            depth = int(token.tag[1])
        elif depth is not None and token.type == "inline":
            del path[depth - 1 :]
            path.append(token.content)
            depth = None
        lines = _block_map(token)
        if lines is None:
            continue
        bounds = _trimmed(starts[lines[0]], starts[lines[1]], markdown)
        if bounds is not None:
            blocks.append(Span(start=bounds[0], end=bounds[1], heading_path=list(path)))
    return blocks


def _structure(span: Span, markdown: str) -> tuple[str, list[int], tuple[int, int] | None]:
    """Re-parse one oversized block to learn its kind and where it may be cut.

    Only reached for a block past ``max_chunk_chars``, which is rare, and it
    answers three questions no :class:`Span` carries: what kind of block this is,
    at which offsets it may legally be cut, and — for a table — where its own
    header row sits.

    The re-parse starts at the beginning of the block's **line**, not at
    ``span.start``: the span was trimmed off its leading whitespace, and an
    indented code block re-parsed without its indentation would come back as a
    paragraph and then be cut like one.

    Returns:
        The first level-0 token type, the legal cut offsets inside the span (row
        boundaries for a table, item boundaries for a list, none otherwise), and
        the table's header-row bounds.
    """
    origin = markdown.rfind("\n", 0, span.start) + 1
    if markdown[origin : span.start].strip():
        origin = span.start
    text = markdown[origin : span.end]
    starts = _line_starts(text)
    tokens = _parser().parse(text)
    kind = next((t.type for t in tokens if t.level == 0 and t.map is not None), "")
    rows = [t.map for t in tokens if t.type == "tr_open" and t.map is not None]
    items = [t.map for t in tokens if t.type == "list_item_open" and t.map is not None]
    # The first row of a GFM table is its header, so it is never a cut point; the
    # first item of a list opens the block, so neither is that.
    lines = [bounds[0] for bounds in (rows or items)[1:]]
    cuts = [offset for offset in (origin + starts[line] for line in lines) if offset > span.start]
    header = None
    if rows:
        header = _trimmed(origin + starts[rows[0][0]], origin + starts[rows[0][1]], markdown)
    return kind, cuts, header


def _with_header(piece: Span, header: tuple[int, int] | None, carry: bool) -> Span:
    """Stamp a table's header-row offsets onto a continuation piece.

    The first piece already contains the header inside its own slice, so it
    carries nothing; every piece after it does.
    """
    if header is None or not carry:
        return piece
    return piece.model_copy(update={"header_start": header[0], "header_end": header[1]})


def _cut_greedy(
    span: Span,
    boundaries: list[int],
    markdown: str,
    params: WorkspaceRagIndex,
    header: tuple[int, int] | None = None,
) -> list[Span]:
    """Cut *span* into pieces at the largest *boundaries* that fit the target.

    The one cutting loop, shared by the structural path (table rows, list items)
    and the textual one (sentences, then whitespace). It never cuts anywhere but
    at a supplied boundary, which is what makes "never inside a row" and "never
    mid-word" properties rather than hopes: with no usable boundary the whole
    span comes back oversized, which rule 1 prefers to a bad cut.

    A repeated header costs the continuation pieces part of their budget, so the
    composed piece stays inside the same ceiling as an uncut one.
    """
    header_len = 0 if header is None else header[1] - header[0] + 1
    pieces: list[Span] = []
    current = span.start
    remaining = [boundary for boundary in boundaries if boundary > span.start]
    while remaining:
        budget = params.chunk_chars - (header_len if pieces else 0)
        if span.end - current <= budget:
            break
        fitting = [boundary for boundary in remaining if boundary - current <= budget]
        cut = fitting[-1] if fitting else remaining[0]
        piece = _piece(span, current, cut, markdown)
        if piece is not None:
            pieces.append(_with_header(piece, header, bool(pieces)))
        current = cut
        remaining = [boundary for boundary in remaining if boundary > cut]
    tail = _piece(span, current, span.end, markdown)
    if tail is not None:
        pieces.append(_with_header(tail, header, bool(pieces)))
    return pieces or [span]


def _boundaries(pattern: re.Pattern[str], span: Span, markdown: str) -> list[int]:
    """Absolute offsets just past each match of *pattern* inside *span*.

    The cut lands **after** the separator, so the piece before it ends on content
    and the piece after it begins on content; the separator itself is then
    trimmed off both. Only whitespace ever falls between two adjacent pieces.
    """
    length = span.end - span.start
    return [
        span.start + match.end()
        for match in pattern.finditer(markdown[span.start : span.end])
        if 0 < match.end() < length
    ]


def _subdivide_paragraph(span: Span, markdown: str, params: WorkspaceRagIndex) -> list[Span]:
    """Cut an oversized paragraph at sentence boundaries, then at whitespace.

    Sentences first, because a chunk that begins mid-sentence reads as noise to
    an embedding model; whitespace second, for the piece a single long sentence
    leaves behind. Never mid-word: a stretch with no whitespace in it at all is
    emitted oversized.
    """
    pieces = _cut_greedy(span, _boundaries(_SENTENCE_END, span, markdown), markdown, params)
    subdivided: list[Span] = []
    for piece in pieces:
        if piece.end - piece.start > params.max_chunk_chars:
            words = _boundaries(_WHITESPACE, piece, markdown)
            subdivided.extend(_cut_greedy(piece, words, markdown, params))
        else:
            subdivided.append(piece)
    return subdivided


def _split_oversized(span: Span, markdown: str, params: WorkspaceRagIndex) -> list[Span]:
    """Apply rule 1: cut a block only when it passes ``max_chunk_chars``.

    Below the ceiling every block is emitted whole, whatever its kind — that is
    the whole of "an atomic block is never split below ``max_chunk_chars``", and
    it is why a table under the ceiling is exactly one chunk.

    At the ceiling, what happens depends on what the block is:

    - a **table** cuts at row boundaries and repeats its header (rule 4);
    - a **list** cuts between items, never inside one;
    - a **paragraph** cuts at sentence, then whitespace boundaries;
    - anything else — a fence, an indented code block, an html block, a
      blockquote — is emitted whole and oversized. A function cut mid-body is
      worse than a chunk over the ceiling.
    """
    if span.end - span.start <= params.max_chunk_chars:
        return [span]
    kind, cuts, header = _structure(span, markdown)
    if cuts:
        return _cut_greedy(span, cuts, markdown, params, header)
    if kind == "paragraph_open":
        return _subdivide_paragraph(span, markdown, params)
    return [span]


def _heading_groups(units: list[Span]) -> list[list[Span]]:
    """Split *units* into runs sharing one heading path — rule 2's whole of it.

    Packing then happens inside a run and never across one, so a chunk belongs to
    exactly one heading path by construction. Under-full chunks are the price and
    are accepted: packing efficiency loses to retrieval precision every time.
    """
    groups: list[list[Span]] = []
    for unit in units:
        if groups and groups[-1][0].heading_path == unit.heading_path:
            groups[-1].append(unit)
        else:
            groups.append([unit])
    return groups


def _fill(group: list[Span], start: int, params: WorkspaceRagIndex) -> int:
    """Index one past the last unit that fits in a chunk starting at *start*.

    ``chunk_chars`` is a soft target: packing stops at the **first** unit that
    would take the chunk past it, and a chunk always holds at least one unit even
    when that one unit is larger than the target.
    """
    size = 0
    stop = start
    while stop < len(group):
        length = group[stop].end - group[stop].start
        if stop > start and size + length > params.chunk_chars:
            break
        size += length
        stop += 1
    return stop


def _overlap_start(group: list[Span], start: int, stop: int, params: WorkspaceRagIndex) -> int:
    """Index the next chunk starts at — rule 3, in whole blocks.

    ``chunk_overlap_chars`` is a **budget**, not a cut point: the next chunk
    begins with as many of this chunk's trailing units as fit inside it, so it
    always begins where a block begins and therefore never mid-sentence.

    **Termination is by construction, not by a timeout.** The result is strictly
    greater than *start* whatever the budget, so every chunk starts strictly
    after the previous one and the loop advances even at the largest legal
    overlap. ``0`` disables overlap and returns *stop*.
    """
    carried = 0
    index = stop
    while index - 1 > start:
        length = group[index - 1].end - group[index - 1].start
        if carried + length > params.chunk_overlap_chars:
            break
        carried += length
        index -= 1
    return index


def _join(units: list[Span]) -> Span:
    """Fuse a run of units into the one span that is the chunk.

    The heading path is the run's — rule 2 guarantees there is exactly one. The
    header offsets come from the first unit that carries any: overlap can begin a
    chunk on a unit that is not a table continuation while a later unit is one,
    and losing the header there would cost that piece its readability for nothing.
    """
    header = next((unit for unit in units if unit.header_start is not None), units[0])
    return units[0].model_copy(
        update={
            "end": units[-1].end,
            "header_start": header.header_start,
            "header_end": header.header_end,
        }
    )


def _pack_group(group: list[Span], params: WorkspaceRagIndex) -> list[Span]:
    """Pack one heading run into chunks, carrying the overlap between them."""
    chunks: list[Span] = []
    start = 0
    while start < len(group):
        stop = _fill(group, start, params)
        chunks.append(_join(group[start:stop]))
        if stop >= len(group):
            break
        start = _overlap_start(group, start, stop, params)
    return chunks


def _merge_short_chunks(chunks: list[Span], params: WorkspaceRagIndex) -> list[Span]:
    """Merge a chunk below ``min_chunk_chars`` into the next one — twice guarded.

    **Forward only, and only under the same heading path.** A chunk below the
    minimum whose next sibling sits under a *different* heading stays small: rule
    2 outranks this, and a reader would otherwise take the merge for
    unconditional.

    **A merge may exceed ``chunk_chars`` and may never exceed
    ``max_chunk_chars``.** The target is soft and the ceiling is hard; a merge
    that would breach the ceiling does not happen and the short chunk is emitted
    as it is. Without that, a document of many tiny sections merges into one
    oversized chunk the embedding model then rejects.
    """
    merged: list[Span] = []
    for chunk in chunks:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and previous.end - previous.start < params.min_chunk_chars
            and previous.heading_path == chunk.heading_path
            and chunk.end - previous.start <= params.max_chunk_chars
        ):
            update: dict[str, int | None] = {"end": max(previous.end, chunk.end)}
            if previous.header_start is None:
                update["header_start"] = chunk.header_start
                update["header_end"] = chunk.header_end
            merged[-1] = previous.model_copy(update=update)
            continue
        merged.append(chunk)
    return merged


def pack_blocks(blocks: list[Span], markdown: str, params: WorkspaceRagIndex) -> list[Span]:
    """Phase 2 — pack *blocks* into chunks under the four rules.

    Module-public but deliberately **not** re-exported, exactly as
    :func:`parse_blocks` is not.

    Three passes, one per concern, so no rule is buried inside another's loop:
    every block is first cut if and only if it passes the ceiling, the result is
    grouped by heading path and packed with overlap, and chunks still under the
    minimum merge forward.

    Args:
        blocks: The output of :func:`parse_blocks`, in document order.
        markdown: The same string those offsets index into.
        params: The chunking configuration, already validated.

    Returns:
        The chunks, in document order, with strictly increasing ``start``
        offsets. Every non-whitespace offset inside a block is covered by the
        union of the chunks; overlap means an offset may be covered twice.
    """
    units: list[Span] = []
    for block in blocks:
        units.extend(_split_oversized(block, markdown, params))
    chunks: list[Span] = []
    for group in _heading_groups(units):
        chunks.extend(_pack_group(group, params))
    return _merge_short_chunks(chunks, params)


class BlockSplitter:
    """The block-structural :class:`TextSplitter`: parse to blocks, pack blocks.

    Declares **no** base class — it satisfies :class:`TextSplitter` structurally,
    which is the point of a Protocol. It is a plain class rather than a Pydantic
    model because it carries no configuration at all: every knob arrives in
    *params*, so there is no state to serialise and two instances are
    interchangeable.
    """

    def split(self, markdown: str, params: WorkspaceRagIndex) -> list[Span]:
        """Return the chunks of *markdown*, as spans into *markdown* itself.

        Args:
            markdown: One document's extracted body.
            params: The chunking configuration.

        Returns:
            The chunks, in document order. Empty for an empty or whitespace-only
            document.
        """
        return pack_blocks(parse_blocks(markdown), markdown, params)
