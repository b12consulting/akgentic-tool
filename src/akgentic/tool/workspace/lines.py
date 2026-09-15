"""The one definition of a line, and the two functions that count by it.

Two capabilities number the lines of a document. The retrieval splitter indexes
``markdown-it``'s token maps, so it must count lines exactly as the parser does;
the read path numbers a gutter for the agent. They held **two** definitions of a
line break and the two disagreed — so a line number minted by one half of the
workspace named a different region when the other half was handed it. What lives
here is the one definition and the two views of it, written once.

**This module imports nothing but the standard library**, and that is
load-bearing rather than tidy, exactly as it is for
:mod:`akgentic.tool.workspace.locks`. ``read/`` may not import ``rag/``'s private
helper and ``rag/`` may not import ``read/``'s: a capability reaching into a peer
is a runtime edge between two modules that are meant to be deletable
independently, and it is precisely the edge this module removes. A module naming
only ``re`` is importable by every capability and creates none.

**It holds nothing else.** No gutter format, no token map, no chunk, no offset
arithmetic over either — those belong to the two capabilities and are what keep
them two. What is shared is *where a line begins*, and that is all that is here.

The invariant the two functions are two faces of, and the reason they live in one
module rather than two::

    len(line_starts(t)) == len(split_lines(t)) + 1      for every t
"""

from __future__ import annotations

import re

_LINE_BREAK = re.compile(r"\r\n?|\n")
"""A line break **as the parser counts them** — its own ``NEWLINES_RE``.

Deliberately not ``str.splitlines``, which is the wider definition: it also
breaks on ``\\v``, ``\\f``, ``\\x1c``-``\\x1e``, ``\\x85``, ``\\u2028`` and
``\\u2029``, none of which start a line for ``markdown-it``. One of those in the
document and the two disagree about how many lines there are, so every token map
past it indexes the wrong line and every offset after it is silently wrong.

It is not a theoretical hazard: ``python-pptx`` renders a soft line break
(``<a:br>``) as a vertical tab, so every extracted deck with a wrapped title
carries several.

Module-private: only the two functions below have callers, and nothing outside
this module needs the pattern object itself. Exactly one of it exists in the
package, which is the whole point.
"""


def split_lines(text: str) -> list[str]:
    """The lines of *text*, counted by :data:`_LINE_BREAK` and nothing wider.

    **Exactly one trailing empty element is dropped, and only when it is empty.**
    A document ending in a break has one line per break, not a phantom empty one
    after the last — but ``"a\\n\\n"`` is genuinely two lines, an ``"a"`` and an
    empty one, and stays two. That is why this pops one element rather than
    looping or stripping: both wider forms flatten that body to one line and
    break the invariant.

    The rule is what makes this agree with :func:`line_starts`, which appends its
    terminating ``len(text)`` only when it is not already there:
    ``len(line_starts(t)) == len(split_lines(t)) + 1`` for every *t*.

    Args:
        text: Any document body. The empty string has no lines and returns ``[]``.

    Returns:
        The lines, carrying none of the breaks that ended them.
    """
    parts = _LINE_BREAK.split(text)
    if parts and parts[-1] == "":
        parts.pop()
    return parts


def line_starts(text: str) -> list[int]:
    """Character offset of the start of every line, plus ``len(text)``.

    Lines are counted with :data:`_LINE_BREAK` — the parser's own definition —
    because this list is indexed by *the parser's* line numbers. Any wider
    definition puts the two out of step and every offset after the first extra
    break is wrong; see :data:`_LINE_BREAK` for which characters do that and
    where they come from.

    The result ends at ``len(text)``, so a token's exclusive ``map[1]`` always
    indexes in range — including for a document with no trailing newline. No
    clamp, which would hide a real indexing bug.

    Appended **only when it is not already the last element**, which is what
    keeps this in step with :func:`split_lines`: ``"a\\nb\\nc"`` and
    ``"a\\nb\\nc\\n"`` are both three lines and both give four starts, so
    ``len(line_starts(t)) == len(split_lines(t)) + 1`` holds either way.
    """
    starts = [0]
    starts.extend(match.end() for match in _LINE_BREAK.finditer(text))
    if starts[-1] != len(text):
        starts.append(len(text))
    return starts
