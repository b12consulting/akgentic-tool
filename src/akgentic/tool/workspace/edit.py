"""Workspace edit utilities — EditMatcher, line ending helpers, unified diff patch.

All functionality required by ``workspace_edit``, ``workspace_multi_edit`` and
``workspace_patch`` lives here. This module has no dependency on ``tool.py`` or
``ToolCard`` — it is pure algorithmic logic, consumed by ``WorkspaceActor``,
which is where the mutations themselves now run (ADR-036 §3).

The substitution and patch helpers moved here from ``tool.py`` for that reason:
``tool.py`` imports ``actor.py``, so leaving them where they were would have made
the actor's import of them a cycle Python refuses at import time.
"""

from __future__ import annotations

import codecs
import difflib
import re
import textwrap
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from akgentic.tool.workspace.workspace import Workspace


@dataclass
class MatchResult:
    start: int  # byte offset in content where match begins
    end: int  # byte offset in content where match ends (exclusive)
    strategy: str  # name of winning strategy, e.g. "exact", "line_trimmed"


class EditMatcher:
    """7-strategy cascade for locating old_string in file content.

    Strategies are tried in order. The first match wins.
    All strategies return MatchResult(start, end, strategy) or None.
    """

    FUZZY_THRESHOLD: float = 0.85

    def find(self, content: str, old_string: str, exact_only: bool = False) -> MatchResult | None:
        """Locate *old_string* in *content*, first match wins.

        Args:
            content: The file's current text.
            old_string: The anchor to locate.
            exact_only: Run strategy 1 alone, skipping the six approximate ones.
                The write gate selects this when the file changed since the
                editing agent read it: approximate matching against text another
                agent has just rewritten is how a plausible edit lands in the
                wrong place (ADR-036 §3, FR6).

        Returns:
            The winning match, or ``None`` when no strategy matched.
        """
        cascade = (
            self._exact,
            self._line_trimmed,
            self._whitespace_normalised,
            self._dedented,
            self._trimmed_boundary,
            self._escape_normalised,
            self._fuzzy,
        )
        for strategy in cascade[:1] if exact_only else cascade:
            result = strategy(content, old_string)
            if result is not None:
                return result
        return None

    # --- Strategy 1: exact ---------------------------------------------------

    def _exact(self, content: str, old: str) -> MatchResult | None:
        idx = content.find(old)
        if idx == -1:
            return None
        return MatchResult(start=idx, end=idx + len(old), strategy="exact")

    # --- Strategy 2: line-trimmed --------------------------------------------

    def _line_trimmed(self, content: str, old: str) -> MatchResult | None:
        """Strip per-line leading/trailing whitespace from old, then exact-match."""
        stripped = "\n".join(line.strip() for line in old.splitlines())
        if stripped == old:
            return None  # no change — nothing new to try
        norm_content = "\n".join(line.strip() for line in content.splitlines())
        idx = norm_content.find(stripped)
        if idx == -1:
            return None
        # Remap idx back to original content
        return self._remap(content, old, idx, norm_content, strategy="line_trimmed")

    # --- Strategy 3: whitespace-normalised -----------------------------------

    def _whitespace_normalised(self, content: str, old: str) -> MatchResult | None:
        """Collapse internal whitespace runs to single space, then exact-match."""
        norm_old = re.sub(r"[ \t]+", " ", old)
        if norm_old == old:
            return None
        norm_content = re.sub(r"[ \t]+", " ", content)
        idx = norm_content.find(norm_old)
        if idx == -1:
            return None
        return self._remap(content, old, idx, norm_content, strategy="whitespace_normalised")

    # --- Strategy 4: dedented ------------------------------------------------

    def _dedented(self, content: str, old: str) -> MatchResult | None:
        """Dedent old_string, then exact-match against dedented content windows."""
        dedented_old = textwrap.dedent(old)
        if dedented_old == old:
            return None
        # Try matching dedented_old against lines in content with various indent levels
        idx = content.find(dedented_old)
        if idx != -1:
            return MatchResult(start=idx, end=idx + len(dedented_old), strategy="dedented")
        # Also try: dedent old, dedent content
        dedented_content = textwrap.dedent(content)
        idx = dedented_content.find(dedented_old)
        if idx == -1:
            return None
        return self._remap(content, old, idx, dedented_content, strategy="dedented")

    # --- Strategy 5: trimmed boundary ----------------------------------------

    def _trimmed_boundary(self, content: str, old: str) -> MatchResult | None:
        """Strip blank lines at edges of old_string, then exact-match."""
        stripped = old.strip("\n")
        if stripped == old:
            return None  # no blank edges
        result = self._exact(content, stripped)
        if result is None:
            return None
        return MatchResult(start=result.start, end=result.end, strategy="trimmed_boundary")

    # --- Strategy 6: escape-normalised ---------------------------------------

    def _escape_normalised(self, content: str, old: str) -> MatchResult | None:
        """Decode escape sequences in old_string, then exact-match."""
        try:
            decoded = codecs.decode(old.encode(), "unicode_escape")
            # In Python 3, codecs.decode with 'unicode_escape' returns str directly
            norm_old: str = (
                decoded if isinstance(decoded, str) else decoded.decode("utf-8", errors="replace")
            )
        except Exception:
            return None
        if norm_old == old:
            return None
        result = self._exact(content, norm_old)
        if result is not None:
            return MatchResult(start=result.start, end=result.end, strategy="escape_normalised")
        # Inverse: decode content, match original old
        try:
            decoded_content = codecs.decode(content.encode(), "unicode_escape")
            norm_content: str = (
                decoded_content
                if isinstance(decoded_content, str)
                else decoded_content.decode("utf-8", errors="replace")
            )
        except Exception:
            return None
        result = self._exact(norm_content, old)
        if result is None:
            return None
        return self._remap(content, old, result.start, norm_content, strategy="escape_normalised")

    # --- Strategy 7: fuzzy ---------------------------------------------------

    def _fuzzy(self, content: str, old: str) -> MatchResult | None:
        """SequenceMatcher fuzzy search; ratio must be >= FUZZY_THRESHOLD."""
        old_lines = old.splitlines()
        content_lines = content.splitlines()
        n = len(old_lines)
        if n == 0:
            return None
        best_ratio = 0.0
        best_start_line = -1
        for i in range(len(content_lines) - n + 1):
            window = content_lines[i : i + n]
            ratio = SequenceMatcher(None, old_lines, window).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start_line = i
        if best_ratio < self.FUZZY_THRESHOLD or best_start_line == -1:
            return None
        # Map line index back to byte offset
        start = sum(len(line) + 1 for line in content_lines[:best_start_line])
        end = start + sum(
            len(line) + 1 for line in content_lines[best_start_line : best_start_line + n]
        )
        # Trim trailing newline overshoot
        end = min(end, len(content))
        return MatchResult(start=start, end=end, strategy="fuzzy")

    # --- Remap helper --------------------------------------------------------

    def _remap(
        self,
        original: str,
        old: str,
        start_in_norm: int,
        norm_content: str,
        strategy: str,
    ) -> MatchResult | None:
        """Map a match offset in norm_content back to the original string.

        Approximation: find the nearest position in original that aligns with
        the normalised match start. Falls back to original line-count heuristic.
        """
        # Count lines before start_in_norm in norm_content
        lines_before = norm_content[:start_in_norm].count("\n")
        # Find same line offset in original
        orig_lines = original.splitlines(keepends=True)
        if lines_before >= len(orig_lines):
            return None
        orig_start = sum(len(line) for line in orig_lines[:lines_before])
        # Count lines in old to determine end
        old_line_count = len(old.splitlines())
        orig_end = sum(
            len(line) for line in orig_lines[lines_before : lines_before + old_line_count]
        )
        orig_end = orig_start + orig_end
        orig_end = min(orig_end, len(original))
        return MatchResult(start=orig_start, end=orig_end, strategy=strategy)


def detect_line_ending(content: str) -> str:
    """Detect dominant line ending in content.

    Returns ``"\\r\\n"`` if CRLF is dominant, ``"\\n"`` otherwise (including
    when content is empty or has no line endings).
    """
    crlf_count = content.count("\r\n")
    lf_count = content.count("\n") - crlf_count  # pure LF only
    return "\r\n" if crlf_count > lf_count else "\n"


def normalise_endings(content: str, line_ending: str) -> str:
    """Convert all line endings in content to line_ending.

    First normalises all CRLF to LF, then converts LF to the target
    line_ending. This prevents double-conversion (``\\r\\r\\n``).
    """
    normalised = content.replace("\r\n", "\n")
    if line_ending == "\r\n":
        return normalised.replace("\n", "\r\n")
    return normalised


class Hunk(BaseModel):
    """Single hunk from a unified diff."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]  # each line prefixed with '+', '-', or ' '


class FilePatch(BaseModel):
    """Parsed unified diff for a single file."""

    path: str
    hunks: list[Hunk]


class EditItem(BaseModel):
    """A single find-and-replace operation for workspace_multi_edit."""

    path: str
    old_string: str
    new_string: str
    replace_all: bool = False


_HUNK_HEADER: re.Pattern[str] = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def hunk_header(hunk: Hunk) -> str:
    """Render *hunk*'s ``@@ -a,b +c,d @@`` header, for naming it in a failure."""
    return f"@@ -{hunk.old_start},{hunk.old_count} +{hunk.new_start},{hunk.new_count} @@"


class HunkContextError(Exception):
    """A hunk's old block does not verify against the file it targets.

    Raised by :func:`render_file_patch` rather than splicing anyway.  A unified
    diff carries its anchor in its context and removal lines; applying a hunk
    whose anchor is not there rewrites whichever lines now occupy those numbers
    and reports success, which is silent corruption — strictly worse than
    refusing, because the file afterwards looks plausible.

    Attributes:
        path: The file the patch names.
        header: The failing hunk's ``@@`` header.
        occurrences: How many times the old block was found in the file — 0 when
            it is absent, 2 or more when it is ambiguous.  Never 1: a single
            occurrence is applied at that offset instead of raising.
    """

    def __init__(self, path: str, hunk: Hunk, occurrences: int) -> None:
        self.path = path
        self.header = hunk_header(hunk)
        self.occurrences = occurrences
        found = "is not in the file" if occurrences == 0 else f"appears {occurrences} times"
        super().__init__(f"hunk {self.header} does not apply to {path}: its context {found}")


def _old_block(hunk: Hunk) -> list[str]:
    """Return the lines *hunk* expects to find, in order, stripped of prefixes.

    Context (``" "``) and removal (``"-"``) lines together are what the hunk was
    cut against; additions are what it produces and verify nothing.
    """
    return [line[1:] for line in hunk.lines if line[:1] in (" ", "-")]


def _locate_hunk(lines: list[str], block: list[str], guess: int, path: str, hunk: Hunk) -> int:
    """Return the index in *lines* where *hunk*'s old *block* actually sits.

    Args:
        lines: The file's current lines.
        block: The hunk's old block, from :func:`_old_block`.
        guess: The offset-adjusted ``old_start - 1`` the diff claims.
        path: The file the patch names, for the failure message.
        hunk: The hunk, for the failure message.

    Returns:
        *guess* when the block verifies there; otherwise the single offset
        elsewhere in the file where it does verify — the lines moved, and the
        patch is still anchored to text that exists exactly once.

    Raises:
        HunkContextError: when the block is absent, or occurs more than once.
            There is deliberately no fuzzy fallback: this function is the
            anchor, and an approximate anchor is not one.
    """
    if not block:
        return guess  # a pure insertion has nothing to verify — see render_file_patch
    if 0 <= guess and lines[guess : guess + len(block)] == block:
        return guess
    matches = [
        index
        for index in range(len(lines) - len(block) + 1)
        if lines[index : index + len(block)] == block
    ]
    if len(matches) != 1:
        raise HunkContextError(path, hunk, len(matches))
    return matches[0]


def _parse_hunk_header(line: str) -> tuple[int, int, int, int] | None:
    """Parse an ``@@ -a,b +c,d @@`` header into ``(old_start, old_count, new_start,
    new_count)``, or ``None`` when *line* is not a hunk header. Omitted counts default to 1.
    """
    match = _HUNK_HEADER.match(line)
    if match is None:
        return None
    return (
        int(match.group(1)),
        int(match.group(2)) if match.group(2) is not None else 1,
        int(match.group(3)),
        int(match.group(4)) if match.group(4) is not None else 1,
    )


class _PatchParser:
    """Line-at-a-time parser accumulating one :class:`FilePatch` per ``+++`` header.

    Holds the partial file/hunk state as attributes so each diff construct gets its
    own named transition instead of a closure over shared locals.
    """

    def __init__(self) -> None:
        """Start with no accumulated patches and no file in progress."""
        self.patches: list[FilePatch] = []
        self._path: str | None = None
        self._hunks: list[Hunk] = []
        self._hunk_lines: list[str] = []
        self._hunk_header: tuple[int, int, int, int] | None = None

    def feed(self, line: str) -> None:
        """Consume one diff line, dispatching to the matching transition."""
        if line.startswith("+++ "):
            self._start_file(line)
        elif line.startswith("--- "):
            return  # skip --- lines; path is taken from the +++ line
        elif (header := _parse_hunk_header(line)) is not None:
            self._start_hunk(header)
        elif self._hunk_header is not None and line[:1] in ("+", "-", " "):
            self._hunk_lines.append(line)

    def finish(self) -> list[FilePatch]:
        """Close off the file in progress and return every parsed patch."""
        self._flush_file()
        return self.patches

    def _start_file(self, line: str) -> None:
        """Close the previous file and begin one for the ``+++`` path on *line*."""
        self._flush_file()
        raw_path = line[4:].strip()
        self._path = raw_path[2:] if raw_path.startswith("b/") else raw_path
        self._hunks = []
        self._hunk_lines = []
        self._hunk_header = None

    def _start_hunk(self, header: tuple[int, int, int, int]) -> None:
        """Close the previous hunk, if any, and begin one under *header*."""
        if self._hunk_header is not None:
            self._flush_hunk()
            self._hunk_lines = []
        self._hunk_header = header

    def _flush_hunk(self) -> None:
        """Append the hunk in progress to the current file's hunk list."""
        if self._hunk_header is None:
            return
        old_start, old_count, new_start, new_count = self._hunk_header
        self._hunks.append(
            Hunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                lines=list(self._hunk_lines),
            )
        )

    def _flush_file(self) -> None:
        """Append the file in progress, with its final hunk, to ``patches``."""
        if self._path is None:
            return
        self._flush_hunk()
        self.patches.append(FilePatch(path=self._path, hunks=list(self._hunks)))


def parse_patch(patch_text: str) -> list[FilePatch]:
    """Parse a GNU unified diff string into a list of FilePatch objects.

    Supports single-file and multi-file diffs. Lines with ``--- /dev/null``
    or ``+++ /dev/null`` are preserved in the path field as-is; consumers
    must handle the sentinel.
    """
    parser = _PatchParser()
    for line in patch_text.splitlines():
        parser.feed(line)
    return parser.finish()


def is_pure_add(file_patch: FilePatch) -> bool:
    """Whether *file_patch* creates a file rather than updating one.

    Note: ``all()`` on an empty sequence returns True, so the empty hunk list is
    guarded explicitly — an empty patch is not a new-file patch.

    Args:
        file_patch: A parsed unified diff for one file.

    Returns:
        True when every hunk line is an addition.
    """
    return bool(file_patch.hunks) and all(
        all(patch_line.startswith("+") for patch_line in hunk.lines if patch_line)
        for hunk in file_patch.hunks
    )


def render_file_patch(raw: str | None, file_patch: FilePatch) -> str:
    """Return the text *file_patch* produces, without touching the filesystem.

    Rendering is separate from writing because the write gate has to run
    *between* the two: the actor computes the new content, checks the live file
    against what the agent observed, and only then publishes.

    **Every hunk is verified before it is spliced.**  Its old block — the
    context and removal lines, in order — must be the text at the line numbers
    the diff names, or occur exactly once elsewhere, in which case it is applied
    at that offset because the lines merely moved.  A hunk whose old block is
    empty is a pure insertion at a position: there is nothing to verify, and it
    goes in at ``old_start`` as it always did.

    Args:
        raw: The file's current text, or ``None`` when it does not exist.
        file_patch: A parsed unified diff for one file.

    Returns:
        The file's full text after every hunk is applied.

    Raises:
        FileNotFoundError: If *raw* is ``None`` and the patch is not a pure add.
        HunkContextError: If any hunk's context is absent from the file, or
            occurs more than once so that no offset is unambiguous.
    """
    if is_pure_add(file_patch):
        added = "\n".join(
            line[1:] for hunk in file_patch.hunks for line in hunk.lines if line.startswith("+")
        )
        return added + "\n"

    if raw is None:
        raise FileNotFoundError(file_patch.path)

    lines = raw.splitlines()
    offset = 0
    for hunk in file_patch.hunks:
        block = _old_block(hunk)
        start = _locate_hunk(lines, block, hunk.old_start - 1 + offset, file_patch.path, hunk)
        new_lines = [line[1:] for line in hunk.lines if line[:1] in ("+", " ")]
        lines[start : start + len(block)] = new_lines
        # Carry the *actual* position forward, not just the length delta: a hunk
        # applied at a relocated offset moves the frame for every hunk after it.
        offset = start + len(new_lines) - (hunk.old_start - 1 + len(block))
    return "\n".join(lines) + "\n"


def patch_label(file_patch: FilePatch) -> str:
    """Return the summary line ``workspace_patch`` reports for *file_patch*."""
    return f"{'created' if is_pure_add(file_patch) else 'updated'}: {file_patch.path}"


def apply_file_patch(workspace: Workspace, file_patch: FilePatch) -> None:
    """Read a workspace file, apply all hunks, write result back.

    Handles add (path="new_file", all hunks are additions),
    update (normal patch), and delete (path is sentinel "/dev/null" — but
    callers handle delete by checking file_patch.path against "/dev/null"
    BEFORE calling this function).

    Ungated by construction: it takes a backend and writes to it. The gated path
    goes through ``WorkspaceActor.apply_patch``, which renders and checks before
    it publishes.

    Args:
        workspace: Workspace backend (Filesystem instance in practice).
        file_patch: Parsed FilePatch with one or more Hunk objects.

    Raises:
        FileNotFoundError: If file does not exist and patch is not a pure add.
        HunkContextError: If a hunk's context does not verify — see
            :func:`render_file_patch`.  A patch whose hunks never matched their
            context used to be applied positionally; that was the defect, and a
            caller relying on the lenient behaviour now sees it here.
        PermissionError: If path escapes the workspace root.
    """
    raw = None if is_pure_add(file_patch) else workspace.read(file_patch.path).decode("utf-8")
    workspace.write(file_patch.path, render_file_patch(raw, file_patch).encode("utf-8"))


def deleted_paths(patch_text: str) -> set[str]:
    """Return the paths a patch deletes, read from the raw diff text.

    ``parse_patch`` derives each path from the ``+++`` line, which is ``/dev/null``
    for a deletion — so the real path must come from the preceding ``--- a/<path>``
    line here.
    """
    delete_paths: set[str] = set()
    lines = patch_text.splitlines()
    for i, line in enumerate(lines):
        if not (line.startswith("+++ /dev/null") or line.startswith("+++ b//dev/null")):
            continue
        for j in range(i - 1, max(i - 5, -1), -1):
            if lines[j].startswith("--- "):
                raw_del = lines[j][4:].strip()
                del_path = raw_del[2:] if raw_del.startswith("a/") else raw_del
                if del_path != "/dev/null":
                    delete_paths.add(del_path)
                break
    return delete_paths


def substitute_edit(
    matcher: EditMatcher, content: str, item: EditItem, exact_only: bool = False
) -> str | None:
    """Apply one edit's substitution to *content*.

    Args:
        matcher: The cascade used to locate ``old_string``.
        content: The text to edit.
        item: The edit to apply.
        exact_only: Restrict the cascade to exact matching — see
            :meth:`EditMatcher.find`.

    Returns:
        The edited content, or ``None`` when ``old_string`` was not found — for
        ``replace_all`` that means not found even once.
    """
    if not item.replace_all:
        match = matcher.find(content, item.old_string, exact_only=exact_only)
        if match is None:
            return None
        return content[: match.start] + item.new_string + content[match.end :]

    result = content
    found_any = False
    while (match := matcher.find(result, item.old_string, exact_only=exact_only)) is not None:
        found_any = True
        result = result[: match.start] + item.new_string + result[match.end :]
    return result if found_any else None


def unified(
    path: str,
    before: str,
    after: str,
    *,
    before_label: str = "a",
    after_label: str = "b",
) -> str:
    """Return the unified diff from *before* to *after*, or ``""`` when equal.

    Args:
        path: Workspace-relative path, used in both file headers.
        before: The left-hand text.
        after: The right-hand text.
        before_label: Header prefix for the left side. A rejection labels it
            ``live`` against ``proposed``, so the direction the agent is being
            shown cannot be misread as an applied change.
        after_label: Header prefix for the right side.
    """
    return "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"{before_label}/{path}",
            tofile=f"{after_label}/{path}",
            lineterm="",
        )
    )


def write_and_diff(backend: Workspace, path: str, raw: str, edited: str) -> tuple[bytes, str]:
    """Write *edited* back to *path* with *raw*'s line endings.

    Args:
        backend: The tree to publish into — the actor's own handle on the gated
            path.
        path: Workspace-relative path.
        raw: The file's text before the edit, and the source of its dominant
            line ending.
        edited: The text to publish.

    Returns:
        The bytes written, and the unified diff — empty when the edit changed
        nothing. The bytes come back because the actor records the writer's own
        observation of them, in the same mailbox turn as the write.
    """
    normalised = normalise_endings(edited, detect_line_ending(raw))
    data = normalised.encode("utf-8")
    backend.write(path, data)
    return data, unified(path, raw, normalised)
