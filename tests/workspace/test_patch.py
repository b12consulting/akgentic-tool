"""Tests for parse_patch and apply_file_patch (Story 5.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from akgentic.tool.workspace.edit import (
    FilePatch,
    Hunk,
    HunkContextError,
    apply_file_patch,
    parse_patch,
    render_file_patch,
)
from akgentic.tool.workspace.workspace import Filesystem

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_DIFF = """\
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,3 +1,3 @@
 line1
-old_line
+new_line
 line3
"""

MULTI_FILE_DIFF = """\
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,2 +1,2 @@
-foo_old
+foo_new
 common
--- a/src/bar.py
+++ b/src/bar.py
@@ -1,2 +1,2 @@
-bar_old
+bar_new
 other
"""

ADD_FILE_DIFF = """\
--- /dev/null
+++ b/src/new_file.py
@@ -0,0 +1,2 @@
+first_line
+second_line
"""

DELETE_FILE_DIFF = """\
--- a/src/old_file.py
+++ /dev/null
@@ -1,2 +0,0 @@
-removed_line1
-removed_line2
"""

MULTI_HUNK_DIFF = """\
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,3 +1,3 @@
 line1
-old1
+new1
 line3
@@ -5,3 +5,3 @@
 line5
-old2
+new2
 line7
"""


# ---------------------------------------------------------------------------
# parse_patch — single file
# ---------------------------------------------------------------------------


def test_parse_patch_single_file() -> None:
    patches = parse_patch(SIMPLE_DIFF)
    assert len(patches) == 1
    assert patches[0].path == "src/foo.py"
    assert len(patches[0].hunks) == 1
    hunk = patches[0].hunks[0]
    assert hunk.old_start == 1
    assert hunk.old_count == 3
    assert hunk.new_start == 1
    assert hunk.new_count == 3


def test_parse_patch_single_file_hunk_lines() -> None:
    patches = parse_patch(SIMPLE_DIFF)
    hunk = patches[0].hunks[0]
    assert " line1" in hunk.lines
    assert "-old_line" in hunk.lines
    assert "+new_line" in hunk.lines


# ---------------------------------------------------------------------------
# parse_patch — multi-file
# ---------------------------------------------------------------------------


def test_parse_patch_multi_file() -> None:
    patches = parse_patch(MULTI_FILE_DIFF)
    assert len(patches) == 2
    assert patches[0].path == "src/foo.py"
    assert patches[1].path == "src/bar.py"


def test_parse_patch_multi_file_paths() -> None:
    patches = parse_patch(MULTI_FILE_DIFF)
    assert patches[0].hunks[0].lines[0] == "-foo_old"
    assert patches[1].hunks[0].lines[0] == "-bar_old"


# ---------------------------------------------------------------------------
# parse_patch — hunk header parsing
# ---------------------------------------------------------------------------


def test_parse_patch_hunk_header_values() -> None:
    patches = parse_patch(SIMPLE_DIFF)
    hunk = patches[0].hunks[0]
    assert hunk.old_start == 1
    assert hunk.old_count == 3
    assert hunk.new_start == 1
    assert hunk.new_count == 3


def test_parse_patch_add_file_path() -> None:
    patches = parse_patch(ADD_FILE_DIFF)
    assert len(patches) == 1
    assert patches[0].path == "src/new_file.py"


def test_parse_patch_delete_file_path() -> None:
    patches = parse_patch(DELETE_FILE_DIFF)
    assert len(patches) == 1
    assert patches[0].path == "/dev/null"


def test_parse_patch_multi_hunk() -> None:
    patches = parse_patch(MULTI_HUNK_DIFF)
    assert len(patches) == 1
    assert len(patches[0].hunks) == 2
    assert patches[0].hunks[0].old_start == 1
    assert patches[0].hunks[1].old_start == 5


def test_parse_patch_hunk_header_missing_count_defaults_to_one() -> None:
    """Cover the branch where hunk header omits count (e.g. @@ -1 +1 @@)."""
    # Unified diff spec: "@@ -start +start @@" without count means count=1
    diff_no_count = """\
--- a/src/foo.py
+++ b/src/foo.py
@@ -1 +1 @@
-old_line
+new_line
"""
    patches = parse_patch(diff_no_count)
    assert len(patches) == 1
    hunk = patches[0].hunks[0]
    assert hunk.old_start == 1
    assert hunk.old_count == 1  # defaulted from missing count
    assert hunk.new_start == 1
    assert hunk.new_count == 1  # defaulted from missing count


# ---------------------------------------------------------------------------
# apply_file_patch — update operation
# ---------------------------------------------------------------------------


def test_apply_file_patch_update(tmp_path: Path) -> None:
    fs = Filesystem(str(tmp_path), "ws")
    fs.write("foo.py", b"line1\nline2\nline3\n")
    patch = FilePatch(
        path="foo.py",
        hunks=[
            Hunk(
                old_start=2,
                old_count=1,
                new_start=2,
                new_count=1,
                lines=["-line2", "+new_line"],
            )
        ],
    )
    apply_file_patch(fs, patch)
    result = fs.read("foo.py").decode()
    assert "new_line" in result
    assert "line2" not in result


def test_apply_file_patch_update_only_targeted_lines(tmp_path: Path) -> None:
    fs = Filesystem(str(tmp_path), "ws")
    fs.write("foo.py", b"line1\nline2\nline3\n")
    patch = FilePatch(
        path="foo.py",
        hunks=[
            Hunk(
                old_start=2,
                old_count=1,
                new_start=2,
                new_count=1,
                lines=["-line2", "+replaced"],
            )
        ],
    )
    apply_file_patch(fs, patch)
    result = fs.read("foo.py").decode()
    assert "line1" in result
    assert "line3" in result
    assert "replaced" in result


# ---------------------------------------------------------------------------
# apply_file_patch — add operation (all-add patch)
# ---------------------------------------------------------------------------


def test_apply_file_patch_add_new_file(tmp_path: Path) -> None:
    fs = Filesystem(str(tmp_path), "ws")
    patch = FilePatch(
        path="new_file.py",
        hunks=[
            Hunk(
                old_start=0,
                old_count=0,
                new_start=1,
                new_count=2,
                lines=["+first_line", "+second_line"],
            )
        ],
    )
    apply_file_patch(fs, patch)
    result = fs.read("new_file.py").decode()
    assert "first_line" in result
    assert "second_line" in result
    # New files must end with a trailing newline (Unix convention)
    assert result.endswith("\n")


def test_apply_file_patch_empty_hunks_does_not_create_file(tmp_path: Path) -> None:
    """Guard against empty hunk list being treated as new-file (all() bug)."""
    fs = Filesystem(str(tmp_path), "ws")
    # Write an existing file first
    fs.write("existing.py", b"content\n")
    patch = FilePatch(path="existing.py", hunks=[])
    # Empty hunk list → is_new_file must be False, update path is taken,
    # splitlines on existing content with no hunks → file written back unchanged
    apply_file_patch(fs, patch)
    result = fs.read("existing.py").decode()
    assert "content" in result


# ---------------------------------------------------------------------------
# apply_file_patch — delete operation (handled by caller — /dev/null sentinel)
# ---------------------------------------------------------------------------


def test_apply_file_patch_delete_sentinel(tmp_path: Path) -> None:
    """apply_file_patch is NOT called for /dev/null path in tool.py.

    This test verifies that callers should check path == "/dev/null" before
    calling apply_file_patch. parse_patch correctly returns path="/dev/null"
    for delete diffs.
    """
    patches = parse_patch(DELETE_FILE_DIFF)
    assert patches[0].path == "/dev/null"
    # Callers (workspace_patch) check for /dev/null and call fs.delete() instead.


# ---------------------------------------------------------------------------
# apply_file_patch — multi-hunk patch applies all hunks in order
# ---------------------------------------------------------------------------


def test_apply_file_patch_multi_hunk(tmp_path: Path) -> None:
    fs = Filesystem(str(tmp_path), "ws")
    # 7 lines; two hunks each replace one line
    fs.write(
        "foo.py",
        b"line1\nold1\nline3\nline4\nline5\nold2\nline7\n",
    )
    patch = FilePatch(
        path="foo.py",
        hunks=[
            Hunk(
                old_start=2,
                old_count=1,
                new_start=2,
                new_count=1,
                lines=["-old1", "+new1"],
            ),
            Hunk(
                old_start=6,
                old_count=1,
                new_start=6,
                new_count=1,
                lines=["-old2", "+new2"],
            ),
        ],
    )
    apply_file_patch(fs, patch)
    result = fs.read("foo.py").decode()
    assert "new1" in result
    assert "new2" in result
    assert "old1" not in result
    assert "old2" not in result
    # Surrounding lines preserved
    assert "line1" in result
    assert "line3" in result
    assert "line7" in result


# ---------------------------------------------------------------------------
# render_file_patch — a hunk is applied only where its context matches
# ---------------------------------------------------------------------------


def _hunk(old_start: int, old_count: int, *lines: str) -> Hunk:
    """A hunk at *old_start* carrying *lines*, with a plausible new-side header."""
    added = len([line for line in lines if line[:1] in ("+", " ")])
    return Hunk(
        old_start=old_start,
        old_count=old_count,
        new_start=old_start,
        new_count=added,
        lines=list(lines),
    )


class TestHunkContextVerification:
    """``render_file_patch`` used to splice at ``old_start`` and check nothing.

    That is the defect: a diff cut against an older revision rewrote whichever
    lines now sat at those numbers and reported success — silent corruption, and
    strictly worse than a refusal, because the file afterwards looks plausible.

    ``apply_file_patch`` is public and this changes it for every caller: a patch
    whose hunks never matched their context used to apply positionally and now
    raises.
    """

    def test_matching_context_applies_where_the_diff_says(self) -> None:
        raw = "line1\nold\nline3\n"
        patch = FilePatch(path="f.py", hunks=[_hunk(2, 1, "-old", "+new")])
        assert render_file_patch(raw, patch) == "line1\nnew\nline3\n"

    def test_a_block_that_moved_is_relocated_by_offset(self) -> None:
        # The lines moved; the patch is still anchored to text that exists
        # exactly once. This is FR6's degradation — a surgical change surviving a
        # concurrent change to an unrelated region of the same file.
        raw = "# banner\nline1\nold\nline3\n"
        patch = FilePatch(path="f.py", hunks=[_hunk(2, 1, "-old", "+new")])
        assert render_file_patch(raw, patch) == "# banner\nline1\nnew\nline3\n"

    def test_absent_context_raises_and_names_the_hunk(self) -> None:
        raw = "wholly different\n"
        patch = FilePatch(path="f.py", hunks=[_hunk(2, 1, "-old", "+new")])
        with pytest.raises(HunkContextError) as failure:
            render_file_patch(raw, patch)
        assert failure.value.path == "f.py"
        assert failure.value.header == "@@ -2,1 +2,1 @@"
        assert failure.value.occurrences == 0

    def test_ambiguous_context_raises_rather_than_guessing(self) -> None:
        raw = "marker\nx\nmarker\n"
        patch = FilePatch(path="f.py", hunks=[_hunk(9, 1, "-marker", "+MARKER")])
        with pytest.raises(HunkContextError) as failure:
            render_file_patch(raw, patch)
        assert failure.value.occurrences == 2

    def test_no_fuzzy_fallback(self) -> None:
        # An approximate anchor is not an anchor. Whitespace that a fuzzy matcher
        # would forgive is a mismatch here, on purpose.
        raw = "def  foo():\n    pass\n"
        patch = FilePatch(path="f.py", hunks=[_hunk(1, 1, "-def foo():", "+def bar():")])
        with pytest.raises(HunkContextError):
            render_file_patch(raw, patch)

    def test_an_insertion_only_hunk_has_nothing_to_verify(self) -> None:
        # An empty old block is a position, not an anchor, so it goes in at
        # old_start as it always did. (A patch whose *every* line is an addition
        # is a file creation — `is_pure_add` — which is why this one carries a
        # second, ordinary hunk.)
        raw = "line1\nline2\nlast\n"
        patch = FilePatch(
            path="f.py",
            hunks=[_hunk(2, 0, "+inserted"), _hunk(3, 1, "-last", "+LAST")],
        )
        assert render_file_patch(raw, patch) == "line1\ninserted\nline2\nLAST\n"

    def test_a_relocated_hunk_moves_the_frame_for_the_ones_after_it(self) -> None:
        # Carrying the *position* forward rather than only the length delta is
        # what keeps a multi-hunk patch coherent once the first hunk relocates.
        raw = "pad\npad\nfirst\nmiddle\nsecond\n"
        patch = FilePatch(
            path="f.py",
            hunks=[_hunk(1, 1, "-first", "+FIRST"), _hunk(3, 1, "-second", "+SECOND")],
        )
        assert render_file_patch(raw, patch) == "pad\npad\nFIRST\nmiddle\nSECOND\n"

    def test_apply_file_patch_propagates_the_failure(self, tmp_path: Path) -> None:
        fs = Filesystem(str(tmp_path), "ws")
        fs.write("foo.py", b"nothing like the patch\n")
        patch = FilePatch(path="foo.py", hunks=[_hunk(1, 1, "-old", "+new")])
        with pytest.raises(HunkContextError):
            apply_file_patch(fs, patch)
        # Nothing was written: the check happens before the splice.
        assert fs.read("foo.py") == b"nothing like the patch\n"
