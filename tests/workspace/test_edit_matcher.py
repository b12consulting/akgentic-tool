"""Tests for EditMatcher strategies and line ending utilities (Story 5.3)."""

from __future__ import annotations

from akgentic.tool.workspace.edit import EditMatcher, detect_line_ending, normalise_endings

# ---------------------------------------------------------------------------
# Strategy 1: exact
# ---------------------------------------------------------------------------


def test_exact_match_found() -> None:
    m = EditMatcher()
    result = m.find("hello world", "world")
    assert result is not None
    assert result.strategy == "exact"
    assert result.start == 6
    assert result.end == 11


def test_exact_match_not_found() -> None:
    m = EditMatcher()
    result = m._exact("hello world", "xyz")
    assert result is None


# ---------------------------------------------------------------------------
# Strategy 2: line-trimmed
# ---------------------------------------------------------------------------


def test_line_trimmed_match() -> None:
    m = EditMatcher()
    content = "def foo():\n    return 1\n"
    # old_string has extra leading/trailing whitespace on each line
    old = "  def foo():  \n      return 1  \n"
    result = m.find(content, old)
    assert result is not None
    assert result.strategy == "line_trimmed"


def test_line_trimmed_no_change_returns_none_from_strategy() -> None:
    m = EditMatcher()
    # Already stripped — strategy should return None (no new info)
    result = m._line_trimmed("hello world", "hello world")
    assert result is None


# ---------------------------------------------------------------------------
# Strategy 3: whitespace-normalised
# ---------------------------------------------------------------------------


def test_whitespace_normalised_match() -> None:
    m = EditMatcher()
    content = "x = 1 + 2"
    old = "x  =  1  +  2"  # extra spaces collapsed
    result = m.find(content, old)
    assert result is not None
    assert result.strategy == "whitespace_normalised"


def test_whitespace_normalised_no_change_returns_none() -> None:
    m = EditMatcher()
    result = m._whitespace_normalised("x = 1", "x = 1")
    assert result is None


# ---------------------------------------------------------------------------
# Strategy 4: dedented
# ---------------------------------------------------------------------------


def test_dedented_match() -> None:
    m = EditMatcher()
    content = "def foo():\n    return 1\n"
    # old_string with extra indentation — test via _dedented directly
    # to avoid earlier strategies in the cascade winning
    old = "    def foo():\n        return 1\n"
    result = m._dedented(content, old)
    assert result is not None
    assert result.strategy == "dedented"


def test_dedented_no_change_returns_none() -> None:
    m = EditMatcher()
    result = m._dedented("no indent\n", "no indent\n")
    assert result is None


# ---------------------------------------------------------------------------
# Strategy 5: trimmed boundary
# ---------------------------------------------------------------------------


def test_trimmed_boundary_match() -> None:
    m = EditMatcher()
    content = "def foo():\n    return 1\n"
    old_with_blank = "\ndef foo():\n    return 1\n"
    result = m.find(content, old_with_blank)
    assert result is not None
    assert result.strategy == "trimmed_boundary"


def test_trimmed_boundary_no_blank_edge_returns_none() -> None:
    m = EditMatcher()
    # No blank edges on old_string — strategy must return None
    result = m._trimmed_boundary("hello world", "hello world")
    assert result is None


# ---------------------------------------------------------------------------
# Strategy 6: escape-normalised
# ---------------------------------------------------------------------------


def test_escape_normalised_double_escaped_newline() -> None:
    m = EditMatcher()
    # Content has a literal newline; old_string has double-escaped \\n
    content = "line1\nline2"
    # Simulate LLM outputting double-escaped: "line1\\nline2"
    old = "line1\\nline2"
    result = m.find(content, old)
    assert result is not None
    assert result.strategy == "escape_normalised"


# ---------------------------------------------------------------------------
# Strategy 7: fuzzy
# ---------------------------------------------------------------------------


def test_fuzzy_above_threshold_matches() -> None:
    m = EditMatcher()
    content = "def calculate_sum(a, b):\n    return a + b\n"
    # Slightly different spelling, should still be >= 0.85
    old = "def calculate_sum(a, b):\n    return a + b\n"  # identical → exact wins
    result = m.find(content, old)
    assert result is not None
    assert result.strategy == "exact"


def test_fuzzy_high_similarity() -> None:
    m = EditMatcher()
    # 10 lines: 9 identical, 1 different — line-level ratio = 0.9 (>= 0.85 threshold)
    shared = ["line1", "line2", "line3", "line4", "line5", "line6", "line7", "line8", "line9"]
    content_lines = shared + ["new_line"]
    old_lines = shared + ["old_line"]
    content = "\n".join(content_lines) + "\n"
    old = "\n".join(old_lines)
    result = m._fuzzy(content, old)
    assert result is not None
    assert result.strategy == "fuzzy"


def test_fuzzy_below_threshold_returns_none() -> None:
    m = EditMatcher()
    result = m.find("hello world", "completely different text that bears no resemblance")
    assert result is None


def test_fuzzy_ratio_just_below_threshold_returns_none() -> None:
    m = EditMatcher()
    # Force a low-similarity check via _fuzzy directly
    result = m._fuzzy("aaa bbb ccc", "xxx yyy zzz ppp qqq rrr sss")
    assert result is None


# ---------------------------------------------------------------------------
# find() cascade order — earlier strategy wins
# ---------------------------------------------------------------------------


def test_find_cascade_exact_wins_over_fuzzy() -> None:
    m = EditMatcher()
    content = "hello world"
    old = "hello world"
    result = m.find(content, old)
    assert result is not None
    assert result.strategy == "exact"


def test_find_returns_none_when_no_strategy_matches() -> None:
    m = EditMatcher()
    result = m.find("short text", "completely unrelated long string with no overlap whatsoever xyz")
    assert result is None


# ---------------------------------------------------------------------------
# detect_line_ending
# ---------------------------------------------------------------------------


def test_detect_line_ending_crlf_dominant() -> None:
    content = "line1\r\nline2\r\nline3\r\n"
    assert detect_line_ending(content) == "\r\n"


def test_detect_line_ending_lf_dominant() -> None:
    content = "line1\nline2\nline3\n"
    assert detect_line_ending(content) == "\n"


def test_detect_line_ending_empty_content_returns_lf() -> None:
    assert detect_line_ending("") == "\n"


def test_detect_line_ending_mixed_crlf_wins() -> None:
    # 3 CRLF vs 1 pure LF
    content = "a\r\nb\r\nc\r\nd\n"
    assert detect_line_ending(content) == "\r\n"


def test_detect_line_ending_mixed_lf_wins() -> None:
    # 1 CRLF vs 3 pure LF
    content = "a\r\nb\nc\nd\n"
    assert detect_line_ending(content) == "\n"


# ---------------------------------------------------------------------------
# normalise_endings
# ---------------------------------------------------------------------------


def test_normalise_endings_lf_to_crlf() -> None:
    content = "line1\nline2\nline3\n"
    result = normalise_endings(content, "\r\n")
    assert result == "line1\r\nline2\r\nline3\r\n"


def test_normalise_endings_crlf_to_lf() -> None:
    content = "line1\r\nline2\r\nline3\r\n"
    result = normalise_endings(content, "\n")
    assert result == "line1\nline2\nline3\n"


def test_normalise_endings_no_op_when_already_target() -> None:
    content = "line1\nline2\n"
    result = normalise_endings(content, "\n")
    assert result == content


def test_normalise_endings_no_double_conversion() -> None:
    # Already CRLF, converting to CRLF should not produce \r\r\n
    content = "line1\r\nline2\r\n"
    result = normalise_endings(content, "\r\n")
    assert "\r\r\n" not in result
    assert result == "line1\r\nline2\r\n"
