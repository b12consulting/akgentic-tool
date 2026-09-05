"""``RagIndexState``: the full render, and the delta that is never a table.

The delta is the whole reason this capability is a ``ContextState`` rather than a
system-prompt line. Re-rendering the table each turn would invalidate the cached
prompt prefix every time one file changed status, which on a tree being indexed is
every turn — so what is asserted here is not only that a delta is produced but
that it is *short*, and keyed on the path.
"""

from __future__ import annotations

from akgentic.tool.core import ContextState
from akgentic.tool.workspace.documents.context import RagFileRow, RagIndexState


def row(path: str, status: str = "embedded", chunk_count: int = 3, reason: str = "") -> RagFileRow:
    """One rendered row."""
    return RagFileRow(path=path, status=status, chunk_count=chunk_count, reason=reason)


def state(*rows: RagFileRow, pending_hidden: int = 0) -> RagIndexState:
    """One snapshot."""
    return RagIndexState(rows=list(rows), pending_hidden=pending_hidden)


class TestItIsAContextState:
    """The contract the agent-side caller relies on."""

    def test_it_implements_the_contract(self) -> None:
        assert issubclass(RagIndexState, ContextState)

    def test_it_round_trips_through_validation(self) -> None:
        """It is delivered on the ``LLM_CONTEXT`` channel, so it is serialised."""
        original = state(row("a.md"), row("b.md", status="pending", chunk_count=0))

        assert RagIndexState.model_validate(original.model_dump()) == original


class TestTheFullRender:
    """What the model sees the first time it carries the card."""

    def test_an_empty_index_renders_a_sentence(self) -> None:
        """A real state, distinct from a provider returning ``None``."""
        assert state().render_full() == "No workspace files are indexed for retrieval."

    def test_every_row_gets_a_line(self) -> None:
        rendered = state(row("a.md"), row("deep/b.md", status="failed", reason="boom"))

        assert "a.md" in rendered.render_full()
        assert "deep/b.md" in rendered.render_full()

    def test_a_failure_reason_is_shown(self) -> None:
        """It is the one thing on a row a model can act on."""
        rendered = state(row("a.md", status="failed", chunk_count=0, reason="rate limited"))

        assert "rate limited" in rendered.render_full()

    def test_hidden_pending_files_are_counted_in_a_tail(self) -> None:
        """A 10,000-file tree must not flood the context window."""
        rendered = state(row("a.md", status="pending", chunk_count=0), pending_hidden=42)

        assert "…and 42 more pending" in rendered.render_full()

    def test_an_index_that_is_all_hidden_pending_still_renders(self) -> None:
        """Zero rows and a positive tail is a legal state, not an empty index."""
        rendered = state(pending_hidden=5).render_full()

        assert rendered != "No workspace files are indexed for retrieval."
        assert "…and 5 more pending" in rendered


class TestTheDelta:
    """Keyed on ``path``, and only what moved."""

    def test_nothing_moved_renders_nothing(self) -> None:
        """``None`` is what tells the caller to push nothing this turn."""
        before = state(row("a.md"), row("b.md"))

        assert before.render_delta(before) is None

    def test_a_status_change_renders_one_short_sentence(self) -> None:
        """``invoice.pdf: splitting → embedded`` — never a re-rendered table."""
        before = state(row("invoice.pdf", status="splitting", chunk_count=0))
        after = state(row("invoice.pdf", status="embedded", chunk_count=12))

        delta = after.render_delta(before)

        assert delta is not None
        assert "invoice.pdf: splitting → embedded." in delta

    def test_a_delta_is_shorter_than_the_table_it_replaces(self) -> None:
        """The property the whole capability turns on."""
        rows = [row(f"file{index}.md") for index in range(20)]
        before = state(*rows)
        after = state(*rows[:-1], row("file19.md", status="stale"))

        delta = after.render_delta(before)

        assert delta is not None
        assert len(delta) < len(after.render_full()) / 4

    def test_an_appearing_path_renders_as_new(self) -> None:
        before = state(row("a.md"))
        after = state(row("a.md"), row("b.md", status="pending", chunk_count=0))

        delta = after.render_delta(before)

        assert delta is not None
        assert "Indexing b.md: pending." in delta

    def test_a_leaving_path_renders_as_removed(self) -> None:
        before = state(row("a.md"), row("b.md"))
        after = state(row("a.md"))

        delta = after.render_delta(before)

        assert delta is not None
        assert "No longer indexed: b.md." in delta

    def test_a_chunk_count_change_renders_on_its_own(self) -> None:
        """A re-index at the same status still moved something worth saying."""
        before = state(row("a.md", chunk_count=3))
        after = state(row("a.md", chunk_count=9))

        delta = after.render_delta(before)

        assert delta is not None
        assert "a.md: 9 chunk(s)." in delta

    def test_a_new_failure_reason_renders(self) -> None:
        before = state(row("a.md", status="embedding", chunk_count=0))
        after = state(row("a.md", status="failed", chunk_count=0, reason="rate limited"))

        delta = after.render_delta(before)

        assert delta is not None
        assert "a.md: rate limited." in delta

    def test_unchanged_rows_are_never_re_listed(self) -> None:
        """Re-listing them is exactly the prefix invalidation this design avoids."""
        before = state(row("a.md"), row("b.md"), row("c.md"))
        after = state(row("a.md"), row("b.md"), row("c.md", status="stale"))

        delta = after.render_delta(before)

        assert delta is not None
        assert "a.md" not in delta
        assert "b.md" not in delta

    def test_the_hidden_pending_count_moving_is_reported(self) -> None:
        """It is the only thing about the pending backlog the model can see."""
        before = state(row("a.md"), pending_hidden=10)
        after = state(row("a.md"), pending_hidden=3)

        delta = after.render_delta(before)

        assert delta is not None
        assert "3 more pending." in delta
