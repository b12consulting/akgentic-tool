"""``workspace_rag_list`` as structured context state (ADR-037 §3, ADR-045 §5).

The retrieval index moves — a file goes ``pending`` → ``splitting`` →
``embedding`` → ``embedded`` over the course of a few turns — and re-rendering
the whole table into the system prompt on every change would invalidate the
cached prompt prefix each time. So it is a :class:`ContextState` instead: the
first turn sees the table, every later turn sees only what moved.

**Nothing here touches an actor, and nothing here reads the tree.**
:func:`render_index_state` reads the tree's *metadata* — one directory scan of
``<meta>/rag/`` plus one parse per record, bounded by ``max_documents`` — through
the :class:`~akgentic.tool.workspace.documents.cache.DocumentCache` its caller
already holds. That is the whole of its I/O, it happens on the calling agent's own
thread, and it is where the render lives because rendering rows into
:class:`RagIndexState` is this module's subject. A snapshot is not a search and is
deliberately not filed under one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from akgentic.core.utils import SerializableBaseModel
from akgentic.tool.core import ContextState
from akgentic.tool.workspace.documents.models import RagStatus

if TYPE_CHECKING:
    from akgentic.tool.workspace.documents.cache import DocumentCache

__all__ = ["RagFileRow", "RagIndexState"]

_EMPTY = "No workspace files are indexed for retrieval."
"""What an index with no rows renders.

A real state, and deliberately distinct from a provider returning ``None``:
"nothing is indexed" is an answer, and an agent that is told it stops asking.
"""


class RagFileRow(SerializableBaseModel):
    """One file's line in the rendered index.

    A projection of :class:`~akgentic.tool.workspace.documents.models.RagFile`,
    carrying what a model can act on and nothing else — no offsets, no chunk ids,
    no batch counters.

    Attributes:
        path: Workspace-relative path.
        status: The row's :class:`~akgentic.tool.workspace.documents.models.RagStatus`
            as its plain string value.
        chunk_count: How many chunks the file currently has in the index.
        reason: Why it failed, or ``""`` — never ``None``, so the delta compares
            two strings rather than branching on absence.
    """

    path: str
    status: str
    chunk_count: int
    reason: str


class RagIndexState(ContextState):
    """The retrieval index at one point in time, diffable file by file.

    Attributes:
        rows: The files to show. Everything that is not ``pending`` is always
            here; ``pending`` files are capped by the card, because a 10,000-file
            tree would otherwise flood the context window with rows that all say
            the same thing.
        pending_hidden: How many ``pending`` files the cap left out.
    """

    rows: list[RagFileRow]
    pending_hidden: int

    def render_full(self) -> str:
        """The whole index, as the model should first see it.

        Returns:
            One line per file, plus a tail naming the pending files the cap left
            out. An empty index renders its own sentence rather than ``""``.
        """
        if not self.rows and self.pending_hidden == 0:
            return _EMPTY
        lines = [f"**Workspace retrieval index:** {len(self.rows)} file(s) shown"]
        lines.extend(_row_line(row) for row in self.rows)
        if self.pending_hidden > 0:
            lines.append(f"…and {self.pending_hidden} more pending")
        return "\n".join(lines)

    def render_delta(self, previous: Self) -> str | None:
        """What moved since *previous*, keyed on ``path``.

        **Never a re-rendered table.** That is the whole reason this capability is
        a ``ContextState`` and not a system-prompt line: re-rendering would
        invalidate the cached prompt prefix on every turn a single file changed
        status.

        Args:
            previous: The state this agent last saw. The caller guarantees it is
                the same concrete type.

        Returns:
            One sentence per file that appeared, left, or changed, or ``None``
            when nothing moved.
        """
        before = {row.path: row for row in previous.rows}
        current = {row.path for row in self.rows}

        parts: list[str] = []
        for row in self.rows:
            old = before.get(row.path)
            if old is None:
                parts.append(f"Indexing {row.path}: {_state_of(row)}.")
            elif old != row:
                parts.extend(_row_changes(old, row))
        parts.extend(
            f"No longer indexed: {row.path}." for row in previous.rows if row.path not in current
        )
        if self.pending_hidden != previous.pending_hidden:
            parts.append(f"{self.pending_hidden} more pending.")
        return " ".join(parts) if parts else None


def render_index_state(cache: DocumentCache, max_pending_shown: int) -> RagIndexState:
    """Return *cache*'s records as rows, capped on ``PENDING`` only.

    **No file access inside the tree, and no tree sweep**: this is taken once per
    turn by every agent carrying the card, and a ``stat`` per candidate file would
    put a tree walk on the hot path for a display. Reading ``<meta>/rag/`` is not
    that — it is one directory scan plus one parse per record, bounded by
    ``max_documents`` (32, or 8 when the vector backend is in-memory). That is
    real where ``<meta>`` is on a network share, and it has no mitigation on
    offer: the directory holds every cross-process lock, so it lives beside the
    tree it belongs to and moves nowhere. **No read-through cache is added here**
    either: it would be exactly the in-memory state the records-on-disk move
    removed.

    The rows are sorted by **path**, so the render is stable across runs. A
    directory glob's order is the file system's, and a display that reordered
    itself between two turns would look like the index had changed.

    Everything that is not ``PENDING`` is always shown — those rows each say
    something different. ``PENDING`` rows all say the same thing, so a
    10,000-file tree would otherwise flood the context window with them.

    Args:
        cache: This tree's document records.
        max_pending_shown: How many ``PENDING`` rows to render.

    Returns:
        The state, never ``None`` and never raising.
    """
    rows: list[RagFileRow] = []
    hidden = 0
    pending_shown = 0
    for entry in sorted(cache.entries(), key=lambda stored: stored.path):
        row = entry.row
        if row is None:
            continue
        if row.status is RagStatus.PENDING:
            if pending_shown >= max_pending_shown:
                hidden += 1
                continue
            pending_shown += 1
        rows.append(
            RagFileRow(
                path=entry.path,
                status=row.status.value,
                chunk_count=row.chunk_count,
                reason=row.reason or "",
            )
        )
    return RagIndexState(rows=rows, pending_hidden=hidden)


def _row_line(row: RagFileRow) -> str:
    """One ``- path [status] …`` line of the full render."""
    suffix = f" — {row.reason}" if row.reason else ""
    return f"- {row.path} [{row.status}] {row.chunk_count} chunk(s){suffix}"


def _state_of(row: RagFileRow) -> str:
    """How a newly appearing row is described, status first."""
    return f"{row.status}{f' — {row.reason}' if row.reason else ''}"


def _row_changes(old: RagFileRow, new: RagFileRow) -> list[str]:
    """One short sentence per field that moved between two rows with the same path."""
    parts: list[str] = []
    if new.status != old.status:
        parts.append(f"{new.path}: {old.status} → {new.status}.")
    if new.chunk_count != old.chunk_count:
        parts.append(f"{new.path}: {new.chunk_count} chunk(s).")
    if new.reason != old.reason and new.reason:
        parts.append(f"{new.path}: {new.reason}.")
    return parts
