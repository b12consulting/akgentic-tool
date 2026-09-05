"""``workspace_rag_list`` as structured context state (ADR-037 §3, ADR-045 §5).

The retrieval index moves — a file goes ``pending`` → ``splitting`` →
``embedding`` → ``embedded`` over the course of a few turns — and re-rendering
the whole table into the system prompt on every change would invalidate the
cached prompt prefix each time. So it is a :class:`ContextState` instead: the
first turn sees the table, every later turn sees only what moved.

Nothing here reads the tree, opens a file or touches an actor. The rows arrive
already shaped from ``#Workspace``'s own dict, which is the whole reason a
per-turn render can be free.
"""

from __future__ import annotations

from typing import Self

from akgentic.core.utils import SerializableBaseModel
from akgentic.tool.core import ContextState

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
