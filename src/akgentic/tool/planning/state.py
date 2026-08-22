"""Structured context state for the planning domain (ADR-037 §5).

``PlanningState`` carries the planning-summary content that used to be
re-rendered into the system prompt. Per-agent shaping is baked in at production
time: ``tasks`` holds only the rows the observing agent should see, while
``total`` and ``owner_counts`` always cover the whole board — the renderers
take no agent argument.
"""

from typing import Self

from akgentic.core.utils import SerializableBaseModel
from akgentic.tool.core import ContextState


class TaskRow(SerializableBaseModel):
    """One visible task as the planning state carries it.

    ``output`` carries the task's output text (empty string when none);
    "has output" is its truthiness.
    """

    id: int
    status: str
    description: str
    owner: str
    creator: str
    output: str


class PlanningState(ContextState):
    """The team planning board at one point in time, diffable task by task.

    ``total`` and ``owner_counts`` cover ALL tasks on the board; ``tasks``
    holds only the rows the observing agent should see (all of them when
    ``filter_by_agent`` is off).
    """

    total: int
    owner_counts: dict[str, int]
    tasks: list[TaskRow]
    agent_name: str
    filter_by_agent: bool

    def render_full(self) -> str:
        """The whole board, byte-identical to the historical ``planning_summary`` prompt.

        An empty board renders the sentinel line — a real state, distinct from
        a provider returning ``None``.
        """
        if self.total == 0:
            return "No current team planning."

        lines = [
            f"**Team planning:** {self.total} task{'s' if self.total != 1 else ''} total",
            f"Owners: {self._breakdown()}",
        ]
        if self.filter_by_agent:
            if self.tasks:
                lines.append(f"\n**Your tasks** (owner or creator: {self.agent_name}):")
                lines.extend(_task_line(row) for row in self.tasks)
            else:
                lines.append(f"\nNo tasks assigned to or created by {self.agent_name} yet.")
        else:
            lines.append("\n**All tasks:**")
            lines.extend(_task_line(row) for row in self.tasks)

        lines.append(
            "\nUse get_planning_task(id) for exact ID lookup or "
            "search_planning(...) to filter tasks."
        )
        return "\n".join(lines)

    def render_delta(self, previous: Self) -> str | None:
        """What moved since ``previous``, keyed on the task ``id`` (ADR-037 §6).

        A task appearing in ``tasks`` renders as new; a task leaving renders as
        removed — whether deleted from the board or merely gone from this
        agent's filtered view. A row on both sides renders only the fields that
        moved. The new total is stated only when ``total`` changed; unchanged
        rows are never re-listed.
        """
        previous_by_id = {row.id: row for row in previous.tasks}
        current_ids = {row.id for row in self.tasks}

        parts: list[str] = []
        if self.total != previous.total:
            parts.append(f"{self.total} task{'s' if self.total != 1 else ''} total.")
        for row in self.tasks:
            old = previous_by_id.get(row.id)
            if old is None:
                parts.append(
                    f"New: ID {row.id} [{row.status}] {row.description} "
                    f"(Owner: {row.owner or 'unassigned'}, Creator: {row.creator})."
                )
            elif old != row:
                parts.extend(_row_changes(old, row))
        parts.extend(
            f"Removed: ID {row.id} [{row.status}] {row.description}."
            for row in previous.tasks
            if row.id not in current_ids
        )
        return " ".join(parts) if parts else None

    def _breakdown(self) -> str:
        """Per-owner counts: named owners alphabetically, ``unassigned`` last."""
        named = sorted((k, v) for k, v in self.owner_counts.items() if k != "unassigned")
        parts = [f"{name}: {count}" for name, count in named]
        unassigned_count = self.owner_counts.get("unassigned", 0)
        if unassigned_count:
            parts.append(f"unassigned: {unassigned_count}")
        return " | ".join(parts)


def _task_line(row: TaskRow) -> str:
    """One ``- ID ...`` task line, byte-identical to the historical prompt."""
    output_part = f" — Output: {row.output}" if row.output else ""
    owner_label = row.owner or "unassigned"
    suffix = f" (Owner: {owner_label}, Creator: {row.creator})"
    return f"- ID {row.id} [{row.status}] {row.description}{output_part}{suffix}"


def _row_changes(old: TaskRow, new: TaskRow) -> list[str]:
    """One short sentence per field that moved between two rows with the same id."""
    owner_label = new.owner or "unassigned"
    parts: list[str] = []
    if new.status != old.status:
        parts.append(f"ID {new.id} [{old.status}] → [{new.status}] (Owner: {owner_label}).")
    if new.owner != old.owner:
        parts.append(f"ID {new.id} owner: {old.owner or 'unassigned'} → {owner_label}.")
    if new.description != old.description:
        parts.append(f"ID {new.id} description: {new.description}.")
    if new.output != old.output:
        if new.output:
            parts.append(f"ID {new.id} output: {new.output}.")
        else:
            parts.append(f"ID {new.id} output cleared.")
    return parts
