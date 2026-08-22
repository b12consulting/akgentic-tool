"""Structured context state for the knowledge-graph domain (ADR-037 §5).

``KnowledgeGraphSummaryState`` carries the compact graph summary that used to
be re-rendered into the system prompt. The ``prompt_include_schema`` /
``prompt_include_roots`` flags are baked in at production time — the renderers
take no configuration argument — and gate their blocks in both renderings.
The state stays O(types + roots), never O(entities): counts, deduplicated
type lists, and root rows only.
"""

from typing import Self

from akgentic.core.utils import SerializableBaseModel
from akgentic.tool.core import ContextState


class RootRow(SerializableBaseModel):
    """One root entity as the summary state carries it."""

    name: str
    entity_type: str
    description: str


class KnowledgeGraphSummaryState(ContextState):
    """The compact knowledge-graph summary at one point in time, diffable.

    ``entity_types`` / ``relation_types`` are sorted and deduplicated;
    ``roots`` is sorted by name. ``include_schema`` and ``include_roots``
    carry the resolved ``GetGraph`` prompt flags: the full type/root data is
    always present, the flags only gate rendering.
    """

    entity_count: int
    relation_count: int
    entity_types: list[str]
    relation_types: list[str]
    roots: list[RootRow]
    include_schema: bool
    include_roots: bool

    def render_full(self) -> str:
        """The whole summary, byte-identical to the historical graph prompt.

        An empty graph renders the sentinel line — a real state, distinct from
        a provider returning ``None``.
        """
        if self.entity_count == 0:
            return "Knowledge graph is empty."

        lines = ["**Knowledge Graph Summary:**"]
        lines.append(f"Entities: {self.entity_count} | Relations: {self.relation_count}")

        if self.include_schema:
            lines.append(f"Entity types: {', '.join(self.entity_types)}")
            if self.relation_types:
                lines.append(f"Relation types: {', '.join(self.relation_types)}")

        if self.include_roots and self.roots:
            lines.append("Root entities:")
            for row in self.roots:
                lines.append(f"- {row.name} ({row.entity_type}): {row.description}")

        lines.append("")
        lines.append("Use the get_graph tool to explore the full graph or subgraphs.")
        return "\n".join(lines)

    def render_delta(self, previous: Self) -> str | None:
        """What moved since ``previous``, computed on the model fields (ADR-037 §6).

        Count movement states the new totals with the signed delta, only for
        counts that moved. Type changes render once each, gated on
        ``include_schema``; root changes render once each, keyed by name and
        gated on ``include_roots``. Unchanged collections are never re-listed.
        """
        parts = _count_movement(previous, self)
        if self.include_schema:
            parts.extend(_type_changes("entity type", previous.entity_types, self.entity_types))
            parts.extend(
                _type_changes("relation type", previous.relation_types, self.relation_types)
            )
        if self.include_roots:
            parts.extend(_root_changes(previous.roots, self.roots))
        return " ".join(parts) if parts else None


def _signed(delta: int) -> str:
    """A signed delta label: ``+3`` or ``-2``."""
    return f"+{delta}" if delta > 0 else str(delta)


def _count_movement(
    previous: KnowledgeGraphSummaryState, current: KnowledgeGraphSummaryState
) -> list[str]:
    """The new totals with signed deltas, one segment per count that moved."""
    segments: list[str] = []
    if current.entity_count != previous.entity_count:
        entity_delta = current.entity_count - previous.entity_count
        segments.append(f"Entities: {current.entity_count} ({_signed(entity_delta)})")
    if current.relation_count != previous.relation_count:
        relation_delta = current.relation_count - previous.relation_count
        segments.append(f"Relations: {current.relation_count} ({_signed(relation_delta)})")
    return [" | ".join(segments) + "."] if segments else []


def _type_changes(label: str, old: list[str], new: list[str]) -> list[str]:
    """One short sentence per type that appeared or disappeared."""
    old_set, new_set = set(old), set(new)
    parts = [f"New {label}: {name}." for name in new if name not in old_set]
    parts.extend(f"Removed {label}: {name}." for name in old if name not in new_set)
    return parts


def _root_changes(old: list[RootRow], new: list[RootRow]) -> list[str]:
    """One short sentence per root added, removed, or changed, keyed by name."""
    old_by_name = {row.name: row for row in old}
    new_names = {row.name for row in new}
    parts: list[str] = []
    for row in new:
        prev = old_by_name.get(row.name)
        if prev is None:
            parts.append(f"New root: {row.name} ({row.entity_type}).")
        elif prev != row:
            parts.append(f"Changed root: {row.name} ({row.entity_type}): {row.description}.")
    parts.extend(f"Removed root: {row.name}." for row in old if row.name not in new_names)
    return parts
