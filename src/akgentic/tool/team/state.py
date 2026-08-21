"""Structured context states for the team domain (ADR-037 §5).

``TeamRosterState`` and ``RoleCatalogState`` carry the roster and role-catalog
content that used to be re-rendered into the system prompt. Per-agent shaping is
baked in at production time: rows never contain ``#``-prefixed tool actors, and
``is_self`` is set by the provider — the renderers take no agent argument.
"""

from typing import Self

from akgentic.core.utils import SerializableBaseModel
from akgentic.tool.core import ContextState


class TeamMemberRow(SerializableBaseModel):
    """One team member as the roster state carries it."""

    name: str
    role: str
    is_self: bool


class TeamRosterState(ContextState):
    """The team roster at one point in time, diffable member by member."""

    members: list[TeamMemberRow]

    def render_full(self) -> str:
        """The whole roster, byte-identical to the historical ``team_members`` prompt."""
        if not self.members:
            return ""
        lines = [
            f"{member.name} (role: {member.role})" + (" - [you]" if member.is_self else "")
            for member in self.members
        ]
        return "**Here is the team member list by name (and role):**\n" + "\n".join(lines)

    def render_delta(self, previous: Self) -> str | None:
        """Who joined and who left since ``previous``, keyed on ``(name, role)``.

        A fire-and-rehire of the same name under a different role surfaces
        honestly as left + joined (ADR-037 §6). Unchanged members are never
        re-listed.
        """
        current_keys = {(member.name, member.role) for member in self.members}
        previous_keys = {(member.name, member.role) for member in previous.members}
        joined = [m for m in self.members if (m.name, m.role) not in previous_keys]
        left = [m for m in previous.members if (m.name, m.role) not in current_keys]
        if not joined and not left:
            return None
        lines = [f"{member.name} (role: {member.role}) joined the team." for member in joined]
        lines += [f"{member.name} (role: {member.role}) left the team." for member in left]
        return "\n".join(lines)


class RoleRow(SerializableBaseModel):
    """One hireable role as the catalog state carries it."""

    role: str
    description: str
    skills: list[str]


class RoleCatalogState(ContextState):
    """The hireable-role catalog at one point in time, diffable role by role."""

    roles: list[RoleRow]

    def render_full(self) -> str:
        """The whole catalog, byte-identical to the historical ``team_roles`` prompt."""
        if not self.roles:
            return ""
        lines = [
            f"{row.role}: {row.description} (Skills: {_skills_str(row)})" for row in self.roles
        ]
        return "**Here is the available team role list (for hiring):**\n" + "\n".join(lines)

    def render_delta(self, previous: Self) -> str | None:
        """Roles added, removed, or re-described since ``previous``, keyed on role name.

        A role present on both sides with a changed description or skills list is
        re-described. Unchanged roles are never re-listed.
        """
        previous_by_role = {row.role: row for row in previous.roles}
        current_by_role = {row.role: row for row in self.roles}
        added = [row for row in self.roles if row.role not in previous_by_role]
        removed = [row for row in previous.roles if row.role not in current_by_role]
        redescribed = [
            row
            for row in self.roles
            if row.role in previous_by_role and previous_by_role[row.role] != row
        ]
        if not added and not removed and not redescribed:
            return None
        lines = [
            f"Role added — {row.role}: {row.description} (Skills: {_skills_str(row)})."
            for row in added
        ]
        lines += [f"Role removed — {row.role}." for row in removed]
        lines += [
            f"Role re-described — {row.role}: {row.description} (Skills: {_skills_str(row)})."
            for row in redescribed
        ]
        return "\n".join(lines)


def _skills_str(row: RoleRow) -> str:
    """Comma-joined skills, or the literal ``none`` when the list is empty."""
    return ", ".join(row.skills) if row.skills else "none"
