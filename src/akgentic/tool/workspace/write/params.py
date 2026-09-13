"""The write capability's parameters — write, delete, edit, multi-edit, patch, mkdir.

Configuration only. Every class here is a :class:`~akgentic.tool.core.BaseToolParam`
subclass carrying what a capability is configured *with*, never what its callable
is *called* with (ADR-020).

**They live here rather than in ``card/params.py``, and the reason is structural
rather than aesthetic.** Importing ``akgentic.tool.workspace.card.params``
executes ``card/__init__.py`` first — that is how Python imports a submodule —
and that module imports the journal, the exec machinery, the RAG mixin, the
vector-store registry and the actor. A capability module that took its parameters
from there would depend on every other capability by construction, and the
property this package's capability modules exist to have would be unobtainable.

``card/params.py`` keeps **re-exporting** these six names, and that is not
tidiness either: :class:`~akgentic.tool.core.BaseToolParam` is a
:class:`~akgentic.core.utils.SerializableBaseModel`, so a card a deployment
persisted with one of them set explicitly carries
``__model__: akgentic.tool.workspace.card.params.<Name>`` and resolves it with
``import_module`` plus ``getattr``. A path that has gone turns a stored team's
tool card into a bad record rather than a card.
"""

from __future__ import annotations

from akgentic.tool.core import TOOL_CALL, BaseToolParam, Channels


class WorkspaceWrite(BaseToolParam):
    """Write content to a file in the team workspace."""

    expose: set[Channels] = {TOOL_CALL}


class WorkspaceDelete(BaseToolParam):
    """Delete a file from the team workspace."""

    expose: set[Channels] = {TOOL_CALL}


class WorkspaceEdit(BaseToolParam):
    """Apply a surgical find-and-replace edit to a workspace file."""

    expose: set[Channels] = {TOOL_CALL}


class WorkspaceMultiEdit(BaseToolParam):
    """Apply a sequence of find-and-replace edits to workspace files."""

    expose: set[Channels] = {TOOL_CALL}


class WorkspacePatch(BaseToolParam):
    """Apply a unified diff patch to the team workspace."""

    expose: set[Channels] = {TOOL_CALL}


class WorkspaceMkdir(BaseToolParam):
    """Create a directory (and parents) in the team workspace."""

    expose: set[Channels] = {TOOL_CALL}
