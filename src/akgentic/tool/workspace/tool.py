"""Compatibility shim for the pre-decomposition ``akgentic.tool.workspace.tool`` module.

The card and its factories moved into :mod:`akgentic.tool.workspace.card`. This
module stays on disk because **the path itself is persisted data**.

``serialize_type()`` writes ``f"{cls.__module__}.{cls.__name__}"`` and the
serializer stamps that string into ``__model__`` on every
:class:`~akgentic.core.utils.SerializableBaseModel` dump. ``BaseToolParam`` is
one, so every ``WorkspaceRead`` / ``WorkspaceExec`` / ``Resource`` / … written
before the decomposition carries the literal string
``akgentic.tool.workspace.tool.<Name>``. Reading such a row back is
``import_module`` plus ``getattr`` on exactly that path. Whether such rows still
exist is a fact about deployed databases, not about this repository, so the path
keeps resolving.

The catalog's ``model_type: akgentic.tool.workspace.WorkspaceTool`` is a
*different* path with a different failure mode, served by the package façade.
Both have to resolve; neither substitutes for the other.

**No deprecation warning, deliberately.** Unlike :mod:`akgentic.tool.vector`,
this is still a documented import path — the package's own ``__init__`` used it
until this decomposition, and the test suite imports from it. Warning here would
fire on healthy code.

Public names only. Private helpers (``_paginate``, ``_grep_rg``,
``_PILLOW_WARN_EMITTED``, …) are **not** re-exported: ``__model__`` markers can
only ever name public models, and a shim that carried private helpers would
invite new code to depend on the old layout.
"""

from __future__ import annotations

from akgentic.tool.workspace.card import (
    ExpandMediaRefs,
    Resource,
    ResourceType,
    WorkspaceDelete,
    WorkspaceEdit,
    WorkspaceExec,
    WorkspaceGlob,
    WorkspaceGrep,
    WorkspaceList,
    WorkspaceMkdir,
    WorkspaceMultiEdit,
    WorkspacePatch,
    WorkspaceRead,
    WorkspaceTool,
    WorkspaceView,
    WorkspaceWrite,
)

# mypy strict implies ``no_implicit_reexport``, and ruff's isort explodes a block of
# ``X as X`` aliases into one import statement per name. ``__all__`` re-exports just
# as deliberately, in a sixteenth of the lines, and doubles as the list of names a
# stored ``__model__`` marker can carry.
__all__ = [
    "ExpandMediaRefs",
    "Resource",
    "ResourceType",
    "WorkspaceDelete",
    "WorkspaceEdit",
    "WorkspaceExec",
    "WorkspaceGlob",
    "WorkspaceGrep",
    "WorkspaceList",
    "WorkspaceMkdir",
    "WorkspaceMultiEdit",
    "WorkspacePatch",
    "WorkspaceRead",
    "WorkspaceTool",
    "WorkspaceView",
    "WorkspaceWrite",
]
