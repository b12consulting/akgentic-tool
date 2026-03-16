"""Workspace module — filesystem backend for team-scoped file operations."""

from akgentic.tool.workspace.edit import (
    EditItem,
    EditMatcher,
    FilePatch,
    Hunk,
    MatchResult,
    apply_file_patch,
    detect_line_ending,
    normalise_endings,
    parse_patch,
)
from akgentic.tool.workspace.tool import (
    WorkspaceGlob,
    WorkspaceGrep,
    WorkspaceList,
    WorkspaceRead,
    WorkspaceReadTool,
)
from akgentic.tool.workspace.workspace import (
    FileEntry,
    Filesystem,
    Workspace,
    get_workspace,
)

__all__ = [
    "EditItem",
    "EditMatcher",
    "FilePatch",
    "Hunk",
    "MatchResult",
    "apply_file_patch",
    "detect_line_ending",
    "normalise_endings",
    "parse_patch",
    "FileEntry",
    "Filesystem",
    "Workspace",
    "get_workspace",
    "WorkspaceRead",
    "WorkspaceList",
    "WorkspaceGlob",
    "WorkspaceGrep",
    "WorkspaceReadTool",
]
