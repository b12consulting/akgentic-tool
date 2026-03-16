"""Workspace module — filesystem backend for team-scoped file operations."""

<<<<<<< feat/45-5-2-workspace-read-tool
from akgentic.tool.workspace.tool import (
    WorkspaceGlob,
    WorkspaceGrep,
    WorkspaceList,
    WorkspaceRead,
    WorkspaceReadTool,
)
=======
>>>>>>> master
from akgentic.tool.workspace.workspace import (
    FileEntry,
    Filesystem,
    Workspace,
    get_workspace,
)

<<<<<<< feat/45-5-2-workspace-read-tool
__all__ = [
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
=======
__all__ = ["FileEntry", "Filesystem", "Workspace", "get_workspace"]
>>>>>>> master
