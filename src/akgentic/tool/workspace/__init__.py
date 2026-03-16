"""Workspace module — filesystem backend for team-scoped file operations."""

from akgentic.tool.workspace.workspace import (
    FileEntry,
    Filesystem,
    Workspace,
    get_workspace,
)

__all__ = ["FileEntry", "Filesystem", "Workspace", "get_workspace"]
