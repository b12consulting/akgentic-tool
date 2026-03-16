"""Workspace Protocol, Filesystem implementation, and get_workspace() factory.

Provides a secure, team-scoped filesystem backend for workspace tools.
All path operations validate that the resolved path stays within the workspace root
to prevent directory traversal attacks.

Unknown WORKSPACE_MODE values raise KeyError intentionally (fail-fast behaviour).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class FileEntry(BaseModel):
    """Metadata for a single filesystem entry inside a workspace."""

    name: str
    is_dir: bool
    size: int  # bytes; 0 for directories


@runtime_checkable
class Workspace(Protocol):
    """Protocol that all workspace backends must satisfy."""

    def read(self, path: str) -> bytes: ...

    def write(self, path: str, data: bytes) -> None: ...

    def delete(self, path: str) -> None: ...

    def list(self, path: str = "") -> list[FileEntry]: ...


class Filesystem:
    """Local filesystem backend for a single team workspace.

    All paths are anchored to ``<base_path>/<workspace_name>``.  Any attempt to
    escape that root (via ``../`` traversal or symlinks that resolve outside) is
    rejected with :exc:`PermissionError`.
    """

    def __init__(self, base_path: str, workspace_name: str) -> None:
        self._root = Path(base_path) / workspace_name
        self._root.mkdir(parents=True, exist_ok=True)

    def _validate_path(self, path: str) -> Path:
        """Resolve *path* relative to the workspace root and validate it.

        Raises:
            PermissionError: if the resolved path escapes the workspace root.
        """
        resolved = (self._root / path).resolve()
        if not str(resolved).startswith(str(self._root.resolve())):
            raise PermissionError(f"Path '{path}' escapes workspace root")
        return resolved

    def read(self, path: str) -> bytes:
        """Return the contents of *path* as bytes.

        Raises:
            FileNotFoundError: if the file does not exist.
            PermissionError: if *path* escapes the workspace root.
        """
        resolved = self._validate_path(path)
        return resolved.read_bytes()

    def write(self, path: str, data: bytes) -> None:
        """Write *data* to *path*, creating missing parent directories.

        Raises:
            PermissionError: if *path* escapes the workspace root.
        """
        resolved = self._validate_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_bytes(data)

    def delete(self, path: str) -> None:
        """Delete the file at *path*.

        Raises:
            FileNotFoundError: if the file does not exist.
            PermissionError: if *path* escapes the workspace root.
        """
        resolved = self._validate_path(path)
        resolved.unlink()

    def list(self, path: str = "") -> list[FileEntry]:
        """List immediate children of *path* (non-recursive).

        Returns directories first (alphabetically), then files (alphabetically).
        ``size`` is 0 for directories and the file byte count for regular files.

        Raises:
            PermissionError: if *path* escapes the workspace root.
        """
        resolved = self._validate_path(path) if path else self._root
        entries: list[FileEntry] = []
        dirs: list[FileEntry] = []
        files: list[FileEntry] = []
        for child in resolved.iterdir():
            if child.is_dir():
                dirs.append(FileEntry(name=child.name, is_dir=True, size=0))
            else:
                files.append(
                    FileEntry(name=child.name, is_dir=False, size=child.stat().st_size)
                )
        dirs.sort(key=lambda e: e.name)
        files.sort(key=lambda e: e.name)
        entries = dirs + files
        return entries


def get_workspace(workspace_name: str) -> Filesystem:
    """Return a :class:`Filesystem` for *workspace_name* using the current mode.

    The base path is selected from ``WORKSPACE_MODE`` environment variable:

    - ``local`` (default, or unset): ``./workspaces``
    - ``docker``: ``/workspaces``

    Raises:
        KeyError: if ``WORKSPACE_MODE`` is set to an unknown value (fail-fast).
    """
    mode = os.environ.get("WORKSPACE_MODE", "local")
    base_path = {
        "local": "./workspaces",
        "docker": "/workspaces",
    }[mode]
    return Filesystem(base_path=base_path, workspace_name=workspace_name)
