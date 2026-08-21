"""Workspace Protocol, Filesystem implementation, and get_workspace() factory.

Provides a secure, team-scoped filesystem backend for workspace tools.
All path operations validate that the resolved path stays within the workspace root
to prevent directory traversal attacks.

The workspace root is derived from the ``AKGENTIC_WORKSPACES_ROOT`` environment
variable (default: ``./workspaces``).
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel

# Creation mode for a newly written file, before the process umask is applied by
# the kernel.  Matching what a plain ``open(path, "wb")`` would request keeps the
# staged file's mode identical to an unstaged write's — see Filesystem.write.
_DEFAULT_FILE_MODE = 0o666

# Budget in bytes for the target's own name inside a staging file's name.  The
# affixes around it cost 38 bytes (two dots, 32 hex digits, ".tmp"), and a name
# over the usual 255-byte limit is rejected outright by ``os.open`` — so copying
# a long target name whole would fail writes that used to succeed.
_STAGED_NAME_BUDGET = 255 - 38

# The staging name ``write`` publishes from, and the predicate that recognises
# one afterwards.  They sit together because they are two halves of one shape:
# ``#Workspace`` sweeps orphaned staging files at start, and a reader-side
# pattern that drifted from the writer would either miss them or delete a
# legitimate file.
_STAGED_NAME_TEMPLATE = ".{stem}.{token}.tmp"
_STAGED_NAME_RE = re.compile(r"^\..+\.[0-9a-f]{32}\.tmp$")


def is_staging_name(name: str) -> bool:
    """Whether *name* is a staging file :meth:`Filesystem.write` left behind.

    The 32 hex digits are load-bearing rather than decoration: they are what
    keeps an agent's own ``.notes.tmp`` — or any hand-written ``.tmp`` file — out
    of a sweep that would otherwise delete it.

    Args:
        name: A single path component, not a path.

    Returns:
        True for the ``.<name>.<32 hex digits>.tmp`` shape, False otherwise.
    """
    return _STAGED_NAME_RE.match(name) is not None


class FileEntry(BaseModel):
    """Metadata for a single filesystem entry inside a workspace."""

    name: str
    is_dir: bool
    size: int  # bytes; 0 for directories


@runtime_checkable
class Workspace(Protocol):
    """Protocol that all workspace backends must satisfy."""

    def read(self, path: str) -> bytes: ...

    def read_bytes(self, path: str) -> bytes: ...

    def write(self, path: str, data: bytes) -> None: ...

    def delete(self, path: str) -> None: ...

    def list(self, path: str = "") -> list[FileEntry]: ...

    def mkdir(self, path: str) -> None: ...

    def exists(self, path: str) -> bool: ...


class Filesystem:
    """Local filesystem backend for a single team workspace.

    All paths are anchored to ``<base_path>/<workspace_name>``.  Any attempt to
    escape that root (via ``../`` traversal or symlinks that resolve outside) is
    rejected with :exc:`PermissionError`.
    """

    def __init__(self, base_path: str, workspace_name: str) -> None:
        self._root = (Path(base_path) / workspace_name).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _validate_path(self, path: str) -> Path:
        """Resolve *path* relative to the workspace root and validate it.

        Uses :meth:`Path.is_relative_to` (Python 3.9+) for component-level
        comparison, which prevents false positives when a sibling workspace name
        begins with the same characters (e.g. ``team-1`` vs ``team-11``).

        Raises:
            PermissionError: if the resolved path escapes the workspace root.
        """
        resolved = (self._root / path).resolve()
        if not resolved.is_relative_to(self._root):
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

    def read_bytes(self, path: str) -> bytes:
        """Return the raw bytes of *path* with no decoding or pagination.

        Raises:
            FileNotFoundError: if the file does not exist.
            PermissionError: if *path* escapes the workspace root.
        """
        resolved = self._validate_path(path)
        return resolved.read_bytes()

    def write(self, path: str, data: bytes) -> None:
        """Atomically write *data* to *path*, creating missing parent directories.

        The bytes are staged in a temporary file and published with
        :func:`os.replace`, so a concurrent reader resolves the path to either the
        complete previous file or the complete new one — never to a truncated
        prefix.  Agents each hold their own :class:`Filesystem` and run workspace
        calls on their own thread, so nothing else serializes a writer against a
        reader (see ADR-036, *Filesystem.write becomes atomic*).

        The staging file is created **in the target's own directory**, which is
        load-bearing rather than cosmetic: ``os.replace`` is atomic only within a
        single filesystem, and staging under the default temp directory would
        silently degrade to copy-then-unlink wherever that is a separate mount.

        Permission bits are preserved: an existing target keeps its own mode, and
        a new file gets the mode a plain write would have produced under the
        current umask.  ``os.replace`` publishes the staged inode, so its mode
        becomes the target's — left unset, every written file would become 0600.
        Nothing else the old inode carried survives the swap: ownership reverts to
        the writing process, extended attributes are dropped, and a hardlink to
        the old file detaches.  That is inherent to publishing by rename, and it
        matters where the workspace is bind-mounted into a sandbox container
        running as another uid.

        No ``fsync`` is performed.  The guarantee offered here is atomicity for
        concurrent readers on one machine, not durability across a crash.

        Raises:
            PermissionError: if *path* escapes the workspace root.
        """
        resolved = self._validate_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        # A ".tmp" suffix keeps the staging file out of the read path's sidecar
        # rule, which claims names that both start with "." and end with ".md".
        stem = resolved.name.encode()[:_STAGED_NAME_BUDGET].decode(errors="ignore")
        staged = resolved.parent / _STAGED_NAME_TEMPLATE.format(stem=stem, token=uuid4().hex)
        try:
            fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _DEFAULT_FILE_MODE)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            if resolved.exists():
                shutil.copymode(resolved, staged)
            os.replace(staged, resolved)
        except BaseException:
            # Cleanup must not mask the failure that caused it — whatever made the
            # publish fail will often make the unlink fail in the same way.
            with contextlib.suppress(OSError):
                staged.unlink(missing_ok=True)
            raise

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
                files.append(FileEntry(name=child.name, is_dir=False, size=child.stat().st_size))
        dirs.sort(key=lambda e: e.name)
        files.sort(key=lambda e: e.name)
        entries = dirs + files
        return entries

    def mkdir(self, path: str) -> None:
        """Create directory *path* and all missing parents within the workspace.

        Idempotent — calling on an existing directory is a no-op.

        Raises:
            PermissionError: if *path* escapes the workspace root.
        """
        resolved = self._validate_path(path)
        resolved.mkdir(parents=True, exist_ok=True)

    def exists(self, path: str) -> bool:
        """Return ``True`` if *path* exists inside the workspace.

        Directories count as existing — :meth:`Path.exists` is not file-only.

        Raises:
            PermissionError: if *path* escapes the workspace root.
        """
        return self._validate_path(path).exists()


def get_workspace(workspace_name: str) -> Filesystem:
    """Return a :class:`Filesystem` for *workspace_name* rooted at the configured base.

    The base path is read from the ``AKGENTIC_WORKSPACES_ROOT`` environment
    variable.  When the variable is unset the default ``./workspaces`` is used.

    Args:
        workspace_name: Team-scoped workspace directory name (e.g. ``"team-1"``).

    Returns:
        A :class:`Filesystem` anchored at ``<AKGENTIC_WORKSPACES_ROOT>/<workspace_name>``.
    """
    base_path = os.environ.get("AKGENTIC_WORKSPACES_ROOT", "./workspaces")
    return Filesystem(base_path=base_path, workspace_name=workspace_name)
