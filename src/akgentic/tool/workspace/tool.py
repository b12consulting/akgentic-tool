"""WorkspaceReadTool — read-only workspace access: read, list, glob, grep.

Provides a ToolCard that exposes four read-only workspace operations as
LLM-callable tools.  All operations are anchored to a team-scoped
:class:`~akgentic.tool.workspace.workspace.Filesystem` backend obtained via
:func:`~akgentic.tool.workspace.workspace.get_workspace`.
"""

from __future__ import annotations

import re as _re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from pydantic import PrivateAttr

from akgentic.tool.core import TOOL_CALL, BaseToolParam, Channels, ToolCard, _resolve
from akgentic.tool.event import ActorToolObserver
from akgentic.tool.workspace.workspace import Filesystem, get_workspace


class WorkspaceRead(BaseToolParam):
    """Read a file from the team workspace with pagination support."""

    expose: set[Channels] = {TOOL_CALL}
    default_limit: int = 2000


class WorkspaceList(BaseToolParam):
    """List immediate children of a directory in the team workspace."""

    expose: set[Channels] = {TOOL_CALL}


class WorkspaceGlob(BaseToolParam):
    """Find files matching a glob pattern in the team workspace."""

    expose: set[Channels] = {TOOL_CALL}
    max_results: int = 100


class WorkspaceGrep(BaseToolParam):
    """Search file contents by regex in the team workspace."""

    expose: set[Channels] = {TOOL_CALL}
    max_results: int = 100
    max_line_length: int = 2000


def _grep_python(
    root: Path,
    pattern: str,
    include_glob: str,
    max_results: int,
    max_line_len: int,
) -> list[tuple[Path, int, str]]:
    """Search files using Python regex — no external dependencies required.

    Args:
        root: Filesystem root to search within.
        pattern: Python regex pattern.
        include_glob: Glob to restrict which files are searched (empty = all).
        max_results: Maximum number of matching lines to return.
        max_line_len: Truncate matching lines to this many characters.

    Returns:
        List of (file_path, line_number, line_text) tuples.
    """
    compiled = _re.compile(pattern)
    results: list[tuple[Path, int, str]] = []
    candidates = sorted(
        root.rglob(include_glob or "*"),
        key=lambda p: p.stat().st_mtime if p.is_file() else 0,
        reverse=True,
    )
    for fpath in candidates:
        if not fpath.is_file():
            continue
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if compiled.search(line):
                results.append((fpath, lineno, line[:max_line_len]))
                if len(results) >= max_results:
                    return results
    return results


def _grep_rg(
    root: Path,
    pattern: str,
    include_glob: str,
    max_results: int,
) -> list[tuple[Path, int, str]] | None:
    """Try ripgrep; return None if rg is not on PATH or exits with error.

    Args:
        root: Filesystem root to search within.
        pattern: Python regex pattern (ripgrep uses the same RE2 syntax).
        include_glob: Glob to restrict which files are searched (empty = all).
        max_results: Maximum number of matching lines to return.

    Returns:
        List of (file_path, line_number, line_text) tuples, or None if rg
        is unavailable or encounters an error.
    """
    if shutil.which("rg") is None:
        return None
    cmd = [
        "rg",
        "--line-number",
        "--no-heading",
        "--hidden",
        "--no-messages",
        "--max-count",
        str(max_results),
    ]
    if include_glob:
        cmd += ["--glob", include_glob]
    cmd += [pattern, str(root)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode not in (0, 1):
        return None
    matches: list[tuple[Path, int, str]] = []
    for line in result.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3:
            try:
                matches.append((Path(parts[0]), int(parts[1]), parts[2]))
            except ValueError:
                continue
    return matches


class WorkspaceReadTool(ToolCard):
    """Read-only workspace access: read, list, glob, grep."""

    name: str = "WorkspaceRead"
    description: str = "Read files, list directories, and search the team workspace"

    workspace_id: str | None = None
    workspace_read: WorkspaceRead | bool = True
    workspace_list: WorkspaceList | bool = True
    workspace_glob: WorkspaceGlob | bool = True
    workspace_grep: WorkspaceGrep | bool = True

    # Private runtime state — not part of the serialised config
    _workspace: Filesystem = PrivateAttr()

    def observer(  # type: ignore[override]
        self, observer: ActorToolObserver
    ) -> "WorkspaceReadTool":
        """Attach observer and initialise the workspace backend.

        Args:
            observer: Actor tool observer; must have a non-None orchestrator.

        Returns:
            Self, enabling method chaining.

        Raises:
            ValueError: If ``observer.orchestrator`` is None.
        """
        self._observer = observer
        if observer.orchestrator is None:
            raise ValueError("WorkspaceReadTool requires access to the orchestrator.")
        ws_name = self.workspace_id or str(observer._team_id)  # type: ignore[attr-defined]
        self._workspace = get_workspace(ws_name)
        return self

    @property
    def workspace(self) -> Filesystem:
        """Return the workspace backend (set after :meth:`observer` is called)."""
        return self._workspace

    def get_tools(self) -> list[Callable[..., Any]]:
        """Return enabled read-only workspace tool callables.

        Returns:
            List of callables for the capabilities not disabled via ``False``.
        """
        tools: list[Callable[..., Any]] = []
        pr = _resolve(self.workspace_read, WorkspaceRead)
        if pr is not None and TOOL_CALL in pr.expose:
            tools.append(self._read_factory(pr))
        pl = _resolve(self.workspace_list, WorkspaceList)
        if pl is not None and TOOL_CALL in pl.expose:
            tools.append(self._list_factory(pl))
        pg = _resolve(self.workspace_glob, WorkspaceGlob)
        if pg is not None and TOOL_CALL in pg.expose:
            tools.append(self._glob_factory(pg))
        pgr = _resolve(self.workspace_grep, WorkspaceGrep)
        if pgr is not None and TOOL_CALL in pgr.expose:
            tools.append(self._grep_factory(pgr))
        return tools

    def _read_factory(self, params: WorkspaceRead) -> Callable[..., Any]:
        """Create the ``workspace_read`` tool callable.

        Args:
            params: Read capability configuration.

        Returns:
            Callable that reads a workspace file with pagination.
        """
        backend = self.workspace

        def workspace_read(path: str, offset: int = 1, limit: int = params.default_limit) -> str:
            """Read a file from the team workspace.

            Args:
                path: Relative path from workspace root (e.g. "src/main.py").
                offset: First line to return, 1-indexed. Defaults to 1.
                limit: Maximum lines to return. Defaults to 2000.

            Returns:
                File contents with 1-indexed line numbers prefixed.
                Truncated files include a trailing notice.

            Raises:
                FileNotFoundError: If the path does not exist.
                PermissionError: If the path escapes the workspace root.
            """
            raw = backend.read(path).decode("utf-8")
            lines = raw.splitlines()
            total = len(lines)
            start = max(0, offset - 1)
            end = min(start + limit, total)
            numbered = "\n".join(
                f"{start + i + 1:<6}{line}" for i, line in enumerate(lines[start:end])
            )
            if end < total:
                numbered += (
                    f"\n[... truncated: {total} lines total, showing {start + 1}-{end} ...]"
                )
            return numbered

        workspace_read.__doc__ = params.format_docstring(workspace_read.__doc__)
        return workspace_read

    def _list_factory(self, params: WorkspaceList) -> Callable[..., Any]:
        """Create the ``workspace_list`` tool callable.

        Args:
            params: List capability configuration.

        Returns:
            Callable that lists workspace directory contents.
        """
        backend = self.workspace

        def workspace_list(path: str = "") -> str:
            """List the contents of a directory in the team workspace.

            Args:
                path: Relative directory path. Defaults to workspace root.

            Returns:
                Newline-separated entries with ``[dir]`` / ``[file]`` prefixes
                and byte sizes for files.  Returns "Empty directory." if the
                directory contains no entries.

            Raises:
                PermissionError: If path escapes the workspace root.
            """
            entries = backend.list(path)
            if not entries:
                return "Empty directory."
            entry_lines: list[str] = []
            for entry in entries:
                if entry.is_dir:
                    entry_lines.append(f"[dir]  {entry.name}")
                else:
                    entry_lines.append(f"[file] {entry.name}  ({entry.size} bytes)")
            return "\n".join(entry_lines)

        workspace_list.__doc__ = params.format_docstring(workspace_list.__doc__)
        return workspace_list

    def _glob_factory(self, params: WorkspaceGlob) -> Callable[..., Any]:
        """Create the ``workspace_glob`` tool callable.

        Args:
            params: Glob capability configuration.

        Returns:
            Callable that searches the workspace via glob patterns.
        """
        backend = self.workspace
        max_results = params.max_results

        def workspace_glob(pattern: str, path: str = "") -> str:
            """Find files matching a glob pattern in the team workspace.

            Args:
                pattern: Glob pattern (e.g. "**/*.py", "src/**/*.ts").
                path: Subdirectory to search within. Defaults to workspace root.

            Returns:
                Newline-separated list of relative file paths, or "No files found."
                Includes truncation notice if more than max_results files matched.

            Raises:
                PermissionError: If path escapes the workspace root.
            """
            if path:
                search_root = (backend._root / path).resolve()
                if not search_root.is_relative_to(backend._root.resolve()):
                    raise PermissionError(f"Path '{path}' escapes workspace root")
            else:
                search_root = backend._root
            all_matches = sorted(
                (match for match in search_root.glob(pattern) if match.is_file()),
                key=lambda match: match.stat().st_mtime,
                reverse=True,
            )
            truncated = len(all_matches) > max_results
            shown = [str(m.relative_to(backend._root)) for m in all_matches[:max_results]]
            if not shown:
                return "No files found."
            result = "\n".join(shown)
            if truncated:
                result += (
                    f"\n[... truncated: {len(all_matches)} total,"
                    f" showing first {max_results} ...]"
                )
            return result

        workspace_glob.__doc__ = params.format_docstring(workspace_glob.__doc__)
        return workspace_glob

    def _grep_factory(self, params: WorkspaceGrep) -> Callable[..., Any]:
        """Create the ``workspace_grep`` tool callable.

        Args:
            params: Grep capability configuration.

        Returns:
            Callable that searches workspace file contents by regex.
        """
        backend = self.workspace
        max_results = params.max_results
        max_line_len = params.max_line_length

        def workspace_grep(pattern: str, path: str = "", include: str = "") -> str:
            """Search file contents using a regex pattern in the team workspace.

            Args:
                pattern: Regular expression pattern (Python re syntax).
                path: Subdirectory to search within. Defaults to workspace root.
                include: Glob pattern to restrict which files are searched
                    (e.g. "*.py", "*.ts"). Empty = all files.

            Returns:
                Formatted results grouped by file, or "No matches found."

            Raises:
                re.error: If pattern is not a valid regex.
                PermissionError: If path escapes the workspace root.
            """
            if path:
                search_root = (backend._root / path).resolve()
                if not search_root.is_relative_to(backend._root.resolve()):
                    raise PermissionError(f"Path '{path}' escapes workspace root")
            else:
                search_root = backend._root

            raw_matches = _grep_rg(search_root, pattern, include, max_results)
            if raw_matches is None:
                raw_matches = _grep_python(
                    search_root, pattern, include, max_results, max_line_len
                )

            if not raw_matches:
                return "No matches found."

            result_lines = [
                f"{fpath.relative_to(backend._root)}:{lineno}: {line}"
                for fpath, lineno, line in raw_matches
            ]
            return "\n".join(result_lines)

        workspace_grep.__doc__ = params.format_docstring(workspace_grep.__doc__)
        return workspace_grep
