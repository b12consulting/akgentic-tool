"""Read-side factories for :class:`WorkspaceTool` — read, list, glob, grep, view.

A read runs on the calling agent's own thread against its own
:class:`~akgentic.tool.workspace.workspace.Filesystem` and reports what it saw
to ``#Workspace`` through a fire-and-forget ``tell`` (ADR-036 §1). Nothing here
asks the actor for anything, so no read can queue behind another agent's write.

:class:`ReadFactories` is a **mixin**: it declares no Pydantic field, and the
two names it consumes off ``self`` are declared under ``if TYPE_CHECKING:`` so
they reach mypy without ever reaching Pydantic's field collection (ADR-045 §1).
"""

from __future__ import annotations

import io
import re as _re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import BinaryContent

from akgentic.tool.errors import RetriableError
from akgentic.tool.workspace.card.params import (
    WorkspaceGlob,
    WorkspaceGrep,
    WorkspaceList,
    WorkspaceRead,
    WorkspaceView,
)
from akgentic.tool.workspace.models import PERM_ERR_MSG
from akgentic.tool.workspace.readers import _MIME_MAP, DocumentReader, MediaContent

if TYPE_CHECKING:
    from akgentic.tool.workspace.workspace import Filesystem

_PERM_ERR_MSG = PERM_ERR_MSG
_REF_RE = _re.compile(r'!!"([^"]+)"|!!(\S+)')
_PILLOW_FMT: dict[str, str] = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
    ".gif": "GIF",
    ".bmp": "BMP",
}
_PILLOW_WARN_EMITTED: bool = False  # guards the one-time Pillow-absent warning


def _maybe_resize(data: bytes, suffix: str, max_dim: int, root: Path, path: str) -> bytes:
    """Resize *data* if longest side exceeds *max_dim*, with sidecar cache.

    When *max_dim* is 0, returns *data* unchanged and writes no sidecar.
    When Pillow is not installed, logs a one-time warning and returns *data* unchanged.

    Sidecar naming: ``.{stem}.{ext}.{max_dim}.{ext}`` colocated with the source file.
    """
    if max_dim == 0:
        return data

    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        global _PILLOW_WARN_EMITTED  # noqa: PLW0603
        if not _PILLOW_WARN_EMITTED:
            import logging  # noqa: PLC0415

            logging.getLogger(__name__).warning(
                "Pillow not installed — sending raw image without resizing. "
                'Install with: pip install "akgentic-tool[vision]"'
            )
            _PILLOW_WARN_EMITTED = True
        return data

    p = Path(path)
    sidecar_name = f".{p.stem}{p.suffix}.{max_dim}{p.suffix}"
    sidecar_path = root / p.parent / sidecar_name

    if sidecar_path.exists():
        return sidecar_path.read_bytes()

    img = Image.open(io.BytesIO(data))
    if max(img.size) <= max_dim:
        return data  # already within limit — no resize, no sidecar

    img.thumbnail((max_dim, max_dim), Image.LANCZOS)  # type: ignore[attr-defined]
    buf = io.BytesIO()
    fmt = _PILLOW_FMT.get(suffix, "JPEG")
    img.save(buf, format=fmt)
    resized = buf.getvalue()
    sidecar_path.write_bytes(resized)
    return resized


def _paginate(raw: str, offset: int, limit: int) -> tuple[str, bool]:
    """Number *raw*'s lines within the requested window.

    Returns:
        The numbered text — with a truncation notice when the window stops short
        — and whether the window covered the **whole** file. The flag is derived
        from the same clamped bounds the text is, so what a read records about
        itself can never disagree with what the agent was shown.
    """
    lines = raw.splitlines()
    total = len(lines)
    start = max(0, offset - 1)
    end = min(start + limit, total)
    numbered = "\n".join(f"{start + i + 1:<6}{line}" for i, line in enumerate(lines[start:end]))
    if end < total:
        numbered += f"\n[... truncated: {total} lines total, showing {start + 1}-{end} ...]"
    return numbered, start == 0 and end == total


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


_BRACE_RE = _re.compile(r"\{([^{}]+)\}")


def _normalize_glob_pattern(pattern: str) -> str:
    """Ensure '**' only appears as a standalone path component.

    Fixes patterns like '**.py' → '**/*.py' that are rejected by Python 3.12
    pathlib.glob() with: ValueError: '**' can only be an entire path component.
    """
    parts = pattern.split("/")
    result: list[str] = []
    for part in parts:
        if "**" in part and part != "**":
            result.append("**")
            remainder = part.replace("**", "*")
            if remainder not in ("", "*"):
                result.append(remainder)
        else:
            result.append(part)
    return "/".join(result)


def _expand_braces(pattern: str) -> list[str]:
    """Expand brace groups in a glob pattern into multiple patterns.

    Handles multiple non-nested brace groups via recursion.
    Patterns without braces are returned as-is (passthrough).

    Args:
        pattern: Glob pattern, potentially containing brace groups like ``{py,js}``.

    Returns:
        List of fully expanded patterns (one entry if no braces found).
    """
    match = _BRACE_RE.search(pattern)
    if not match:
        return [pattern]
    prefix = pattern[: match.start()]
    suffix = pattern[match.end() :]
    alternatives = match.group(1).split(",")
    expanded: list[str] = []
    for alt in alternatives:
        expanded.extend(_expand_braces(f"{prefix}{alt.strip()}{suffix}"))
    return expanded


def _build_tree(
    root: Path,
    prefix: str = "",
    current_depth: int = 0,
    max_depth: int = 0,
) -> list[str]:
    """Render directory entries as an ASCII tree recursively.

    Args:
        root: Filesystem path of the directory to render.
        prefix: Current indentation prefix string for rendering.
        current_depth: Current recursion depth (0 = top level).
        max_depth: Max depth to recurse (0 = unlimited, N = stop at N).

    Returns:
        List of rendered lines (one per entry, no trailing newline).
    """
    try:
        children = list(root.iterdir())
    except PermissionError:
        return []

    dirs = sorted([c for c in children if c.is_dir()], key=lambda c: c.name)
    files = sorted([c for c in children if c.is_file()], key=lambda c: c.name)
    entries = dirs + files

    lines: list[str] = []
    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        if entry.is_dir():
            lines.append(f"{prefix}{connector}{entry.name}/")
            # Recurse if max_depth is unlimited (0) or we haven't reached the limit
            if max_depth == 0 or current_depth + 1 < max_depth:
                extension = "    " if is_last else "│   "
                lines.extend(_build_tree(entry, prefix + extension, current_depth + 1, max_depth))
        else:
            size = entry.stat().st_size
            lines.append(f"{prefix}{connector}{entry.name} ({size} bytes)")

    return lines


class ReadFactories:
    """The read-side factory bodies of :class:`WorkspaceTool`.

    Declares **no Pydantic field**: the annotations below are inside
    ``if TYPE_CHECKING:``, so they are never executed and never reach
    ``__annotations__``, which is where Pydantic v2 collects fields from across
    the MRO. mypy reads them normally.

    ``workspace`` is declared as a read-only *property* rather than a plain
    attribute because that is what the card provides, and mypy refuses to
    override a writeable attribute with a property.
    """

    if TYPE_CHECKING:

        @property
        def workspace(self) -> Filesystem: ...

        def _observation_recorder(self) -> Callable[[str, bytes, bool], None]: ...

    def _expand_media_refs(self, prompt: str) -> list[str | MediaContent]:
        """Expand ``!!glob_pattern`` tokens in a prompt into binary image content.

        Supports both ``!!pattern`` (no spaces) and ``!!"pattern with spaces"`` (quoted).

        For each ``!!pattern`` or ``!!"pattern"`` token:
        - Image matches (extension in ``_MIME_MAP``) → ``MediaContent`` objects (sorted by path)
        - Document-only matches (extension in ``DocumentReader.extensions`` but NOT in
          ``_MIME_MAP``) → ``"!!name[=> Use workspace_read tool]"`` hint strings
        - No matches at all → ``"!!_pattern_[Error: no image found]"``

        Pure-text prompts (no ``!!`` tokens) return ``[prompt]``.

        .. note::
            The returned list may contain trailing empty strings (``""``) when the
            prompt ends with a ``!!token`` with no text following it.  Consumers
            that only care about non-empty parts should filter out empty strings::

                parts = [p for p in result if p != ""]

        Args:
            prompt: Input prompt string potentially containing ``!!glob_pattern`` tokens.

        Returns:
            Mixed list of plain strings and ``MediaContent`` objects.  May include
            trailing ``""`` entries when the last character of *prompt* is part of
            an expanded token.

        Raises:
            RuntimeError: If :meth:`observer` has not been called yet (workspace
                not initialised).
        """
        parts: list[str | MediaContent] = []
        last = 0
        for m in _REF_RE.finditer(prompt):
            if m.start() > last:
                parts.append(prompt[last : m.start()])
            pattern = m.group(1) or m.group(2)
            all_matches = sorted(p for p in self.workspace._root.glob(pattern) if p.is_file())
            image_matches = [p for p in all_matches if p.suffix.lower() in _MIME_MAP]
            doc_matches = [
                p
                for p in all_matches
                if p.suffix.lower() in DocumentReader.extensions
                and p.suffix.lower() not in _MIME_MAP
            ]
            if image_matches:
                for path in image_matches:
                    try:
                        data = path.read_bytes()
                    except OSError:
                        parts.append(f"!!{path.name}[Error: file unreadable]")
                        continue
                    parts.append(
                        MediaContent(
                            data=data,
                            media_type=_MIME_MAP[path.suffix.lower()],
                        )
                    )
            elif doc_matches:
                for path in doc_matches:
                    parts.append(f"!!{path.name}[=> Use workspace_read tool]")
            else:
                parts.append(f"!!{pattern}[Error: no image found in the workspace]")
            last = m.end()
        parts.append(prompt[last:])
        return parts

    def _read_factory(self, params: WorkspaceRead) -> Callable[..., Any]:
        """Create the ``workspace_read`` tool callable.

        Args:
            params: Read capability configuration.

        Returns:
            Callable that reads a workspace file with pagination.
        """
        backend = self.workspace
        record = self._observation_recorder()
        _dr_cfg = params.document_reader
        if _dr_cfg is True:
            document_reader: DocumentReader | None = DocumentReader()
        elif _dr_cfg is False:
            document_reader = None
        else:
            document_reader = _dr_cfg

        def workspace_read(
            path: str,
            offset: int = 1,
            limit: int = params.default_limit,
            force_document_regeneration: bool = params.force_document_regeneration,
        ) -> str:
            """Read a file from the team workspace.

            Args:
                path: Relative path from workspace root (e.g. "src/main.py").
                offset: First line to return, 1-indexed. Defaults to 1.
                limit: Maximum lines to return. Defaults to 2000.
                force_document_regeneration: If True, re-extract binary files
                    even if a cached sidecar exists. Defaults to False.

            Returns:
                File contents with 1-indexed line numbers prefixed.
                Truncated files include a trailing notice.

            Raises:
                RetriableError: If the path does not exist or escapes the
                    workspace root.
                ValueError: If the file is a binary format and
                    ``document_reader`` is not configured.
            """
            ext = Path(path).suffix.lower()
            p = Path(path)

            # Sidecar self-read guard: .report.pdf.md -> plain text
            is_sidecar = p.name.startswith(".") and p.name.endswith(".md")

            # ValueError check outside try -- configuration error, not retryable
            if not is_sidecar and document_reader is None and ext in DocumentReader.extensions:
                raise ValueError(
                    "Binary file reading requires document_reader. "
                    'Install: pip install "akgentic-tool[docs]"'
                )

            try:
                # Raw source bytes, and only on the branch that actually read them.
                # A document read shows extracted Markdown rather than the file, and
                # on a sidecar cache hit the source is never opened at all — hashing
                # it would put a full file read back onto the path NFR1 keeps free.
                observed: bytes | None = None
                if (
                    not is_sidecar
                    and document_reader is not None
                    and ext in document_reader.extensions
                ):
                    # Binary path: sidecar cache or extraction
                    sidecar = backend._root / p.parent / f".{p.name}.md"
                    if sidecar.exists() and not force_document_regeneration:
                        raw = sidecar.read_text(encoding="utf-8")
                    else:
                        content_bytes = backend.read(path)
                        raw = document_reader.extract_text(content_bytes, path)
                        sidecar.write_text(raw, encoding="utf-8")
                else:
                    # Text path (existing logic)
                    observed = backend.read(path)
                    raw = observed.decode("utf-8")

                numbered, full = _paginate(raw, offset, limit)
                if observed is not None:
                    # One recording call per invocation, whatever the file's size.
                    record(path, observed, full)
                return numbered
            except FileNotFoundError:
                raise RetriableError(f"File not found: {path}")
            except PermissionError:
                raise RetriableError(_PERM_ERR_MSG)

        workspace_read.__doc__ = params.format_docstring(workspace_read.__doc__)
        return workspace_read

    def _list_factory(self, params: WorkspaceList) -> Callable[..., Any]:
        """Create the ``workspace_list`` tool callable.

        Args:
            params: List capability configuration.

        Returns:
            Callable that lists workspace directory contents (flat or tree).
        """
        backend = self.workspace

        def workspace_list(path: str = "", depth: int = params.max_depth) -> str:
            """List the contents of a directory in the team workspace.

            Args:
                path: Relative directory path. Defaults to workspace root.
                depth: Tree depth. 1 = flat list (default), 0 = unlimited tree,
                    N > 1 = tree N levels deep.

            Returns:
                Flat list or ASCII tree of entries. Directories shown as ``name/``,
                files as ``name (N bytes)``. Returns "Empty directory." if no entries.

            Raises:
                RetriableError: If the directory does not exist, the path points at
                    a file rather than a directory, or the path escapes the
                    workspace root.
            """
            try:
                if path:
                    resolved = backend._validate_path(path)
                else:
                    resolved = backend._root

                entries = backend.list(path)
                if not entries:
                    return "Empty directory."

                if depth == 1:
                    # Flat list — no tree connectors
                    lines: list[str] = []
                    for entry in entries:
                        if entry.is_dir:
                            lines.append(f"{entry.name}/")
                        else:
                            lines.append(f"{entry.name} ({entry.size} bytes)")
                    return "\n".join(lines)
                else:
                    # ASCII tree — depth=0 means unlimited, depth>1 means N levels
                    tree_lines = _build_tree(resolved, max_depth=depth)
                    return ".\n" + "\n".join(tree_lines) if tree_lines else "Empty directory."
            except (FileNotFoundError, NotADirectoryError):
                raise RetriableError(f"Directory not found: {path}")
            except PermissionError:
                raise RetriableError(_PERM_ERR_MSG)

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
                RetriableError: If path escapes the workspace root.
            """
            try:
                if path:
                    search_root = (backend._root / path).resolve()
                    if not search_root.is_relative_to(backend._root):
                        raise PermissionError(f"Path '{path}' escapes workspace root")
                else:
                    search_root = backend._root
                seen: set[Path] = set()
                raw_matches: list[Path] = []
                for expanded_pattern in _expand_braces(pattern):
                    safe_pattern = _normalize_glob_pattern(expanded_pattern)
                    for m in search_root.glob(safe_pattern, case_sensitive=False):
                        if m.is_file() and m not in seen:
                            seen.add(m)
                            raw_matches.append(m)
                all_matches = sorted(
                    raw_matches,
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
            except PermissionError:
                raise RetriableError(_PERM_ERR_MSG)

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
                RetriableError: If pattern is not a valid regex or path escapes workspace root.
            """
            try:
                if path:
                    search_root = (backend._root / path).resolve()
                    if not search_root.is_relative_to(backend._root):
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
            except PermissionError:
                raise RetriableError(_PERM_ERR_MSG)
            except _re.error as e:
                raise RetriableError(f"Invalid regex pattern: {e}")

        workspace_grep.__doc__ = params.format_docstring(workspace_grep.__doc__)
        return workspace_grep

    def _view_factory(self, params: WorkspaceView) -> Callable[..., Any]:
        """Create the ``workspace_view`` tool callable.

        Args:
            params: View capability configuration.

        Returns:
            Callable that reads an image from the workspace as BinaryContent.
        """
        backend = self.workspace
        max_dim = params.max_dimension

        def workspace_view(path: str) -> BinaryContent:
            """View an image file from the workspace. Returns the image for vision analysis.

            Use this when you need to visually inspect an image (screenshot, diagram, photo).
            For text extraction from documents (PDF, DOCX), use workspace_read instead.

            Supported formats: PNG, JPEG, GIF, WebP, BMP.

            Args:
                path: Relative path to the image file within the workspace.

            Returns:
                BinaryContent with the image bytes and MIME type.

            Raises:
                RetriableError: If the path does not exist, escapes the workspace root,
                    or the file extension is not a supported image format.
            """
            try:
                data = backend.read_bytes(path)
            except FileNotFoundError:
                raise RetriableError(f"File not found: {path}")
            except PermissionError:
                raise RetriableError(_PERM_ERR_MSG)
            try:
                suffix = PurePosixPath(path).suffix.lower()
                mime = _MIME_MAP.get(suffix)
                if mime is None:
                    raise RetriableError(
                        f"Unsupported image format '{suffix}'. "
                        f"Supported: {', '.join(sorted(_MIME_MAP))}. "
                        f"For documents, use workspace_read instead."
                    )
                data = _maybe_resize(data, suffix, max_dim, backend._root, path)
                return BinaryContent(data=data, media_type=mime)
            except RetriableError:
                raise

        workspace_view.__doc__ = params.format_docstring(workspace_view.__doc__)
        return workspace_view
