"""Workspace Protocol, Filesystem implementation, and get_workspace() factory.

Provides a secure, team-scoped filesystem backend for workspace tools.
All path operations validate that the resolved path stays within the workspace root
to prevent directory traversal attacks.

The workspace root is derived from the ``AKGENTIC_WORKSPACES_ROOT`` environment
variable (default: ``./workspaces``).

:func:`resolve_workspace_path` also lives here — the **one** place a workspace
directory is derived (ADR-048 Decision 5). It sits beside :func:`get_workspace`
because that is the only other function that turns a name into a tree.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import string
from pathlib import Path, PurePosixPath
from typing import Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel

from akgentic.core.utils.serializer import SerializableBaseModel

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


class PathEscapeError(PermissionError):
    """A path that resolved outside the workspace root.

    A subclass rather than a new exception so that every existing
    ``except PermissionError`` keeps catching it and nothing outside changes by
    accident. What it buys is the one distinction the workspace could not make
    before: an *escaping path* and an *OS-denied write* both arrive as
    ``PermissionError``, and telling an agent its path escaped the workspace
    when the path is fine and the file is simply not replaceable sends it
    rewriting a correct path for ever.

    The second case stopped being hypothetical with ``workspace_exec``: a run in
    a container writes as another uid, and publication by rename means the host
    process must be able to replace that inode on the next write.
    """


class FileEntry(BaseModel):
    """Metadata for a single filesystem entry inside a workspace."""

    name: str
    is_dir: bool
    size: int  # bytes; 0 for directories


class WriteEntry(BaseModel):
    """One file's contribution to a batch write.

    A model rather than a tuple because it crosses a boundary: the actor builds
    the batch from what a ``multi_edit`` or a ``patch`` computed in memory, and
    hands it to the backend to publish as a unit.

    Attributes:
        path: Workspace-relative path.
        data: The exact bytes to publish there.
    """

    path: str
    data: bytes


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
            PathEscapeError: if the resolved path escapes the workspace root. It
                is a ``PermissionError``, so existing handlers are unaffected —
                but it lets a caller tell an escaping path apart from an
                OS-level denial on a path that is perfectly legal.
        """
        resolved = (self._root / path).resolve()
        if not resolved.is_relative_to(self._root):
            raise PathEscapeError(f"Path '{path}' escapes workspace root")
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
        staged, resolved = self._stage(path, data)
        try:
            os.replace(staged, resolved)
        except BaseException:
            # Cleanup must not mask the failure that caused it — whatever made the
            # publish fail will often make the unlink fail in the same way.
            with contextlib.suppress(OSError):
                staged.unlink(missing_ok=True)
            raise

    def write_many(self, entries: list[WriteEntry]) -> None:
        """Stage **every** entry, then publish every entry.

        This is the one publication mechanism for a batch — ``multi_edit`` and
        ``patch`` both use it, because two publication paths that differ is how
        one of them drifts.  Staging everything first moves the failures that
        actually happen — a permission error, ENOSPC, a path that escapes the
        root — to a point where **nothing** is visible yet: a batch that fails on
        its second file leaves the first file's previous bytes untouched, and
        every staged file is removed on the way out.

        **What this does not offer, stated plainly.**  N renames are not one
        atomic operation, and nothing on POSIX makes them so.  A rename that
        fails *after* an earlier rename in the same batch succeeded leaves that
        earlier file published, and there is no way to take it back from here:
        the previous inode is already gone.  The remaining staged files are
        cleaned up and the exception is re-raised, so the caller — which still
        holds the original bytes it read — is the only party able to attempt a
        restore.  The window is small and the failure mode is rare; a claim of
        atomicity a reader could disprove would be worse than this note.

        Args:
            entries: The files to publish, in order.  An empty list is a no-op.

        Raises:
            PermissionError: if any entry's path escapes the workspace root.
            OSError: whatever staging or publishing raised.
        """
        staged: list[tuple[Path, Path]] = []
        try:
            for entry in entries:
                staged.append(self._stage(entry.path, entry.data))
        except BaseException:
            for pending, _ in staged:
                with contextlib.suppress(OSError):
                    pending.unlink(missing_ok=True)
            raise
        for index, (source, target) in enumerate(staged):
            try:
                os.replace(source, target)
            except BaseException:
                for pending, _ in staged[index:]:
                    with contextlib.suppress(OSError):
                        pending.unlink(missing_ok=True)
                raise

    def _stage(self, path: str, data: bytes) -> tuple[Path, Path]:
        """Write *data* to a staging file beside *path*, ready to be published.

        Shared by :meth:`write` and :meth:`write_many` so the two cannot drift on
        the details that matter: the staging name ``#Workspace`` sweeps, the
        same-directory placement ``os.replace`` needs, and the mode the target
        keeps.

        Returns:
            The staging file and the resolved target, in that order.

        Raises:
            PermissionError: if *path* escapes the workspace root.
        """
        resolved = self._validate_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        # The ".tmp" suffix is part of the shape three other things match on, and
        # the shape is what makes a staging file recognisable as one: the startup
        # sweep only unlinks a name carrying the full ".<name>.<32 hex>.tmp" form,
        # so a user's own ".notes.tmp" survives; `is_staging_name` is the one
        # definition of it; and the retrieval index's tree walk skips it along
        # with every other dot-prefixed name, so a half-written file is never
        # embedded. (It also kept the file out of the read path's old sidecar
        # rule, which claimed names both starting with "." and ending ".md" —
        # that rule is gone since ADR-045, and the three reasons above are not.)
        stem = resolved.name.encode()[:_STAGED_NAME_BUDGET].decode(errors="ignore")
        staged = resolved.parent / _STAGED_NAME_TEMPLATE.format(stem=stem, token=uuid4().hex)
        try:
            fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _DEFAULT_FILE_MODE)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            if resolved.exists():
                shutil.copymode(resolved, staged)
        except BaseException:
            with contextlib.suppress(OSError):
                staged.unlink(missing_ok=True)
            raise
        return staged, resolved

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


# ---------------------------------------------------------------------------
# The one place a workspace path is derived (ADR-048)
# ---------------------------------------------------------------------------

ANONYMOUS = "anonymous"
"""The ``<scope>`` a principal-less caller lands in.

Not a gap to close: where there is no principal there is no isolation, by
construction. A deployment that supplies no user id gets exactly one scope.
"""

METADATA_SCOPE = "_meta"
"""The ``<scope>`` a metadata-keyed workspace lives under.

Shared across teams and across users — that is its purpose — so it sits under a
reserved scope rather than under anybody's principal.
"""

RESERVED_SCOPES = frozenset({METADATA_SCOPE})
"""Scopes no principal may occupy — reserved for the metadata layout.

**Derived from :data:`METADATA_SCOPE`, never spelled again.** Two literals that
have to agree is the defect this whole module removes, one scale down: a scope
renamed in one of them and not the other would leave the metadata namespace
open to a principal, which is a silent isolation failure.

Matched **exactly**, never as a ``_`` prefix. The Azure AD ``sub`` is base64url
and its alphabet includes ``_``, so reserving the whole underscore namespace
would refuse roughly one user in sixty-four at team creation; a 43-character
``sub`` never equals ``_meta``.
"""

# The only shapes that are not a directory name. Everything else — emails, dots
# mid-name, ``+``, ``=``, unicode — is a legal filename and goes in verbatim.
_UNSAFE = re.compile(r"[/\\\x00]")

# The metadata identifier's two separators, and the safe set that keeps them
# unforgeable. Every character outside the set is percent-encoded, so neither
# separator can occur inside a value.
_META_PAIR_SEP = "-"
_META_JOIN_SEP = "__"
_META_SAFE = frozenset(string.ascii_letters + string.digits + ".")

# The usual filesystem limit for a single name, in **bytes** rather than
# characters: a multibyte value blows it well before 255 characters.
_MAX_LEAF_BYTES = 255


def _unusable_as_segment(value: str) -> bool:
    """Whether *value* cannot be a single directory segment.

    The four shapes a scope and a leaf both reject, in one predicate rather than
    two hand-copied guards that must agree except in one detail — which is the
    defect this whole change removes, one scale down.

    ``.`` and ``..`` are covered by the leading-dot test, which also keeps a
    workspace out of the hidden-directory namespace.
    """
    return not value or value.startswith(".") or _UNSAFE.search(value) is not None


def user_segment(user_id: str | None) -> str:
    """The ``<scope>`` segment for a principal — the user id itself (ADR-048 Decision 4).

    No encoding, no digest, no inverse function: every configured producer
    already emits a directory name. The Azure AD ``sub`` is base64url, an
    unauthenticated deployment yields the literal ``anonymous``, a service
    principal's id is a dashed UUID, and the one free-form producer is an
    admin-supplied ``owner_id`` — typically an email, which is a perfectly legal
    filename. Keeping the raw value is also what makes reading the tree back
    against a team's stored ``user_id`` work with no decoding step.

    Args:
        user_id: The owning principal, or ``None`` for an unauthenticated one.

    Returns:
        The scope segment.

    Raises:
        ValueError: If the value cannot be a directory name (empty, leading
            ``.``, or containing ``/``, ``\\`` or NUL), or if it is a reserved
            scope. Both are silent isolation failures if allowed through: the
            empty string yields a hidden directory every affected user shares,
            a ``/`` yields a nested path where two distinct ids alias, and
            ``_meta`` lands a principal in the shared metadata namespace. The
            empty case is reachable from an OIDC token carrying no ``sub``
            without anyone misbehaving, and it must still raise.
    """
    principal = ANONYMOUS if user_id is None else user_id
    if _unusable_as_segment(principal):
        raise ValueError(f"user_id is not usable as a workspace directory name: {principal!r}")
    if principal in RESERVED_SCOPES:
        raise ValueError(f"user_id may not be a reserved scope: {principal!r}")
    return principal


def leaf_segment(value: str) -> str:
    """The ``<leaf>`` segment — a ``workspace_id``, a team id, or a joined metadata key.

    Identical to :func:`user_segment` except that ``_meta`` is **not** reserved:
    ``_meta`` is a scope, and a workspace legitimately named ``_meta`` under some
    principal collides with nothing.

    A team id is a UUID and a joined metadata leaf is percent-encoded, so both
    pass by construction. The guard exists for the one value an author types by
    hand: without it a ``workspace_id`` of ``../x`` yields ``<user>/../x`` — one
    segment, outside the principal's directory — and ``Filesystem`` validates the
    name it is given not at all.

    Args:
        value: The candidate leaf.

    Returns:
        The leaf segment, unchanged.

    Raises:
        ValueError: If the value cannot be a single directory segment.
    """
    if _unusable_as_segment(value):
        raise ValueError(f"workspace leaf is not usable as a directory name: {value!r}")
    return value


def _encode_metadata_value(value: str) -> str:
    """Percent-encode everything outside ``[A-Za-z0-9.]`` (ADR-048 Decision 3).

    Hand-rolled, and it has to be. ``urllib.parse.quote`` cannot implement this:
    its always-safe set is hard-coded as ``ascii_letters + digits + "_.-~"`` and
    **no** argument — ``safe=""`` included — forces ``_``, ``-`` or ``~`` to
    encode. Those are exactly the characters the separators are built from, so
    ``quote`` would leave the join forgeable while looking like it solved the
    problem: a ``customer_id`` of ``ACME__case_id-42`` under ``["customer_id"]``
    would yield the same string as ``["customer_id", "case_id"]`` over ``ACME``
    and ``42``, and two key sets would silently address one tree.
    """
    return "".join(
        char
        if char in _META_SAFE
        else "".join(f"%{byte:02X}" for byte in char.encode("utf-8"))
        for char in value
    )


def _metadata_leaf(keys: list[str], metadata: SerializableBaseModel | None) -> str:
    """Join the declared keys into one segment: ``<key>-<enc(value)>__<key>-<enc(value)>``.

    **Keys are sorted, not declaration-ordered.** The declaration is a *set*: two
    cards naming the same keys in either order must reach the same workspace.
    ``case_id`` sorting before ``customer_id`` reads less naturally than an
    author would write it, and canonical beats readable — a readable form that
    yields two ids for one key set is the failure being removed here.

    Every failure below is a hard error, never a fallback to a user path. The
    reasons differ but the shape does not: each fallback would be a *silent*
    isolation failure, and the raise happens at team creation in front of the
    admin who caused it.

    Args:
        keys: The metadata fields the card declared. Non-empty.
        metadata: The team's metadata — plain data, whose type this package
            never learns.

    Returns:
        The joined leaf.

    Raises:
        ValueError: If the team carries no metadata (falling back would silently
            *un-share* a workspace declared to be shared), if a declared key is
            not a field of the model (a typo must not resolve to a tree), if a
            value is ``None`` or empty (``case_id-`` would be a real directory
            shared by every team that left it blank), or if the joined leaf
            exceeds 255 bytes (truncating collides, and a collision here is an
            isolation failure that looks like success).
    """
    if metadata is None:
        raise ValueError(
            f"workspace_metadata_keys {sorted(set(keys))!r} were declared, but the team "
            "carries no metadata"
        )
    declared = type(metadata).model_fields
    pairs: list[str] = []
    for key in sorted(set(keys)):
        if key not in declared:
            raise ValueError(
                f"workspace_metadata_keys names {key!r}, which is not a field of "
                f"{type(metadata).__name__}"
            )
        value = getattr(metadata, key)
        if value is None or str(value) == "":
            raise ValueError(f"the team's metadata carries no value for {key!r}")
        pairs.append(f"{key}{_META_PAIR_SEP}{_encode_metadata_value(str(value))}")
    leaf = _META_JOIN_SEP.join(pairs)
    if len(leaf.encode("utf-8")) > _MAX_LEAF_BYTES:
        raise ValueError(
            f"the metadata workspace name exceeds {_MAX_LEAF_BYTES} bytes: {leaf!r}"
        )
    return leaf


def resolve_workspace_path(
    *,
    workspace_id: str | None,
    workspace_metadata_keys: list[str],
    team_id: str,
    user_id: str | None,
    metadata: SerializableBaseModel | None,
) -> PurePosixPath:
    """The workspace's two-segment path. The only place a path is derived.

    ``<scope>`` answers *whose is this*, ``<leaf>`` answers *which of theirs*,
    and the leaf is always a discriminator unique to one workspace — a team id,
    a ``workspace_id``, or a joined metadata key, never a category:

    ==========================================  ==================================
    Card                                        Path
    ==========================================  ==================================
    ``WorkspaceTool()``                         ``<user_segment(user_id)>/<team_id>``
    ``WorkspaceTool(workspace_id="notes")``     ``<user_segment(user_id)>/notes``
    ``WorkspaceTool(workspace_metadata_keys=…)``  ``_meta/case_id-42__customer_id-ACME``
    ==========================================  ==================================

    **Depth is fixed at two, and the invariant that buys is that no workspace
    path is a prefix of another.** ``Filesystem._validate_path`` rejects only
    paths resolving *outside* the root, so a workspace at ``ACME/`` would read
    and write everything under ``ACME/42/`` as ordinary in-tree activity, with
    the per-path write gate none the wiser. That is containment rather than a
    name collision, and it is worse. At depth two with a unique leaf the
    antichain holds by construction; an N-segment layout could be proved one
    only by knowing every other workspace in the deployment.

    The two card fields are mutually exclusive, enforced at card construction
    rather than by precedence here (ADR-048 Decision 2).

    Args:
        workspace_id: The card's named workspace, or ``None`` for the team's own.
        workspace_metadata_keys: The metadata fields keying a shared workspace.
            Empty for the two per-user layouts.
        team_id: The owning team, used as the leaf when no name is declared.
        user_id: The owning principal, or ``None`` for an unauthenticated one.
        metadata: The team's metadata, consulted only when keys are declared.

    Returns:
        A relative two-segment path, to be joined to the workspaces root.

    Raises:
        ValueError: On any input that cannot yield a safe two-segment path — see
            :func:`user_segment`, :func:`leaf_segment` and the metadata
            conditions. **Never** a fallback: a raise out of card binding fails
            team creation in front of the admin who caused it, where a fallback
            would silently collapse several principals into one tree.
    """
    if workspace_metadata_keys:
        return PurePosixPath(METADATA_SCOPE) / _metadata_leaf(workspace_metadata_keys, metadata)
    # Tested against ``None`` rather than for truthiness: a card carrying
    # ``workspace_id=""`` named a workspace and got the name wrong, and falling
    # through to the team id would answer that mistake silently.
    named = team_id if workspace_id is None else workspace_id
    return PurePosixPath(user_segment(user_id)) / leaf_segment(named)
