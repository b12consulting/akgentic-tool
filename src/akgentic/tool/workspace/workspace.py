"""Workspace Protocol, Filesystem implementation, and get_workspace() factory.

Provides a secure, team-scoped filesystem backend for workspace tools.
All path operations validate that the resolved path stays within the workspace root
to prevent directory traversal attacks.

The workspace root is derived from the ``AKGENTIC_WORKSPACES_ROOT`` environment
variable (default: ``./workspaces``).  :func:`meta_dir_for` derives the metadata
directory that sits **beside** each tree, from ``AKGENTIC_WORKSPACE_META_ROOT``
when it is set and from the same workspaces root when it is not.

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
from akgentic.tool.workspace.models import GIT_DIR_SUFFIX, META_DIR_SUFFIX

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


def _workspaces_root() -> str:
    """The base directory every workspace tree hangs off.

    The ``./workspaces`` default is spelled **here and nowhere else**. Two
    literals that have to agree is this module's own recurring defect — see
    :data:`RESERVED_SCOPES` — and here it would be worse than usual: a metadata
    directory derived from one default and a tree derived from the other would
    not be siblings at all, which is the single property
    :func:`meta_dir_for` exists to hold.
    """
    return os.environ.get("AKGENTIC_WORKSPACES_ROOT", "./workspaces")


def get_workspace(workspace_name: str) -> Filesystem:
    """Return a :class:`Filesystem` for *workspace_name* rooted at the configured base.

    The base path is read from the ``AKGENTIC_WORKSPACES_ROOT`` environment
    variable.  When the variable is unset the default ``./workspaces`` is used.

    Args:
        workspace_name: Team-scoped workspace directory name (e.g. ``"team-1"``).

    Returns:
        A :class:`Filesystem` anchored at ``<AKGENTIC_WORKSPACES_ROOT>/<workspace_name>``.
    """
    return Filesystem(base_path=_workspaces_root(), workspace_name=workspace_name)


def meta_dir_for(workspace_path: str) -> Path:
    """Return the metadata directory belonging to the tree at *workspace_path*.

    A **sibling** of the tree, never a child of it: workspace
    ``<scope>/<kind>/notes`` owns ``<scope>/<kind>/notes.akgentic``, exactly as
    it owns the journal's ``<scope>/<kind>/notes.git``
    (:func:`~akgentic.tool.workspace.journal.git_dir_for`, ADR-051 Decision 9).
    Both land in ``<scope>/<kind>/``, beside the tree, whatever the depth: the
    rule is "sibling of the resolved tree", and it needed no change when the
    kind segment was added.

    **The placement is a containment rule, not a naming preference.**
    :meth:`Filesystem._validate_path` rejects everything that does not resolve
    inside the root, so a path beside the tree is a path no read capability can
    name — not by ``workspace_list``, not by ``workspace_glob``, not by a
    ``../`` traversal. Inside the tree it would be all of those, and one thing
    worse than the journal already faces: the sandbox mounts the tree, so an
    ``rm -rf`` from a sandboxed run could delete the exec lock guarding *that
    very run* — the lock removed by the thing it exists to serialise, with
    nothing raising.

    **Nothing is created here.** The returned directory does not exist unless
    something else made it; whichever caller first needs it creates it.

    Args:
        workspace_path: The three-segment ``<scope>/<kind>/<leaf>`` path
            :func:`get_workspace` takes — not a resolved root. The resolved root
            alone cannot survive ``AKGENTIC_WORKSPACE_META_ROOT``, which
            relocates the metadata *parent*: the scope and kind segments have to
            be carried across the move, and they are unrecoverable from an
            absolute path without also knowing which workspaces root it came
            from.

    Returns:
        The absolute ``<parent>/<scope>/<kind>/<leaf>.akgentic``, where ``<parent>`` is
        ``AKGENTIC_WORKSPACE_META_ROOT`` when it carries a value and the
        workspaces root otherwise — an **empty** value falls back rather than
        being honoured, see below.
    """
    # Derived in the ``git_dir_for`` shape — resolve the tree, then append the
    # suffix to its name — because two derivations that drift give two metadata
    # directories over one tree.
    #
    # ``or`` rather than a ``get`` default, because an empty value is set: a
    # compose file interpolating an unset variable, or a bare ``FOO=`` in an env
    # file, both arrive here as ``""``. Honouring that would resolve the parent
    # against the process cwd while the tree stayed under the workspaces root —
    # the metadata directory detached from the tree it belongs to, with nothing
    # raising. An empty ``AKGENTIC_WORKSPACES_ROOT`` is a different case and
    # keeps its existing meaning (:func:`_workspaces_root`): there the tree and
    # everything derived from it move together.
    parent = os.environ.get("AKGENTIC_WORKSPACE_META_ROOT") or _workspaces_root()
    resolved = (Path(parent) / workspace_path).resolve()
    return resolved.parent / f"{resolved.name}{META_DIR_SUFFIX}"


# ---------------------------------------------------------------------------
# The one place a workspace path is derived (ADR-048)
# ---------------------------------------------------------------------------

ANONYMOUS = "anonymous"
"""The ``<scope>`` a principal-less caller lands in.

Not a gap to close: where there is no principal there is no isolation, by
construction. A deployment that supplies no user id gets exactly one scope.
"""

SHARED_SCOPE = "_shared"
"""The ``<scope>`` of every tree a card declares ``workspace_sharable=True``.

The one scope that is not a principal: a tree under it belongs to nobody's
user id, so every principal whose card declares the same kind and leaf reaches
the same tree. Reserved against principals for exactly that reason — see
:data:`RESERVED_SCOPES`.
"""

TEAM_KIND = "_team"
"""The ``<kind>`` of a card that names no workspace: the leaf is the team id."""

ID_KIND = "_id"
"""The ``<kind>`` of a card declaring ``workspace_id``: the leaf is that name."""

METADATA_KIND = "_meta"
"""The ``<kind>`` of a card declaring ``workspace_metadata_keys``.

``_meta`` is a **kind** now, never a scope: it says how the leaf was derived —
from the team's metadata values — and nothing about who may reach the tree. A
metadata tree is per-principal by default like any other; sharing is
``workspace_sharable``'s to declare, not the kind's to imply.
"""

RESERVED_KINDS = frozenset({TEAM_KIND, ID_KIND, METADATA_KIND})
"""The three ``<kind>`` names — refused as a leaf by :func:`leaf_segment`.

**Derived from the three constants, never spelled again**, on the pattern of
``_SIDECAR_SUFFIXES``: two literals that have to agree is the defect this whole
module removes, one scale down.
"""

RESERVED_SCOPES = frozenset({SHARED_SCOPE}) | RESERVED_KINDS
"""Scopes no principal may occupy: the shared scope, and every kind name.

``_shared`` because a principal of that name would **be** the shared cell — its
per-principal ``_shared/_id/notes`` is the shared ``_shared/_id/notes``. The kind
names so that each reserved name means one thing at every position of a path,
which is what lets a reader classify a tree from its shape alone.

**Derived, never spelled again**, for the reason :data:`RESERVED_KINDS` is: a
name renamed in one set and not the other would open a reserved namespace to a
principal, which is a silent isolation failure.

Matched **exactly**, never as a ``_`` prefix. The Azure AD ``sub`` is base64url
and its alphabet includes ``_``, so reserving the whole underscore namespace
would refuse roughly one user in sixty-four at team creation; a 43-character
``sub`` never equals ``_shared``. Matched **case-insensitively**, because macOS
and Windows filesystems are: ``_SHARED/`` and ``_shared/`` are one directory
there.
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

# Every sibling directory a workspace owns beside its tree, mapped to what it
# holds — one entry per derivation, each keyed on that derivation's **own**
# constant rather than on a second literal that would have to agree with it.
# ``leaf_segment`` refuses a leaf ending in any of them; see its docstring for
# why a name clash here is a containment failure.
_SIDECAR_SUFFIXES = {
    GIT_DIR_SUFFIX: "journal",
    META_DIR_SUFFIX: "metadata",
}


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
            scope in any letter case. Both are silent isolation failures if
            allowed through: the empty string yields a hidden directory every
            affected user shares, a ``/`` yields a nested path where two
            distinct ids alias, and ``_shared`` lands a principal **in** the
            shared cell — its per-principal ``_shared/_id/notes`` is the shared
            ``_shared/_id/notes``, reachable by every card that declares it.
            The empty case is reachable from an OIDC token carrying no ``sub``
            without anyone misbehaving, and it must still raise.
    """
    principal = ANONYMOUS if user_id is None else user_id
    if _unusable_as_segment(principal):
        raise ValueError(f"user_id is not usable as a workspace directory name: {principal!r}")
    if principal.lower() in RESERVED_SCOPES:
        raise ValueError(f"user_id may not be a reserved scope: {principal!r}")
    return principal


def leaf_segment(value: str) -> str:
    """The ``<leaf>`` segment — a ``workspace_id``, a team id, or a joined metadata key.

    Like :func:`user_segment` except in what it reserves: a leaf may not be one
    of the three **kind** names (``_team``, ``_id``, ``_meta``), and it may not
    end in ``.git`` or ``.akgentic``. ``_shared`` is not reserved here — it is a
    scope, and a leaf of that name sits at the third position, where it collides
    with nothing.

    A team id is a UUID, so it passes by construction, and a joined metadata
    leaf always contains ``-``, so it never equals a kind name. The guards exist
    for the values that do not pass by construction: a ``workspace_id`` an author
    types by hand, and a joined metadata leaf built out of business data.

    **Why a kind name is refused as a leaf — containment across depths, not a
    collision.** Among three-segment paths a kind-named leaf meets nothing:
    ``alice/_id/_team`` is the only path of that name. What it would reach is a
    tree of a *different depth*. The only shorter path that can contain a
    three-segment tree is ``<scope>/<kind>`` — a two-segment path whose leaf is a
    kind name — and the layout this one replaced accepted exactly that:
    ``workspace_id="_meta"`` resolved to ``alice/_meta``, a legal tree on disk
    which is the **parent** of every ``alice/_meta/*`` tree minted now. An agent
    anchored there reads and writes all of them as ordinary in-tree activity,
    for the reason given below for the suffixes. Refusing kind names as leaves
    means nothing from here on can mint such a path, and keeps each reserved name
    meaning one thing at every position — which is what lets a reader classify a
    tree from its shape. The trees already on disk are the migration's to find.

    The kind match is **exact** — ``_teams``, ``_ids`` and ``_metadata`` are
    ordinary names — and **case-insensitive**, for the reason the suffix match
    below is.

    **Why these suffixes are a containment failure and not a name clash.** Both
    of a workspace's sidecar directories are *siblings of the tree, in the same
    directory*: ``git_dir_for`` returns ``<root>.git`` and :func:`meta_dir_for`
    returns ``<root>.akgentic``, so ``workspace_id="notes"`` owns
    ``<scope>/_id/notes``, ``<scope>/_id/notes.git`` and
    ``<scope>/_id/notes.akgentic``. A second card declaring
    ``workspace_id="notes.git"`` therefore roots its **tree** at the first
    workspace's **git repository**, and its agent lists, reads, writes and
    deletes inside another workspace's history as ordinary in-tree activity —
    ``Filesystem._validate_path`` rejects only what resolves *outside* the root,
    and that root is a perfectly real directory. From the other side, the first
    workspace's commits surface as files in the second's tree. Nothing raises.
    It is the same failure the fixed three-segment depth removes, arriving
    through a suffix instead of through a slash.

    ``workspace_id="notes.akgentic"`` is that failure again, over the directory
    holding the exec lock, the document cache and the retrieval index — every
    one of which is placed outside the tree *precisely* so that no agent can
    reach it, and all of which this card would then hold as its own files.

    **Rejecting beats renaming.** No suffix-stripping and no relocate-and-log: a
    deployment that genuinely has a workspace named ``foo.git`` must be told, at
    team creation, in front of the admin who caused it. Silently moving somebody
    else's tree is the failure, not the remedy.

    **The match is case-insensitive** although both derivations only ever emit
    lowercase, because macOS and Windows filesystems are case-insensitive by
    default: ``<scope>/_id/notes.GIT`` and ``<scope>/_id/notes.git`` are one
    directory there, so an exact-match guard would pass the collision straight
    through on the platform most of this is developed on.

    **This does not make the journal's own guard redundant.** ``GitJournal``
    refuses to initialise when the root it was handed ends in ``.git``, and that
    covers the layer this function cannot see: ``Filesystem`` and
    :func:`get_workspace` take a name that never passes through here — which is
    how ``akgentic-infra`` calls them today.

    Args:
        value: The candidate leaf.

    Returns:
        The leaf segment, unchanged.

    Raises:
        ValueError: If the value cannot be a single directory segment, if it is
            a kind name in any letter case, or if it ends in ``.git`` or
            ``.akgentic``.
    """
    if _unusable_as_segment(value):
        raise ValueError(f"workspace leaf is not usable as a directory name: {value!r}")
    if value.lower() in RESERVED_KINDS:
        raise ValueError(
            f"workspace leaf may not be {value!r}: it is a reserved kind name, and a "
            "reserved name means one thing at every position of a workspace path"
        )
    for suffix, owner in _SIDECAR_SUFFIXES.items():
        if value.lower().endswith(suffix):
            raise ValueError(
                f"workspace leaf may not end in {suffix!r}, which is another "
                f"workspace's {owner} directory: {value!r}"
            )
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

    **Keys stay in declaration order** (ADR-048 Decision 3). The declaration is a
    *sequence*, not a set: it is an ordered refinement path whose first key is the
    coarsest scope, so ``["customer_id", "case_id"]`` yields
    ``customer_id-ACME__case_id-42`` and ``ls <scope>/_meta/`` groups a
    customer's workspaces beside each other instead of scattering them under
    whichever key happened to sort first.

    The trade, stated so nobody reads it as a defect: two cards naming the same
    keys in **different orders** address **different** workspaces. Under the
    sequence model that is honest — they declared different scopes — and unlike a
    silent collision it is visible in the directory name.

    The prefix this creates is a **string** prefix between two *siblings*, never a
    path prefix: ``<scope>/_meta/customer_id-ACME`` and
    ``<scope>/_meta/customer_id-ACME__case_id-42`` are two leaves under one
    ``<scope>/_meta``, and a sibling cannot contain a sibling. The antichain is
    about one path *containing* another and is untouched here.

    Duplicates are removed keeping the **first** occurrence, so a repeated key
    adds no scope and names the tree it named once.

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
            re-home the tree onto a path the card never declared), if a declared
            key is not a field of the model (a typo must not resolve to a tree), if a
            value is ``None`` or empty (``case_id-`` would be a real directory
            shared by every team that left it blank), or if the joined leaf
            exceeds 255 bytes (truncating collides, and a collision here is an
            isolation failure that looks like success).
    """
    # One ordered, deduped list, computed once and used by both the raise below
    # and the join: two spellings of one rule is this module's own recurring
    # defect one scale down. ``dict.fromkeys`` keeps the first occurrence, which
    # is what makes the dedupe stable — ``set`` is what must not appear.
    ordered = list(dict.fromkeys(keys))
    if metadata is None:
        raise ValueError(
            f"workspace_metadata_keys {ordered!r} were declared, but the team "
            "carries no metadata"
        )
    declared = type(metadata).model_fields
    pairs: list[str] = []
    for key in ordered:
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
    workspace_sharable: bool,
) -> PurePosixPath:
    """The workspace's three-segment path. The only place a path is derived.

    ``<scope>`` answers *who may reach this*, ``<kind>`` answers *how was the
    leaf derived*, and ``<leaf>`` answers *which one* — always a discriminator
    unique to one workspace of that kind, never a category:

    ==============================================  =============================
    Card                                            Path
    ==============================================  =============================
    ``WorkspaceTool()``                             ``alice/_team/<team_id>``
    ``WorkspaceTool(workspace_id="notes")``         ``alice/_id/notes``
    ``WorkspaceTool(workspace_metadata_keys=…)``    ``alice/_meta/customer_id-ACME``
    ``… workspace_sharable=True`` (each of above)   ``_shared/<kind>/<leaf>``
    ==============================================  =============================

    ``alice`` is :func:`user_segment` of the principal. Per-principal is the
    default for **every** kind, metadata included; ``workspace_sharable=True``
    swaps the principal for :data:`SHARED_SCOPE` and changes nothing else.

    **Depth is fixed at three, and the invariant that buys is that no workspace
    path is a proper prefix of another.** No path of exactly three segments can
    be a proper prefix of another, because a proper prefix has strictly fewer
    segments. Fixed depth buys the property; two was never load-bearing, only
    fixed. The hazard it removes is **containment**, not collision:
    ``Filesystem._validate_path`` rejects only paths resolving *outside* the
    root, so a workspace anchored at a parent would read and write everything
    under a child's tree as ordinary in-tree activity, with the per-path write
    gate none the wiser.

    **``workspace_sharable`` is a request.** This resolver honours it for every
    kind; whether a deployment permits a shared tree of a given kind is a
    separate check at bind time.

    The two layout fields are mutually exclusive, enforced at card construction
    rather than by precedence here (ADR-048 Decision 2); ``workspace_sharable``
    is orthogonal to both.

    Args:
        workspace_id: The card's named workspace, or ``None`` for the team's own.
        workspace_metadata_keys: The metadata fields keying the workspace. Empty
            for the other two kinds.
        team_id: The owning team, used as the leaf when no name is declared.
        user_id: The owning principal, or ``None`` for an unauthenticated one.
            Not consulted at all when *workspace_sharable* is set.
        metadata: The team's metadata, consulted only when keys are declared.
        workspace_sharable: Whether the tree lives under the shared scope rather
            than under the principal. **Required, with no default**: a caller
            that does not know about sharing must fail here with a
            ``TypeError``, not silently resolve a shared card's tree to a
            per-principal path nobody else can see.

    Returns:
        A relative three-segment path, to be joined to the workspaces root.

    Raises:
        ValueError: On any input that cannot yield a safe three-segment path —
            see :func:`user_segment`, :func:`leaf_segment` and the metadata
            conditions. **Never** a fallback: a raise out of card binding fails
            team creation in front of the admin who caused it, where a fallback
            would silently collapse several principals into one tree.
    """
    if workspace_metadata_keys:
        # Through ``leaf_segment`` like any other leaf, rather than a second
        # suffix check beside it — duplicating a rule is the defect this module
        # exists to remove. ``.`` is inside the encoder's safe set, so a
        # ``customer_id`` of ``x.git`` survives encoding whole and yields the leaf
        # ``customer_id-x.git``, which *is* the journal directory of
        # ``<scope>/_meta/customer_id-x``. The same collision as the hand-typed
        # ``workspace_id="notes.git"``, reached from business data rather than
        # from a card field anybody chose.
        kind, leaf = METADATA_KIND, _metadata_leaf(workspace_metadata_keys, metadata)
    elif workspace_id is not None:
        # Tested against ``None`` rather than for truthiness: a card carrying
        # ``workspace_id=""`` named a workspace and got the name wrong, and
        # falling through to the team id would answer that mistake silently.
        kind, leaf = ID_KIND, workspace_id
    else:
        kind, leaf = TEAM_KIND, team_id
    # The shared scope never consults the principal: a resolver must not refuse
    # on an input it does not use.
    scope = SHARED_SCOPE if workspace_sharable else user_segment(user_id)
    return PurePosixPath(scope) / kind / leaf_segment(leaf)
