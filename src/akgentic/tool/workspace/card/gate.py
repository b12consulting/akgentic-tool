"""The six mutations, the live-hash gate, the observation map and the refusal vocabulary.

Every mutation the card accepts runs **here**, on the card's own tree handle, and
only after the live file still matches what this agent observed (ADR-036 §3).
The check and the write are one *locked region* — returning a verdict and letting
the caller write would reopen the window the gate exists to close.

**What closes that window is an ``fcntl.flock`` on a file, not a mailbox**
(ADR-051 Decision 3). An actor's mailbox serialises one process; two workers over
one mounted tree run two actors that cannot see each other, and the file lock is
exclusive exactly where the mailbox is not. It follows that **no single-process
test can prove this works**: a ``threading.Lock``, one actor's mailbox and the
GIL all serialise one interpreter, so a spec that contends two threads stays
green with the ``flock`` deleted outright. The guards that bite spawn real
interpreters.

**The hash is read from disk on every check, never cached.** A
``{path -> current_sha}`` map would pass almost every test written against this
module and fail exactly one: the file written behind the gate's back. The lock
orders the writers that come *through* here; the hash is what catches everyone
else — an upload, a sandbox run, another team, a human with an editor. Do not
optimise :meth:`CardGate._live` into a cache.

:meth:`CardGate._gated` is the one point all six mutations converge on, so a
seventh added later cannot forget the busy check, the out-of-band commit, the
lock, the agent's own commit or the stale-mark.

**The observation map is this agent's and this agent's only**, an
``OrderedDict`` that is the whole of the LRU — recording moves an entry to the
end, eviction pops from the front. There is no lock on it and none is needed:
one card belongs to one agent, and only that agent's own calls touch it
(ADR-051 Decision 2).

:class:`CardGate` is a **mixin**: it declares no Pydantic field, and the names it
consumes off ``self`` are declared under ``if TYPE_CHECKING:`` so they reach mypy
without ever reaching Pydantic's field collection (ADR-045 §1).
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import logging
import os
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from akgentic.tool.workspace.edit import (
    EditItem,
    EditMatcher,
    FilePatch,
    HunkContextError,
    deleted_paths,
    detect_line_ending,
    is_pure_add,
    normalise_endings,
    parse_patch,
    patch_label,
    render_file_patch,
    substitute_edit,
    unified,
    write_and_diff,
)
from akgentic.tool.workspace.execution import mutation_busy
from akgentic.tool.workspace.journal import GitJournal, Identity
from akgentic.tool.workspace.lock import LockBackend
from akgentic.tool.workspace.models import (
    MAX_REJECTION_DIFF_LINES,
    OUT_OF_BAND_AUTHOR,
    PERM_ERR_MSG,
    PUBLISH_LOST_MSG,
    WRITE_DENIED_MSG,
    MutationOutcome,
    MutationStatus,
    Observation,
    Precondition,
    content_sha,
)
from akgentic.tool.workspace.workspace import (
    Filesystem,
    PathEscapeError,
    WriteEntry,
)

if TYPE_CHECKING:
    from akgentic.tool.workspace.actor import WorkspaceActor

logger = logging.getLogger(__name__)

LOCKS_DIR_NAME = "locks"
"""The directory under ``<meta>`` holding one lock file per contended path."""

PATH_LOCK_PREFIX = "path-"
"""What a path's lock file is named, before the digest of the path itself."""

_LOCK_FILE_MODE = 0o600
"""Mode a lock file is created with — nothing outside the owner needs it."""

_MATCHER = EditMatcher()
"""The anchor cascade, shared by every card in the process.

**A module-level instance, not a card attribute**, because the cascade is pure:
seven strategies over the two strings it is handed, one class constant, and not a
byte of per-card state. One per card would be a private attribute whose default
differs between two otherwise identical cards — which breaks ``WorkspaceTool``
equality, and therefore the round-trip a catalog depends on, for nothing.
"""

_UNBOUND_MSG = (
    "The workspace is not bound — a mutating WorkspaceTool must be wired "
    "through observer() with a live orchestrator."
)

##
## Refusal wording.  One builder composes every rejection from these, so the six
## mutation methods cannot drift apart — and the LLM reads them, so they say what
## to do next before they say what went wrong.
##
_REASON_UNREAD_WRITE = "it already exists and you have not read it — read it before overwriting"
_REASON_UNREAD_EDIT = "it already exists and you have not read it — read it before editing"
_REASON_CHANGED = "it changed since you read it"
_REASON_GONE = "it was deleted since you read it"
_REASON_PARTIAL = "you read only part of it, and a page is not a licence to replace the whole file"
_REASON_EXACT_MISS = (
    "it changed since you read it, and your old_string no longer matches it exactly — "
    "approximate matching is disabled on a file another writer has touched"
)
_REASON_PATCH_STALE = (
    "it changed since you read it, and none of your patch's context could be found in "
    "what is there now — its hunks address line numbers that have since moved"
)

_CHANGE_REASONS = frozenset({_REASON_CHANGED, _REASON_EXACT_MISS, _REASON_PATCH_STALE})
"""Reasons that earn a provenance line at all.

Each describes a file whose *content* moved under the agent while it still
exists, so the live bytes have an author and the journal either names them or
establishes that no agent wrote them.

``_REASON_GONE`` is deliberately absent. A deleted file has no live bytes to
attribute, so nothing distinguishes another agent's ``workspace_delete`` from an
outside removal. Claiming "it came from outside the workspace tools" there would
be a guess stated as a fact, and would misattribute a teammate's delete exactly
as naming the wrong writer would.
"""

_OUT_OF_BAND = (
    "No agent in this team wrote what is there now — it came from outside the "
    "workspace tools (an upload, a sandbox run, or another team)."
)
_NEXT_STEP = "Read the file again, reconsider your change against what is there now, then retry."


def _accepted(message: str) -> MutationOutcome:
    """The mutation happened; *message* is what the agent is told."""
    return MutationOutcome(status=MutationStatus.ACCEPTED, message=message)


def _rejected(message: str) -> MutationOutcome:
    """The mutation did not happen and the agent must react — raised as retriable."""
    return MutationOutcome(status=MutationStatus.REJECTED, message=message)


def _failed(message: str) -> MutationOutcome:
    """The mutation did not happen; *message* is **returned**, not raised."""
    return MutationOutcome(status=MutationStatus.FAILED, message=message)


def _denied(exc: PermissionError) -> MutationOutcome:
    """Refuse a mutation the filesystem would not allow, saying which kind of denial.

    Both arrive as ``PermissionError`` and they mean opposite things. A path
    escape is the agent's fault and is fixed by naming a different path; an
    OS-level denial means the path was right and the file could not be replaced,
    which no amount of rewriting the path will fix. Told the first when the
    second is true, an agent loops.

    Args:
        exc: Whatever the backend raised.

    Returns:
        The matching refusal. Both are retriable — the second because the file's
        owner may change, or the agent may pick another path knowing why.
    """
    return _rejected(PERM_ERR_MSG if isinstance(exc, PathEscapeError) else WRITE_DENIED_MSG)


def _precondition(seen: Observation | None) -> Precondition:
    """Derive what must hold of a file before *seen*'s agent may replace it.

    Args:
        seen: What the agent last observed of the path, or ``None``.

    Returns:
        The digest the live file must still carry, or ``"absent"`` — an agent
        that has not read a file may only create it.
    """
    return "absent" if seen is None else seen.sha


def _capped(diff: str) -> str:
    """Trim *diff* to the cap a refusal may carry, noting what was cut.

    The refusal travels back into the model's next turn, so an uncapped diff of
    a large file makes the *refusal* the thing that breaks the turn.

    Args:
        diff: A unified diff.

    Returns:
        *diff* unchanged when it is short enough, otherwise its first
        ``MAX_REJECTION_DIFF_LINES`` lines followed by a one-line notice.
    """
    lines = diff.splitlines()
    if len(lines) <= MAX_REJECTION_DIFF_LINES:
        return diff
    elided = len(lines) - MAX_REJECTION_DIFF_LINES
    kept = "\n".join(lines[:MAX_REJECTION_DIFF_LINES])
    return f"{kept}\n... {elided} more diff line(s) not shown — read the file to see the rest."


def _preserve_endings(content: str, live: bytes | None) -> str:
    """Give *content* the dominant line ending of the file it replaces.

    The live bytes are passed in rather than read again: the gate has just read
    them to hash them, and one mutation must cost one file read, not two.

    Args:
        content: The text the agent proposed.
        live: The file's current bytes, or ``None`` when it does not exist.

    Returns:
        *content* verbatim for a new or non-UTF-8 file, so a Windows-authored
        file is never silently converted; otherwise *content* with the existing
        file's line endings.
    """
    if live is None:
        return content
    try:
        existing = live.decode("utf-8")
    except UnicodeDecodeError:
        return content
    return normalise_endings(content, detect_line_ending(existing))


def lock_file_for(meta_dir: Path, path: str) -> Path:
    """The lock file writers of *path* contend on, under *meta_dir*.

    The name is a digest **for safety, not for identity**: a workspace-relative
    path carries slashes and may be longer than a filename may be, and the same
    reasoning ``YamlDocumentStore`` gives for its own names applies unchanged.
    Nothing reads the digest back, and two paths colliding would cost a spurious
    serialisation and never a lost update.

    Args:
        meta_dir: The tree's metadata directory — a **sibling** of the tree, so
            no read capability can name a lock file and an ``rm -rf`` inside a
            sandboxed run cannot delete the lock guarding that very run.
        path: Workspace-relative path.

    Returns:
        ``<meta>/locks/path-<sha256 of path>``.
    """
    digest = hashlib.sha256(path.encode("utf-8")).hexdigest()
    return meta_dir / LOCKS_DIR_NAME / f"{PATH_LOCK_PREFIX}{digest}"


@dataclass
class _Staged:
    """One file's in-memory state during an all-or-nothing multi-edit.

    Attributes:
        live: The bytes on disk when the file was first gated.
        text: The text after every edit applied so far — later edits on the same
            path see earlier ones.
        exact_only: Whether the file had changed since the agent read it, which
            restricts every edit on it to exact matching.
    """

    live: bytes
    text: str
    exact_only: bool


@dataclass
class _PatchBatch:
    """One ``workspace_patch`` call, gated in full before anything is published.

    A patch that applied file by file could commit a state no agent intended —
    half its files updated, the rest refused — and the journal would then record
    that state as one agent's deliberate change. Gating everything into this
    batch first is what makes "one mutation is one commit" true of ``patch``.

    Attributes:
        removals: The paths a ``+++ /dev/null`` section names, read from the raw
            diff text because ``parse_patch`` only sees ``/dev/null``.
        writes: Rendered file contents, ready for one batch publication.
        deletions: Paths to unlink once every write has landed.
        labels: The per-file summary lines, in the order the patch names them.
    """

    removals: set[str]
    writes: list[WriteEntry] = field(default_factory=list)
    deletions: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


class CardGate:
    """The six mutations, the live-hash check, the observation map, the refusal.

    Declares **no Pydantic field**: the annotations below are inside
    ``if TYPE_CHECKING:``, so they are never executed and never reach
    ``__annotations__``, which is where Pydantic v2 collects fields from across
    the MRO. mypy reads them normally.
    """

    if TYPE_CHECKING:
        # Every one of these is a real ``PrivateAttr`` — or, for the cap, a real
        # field — on ``WorkspaceTool``. Declared, never defined.
        _workspace: Filesystem | None
        _workspace_tell: WorkspaceActor | None
        _agent_id: str
        _agent_name: str
        _workspace_path: str
        _meta_dir: Path | None
        _journal: GitJournal | None
        _observations: OrderedDict[str, Observation]
        _touched: list[str]
        _lock_backend: LockBackend | None
        _exec_budget_s: float
        max_observations_per_agent: int

    ##
    ## The observation map — this agent's slice, and nobody else's
    ##
    def record_observation(self, path: str, observation: Observation) -> None:
        """Record that this agent observed *path* as described by *observation*.

        Re-recording a known path refreshes its recency instead of adding an
        entry. Over the cap, the least recently observed path is evicted.

        **Only the path dimension is capped, and there is no other dimension.**
        The agent dimension the actor's map carried is gone with the actor: one
        card belongs to one agent, so the map holds this agent's slice by
        construction rather than by keying. Eviction on the path dimension is
        safe for the reason it always was — a lost observation makes the gate
        *refuse* a write, which is a correctness-preserving degradation.

        **No lock, and none is to be added.** This map is reached only from its
        own agent's calls, so there is nothing to serialise it against
        (ADR-051 Decision 2). The cross-process hazard the ``flock`` answers is
        about the *tree*, which several agents share; this map is shared with
        nobody.

        Args:
            path: Workspace-relative path that was read.
            observation: Digest of the file's bytes, and whether it was whole.
        """
        seen = self._observations
        seen[path] = observation
        seen.move_to_end(path)
        while len(seen) > self.max_observations_per_agent:
            seen.popitem(last=False)

    def observation_for(self, path: str) -> Observation | None:
        """Return what this agent last observed of *path*, or ``None``.

        A lookup does not refresh recency: the gate consults this on every
        mutation, and letting a write extend a path's lifetime would evict the
        paths an agent is actively reading in favour of the ones it writes.

        Args:
            path: Workspace-relative path.

        Returns:
            The recorded observation, or ``None`` when there is none — which the
            gate reads as "you have not read this".
        """
        return self._observations.get(path)

    def _accept(self, path: str, data: bytes) -> None:
        """Record that this agent wrote *data* to *path*, inside the held lock.

        The writer's own observation is refreshed because an agent that has just
        written a file has by definition observed it in full — without this, its
        *next* write to the same path would be refused with a diff against its
        own content.

        **It has to happen inside the lock the write happened under.** Refresh
        it after the release and another writer can land between the two, so
        this agent's recorded digest is of bytes that are no longer there; the
        failure surfaces one turn later, as a gate refusing an agent against its
        own content, and looks nothing like the mistake that caused it.

        This is also where the mutation's write set is collected, for the commit
        :meth:`_gated` makes once the body returns. The list is plain state
        rather than a return value because the alternative is widening six
        signatures.
        """
        self._touched.append(path)
        self.record_observation(path, Observation(sha=content_sha(data), full=True))

    def _forget(self, path: str) -> None:
        """Drop what an accepted delete invalidated.

        Only this agent's observation goes: another agent still holding one is
        meant to be refused, because from its point of view the file vanished
        under it — and its observation lives in its own card, which this one
        cannot reach and must not.

        A delete is a change to the path like any other, so it joins the write
        set — ``git add -A -- <path>`` stages a removal as readily as a write.
        """
        self._touched.append(path)
        self._observations.pop(path, None)

    ##
    ## The tree handle, the hold, and the cross-process lock
    ##
    def _tree(self) -> Filesystem:
        """The handle every mutation writes through, or a refusal to write ungated.

        Raises:
            RuntimeError: When the card was never wired through ``observer()``.
                There is deliberately no ungated path to fall back to: one would
                be a bypass of the gate reachable from any harness that skipped
                the binding.
        """
        workspace = self._workspace
        if workspace is None:
            raise RuntimeError(_UNBOUND_MSG)
        return workspace

    @contextlib.contextmanager
    def _hold(self, paths: Sequence[str]) -> Iterator[None]:
        """Hold every lock in *paths* for the duration of the block.

        The idiom is
        :meth:`~akgentic.tool.vector_store.backends.local.LocalBackend._hold`'s:
        an exclusive ``flock`` on a lazily created file, unlocked and closed in
        ``finally``, never unlinked — unlinking would let a second process
        create a fresh inode and take a hold that excludes nobody, which is the
        classic way a file lock stops locking.

        **Acquired in sorted path order, and that is what makes a deadlock
        impossible.** Two agents that both touch ``a.md`` and ``b.md`` take them
        in the same order whatever order their patches name them in, so neither
        can hold one while waiting for the other's.

        **This, not ``O_EXCL``, is what makes two agents creating one path
        produce one winner.** ``Filesystem._stage`` publishes by rename and
        would happily let the loser's rename land on top of the winner's file;
        what stops it is that the loser cannot even read the live file until the
        winner has published, by which time the gate sees a file it was never
        shown and refuses.

        An empty *paths* takes nothing: ``mkdir`` has no content and therefore
        no check-then-write window to close.

        A metadata directory that cannot be created degrades to an unlocked
        mutation with one warning, rather than to a refused one: the tree is
        still gated by the live hash, which is the correctness property, and the
        same choice :meth:`_busy_refusal` makes for a metadata directory it
        cannot read. Failing closed instead would wedge every mutation on a tree
        whose ``<meta>`` went read-only mid-session, and the bytes a mutation
        writes are safe either way — only the ordering between two writers is
        lost, which is what was on offer before the lock existed at all.
        """
        meta_dir = self._meta_dir
        if not paths or meta_dir is None:
            yield
            return
        handles: list[int] = []
        try:
            try:
                for path in sorted(set(paths)):
                    handles.append(self._open_lock(lock_file_for(meta_dir, path)))
            except OSError:
                logger.warning(
                    "Workspace %s: could not take the path locks under %s — mutating unserialised",
                    self._workspace_path,
                    meta_dir / LOCKS_DIR_NAME,
                    exc_info=True,
                )
            yield
        finally:
            for handle in reversed(handles):
                with contextlib.suppress(OSError):
                    fcntl.flock(handle, fcntl.LOCK_UN)
                os.close(handle)

    @staticmethod
    def _open_lock(lock_path: Path) -> int:
        """Create *lock_path* if absent, take its exclusive ``flock``, return the fd."""
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = os.open(lock_path, os.O_RDWR | os.O_CREAT, _LOCK_FILE_MODE)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
        except BaseException:
            os.close(handle)
            raise
        return handle

    ##
    ## The six mutations — each one thin, all six through one point
    ##
    def _gated(
        self, capability: str, paths: Sequence[str], run: Callable[[], MutationOutcome]
    ) -> MutationOutcome:
        """Run one mutation between the busy check, the lock, and the two commits.

        Order: refuse if a run holds the tree → commit anything nobody claimed →
        **take every path lock** → run the gate and the write → release → stage
        the mutation's own paths and commit them as this agent → tell the index
        what went stale.

        **The busy check is first, and it is here rather than in six places.**
        This is the one point all six mutations converge on, so a seventh added
        later cannot forget it — nor the lock, which is the reason this method
        still exists now that the mailbox does not serialise anything. A
        forgotten ``flock`` fails silently, under load, on a shared mount.
        Being ahead of the body it is also ahead of the gate's file read: a
        refusal that first opens a file is doing work it is about to throw away.

        **The out-of-band commit runs before the gate decides**, which is
        correct — the dirt exists whether the mutation is accepted or refused,
        and it belongs to nobody either way. It is also what makes a refusal's
        attribution sound: by the time the gate reads the file, the live bytes
        are committed under an agent's identity or under ``out-of-band``, and
        :meth:`_rejection` never has to guess between them.

        **The commits are outside the path locks, deliberately.** The journal has
        an exclusive lock of its own, and holding one family of lock while taking
        the other is how a deadlock is built. The residual window is named rather
        than closed: between the release and the commit another writer can
        publish the same path, and this agent's ``git add -- <path>`` would then
        stage those bytes under this agent's name. It costs one misattributed
        commit in the log and never a lost update, because the gate is the live
        hash and not the journal.

        **The index is told last, over the tell proxy.** ``_touched`` already
        holds exactly the paths this mutation changed, deletes included, so a
        seventh mutation gets the signal for free. A lost signal degrades to a
        stale index row; it never fails a mutation whose bytes are already on
        disk.

        Args:
            capability: The mutation's short name, which becomes the commit
                subject's first field.
            paths: Every path this mutation may write, for the lock. Empty for
                ``mkdir``.
            run: The mutation body.

        Returns:
            Whatever the body returned, untouched. No journal or index failure
            can change a mutation's outcome.
        """
        busy = self._busy_refusal()
        if busy is not None:
            return _rejected(busy)
        journal = self._journal
        if journal is not None:
            journal.commit_out_of_band()
        self._touched = []
        with self._hold(paths):
            outcome = run()
        touched = list(self._touched)
        if outcome.status is not MutationStatus.ACCEPTED or not touched:
            return outcome
        if journal is not None:
            journal.commit_paths(touched, Identity(self._agent_name, self._agent_id), capability)
        self._mark_stale(touched)
        return outcome

    def _busy_refusal(self) -> str | None:
        """Refuse a mutation while an exec run holds the tree, or allow it.

        **Answered from the tree, not from a process.** The actor used to read
        its own ``_running``, an attribute of one instance in one interpreter —
        so two workers over one mount held two of them and neither saw the
        other's run, which is the whole reason ADR-051 exists. The marker at
        ``<meta>/exec.lock`` *is* the cross-process truth, and it carries the
        holder's run id and name so the text below is byte-identical whichever
        process composes it.

        Fail fast, never stall: ten seconds of silence inside a tool call is
        indistinguishable from a hang, where an immediate refusal naming the
        holder lets the model read a file, answer the user, or ask the holder.

        A tree whose marker cannot be read at all **allows** the mutation, with
        one debug line. The gate's correctness rests on the live hash; the hold
        is a courtesy to a command whose write set is unknowable, and failing
        closed here would wedge every mutation on a tree with an unreadable
        metadata directory.

        Returns:
            The refusal text, or ``None`` when the tree is free.
        """
        backend = self._lock_backend
        if backend is None:
            return None
        try:
            held = backend.holder(self._workspace_path, self._exec_budget_s)
        except OSError:
            logger.debug("Could not read the exec hold for %s", self._workspace_path, exc_info=True)
            return None
        if held is None:
            return None
        return mutation_busy(held.run_id, held.agent_name or held.agent_id)

    def _mark_stale(self, paths: list[str]) -> None:
        """Tell the actor which paths an accepted mutation changed — fire and forget.

        A **tell**: the mutation is already on disk and nothing comes back, so a
        slow or dead actor must not hold the agent's turn open. It marks stale
        and does not re-index, because an agent mid-task rewrites the same file
        repeatedly and indexing each accepted write would spend embedding
        credits on every save.
        """
        tell = self._workspace_tell
        if tell is None:
            return
        try:
            tell.mark_paths_stale(paths)
        except Exception:
            logger.debug("Could not mark %s stale for retrieval", paths, exc_info=True)

    def apply_write(self, path: str, content: str) -> MutationOutcome:
        """Replace *path* wholesale with *content*, if the live file still matches.

        Args:
            path: Workspace-relative path.
            content: The text the agent proposed.

        Returns:
            ``Written: <path>`` on acceptance, or a refusal carrying the diff of
            the live file against *content* — what the write would have destroyed.
        """
        return self._gated("write", [path], lambda: self._write(path, content))

    def _write(self, path: str, content: str) -> MutationOutcome:
        """Gate and perform one whole-file write — see :meth:`apply_write`."""
        try:
            live = self._live(path)
            refusal, _ = self._check(path, whole_file=True, live=live, proposed=content)
            if refusal is not None:
                return _rejected(refusal)
            data = _preserve_endings(content, live).encode("utf-8")
            self._tree().write(path, data)
        except PermissionError as exc:
            return _denied(exc)
        except FileNotFoundError:
            return _rejected(PUBLISH_LOST_MSG)
        self._accept(path, data)
        return _accepted(f"Written: {path}")

    def apply_delete(self, path: str) -> MutationOutcome:
        """Delete *path*, if this agent has read it whole and it has not moved.

        Args:
            path: Workspace-relative path.

        Returns:
            ``Deleted: <path>`` on acceptance, or a refusal. An accepted delete
            drops the agent's observation, so its next write to the path is a
            create rather than a stale rejection.
        """
        return self._gated("delete", [path], lambda: self._delete(path))

    def _delete(self, path: str) -> MutationOutcome:
        """Gate and perform one delete — see :meth:`apply_delete`."""
        try:
            live = self._live(path)
            refusal, _ = self._check(path, whole_file=True, live=live)
            if refusal is not None:
                return _rejected(refusal)
            if live is None:
                return _rejected(f"File not found: {path}")
            self._tree().delete(path)
        except PermissionError as exc:
            return _denied(exc)
        except FileNotFoundError:
            return _rejected(PUBLISH_LOST_MSG)
        self._forget(path)
        return _accepted(f"Deleted: {path}")

    def apply_edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> MutationOutcome:
        """Substitute *old_string* for *new_string* in *path*, governed by the anchor.

        An anchored edit is admitted on a file that changed since the agent read
        it — that is the whole reason to prefer ``edit`` over ``write`` — but the
        7-strategy cascade drops to exact matching there. Approximate matching
        against text another agent has just rewritten is how a plausible edit
        lands in the wrong place.

        Args:
            path: Workspace-relative path.
            old_string: The anchor to replace.
            new_string: Its replacement.
            replace_all: Replace every occurrence rather than the first.

        Returns:
            The unified diff on acceptance, ``[ERROR] old_string not found …``
            when the anchor simply is not there, or a refusal.
        """
        return self._gated(
            "edit",
            [path],
            lambda: self._edit(path, old_string, new_string, replace_all),
        )

    def _edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool,
    ) -> MutationOutcome:
        """Gate and perform one anchored edit — see :meth:`apply_edit`."""
        try:
            live = self._live(path)
            refusal, exact_only = self._check(path, whole_file=False, live=live)
            if refusal is not None:
                return _rejected(refusal)
            if live is None:
                return _rejected(f"File not found: {path}")
            raw = live.decode("utf-8")
            item = EditItem(
                path=path,
                old_string=old_string,
                new_string=new_string,
                replace_all=replace_all,
            )
            edited = substitute_edit(_MATCHER, raw, item, exact_only=exact_only)
            if edited is None:
                return self._anchor_miss(path, live, exact_only)
            data, diff = write_and_diff(self._tree(), path, raw, edited)
        except PermissionError as exc:
            return _denied(exc)
        except FileNotFoundError:
            return _rejected(PUBLISH_LOST_MSG)
        self._accept(path, data)
        return _accepted(diff or f"(no change) {path}")

    def apply_multi_edit(self, edits: list[EditItem]) -> MutationOutcome:
        """Apply *edits* across one or more files, all-or-nothing.

        Every distinct path is gated and every substitution computed in memory
        before anything is published, so a refusal or a missing anchor anywhere
        leaves every file in the batch untouched on disk. Later edits on one path
        still see the result of earlier ones.

        **Every path in the batch is locked for the whole mutation**, in sorted
        order, so the gate's all-or-nothing promise holds against a concurrent
        writer rather than only against this agent's own sequence.

        Args:
            edits: The ordered batch.

        Returns:
            The combined diff on acceptance, or the first failure or refusal.
        """
        return self._gated(
            "multi_edit", [item.path for item in edits], lambda: self._multi_edit(edits)
        )

    def _multi_edit(self, edits: list[EditItem]) -> MutationOutcome:
        """Gate every edit in memory, then publish the batch — see :meth:`apply_multi_edit`."""
        staged: dict[str, _Staged] = {}
        try:
            for item in edits:
                blocked = self._stage_edit(item, staged)
                if blocked is not None:
                    return blocked
            return self._publish_staged(staged)
        except PermissionError as exc:
            return _denied(exc)
        except FileNotFoundError:
            return _rejected(PUBLISH_LOST_MSG)

    def apply_patch(self, patch_text: str) -> MutationOutcome:
        """Apply a unified diff, gating every file it touches, all-or-nothing.

        A file refused by the gate, or a hunk that does not apply, leaves
        **every** file the patch names unchanged on disk and adds no commit. The
        journal is what forces this: from FR7 one mutation is one commit, and a
        half-applied patch produces a commit of a state no agent intended.

        The patch is parsed **once, here**, because its file list is also the
        lock set — the locks have to be held before the first file is gated, and
        a second parse to discover them would be a second chance for the two
        lists to disagree.

        **The parse keeps the guard it had when it sat inside the body.** It is
        the one step that validates a path before anything is gated, so a patch
        naming ``../../etc`` raises out of ``parse_patch``; moving the call out
        of :meth:`_patch`'s ``try`` without moving its ``except`` would turn that
        refusal into a crash in the agent's tool call.

        Args:
            patch_text: A GNU unified diff.

        Returns:
            The per-file summary on acceptance, or the first failure or refusal.
        """
        try:
            parsed = list(parse_patch(patch_text))
            batch = _PatchBatch(removals=deleted_paths(patch_text))
        except PermissionError as exc:
            return _denied(exc)
        named = {entry.path for entry in parsed if entry.path != "/dev/null"}
        return self._gated(
            "patch", sorted(named | batch.removals), lambda: self._patch(parsed, batch)
        )

    def _patch(self, parsed: list[FilePatch], batch: _PatchBatch) -> MutationOutcome:
        """Gate the whole patch, then publish it in one step — see :meth:`apply_patch`."""
        try:
            for file_patch in parsed:
                blocked = self._gate_patch_entry(file_patch, batch)
                if blocked is not None:
                    return blocked
            if not batch.labels:
                return _accepted("(no patches applied)")
            return self._publish_patch(batch)
        except PermissionError as exc:
            return _denied(exc)
        except FileNotFoundError:
            return _rejected(PUBLISH_LOST_MSG)

    def apply_mkdir(self, path: str) -> MutationOutcome:
        """Create *path* and its missing parents — converged, but neither gated nor locked.

        A directory has no content to clobber, so there is no digest to compare
        and no check-then-write window for a lock to close, and
        ``Filesystem.mkdir`` is ``parents=True, exist_ok=True``: applying the
        "read it first" rule here would break idempotent creation for no safety
        gain. It still goes through :meth:`_gated` so that the busy check and the
        out-of-band commit reach it like every other mutation.

        Args:
            path: Workspace-relative directory path.

        Returns:
            ``Created: <path>``, or a refusal for a path outside the root.
        """
        return self._gated("mkdir", [], lambda: self._mkdir(path))

    def _mkdir(self, path: str) -> MutationOutcome:
        """Create a directory — see :meth:`apply_mkdir`.

        Nothing is recorded as touched, so the journal makes no commit of its
        own: git does not track empty directories, and asking it to commit one
        would be a failure where there is none. A dirty tree still commits as
        ``out-of-band`` beforehand, because that dirt is real.
        """
        try:
            self._tree().mkdir(path)
        except PermissionError as exc:
            return _denied(exc)
        return _accepted(f"Created: {path}")

    ##
    ## The gate
    ##
    def _live(self, path: str) -> bytes | None:
        """Return *path*'s current bytes, read from disk, or ``None`` if absent.

        Called once per mutation and **never memoised**. This single line is what
        makes the gate correct against writers that never pass through this card
        — an upload, a sandbox run, another team, a human with an editor — and a
        cache here would be blind to all of them. The ``flock`` is not a reason
        to start caching: it orders the writers that come *through* the gate and
        says nothing about the rest.
        """
        try:
            return self._tree().read(path)
        except FileNotFoundError:
            return None

    def _check(
        self,
        path: str,
        *,
        whole_file: bool,
        live: bytes | None,
        proposed: str | None = None,
    ) -> tuple[str | None, bool]:
        """Decide whether this agent may mutate *path*.

        Args:
            path: Workspace-relative path.
            whole_file: True for ``write`` and ``delete``, which replace or
                remove everything; False for the anchored mutations, which are
                governed by their anchor instead.
            live: The file's current bytes, already read by the caller.
            proposed: The whole-file content the agent proposed, when there is
                one — it is what the refusal diffs the live file against.

        Returns:
            The rejection text or ``None`` to proceed, and whether an anchored
            mutation must restrict itself to exact matching. A ``None`` rejection
            with ``live is None`` means "nothing to clobber" — the caller decides
            whether that is a create or a not-found.
        """
        seen = self.observation_for(path)
        if live is None:
            if seen is None:
                return None, False
            return self._gone(path), False
        if whole_file:
            return self._check_whole(path, seen, live, proposed), False
        if seen is None:
            return self._rejection(path, _REASON_UNREAD_EDIT, live, None), False
        return None, content_sha(live) != seen.sha

    def _check_whole(
        self, path: str, seen: Observation | None, live: bytes, proposed: str | None
    ) -> str | None:
        """Apply the whole-file table to a file that exists.

        The predicate is *the file has not changed*, never *this agent's last
        operation on the path was a read*. An operation-order rule admits
        ``read(A) -> write(B) -> write(A)`` and lets A destroy B's work, which is
        the exact lost update the gate exists to prevent.
        """
        expected = _precondition(seen)
        if expected == "absent":
            return self._rejection(path, _REASON_UNREAD_WRITE, live, proposed)
        if content_sha(live) != expected:
            return self._rejection(path, _REASON_CHANGED, live, proposed)
        if seen is not None and not seen.full:
            return self._rejection(path, _REASON_PARTIAL, live, proposed)
        return None

    def _gone(self, path: str) -> str:
        """Refuse a mutation on a vanished file, and clear the observation that refused it.

        The refusal has to be recoverable, and this is the one row of either
        table whose stated next step cannot be taken: ``workspace_read`` on a
        missing file raises and records nothing, so a retained observation would
        refuse **every** later mutation of the path — write and delete alike —
        for the life of the team. The agent could never recreate a file a
        teammate or an outside writer removed.

        Dropping it makes the refusal a one-time warning. The agent is told the
        file went; its next write is judged as a create against whatever is on
        disk at that moment, so a file that reappeared in the meantime is still
        protected by the "read it before overwriting" row.

        **It mutates on the refusal path, and it is the only row that does** —
        which is why it needs no lock of its own: the map it changes is this
        card's, reached by nobody else.
        """
        self._observations.pop(path, None)
        return self._rejection(path, _REASON_GONE, None, None)

    def _anchor_miss(self, path: str, live: bytes, exact_only: bool) -> MutationOutcome:
        """What to say when ``old_string`` did not match.

        On an unchanged file this is the plain, returned ``[ERROR]`` string it
        has always been. On a **changed** file it is a refusal instead: the agent
        must be told the file moved under it, or it will retry the identical edit
        against text that no longer exists.
        """
        if not exact_only:
            return _failed(f"[ERROR] old_string not found in {path}")
        return _rejected(self._rejection(path, _REASON_EXACT_MISS, live, None))

    def _rejection(self, path: str, reason: str, live: bytes | None, proposed: str | None) -> str:
        """Compose the one refusal text every rejection uses.

        Three ingredients in order of value to the agent: what to do next, who
        else wrote, and what changed. A bare refusal makes the agent retry the
        identical write, so the message is the product.

        **The first and last lines are what every configuration gets**, journal
        or no journal: the actionable half — *it changed, read it again* — needs
        no attribution at all. The middle line is a best-effort extra, and with
        no journal there is simply no middle line. Guessing one would be worse
        than silence: without a history, a teammate's write and an upload are
        indistinguishable, and naming either is a guess stated as a fact.
        """
        lines = [f"Refused to modify {path}: {reason}."]
        provenance = self._provenance(path, reason, live)
        if provenance is not None:
            lines.append(provenance)
        lines.append(_NEXT_STEP)
        evidence = self._evidence(path, live, proposed)
        if evidence:
            lines.append(evidence)
        return "\n".join(lines)

    def _provenance(self, path: str, reason: str, live: bytes | None) -> str | None:
        """The refusal's middle line — who wrote what is there now, or nothing.

        Asked only when there are **live bytes** to attribute, which is what
        keeps a deleted file out of it: git can name whoever committed the
        removal, and naming them would turn a teammate's ``workspace_delete``
        into an accusation that something came from outside the team.

        Two branches, the same two the refusal has always had: a real name
        becomes *"It was last written by agent 'x'."*, and the out-of-band
        identity becomes the sentence saying no agent in this team wrote it —
        the latter only for a ``_CHANGE_REASONS`` reason, for the reason that
        frozen set documents.
        """
        journal = self._journal
        if live is None or journal is None:
            return None
        author = journal.last_author(path)
        if author is None:
            return None
        if author != OUT_OF_BAND_AUTHOR:
            # The agent's *name*, not its UUID. The rejection message is the
            # product; "last written by agent '3f2a…'" is something a model can
            # read and nothing it can act on — which is why the journal is
            # authored by name in the first place.
            return f"It was last written by agent '{author}'."
        return _OUT_OF_BAND if reason in _CHANGE_REASONS else None

    def _evidence(self, path: str, live: bytes | None, proposed: str | None) -> str:
        """Show what the refused mutation was up against.

        A whole-file write has a proposed content, so the refusal carries the
        diff of *live* against it — what the write would have destroyed. It is
        deliberately not a diff against what the agent observed: this card
        stores a digest and a boolean and never file content, and the agent's
        own read is still in its context to reconcile against. ``edit`` and
        ``delete`` have no proposed whole-file content, so they carry the live
        state instead.
        """
        if live is None:
            return ""
        text = live.decode("utf-8", errors="replace")
        if proposed is None:
            lines = len(text.splitlines())
            return f"The live file has {lines} line(s) and digest {content_sha(live)}."
        diff = unified(path, text, proposed, before_label="live", after_label="proposed")
        if not diff:
            return ""
        return f"Your content would have replaced the live file:\n{_capped(diff)}"

    ##
    ## multi_edit — stage everything, publish nothing until it all passes
    ##
    def _stage_edit(self, item: EditItem, staged: dict[str, _Staged]) -> MutationOutcome | None:
        """Gate and apply one edit in memory; return an outcome only on failure."""
        entry = staged.get(item.path)
        if entry is None:
            live = self._live(item.path)
            refusal, exact_only = self._check(item.path, whole_file=False, live=live)
            if refusal is not None:
                return _rejected(refusal)
            if live is None:
                return _rejected(f"File not found: {item.path}")
            entry = _Staged(live=live, text=live.decode("utf-8"), exact_only=exact_only)
            staged[item.path] = entry
        edited = substitute_edit(_MATCHER, entry.text, item, exact_only=entry.exact_only)
        if edited is None:
            return self._anchor_miss(item.path, entry.live, entry.exact_only)
        entry.text = edited
        return None

    def _publish_staged(self, staged: dict[str, _Staged]) -> MutationOutcome:
        """Publish every staged file through the batch write and return the combined diff.

        ``multi_edit`` was already atomic against the *gate*; it was not atomic
        against the *filesystem*, because a failure on the second of three files
        left the first one written. It now shares one publication mechanism with
        ``patch`` — two publication paths that differ is how one of them drifts.
        """
        entries: list[WriteEntry] = []
        diffs: list[str] = []
        for path, entry in staged.items():
            raw = entry.live.decode("utf-8")
            text = normalise_endings(entry.text, detect_line_ending(raw))
            entries.append(WriteEntry(path=path, data=text.encode("utf-8")))
            diff = unified(path, raw, text)
            if diff:
                diffs.append(diff)
        self._tree().write_many(entries)
        for written in entries:
            self._accept(written.path, written.data)
        return _accepted("\n".join(diffs) if diffs else "(no changes applied)")

    ##
    ## patch — gate everything, publish once
    ##
    def _gate_patch_entry(
        self, file_patch: FilePatch, batch: _PatchBatch
    ) -> MutationOutcome | None:
        """Gate one section of a patch into *batch*, publishing nothing.

        The blanket ``except`` is the shape ``workspace_patch`` has always had:
        anything a single file raises — a missing target, an escaping path —
        becomes that file's ``[ERROR]`` line rather than an exception out of the
        tool call. What has changed is that it can no longer leave earlier files
        written, because no file is written until every one of them has passed.

        Returns:
            ``None`` when the section was accepted into *batch*, otherwise the
            refusal or failure that ends the whole patch.
        """
        try:
            if file_patch.path == "/dev/null":
                return self._gate_removals(batch)
            return self._gate_file_patch(file_patch, batch)
        except Exception as exc:
            return _failed(f"[ERROR] {file_patch.path}: {exc}")

    def _gate_removals(self, batch: _PatchBatch) -> MutationOutcome | None:
        """Gate every path a ``+++ /dev/null`` section names, deleting nothing yet.

        ``deleted_paths`` reads the whole diff at once, so a patch carrying two
        deletion sections would arrive here twice with the same set. Publication
        is deferred, so a duplicate would reach ``delete`` a second time on a
        file that is already gone — skip what is already staged.
        """
        for path in sorted(batch.removals):
            if path in batch.deletions:
                continue
            live = self._live(path)
            refusal, _ = self._check(path, whole_file=True, live=live)
            if refusal is not None:
                return _rejected(refusal)
            if live is None:
                raise FileNotFoundError(path)  # the blanket except reports it, as before
            batch.deletions.append(path)
            batch.labels.append(f"deleted: {path}")
        return None

    def _gate_file_patch(self, file_patch: FilePatch, batch: _PatchBatch) -> MutationOutcome | None:
        """Gate one file's patch and render it into *batch*.

        **The check runs before the render**, which is what makes a patch against
        a file that vanished say *"it was deleted since you read it"* rather than
        ``[ERROR] <path>: <path>`` — and, being a refusal, clears the observation
        so the agent's next write to that path is judged as a create.

        A pure-add patch replaces the file wholesale, so it answers to the
        whole-file table — otherwise a patch would be a way around the gate.

        An update patch is on the **anchored** table, and earns its place there:
        ``render_file_patch`` verifies each hunk's context before splicing it. On
        a file that changed since the agent read it, a patch whose hunks still
        verify is applied (that is FR6's degradation, which 29-3 had to refuse
        wholesale); one whose hunks do not is stale. On an *unchanged* file the
        same failure is a bad patch, not a staleness problem, and stays the
        returned ``[ERROR]`` string it has always been.
        """
        live = self._live(file_patch.path)
        pure_add = is_pure_add(file_patch)
        refusal, changed = self._check(
            file_patch.path,
            whole_file=pure_add,
            live=live,
            proposed=render_file_patch(None, file_patch) if pure_add else None,
        )
        if refusal is not None:
            return _rejected(refusal)
        raw = None if live is None else live.decode("utf-8")
        try:
            # raw is None only for a create; an update patch raises FileNotFoundError
            # here, which the caller turns into the [ERROR] line it always was.
            proposed = render_file_patch(raw, file_patch)
        except HunkContextError as miss:
            if changed:
                return _rejected(self._rejection(file_patch.path, _REASON_PATCH_STALE, live, None))
            return _failed(f"[ERROR] {file_patch.path}: {miss}")
        batch.writes.append(WriteEntry(path=file_patch.path, data=proposed.encode("utf-8")))
        batch.labels.append(patch_label(file_patch))
        return None

    def _publish_patch(self, batch: _PatchBatch) -> MutationOutcome:
        """Publish a fully gated patch: every write together, then every deletion.

        The all-or-nothing guarantee is against the **gate** and against staging:
        by the time this runs, every file has passed and every hunk has rendered.
        It is not a transaction over the filesystem, and one residual window says
        so — a deletion failing after the writes have landed refuses the mutation
        with those writes on disk. They are then committed as ``out-of-band`` by
        the next mutation, which is the honest record: no agent owns a state no
        agent asked for. Nothing on POSIX makes the pair atomic.
        """
        tree = self._tree()
        tree.write_many(batch.writes)
        for entry in batch.writes:
            self._accept(entry.path, entry.data)
        for path in batch.deletions:
            tree.delete(path)
            self._forget(path)
        return _accepted("\n".join(batch.labels))
