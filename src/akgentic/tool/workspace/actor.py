"""``#Workspace-<workspace_name>``: the team singleton that owns one workspace tree.

29-2 wired an actor that knew what everyone had read and decided nothing. This
module is where it decides: every mutation now runs **here**, on the actor's own
tree handle, and only after the live file still matches what the writing agent
observed (ADR-036 §3).

**The check and the write are one mailbox turn.** Returning a verdict and letting
the agent write would reopen the window the gate exists to close — between the
answer and the write, a third agent can land a mutation. That is why the
mutation helpers moved into :mod:`akgentic.tool.workspace.edit` and why this
actor performs real file I/O rather than adjudicating from a distance.

**The hash is read from disk on every check, never cached.** A
``{path -> current_sha}`` map would pass almost every test written against this
module and fail exactly one: the file written behind the actor's back. That case
is not exotic — it is the frontend upload, resource seeding, a sandbox run, and a
second team sharing a ``workspace_id``. Four writers that never call this actor,
all caught for free, because the check consults the *file* rather than a record
of who wrote it. Do not optimise this into a cache.

**Reads never come here for content.** They report what they saw through a
fire-and-forget ``tell`` and go straight to the agent's own ``Filesystem``. From
this story the ask path hashes files, so a reader that waited on it would queue
behind another agent's mutation hashing a large file — and an ask carries no
timeout.

**The name carries the workspace, and that is load-bearing.**
``getChildrenOrCreate`` keys on the actor *name*, so a fixed ``#Workspace`` would
collapse two cards carrying different ``workspace_id`` values onto one actor
owning one of the two trees — silently, in a team where
``WorkspaceTool(workspace_id="shared")`` is a documented configuration. The
unicity domain of an actor must equal the resource it owns, and the resource is a
tree. The ``#`` prefix stays: it is the orchestrator's two-phase-stop invariant.

The deferred-result mechanism (ADR-033) stays disengaged. What the ask path does
is bounded — one file read, one write, dict operations — and never external.
"""

from __future__ import annotations

import contextlib
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from akgentic.core.agent import Akgent
from akgentic.tool.workspace.edit import (
    EditItem,
    EditMatcher,
    FilePatch,
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
from akgentic.tool.workspace.models import (
    MAX_REJECTION_DIFF_LINES,
    PERM_ERR_MSG,
    LastWrite,
    MutationOutcome,
    MutationStatus,
    Observation,
    Precondition,
    WorkspaceConfig,
    WorkspaceState,
    content_sha,
)
from akgentic.tool.workspace.workspace import Filesystem, get_workspace, is_staging_name

logger = logging.getLogger(__name__)

WORKSPACE_ACTOR_NAME = "#Workspace"
"""Base actor name. The live name appends the workspace — see :func:`workspace_actor_name`.

The ``#`` prefix is the orchestrator's teardown invariant: it is what classifies
the actor as a tool actor during the two-phase stop.
"""

WORKSPACE_ACTOR_ROLE = "ToolActor"

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
    "it changed since you read it, and a unified diff carries no anchor that can be "
    "verified — its hunks address line numbers that have since moved"
)

_CHANGE_REASONS = frozenset({_REASON_CHANGED, _REASON_EXACT_MISS, _REASON_PATCH_STALE})
"""Reasons that earn the out-of-band sentence.

Each describes a file whose *content* moved under the agent while it still
exists, so the live bytes have an author and :meth:`WorkspaceActor._writer_of`
either names them or establishes that no agent wrote them.

``_REASON_GONE`` is deliberately absent. A deleted file has no live bytes to
attribute, so nothing distinguishes another agent's ``workspace_delete`` from an
outside removal — an accepted delete drops its own last-writer entry. Claiming
"it came from outside the workspace tools" there would be a guess stated as a
fact, and would misattribute a teammate's delete exactly as naming the wrong
writer would.
"""

_OUT_OF_BAND = (
    "No agent in this team wrote what is there now — it came from outside the "
    "workspace tools (an upload, a sandbox run, or another team)."
)
_NEXT_STEP = "Read the file again, reconsider your change against what is there now, then retry."


def workspace_actor_name(workspace_name: str) -> str:
    """Return the singleton actor name owning *workspace_name*.

    Args:
        workspace_name: The resolved workspace — a card's ``workspace_id``, or
            the team id when it has none.

    Returns:
        ``#Workspace-<workspace_name>``.
    """
    return f"{WORKSPACE_ACTOR_NAME}-{workspace_name}"


def _accepted(message: str) -> MutationOutcome:
    """The mutation happened; *message* is what the agent is told."""
    return MutationOutcome(status=MutationStatus.ACCEPTED, message=message)


def _rejected(message: str) -> MutationOutcome:
    """The mutation did not happen and the agent must react — raised as retriable."""
    return MutationOutcome(status=MutationStatus.REJECTED, message=message)


def _failed(message: str) -> MutationOutcome:
    """The mutation did not happen; *message* is **returned**, not raised."""
    return MutationOutcome(status=MutationStatus.FAILED, message=message)


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


class WorkspaceActor(Akgent[WorkspaceConfig, WorkspaceState]):
    """Team singleton owning one workspace tree, the observations, and the gate.

    Neither map is a state field: recording is not persisted state (see
    :class:`WorkspaceState`). The observation map is keyed
    ``agent_id -> path -> Observation`` and each inner map is an
    :class:`~collections.OrderedDict`, which is the whole of the LRU — recording
    moves an entry to the end, eviction pops from the front. The last-writer map
    is keyed by path across all agents and is capped independently.
    """

    def on_start(self) -> None:
        """Initialise state, take the tree handle, and sweep orphaned staging files."""
        super().on_start()
        self.state = WorkspaceState()
        self.state.observer(self)
        self._observations: dict[str, OrderedDict[str, Observation]] = {}
        self._last_writers: OrderedDict[str, LastWrite] = OrderedDict()
        self._matcher = EditMatcher()
        self._workspace: Filesystem = get_workspace(self.config.workspace_name)
        self._sweep_staging_files()

    ##
    ## The observation map — reached through the card's **tell** proxy
    ##
    def record_observation(self, agent_id: str, path: str, observation: Observation) -> None:
        """Record that *agent_id* observed *path* as described by *observation*.

        Fire-and-forget: the card sends this through ``proxy_tell``, because the
        reader needs nothing back. An ask would make every reader wait on a
        mailbox that, from this story on, hashes files — and an ask carries no
        timeout, so a merely slow actor would stall reads rather than refuse
        writes.

        Re-recording a known path refreshes its recency instead of adding an
        entry. Over the cap, the least recently observed path is evicted.

        Only the **path** dimension is capped. The agent dimension is deliberately
        unbounded: a team's roster is small and bounded by the team itself, and
        evicting an absent agent's observations would silently re-open the
        "never read it" hole for an agent that comes back. Eviction on the path
        dimension is safe for the opposite reason — a lost observation makes the
        gate *refuse* a write, which is a correctness-preserving degradation.

        Args:
            agent_id: Identity of the reading agent, as a string.
            path: Workspace-relative path that was read.
            observation: Digest of the file's bytes, and whether it was whole.
        """
        seen = self._observations.setdefault(agent_id, OrderedDict())
        seen[path] = observation
        seen.move_to_end(path)
        while len(seen) > self.config.max_observations_per_agent:
            seen.popitem(last=False)

    def observation_for(self, agent_id: str, path: str) -> Observation | None:
        """Return what *agent_id* last observed of *path*, or ``None``.

        A lookup does not refresh recency: the gate consults this on every
        mutation, and letting a write extend a path's lifetime would evict the
        paths an agent is actively reading in favour of the ones it writes.

        Args:
            agent_id: Identity of the agent, as a string.
            path: Workspace-relative path.

        Returns:
            The recorded observation, or ``None`` when there is none — which the
            gate reads as "you have not read this".
        """
        seen = self._observations.get(agent_id)
        return None if seen is None else seen.get(path)

    ##
    ## The six mutations — ask path.  Each gates, then writes, in one turn.
    ##
    def apply_write(self, agent_id: str, path: str, content: str) -> MutationOutcome:
        """Replace *path* wholesale with *content*, if the live file still matches.

        Args:
            agent_id: Identity of the writing agent, as a string.
            path: Workspace-relative path.
            content: The text the agent proposed.

        Returns:
            ``Written: <path>`` on acceptance, or a refusal carrying the diff of
            the live file against *content* — what the write would have destroyed.
        """
        try:
            live = self._live(path)
            refusal, _ = self._check(agent_id, path, whole_file=True, live=live, proposed=content)
            if refusal is not None:
                return _rejected(refusal)
            data = _preserve_endings(content, live).encode("utf-8")
            self._workspace.write(path, data)
        except PermissionError:
            return _rejected(PERM_ERR_MSG)
        self._accept(agent_id, path, data)
        return _accepted(f"Written: {path}")

    def apply_delete(self, agent_id: str, path: str) -> MutationOutcome:
        """Delete *path*, if the deleting agent has read it whole and it has not moved.

        Args:
            agent_id: Identity of the deleting agent, as a string.
            path: Workspace-relative path.

        Returns:
            ``Deleted: <path>`` on acceptance, or a refusal. An accepted delete
            drops the agent's observation, so its next write to the path is a
            create rather than a stale rejection.
        """
        try:
            live = self._live(path)
            refusal, _ = self._check(agent_id, path, whole_file=True, live=live)
            if refusal is not None:
                return _rejected(refusal)
            if live is None:
                return _rejected(f"File not found: {path}")
            self._workspace.delete(path)
        except PermissionError:
            return _rejected(PERM_ERR_MSG)
        self._forget(agent_id, path)
        return _accepted(f"Deleted: {path}")

    def apply_edit(
        self,
        agent_id: str,
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
            agent_id: Identity of the editing agent, as a string.
            path: Workspace-relative path.
            old_string: The anchor to replace.
            new_string: Its replacement.
            replace_all: Replace every occurrence rather than the first.

        Returns:
            The unified diff on acceptance, ``[ERROR] old_string not found …``
            when the anchor simply is not there, or a refusal.
        """
        try:
            live = self._live(path)
            refusal, exact_only = self._check(agent_id, path, whole_file=False, live=live)
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
            edited = substitute_edit(self._matcher, raw, item, exact_only=exact_only)
            if edited is None:
                return self._anchor_miss(path, live, exact_only)
            data, diff = write_and_diff(self._workspace, path, raw, edited)
        except PermissionError:
            return _rejected(PERM_ERR_MSG)
        self._accept(agent_id, path, data)
        return _accepted(diff or f"(no change) {path}")

    def apply_multi_edit(self, agent_id: str, edits: list[EditItem]) -> MutationOutcome:
        """Apply *edits* across one or more files, all-or-nothing.

        Every distinct path is gated and every substitution computed in memory
        before anything is published, so a refusal or a missing anchor anywhere
        leaves every file in the batch untouched on disk. Later edits on one path
        still see the result of earlier ones.

        Args:
            agent_id: Identity of the editing agent, as a string.
            edits: The ordered batch.

        Returns:
            The combined diff on acceptance, or the first failure or refusal.
        """
        staged: dict[str, _Staged] = {}
        try:
            for item in edits:
                blocked = self._stage_edit(agent_id, item, staged)
                if blocked is not None:
                    return blocked
            return self._publish_staged(agent_id, staged)
        except PermissionError:
            return _rejected(PERM_ERR_MSG)

    def apply_patch(self, agent_id: str, patch_text: str) -> MutationOutcome:
        """Apply a unified diff, gating every file it touches.

        Partial by design, as it has always been: application stops at the first
        file that fails and earlier files stay applied. The all-or-nothing
        treatment belongs to ``multi_edit``.

        Args:
            agent_id: Identity of the patching agent, as a string.
            patch_text: A GNU unified diff.

        Returns:
            The per-file summary on acceptance, or the first failure or refusal.
        """
        try:
            removals = deleted_paths(patch_text)
            results: list[str] = []
            for file_patch in parse_patch(patch_text):
                outcome = self._patch_one(agent_id, file_patch, removals)
                if outcome.status is not MutationStatus.ACCEPTED:
                    return outcome
                if outcome.message:
                    results.append(outcome.message)
        except PermissionError:
            return _rejected(PERM_ERR_MSG)
        return _accepted("\n".join(results) if results else "(no patches applied)")

    def apply_mkdir(self, agent_id: str, path: str) -> MutationOutcome:
        """Create *path* and its missing parents — serialized, but not gated.

        A directory has no content to clobber, so there is no digest to compare,
        and ``Filesystem.mkdir`` is ``parents=True, exist_ok=True``: applying the
        "read it first" rule here would break idempotent creation for no safety
        gain. It runs on the actor anyway so that one serialization domain owns
        the tree.

        Args:
            agent_id: Identity of the creating agent — unused by the check, kept
                so every mutation carries its author.
            path: Workspace-relative directory path.

        Returns:
            ``Created: <path>``, or a refusal for a path outside the root.
        """
        del agent_id  # no content, no precondition — see the docstring
        try:
            self._workspace.mkdir(path)
        except PermissionError:
            return _rejected(PERM_ERR_MSG)
        return _accepted(f"Created: {path}")

    ##
    ## The gate
    ##
    def _live(self, path: str) -> bytes | None:
        """Return *path*'s current bytes, read from disk, or ``None`` if absent.

        Called once per mutation and never memoised. This single line is what
        makes the gate correct against writers that never pass through this
        actor; a cache here would be blind to all of them.
        """
        try:
            return self._workspace.read(path)
        except FileNotFoundError:
            return None

    def _check(
        self,
        agent_id: str,
        path: str,
        *,
        whole_file: bool,
        live: bytes | None,
        proposed: str | None = None,
    ) -> tuple[str | None, bool]:
        """Decide whether *agent_id* may mutate *path*.

        Args:
            agent_id: Identity of the mutating agent.
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
        seen = self.observation_for(agent_id, path)
        if live is None:
            if seen is None:
                return None, False
            return self._gone(agent_id, path), False
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

    def _gone(self, agent_id: str, path: str) -> str:
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
        """
        seen = self._observations.get(agent_id)
        if seen is not None:
            seen.pop(path, None)
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
        """
        lines = [f"Refused to modify {path}: {reason}."]
        writer = self._writer_of(path, live)
        if writer is not None:
            lines.append(f"It was last written by agent '{writer}'.")
        elif reason in _CHANGE_REASONS:
            lines.append(_OUT_OF_BAND)
        lines.append(_NEXT_STEP)
        evidence = self._evidence(path, live, proposed)
        if evidence:
            lines.append(evidence)
        return "\n".join(lines)

    def _writer_of(self, path: str, live: bytes | None) -> str | None:
        """Name the agent whose accepted write produced *live*, if any.

        Attribution is granted only while the live bytes still hash to what that
        agent wrote. Once anything else has touched the path the last accepted
        writer is no longer the author of what is on disk, and naming them would
        pin an out-of-band change on whichever agent happened to write last.
        """
        if live is None:
            return None
        entry = self._last_writers.get(path)
        if entry is None or entry.sha != content_sha(live):
            return None
        return entry.agent_id

    def _evidence(self, path: str, live: bytes | None, proposed: str | None) -> str:
        """Show what the refused mutation was up against.

        A whole-file write has a proposed content, so the refusal carries the
        diff of *live* against it — what the write would have destroyed. It is
        deliberately not a diff against what the agent observed: this actor
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
    ## Bookkeeping on the accept path
    ##
    def _accept(self, agent_id: str, path: str, data: bytes) -> None:
        """Record that *agent_id* wrote *data* to *path*, in the same mailbox turn.

        The writer's own observation is refreshed because an agent that has just
        written a file has by definition observed it in full — without this, its
        *next* write to the same path would be refused with a diff against its
        own content. Doing it here rather than in the closure keeps it inside the
        one turn; a second round trip would be a second window.
        """
        sha = content_sha(data)
        self.record_observation(agent_id, path, Observation(sha=sha, full=True))
        self._last_writers[path] = LastWrite(agent_id=agent_id, sha=sha)
        self._last_writers.move_to_end(path)
        while len(self._last_writers) > self.config.max_tracked_writers:
            self._last_writers.popitem(last=False)

    def _forget(self, agent_id: str, path: str) -> None:
        """Drop what an accepted delete invalidated.

        Only the deleting agent's observation goes: another agent still holding
        one is meant to be refused, because from its point of view the file
        vanished under it.
        """
        seen = self._observations.get(agent_id)
        if seen is not None:
            seen.pop(path, None)
        self._last_writers.pop(path, None)

    ##
    ## multi_edit — stage everything, publish nothing until it all passes
    ##
    def _stage_edit(
        self, agent_id: str, item: EditItem, staged: dict[str, _Staged]
    ) -> MutationOutcome | None:
        """Gate and apply one edit in memory; return an outcome only on failure."""
        entry = staged.get(item.path)
        if entry is None:
            live = self._live(item.path)
            refusal, exact_only = self._check(agent_id, item.path, whole_file=False, live=live)
            if refusal is not None:
                return _rejected(refusal)
            if live is None:
                return _rejected(f"File not found: {item.path}")
            entry = _Staged(live=live, text=live.decode("utf-8"), exact_only=exact_only)
            staged[item.path] = entry
        edited = substitute_edit(self._matcher, entry.text, item, exact_only=entry.exact_only)
        if edited is None:
            return self._anchor_miss(item.path, entry.live, entry.exact_only)
        entry.text = edited
        return None

    def _publish_staged(self, agent_id: str, staged: dict[str, _Staged]) -> MutationOutcome:
        """Write every staged file and return the combined diff."""
        diffs: list[str] = []
        for path, entry in staged.items():
            raw = entry.live.decode("utf-8")
            data, diff = write_and_diff(self._workspace, path, raw, entry.text)
            self._accept(agent_id, path, data)
            if diff:
                diffs.append(diff)
        return _accepted("\n".join(diffs) if diffs else "(no changes applied)")

    ##
    ## patch
    ##
    def _patch_one(
        self, agent_id: str, file_patch: FilePatch, removals: set[str]
    ) -> MutationOutcome:
        """Apply one file's patch, reporting any failure as the returned ``[ERROR]``.

        The blanket ``except`` is the shape ``workspace_patch`` has always had:
        anything a single file's application raises — a missing target, an
        escaping path — becomes that file's ``[ERROR]`` line rather than an
        exception out of the tool call.
        """
        try:
            if file_patch.path == "/dev/null":
                return self._patch_removals(agent_id, removals)
            return self._patch_file(agent_id, file_patch)
        except Exception as exc:
            return _failed(f"[ERROR] {file_patch.path}: {exc}")

    def _patch_removals(self, agent_id: str, removals: set[str]) -> MutationOutcome:
        """Delete every path a ``+++ /dev/null`` batch names, each one gated."""
        lines: list[str] = []
        for path in sorted(removals):
            live = self._live(path)
            refusal, _ = self._check(agent_id, path, whole_file=True, live=live)
            if refusal is not None:
                return _rejected(refusal)
            self._workspace.delete(path)  # a missing file raises, exactly as before
            self._forget(agent_id, path)
            lines.append(f"deleted: {path}")
        return _accepted("\n".join(lines))

    def _patch_file(self, agent_id: str, file_patch: FilePatch) -> MutationOutcome:
        """Render one file's hunks, gate the result, then publish it.

        A pure-add patch replaces the file wholesale, so it is gated by the
        whole-file table rather than as an anchored mutation — otherwise a patch
        could create over a file the agent had never read.

        An update patch is admitted onto the anchored table, but unlike ``edit``
        it cannot degrade to exact matching on a changed file: ``render_file_patch``
        splices each hunk at ``old_start`` and verifies no context, so a diff cut
        against an older revision is applied at line numbers that have moved —
        silently destroying the very lines the other writer added and reporting
        ``updated:``. The anchored table's admission is earned by an anchor the
        actor can check; a unified diff offers none, so a changed file is refused.
        """
        live = self._live(file_patch.path)
        proposed = render_file_patch(None if live is None else live.decode("utf-8"), file_patch)
        refusal, changed = self._check(
            agent_id,
            file_patch.path,
            whole_file=is_pure_add(file_patch),
            live=live,
            proposed=proposed,
        )
        if refusal is not None:
            return _rejected(refusal)
        if changed:
            return _rejected(self._rejection(file_patch.path, _REASON_PATCH_STALE, live, proposed))
        data = proposed.encode("utf-8")
        self._workspace.write(file_patch.path, data)
        self._accept(agent_id, file_patch.path, data)
        return _accepted(patch_label(file_patch))

    ##
    ## Startup housekeeping
    ##
    def _sweep_staging_files(self) -> None:
        """Delete staging files an interrupted write left behind, anywhere in the tree.

        ``Filesystem.write`` publishes by rename from ``.<name>.<32 hex>.tmp`` in
        the target's own directory. A process killed between the two steps leaves
        one behind for good, and nothing else ever removes it.

        The sweep runs **at actor start only** — never on a timer, never per
        mutation — and matches the full staging shape, so a user's own
        ``.notes.tmp`` survives. Every failure is suppressed: a directory this
        process cannot clean must not stop the team's workspace from starting.
        """
        root = self._workspace._root
        staged: list[Path] = []
        with contextlib.suppress(OSError):
            staged = [
                entry
                for entry in root.rglob("*")
                if is_staging_name(entry.name) and entry.is_file()
            ]
        removed = 0
        for entry in staged:
            with contextlib.suppress(OSError):
                entry.unlink()
                removed += 1
        if removed:
            logger.info(
                "Workspace %s: swept %d orphaned staging file(s) at start",
                self.config.workspace_name,
                removed,
            )
