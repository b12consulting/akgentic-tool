"""Relocate a pre-two-segment workspaces root under its owners (ADR-048 §Migration).

An **operator tool**, not part of the card's public API: nothing in the card, the
actor, the gate, the journal or the backends imports this module, and it is
deliberately absent from ``workspace/__init__.py``.  Run it as::

    python -m akgentic.tool.workspace.migrate --root ./workspaces --owners owners.json
    python -m akgentic.tool.workspace.migrate --root ./workspaces --owners owners.json --apply

Three properties are what make it safe to point at a production root, and each of
them is a decision rather than an implementation detail:

**It never derives an owner.**  The team-id → user-id mapping is an *input*
(``--owners`` or ``--owner``), because ``Process`` lives in ``akgentic-team`` and
this package may import ``akgentic-core`` only — so the lookup happens outside and
its result is handed in.  An unmapped team id is **refused**, never defaulted to
:data:`~akgentic.tool.workspace.workspace.ANONYMOUS`: defaulting would file one
user's tree under the shared anonymous scope, which is the exposure ADR-048
closes.

**A candidate is one unit carrying two directories.**  The journal is a *sibling*
of the tree — :func:`~akgentic.tool.workspace.journal.git_dir_for` returns
``<root>.git`` in the same directory — so ``workspaces/<team_id>`` and
``workspaces/<team_id>.git`` move together or neither moves.  A migration that
relocated only the tree would lose that workspace's history **silently**: nothing
raises, and the actor initialises an empty repository at the new sibling on next
start.

**The whole plan is validated before anything moves.**  Both destinations are
conflict-checked in the same plan, so the ordinary collision case never reaches
:func:`apply` at all.  The tree-move rollback in :func:`apply` exists for what a
plan *cannot* foresee — a permission change, ENOSPC, a concurrent writer — and is
explicitly not the primary defence.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from akgentic.tool.workspace.models import GIT_DIR_SUFFIX
from akgentic.tool.workspace.workspace import METADATA_SCOPE, user_segment

__all__ = [
    "MigrationConflictError",
    "MigrationEntry",
    "MigrationPlan",
    "Verdict",
    "apply",
    "main",
    "plan",
    "render",
]


class Verdict(StrEnum):
    """What the plan decided about one directory at the root."""

    MOVE = "MOVE"
    """A mapped team workspace: the tree and its journal move together."""

    ALREADY_MIGRATED = "ALREADY_MIGRATED"
    """The source is gone and the destination is there — a previous run did it.

    Deliberately **not** a conflict.  Treating it as one would make the script
    non-idempotent, so a half-finished run could never be re-driven to
    completion.
    """

    CONFLICT = "CONFLICT"
    """The destination already exists.  One of these refuses the **whole** plan."""

    UNMAPPED = "UNMAPPED"
    """A team id absent from the mapping.  Never moved, and the run exits non-zero."""

    MANUAL = "MANUAL"
    """A named workspace, or any ``.git`` the mapping did not account for.

    None of them can be placed without human judgement: the mapping from a *name*
    to a principal exists nowhere on disk, and a ``.git`` sibling standing alone
    has nothing to prove which workspace it belonged to.  A ``.git`` whose tree
    *is* beside it is equally manual — it moves with that tree — and the entry's
    ``detail`` says which of the two cases this row is.
    """

    SKIPPED = "SKIPPED"
    """Already a scope directory, not a workspace."""


class MigrationEntry(BaseModel):
    """One candidate, carrying **both** of its directories.

    Frozen, and a model rather than a tuple, because it crosses the boundary
    between the planner and the applier — and because the pairing of a tree with
    its journal is the whole point: two independent entries could be applied
    independently, which is the failure this shape makes unrepresentable.

    Attributes:
        name: The directory's name at the root.
        verdict: What the plan decided.
        source: The tree as it stands now.
        destination: Where the tree goes.  Equal to *source* when nothing moves.
        journal_source: The sibling ``<name>.git``, or ``None`` when there is
            none.  A workspace with no history is the ordinary case, not an
            error.
        journal_destination: Where that sibling goes, or ``None``.
        detail: One line of context for the printed plan — why this verdict.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    verdict: Verdict
    source: Path
    destination: Path
    journal_source: Path | None = None
    journal_destination: Path | None = None
    detail: str = ""


class MigrationPlan(BaseModel):
    """Every candidate under one root, and the verdicts computed for them."""

    model_config = ConfigDict(frozen=True)

    root: Path
    entries: list[MigrationEntry]

    @property
    def conflicts(self) -> list[MigrationEntry]:
        """Entries whose destination is occupied.  Any one refuses the plan."""
        return [entry for entry in self.entries if entry.verdict is Verdict.CONFLICT]

    @property
    def unmapped(self) -> list[MigrationEntry]:
        """Team ids the mapping does not name.  Never moved, never guessed."""
        return [entry for entry in self.entries if entry.verdict is Verdict.UNMAPPED]

    @property
    def moves(self) -> list[MigrationEntry]:
        """The entries :func:`apply` will actually move."""
        return [entry for entry in self.entries if entry.verdict is Verdict.MOVE]


class MigrationConflictError(RuntimeError):
    """Raised by :func:`apply` when the plan holds any :attr:`Verdict.CONFLICT`.

    A ``RuntimeError`` rather than a ``ValueError``: the plan was computed
    correctly, and what refuses is the state of the destination tree.
    """


def _is_team_id(name: str) -> bool:
    """Whether *name* parses as a UUID, and is therefore a team workspace.

    The only mechanical signal that separates a team's default workspace from a
    *named* one.  A named workspace has no derivable owner, so admitting one here
    would be the guess this script exists not to make.
    """
    try:
        uuid.UUID(name)
    except ValueError:
        return False
    return True


def _validated_scopes(owners: Mapping[str, str]) -> set[str]:
    """Push every mapping **value** through :func:`user_segment`, and report all failures.

    The mapping is operator-supplied, so it is exactly as trustworthy as a
    hand-edited JSON file.  Reusing the resolver's own predicate rather than
    re-implementing it is what keeps a destination the script creates reachable by
    the card that will later resolve it: a scope of ``_meta``, ``""`` or one
    containing ``/`` would be refused at bind time, so writing a tree there would
    strand it.

    Every bad value is collected before raising, because an operator fixing a
    mapping file one error per run is an operator who stops reading them.

    Returns:
        The distinct, valid scope names.

    Raises:
        ValueError: If any value cannot be a scope segment.
    """
    failures: list[str] = []
    scopes: set[str] = set()
    for team_id, user_id in sorted(owners.items()):
        try:
            scopes.add(user_segment(user_id))
        except ValueError as exc:
            failures.append(f"  {team_id} -> {user_id!r}: {exc}")
    if failures:
        joined = "\n".join(failures)
        raise ValueError(f"the owner mapping carries unusable user ids:\n{joined}")
    return scopes


def _mapped_entry(root: Path, team_id: str, user_id: str) -> MigrationEntry | None:
    """The verdict for one mapped team id, or ``None`` when there is nothing to do.

    ``None`` covers the ordinary case of a mapping that names every team in the
    deployment while the root holds directories for only some of them: neither the
    source nor the destination exists, so the team has no workspace and emitting a
    row per absent team would bury the rows that matter.

    Both destinations are checked here, in the plan, which is what keeps the
    collision case away from :func:`apply` entirely.
    """
    source = root / team_id
    destination = root / user_id / team_id
    journal_source = root / f"{team_id}{GIT_DIR_SUFFIX}"
    journal_destination = root / user_id / f"{team_id}{GIT_DIR_SUFFIX}"
    has_journal = journal_source.is_dir()
    pair = {
        "journal_source": journal_source if has_journal else None,
        "journal_destination": journal_destination if has_journal else None,
    }

    if not source.is_dir():
        if not destination.is_dir():
            return None
        if has_journal:
            # The tree is at its destination while its journal is still at the
            # root — a hand-migration that moved one of the pair, or a run killed
            # between the two moves.  Nothing else in the plan would mention it:
            # the leftover scan skips ``<team>.git`` precisely because the mapping
            # named this team, so without this row the operator is told the root
            # is clean while that workspace's history sits orphaned.
            return MigrationEntry(
                name=team_id,
                verdict=Verdict.ALREADY_MIGRATED,
                source=source,
                destination=destination,
                detail=(
                    f"already at {user_id}/{team_id}, but its journal is still at "
                    f"{team_id}{GIT_DIR_SUFFIX} — move it beside the tree by hand"
                ),
                **pair,
            )
        return MigrationEntry(
            name=team_id,
            verdict=Verdict.ALREADY_MIGRATED,
            source=source,
            destination=destination,
            detail=f"already at {user_id}/{team_id}",
        )

    if destination.exists():
        return MigrationEntry(
            name=team_id,
            verdict=Verdict.CONFLICT,
            source=source,
            destination=destination,
            detail=f"destination {user_id}/{team_id} already exists",
            **pair,
        )
    if has_journal and journal_destination.exists():
        return MigrationEntry(
            name=team_id,
            verdict=Verdict.CONFLICT,
            source=source,
            destination=destination,
            detail=f"journal destination {user_id}/{team_id}{GIT_DIR_SUFFIX} already exists",
            **pair,
        )
    return MigrationEntry(
        name=team_id,
        verdict=Verdict.MOVE,
        source=source,
        destination=destination,
        detail=f"-> {user_id}/{team_id}" + (" (+ journal)" if has_journal else ""),
        **pair,
    )


def _leftover_entry(root: Path, name: str, scopes: set[str]) -> MigrationEntry:
    """The verdict for a root directory the mapping did not account for.

    Order matters here and is not arbitrary: a scope is recognised **before** the
    UUID test, because a service principal's user id *is* a dashed UUID — so a
    scope directory named after one would otherwise be read as an unmapped team
    and refuse the run.
    """
    source = root / name
    if name == METADATA_SCOPE or name in scopes:
        return MigrationEntry(
            name=name,
            verdict=Verdict.SKIPPED,
            source=source,
            destination=source,
            detail="already a scope",
        )
    if any(child.is_dir() and _is_team_id(child.name) for child in source.iterdir()):
        return MigrationEntry(
            name=name,
            verdict=Verdict.SKIPPED,
            source=source,
            destination=source,
            detail="already a scope (holds team workspaces)",
        )
    if name.endswith(GIT_DIR_SUFFIX):
        # Whether the tree is beside it decides what to tell the operator, and the
        # two answers are opposite instructions. A journal standing alone cannot be
        # placed at all; one whose tree is right here — a named workspace, or an
        # unmapped team — must move *with* that tree, which is what the README's
        # manual path says. Asserting the orphan case without looking would
        # contradict that instruction on the very rows it applies to.
        tree = root / name[: -len(GIT_DIR_SUFFIX)]
        detail = (
            f"the journal of {tree.name} — move it with its tree, never alone"
            if tree.is_dir()
            else "a journal with no tree beside it — nothing proves whose it is"
        )
        return MigrationEntry(
            name=name,
            verdict=Verdict.MANUAL,
            source=source,
            destination=source,
            detail=detail,
        )
    if _is_team_id(name):
        return MigrationEntry(
            name=name,
            verdict=Verdict.UNMAPPED,
            source=source,
            destination=source,
            detail="team id absent from the owner mapping",
        )
    return MigrationEntry(
        name=name,
        verdict=Verdict.MANUAL,
        source=source,
        destination=source,
        detail="a named workspace — move it by hand, per README",
    )


def plan(root: Path, owners: Mapping[str, str]) -> MigrationPlan:
    """Compute every verdict without touching a single directory.

    Pure with respect to the tree: it reads the root's listing and nothing else,
    which is what makes the whole of this module's behaviour testable against a
    ``tmp_path`` root.

    Args:
        root: The workspaces root, as it stands today.
        owners: Team id → owning user id.  An *input*, never a lookup — see the
            module docstring.

    Returns:
        The plan.  Ordered mapped entries first, then whatever else the root
        holds, each set sorted by name so two runs print identically.

    Raises:
        ValueError: If *root* is not a directory, or if any mapping value cannot
            be a scope segment.
    """
    if not root.is_dir():
        raise ValueError(f"workspaces root is not a directory: {root}")
    scopes = _validated_scopes(owners)

    entries: list[MigrationEntry] = []
    consumed: set[str] = set()
    for team_id in sorted(owners):
        consumed.add(team_id)
        consumed.add(f"{team_id}{GIT_DIR_SUFFIX}")
        entry = _mapped_entry(root, team_id, owners[team_id])
        if entry is not None:
            entries.append(entry)

    for child in sorted(root.iterdir(), key=lambda path: path.name):
        if not child.is_dir() or child.name in consumed:
            continue
        entries.append(_leftover_entry(root, child.name, scopes))

    return MigrationPlan(root=root, entries=entries)


def _move(source: Path, destination: Path) -> None:
    """Move one directory, refusing an occupied destination.

    The existence check is deliberately repeated here even though the plan made
    it: the plan was computed at some earlier moment, and this is the last point
    at which a destination that appeared in between can be caught before
    ``shutil.move`` would quietly move *into* it rather than *onto* it.
    """
    if destination.exists():
        raise MigrationConflictError(
            f"destination appeared since the plan was computed: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))


def apply(plan: MigrationPlan) -> None:
    """Perform the plan's moves, or refuse the plan whole.

    **The refusal is all-or-nothing and comes first.** A collision discovered
    halfway through a run leaves a half-migrated root, which is the state hardest
    to reason about — so every conflict is reported before anything moves.

    Within one candidate the tree moves first and the journal second.  If the
    journal move raises, **the tree is moved back** before the error propagates,
    so a candidate that cannot be fully migrated ends the run exactly as it
    started it.  A tree that arrived without its history would lose that history
    silently; leaving the tree at its new path with its journal at the old one
    would be worse still, because nothing on disk would record the pairing.

    Args:
        plan: The plan to apply.  Only :attr:`Verdict.MOVE` entries are touched.

    Raises:
        MigrationConflictError: If the plan holds any conflict.  Nothing moves.
        OSError: Whatever a move raised, after the candidate has been restored.
    """
    conflicts = plan.conflicts
    if conflicts:
        detail = "\n".join(f"  {entry.name}: {entry.detail}" for entry in conflicts)
        raise MigrationConflictError(
            f"{len(conflicts)} destination(s) already exist; nothing was moved:\n{detail}"
        )

    for entry in plan.moves:
        _move(entry.source, entry.destination)
        if entry.journal_source is None or entry.journal_destination is None:
            continue
        try:
            _move(entry.journal_source, entry.journal_destination)
        except BaseException:
            # The tree is already at its new path.  Put it back, so the candidate
            # ends the run where it started: a live workspace whose history is
            # orphaned under the old path is worse than one that never moved.
            shutil.move(str(entry.destination), str(entry.source))
            raise


def render(plan: MigrationPlan) -> str:
    """The plan as a readable table, printed on a dry run **and** on ``--apply``.

    An operator who applied a plan should see the same rows as one who did not;
    a run that reports only its failures leaves nobody able to say what happened.
    """
    lines = [f"workspaces root: {plan.root}", ""]
    if not plan.entries:
        lines.append("nothing to do — the root holds no workspace directories")
        return "\n".join(lines)
    width = max(len(entry.name) for entry in plan.entries)
    verdict_width = max(len(entry.verdict.value) for entry in plan.entries)
    for entry in plan.entries:
        lines.append(
            f"{entry.name:<{width}}  {entry.verdict.value:<{verdict_width}}  {entry.detail}"
        )
    counts = ", ".join(
        f"{verdict.value}={sum(1 for e in plan.entries if e.verdict is verdict)}"
        for verdict in Verdict
        if any(e.verdict is verdict for e in plan.entries)
    )
    lines.extend(["", counts])
    return "\n".join(lines)


def _default_root() -> Path:
    """``AKGENTIC_WORKSPACES_ROOT``, then ``./workspaces`` — the same order the card uses."""
    return Path(os.environ.get("AKGENTIC_WORKSPACES_ROOT", "./workspaces"))


def _owners_from_file(path: Path) -> dict[str, str]:
    """Read ``--owners``: a flat JSON object mapping team id to user id.

    Raises:
        ValueError: If the document is not an object of strings.  A list, or a
            nested object, is a file the operator built wrong, and reading it
            leniently would migrate directories under names taken from the wrong
            level.
    """
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected a JSON object mapping team id to user id")
    owners: dict[str, str] = {}
    for team_id, user_id in loaded.items():
        if not isinstance(team_id, str) or not isinstance(user_id, str):
            raise ValueError(f"{path}: every key and value must be a string")
        owners[team_id] = user_id
    return owners


def _owners_from_single_principal(root: Path, owner: str) -> dict[str, str]:
    """``--owner``: one principal owns the whole root — the community and CLI tiers.

    The mapping is derived from the *listing*, not invented: every UUID-named
    directory that is not already this principal's own scope is one of their
    teams.  Named workspaces are still left out, because ``--owner`` says who owns
    the root and not that every directory in it is a team.
    """
    if not root.is_dir():
        raise ValueError(f"workspaces root is not a directory: {root}")
    return {
        child.name: owner
        for child in root.iterdir()
        if child.is_dir()
        and child.name != owner
        and not child.name.endswith(GIT_DIR_SUFFIX)
        and _is_team_id(child.name)
    }


def _parser() -> argparse.ArgumentParser:
    """The command line.  ``--owners`` and ``--owner`` are exclusive and one is required."""
    parser = argparse.ArgumentParser(
        prog="python -m akgentic.tool.workspace.migrate",
        description=(
            "Move a pre-ADR-048 workspaces root under its owners. Dry run by default; "
            "named workspaces are reported and never touched."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_default_root(),
        help="the workspaces root (default: $AKGENTIC_WORKSPACES_ROOT, else ./workspaces)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--owners",
        type=Path,
        help="a JSON object mapping team id to owning user id",
    )
    group.add_argument(
        "--owner",
        help="a single owning user id for the whole root (community / CLI tiers)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the moves. Without it nothing is written.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Plan the migration, print it, and — only with ``--apply`` — perform it.

    Args:
        argv: Arguments without the program name.  ``None`` reads ``sys.argv``.

    Returns:
        ``0`` when the plan is clean, ``1`` when it holds a conflict or an
        unmapped team id, ``2`` when the inputs themselves are unusable.  An
        unmapped team id fails the run even on a dry run, because a migration
        that silently left one principal's tree at the root is the failure this
        script exists to avoid.
    """
    args = _parser().parse_args(argv)
    root: Path = args.root
    try:
        owners = (
            _owners_from_file(args.owners)
            if args.owners is not None
            else _owners_from_single_principal(root, args.owner)
        )
        computed = plan(root, owners)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(render(computed))

    if computed.conflicts or computed.unmapped:
        print("\nrefusing: the plan is not clean. Nothing was moved.", file=sys.stderr)
        return 1
    if not args.apply:
        print("\ndry run — nothing was moved. Re-run with --apply to perform it.")
        return 0

    try:
        apply(computed)
    except (MigrationConflictError, OSError) as exc:
        print(f"error while applying: {exc}", file=sys.stderr)
        return 1
    print(f"\nmoved {len(computed.moves)} workspace(s).")
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main(sys.argv[1:]))
