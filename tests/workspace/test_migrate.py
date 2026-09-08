"""The pre-ADR-048 workspaces-root migration: what it moves, and what it refuses.

Every spec here builds a root by hand under ``tmp_path`` and asserts against the
directories that exist afterwards, never against printed text — the plan's table
is for an operator, and a spec that read it would be testing a format.

Four properties carry the weight, and each of them has a mutation in the story's
matrix aimed squarely at it:

- **the journal moves with its tree** — ``<team>.git`` is a *sibling*, so a
  migration that relocated only the tree would lose every workspace's history
  silently;
- **a candidate that cannot be fully migrated ends the run where it started** —
  if the journal move raises after the tree has moved, the tree is moved back;
- **the whole plan is validated before anything moves** — one occupied
  destination refuses the run, and the *other*, perfectly valid candidate is
  still at its original path afterwards;
- **an owner is never derived** — an unmapped team id is refused rather than
  filed under the shared anonymous scope.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from akgentic.tool.workspace.migrate import (
    MigrationConflictError,
    Verdict,
    apply,
    main,
    plan,
    render,
)

ALICE = "2R0bQV8j9zX8CEsBl6APi7MXgAn4_laOa8vd9ZoIHIQ"
BOB = "geoffroy.piroux@example.com"


def _workspace(root: Path, name: str, *, journal: bool = True, marker: str = "x") -> Path:
    """Create a workspace tree at *root/name*, with its ``.git`` sibling by default.

    The marker file is what lets a spec prove a *specific* tree arrived, rather
    than a directory of the right name.
    """
    tree = root / name
    tree.mkdir(parents=True)
    (tree / "note.md").write_text(marker, encoding="utf-8")
    if journal:
        git_dir = root / f"{name}.git"
        git_dir.mkdir(parents=True)
        (git_dir / "HEAD").write_text(f"ref: {marker}\n", encoding="utf-8")
    return tree


def _verdicts(root: Path, owners: dict[str, str]) -> dict[str, Verdict]:
    """The plan as ``{name: verdict}`` — the shape most assertions actually want."""
    return {entry.name: entry.verdict for entry in plan(root, owners).entries}


@pytest.fixture
def team_id() -> str:
    """A fresh team id, so no two specs can collide through a shared literal."""
    return str(uuid4())


class TestPlan:
    """What the planner decides, without touching the tree."""

    def test_a_mapped_team_directory_is_a_move(self, tmp_path: Path, team_id: str) -> None:
        _workspace(tmp_path, team_id)

        entries = plan(tmp_path, {team_id: ALICE}).entries

        assert len(entries) == 1
        assert entries[0].verdict is Verdict.MOVE
        assert entries[0].destination == tmp_path / ALICE / team_id
        assert entries[0].journal_source == tmp_path / f"{team_id}.git"
        assert entries[0].journal_destination == tmp_path / ALICE / f"{team_id}.git"

    def test_a_workspace_without_a_journal_carries_no_journal_pair(
        self, tmp_path: Path, team_id: str
    ) -> None:
        _workspace(tmp_path, team_id, journal=False)

        entry = plan(tmp_path, {team_id: ALICE}).entries[0]

        assert entry.verdict is Verdict.MOVE
        assert entry.journal_source is None
        assert entry.journal_destination is None

    def test_a_named_workspace_is_manual(self, tmp_path: Path) -> None:
        _workspace(tmp_path, "notes")

        assert _verdicts(tmp_path, {}) == {"notes": Verdict.MANUAL, "notes.git": Verdict.MANUAL}

    def test_an_unmapped_team_id_is_unmapped(self, tmp_path: Path, team_id: str) -> None:
        _workspace(tmp_path, team_id, journal=False)

        assert _verdicts(tmp_path, {}) == {team_id: Verdict.UNMAPPED}

    def test_a_git_directory_with_no_tree_beside_it_is_manual(self, tmp_path: Path) -> None:
        (tmp_path / "orphan.git").mkdir()

        assert _verdicts(tmp_path, {}) == {"orphan.git": Verdict.MANUAL}

    def test_only_a_genuinely_orphaned_journal_is_called_orphaned(self, tmp_path: Path) -> None:
        """The two cases carry opposite instructions, so the row must tell them apart.

        A journal whose tree is right beside it moves *with* that tree — the
        README's manual path. Saying "no tree beside it" on that row contradicts
        the instruction on exactly the rows it governs.
        """
        _workspace(tmp_path, "notes")
        (tmp_path / "orphan.git").mkdir()

        details = {entry.name: entry.detail for entry in plan(tmp_path, {}).entries}

        assert "no tree beside it" in details["orphan.git"]
        assert "no tree beside it" not in details["notes.git"]
        assert "with its tree" in details["notes.git"]

    def test_an_existing_scope_directory_is_skipped(self, tmp_path: Path, team_id: str) -> None:
        _workspace(tmp_path / ALICE, team_id)

        assert _verdicts(tmp_path, {team_id: ALICE}) == {
            team_id: Verdict.ALREADY_MIGRATED,
            ALICE: Verdict.SKIPPED,
        }

    def test_a_scope_named_like_a_uuid_is_a_scope_not_an_unmapped_team(
        self, tmp_path: Path, team_id: str
    ) -> None:
        """A service principal's user id *is* a dashed UUID (ADR-048 Decision 4)."""
        principal = str(uuid4())
        _workspace(tmp_path / principal, team_id)

        assert _verdicts(tmp_path, {team_id: principal})[principal] is Verdict.SKIPPED

    def test_the_reserved_metadata_scope_is_skipped(self, tmp_path: Path) -> None:
        _workspace(tmp_path / "_meta", "case_id-42__customer_id-ACME")

        assert _verdicts(tmp_path, {})["_meta"] is Verdict.SKIPPED

    def test_an_occupied_destination_is_a_conflict(self, tmp_path: Path, team_id: str) -> None:
        _workspace(tmp_path, team_id)
        (tmp_path / ALICE / team_id).mkdir(parents=True)

        assert _verdicts(tmp_path, {team_id: ALICE}) == {
            team_id: Verdict.CONFLICT,
            ALICE: Verdict.SKIPPED,
        }

    def test_an_occupied_journal_destination_is_also_a_conflict(
        self, tmp_path: Path, team_id: str
    ) -> None:
        """The pair is one unit: either destination being taken refuses the candidate."""
        _workspace(tmp_path, team_id)
        (tmp_path / ALICE / f"{team_id}.git").mkdir(parents=True)

        assert _verdicts(tmp_path, {team_id: ALICE})[team_id] is Verdict.CONFLICT

    def test_a_source_that_is_gone_with_its_destination_present_is_already_migrated(
        self, tmp_path: Path, team_id: str
    ) -> None:
        _workspace(tmp_path / ALICE, team_id)

        entry = next(e for e in plan(tmp_path, {team_id: ALICE}).entries if e.name == team_id)

        assert entry.verdict is Verdict.ALREADY_MIGRATED
        assert entry.verdict is not Verdict.CONFLICT

    def test_a_tree_migrated_without_its_journal_is_reported(
        self, tmp_path: Path, team_id: str
    ) -> None:
        """The half-moved pair a hand-migration or a killed run leaves behind.

        The leftover scan cannot catch it: it skips ``<team>.git`` precisely
        because the mapping named this team. Without a row here the plan renders
        clean and exits zero while that workspace's history sits orphaned at the
        root — AC 9's silent-history-loss, surviving a run that reported success.
        """
        _workspace(tmp_path / ALICE, team_id, journal=False)
        (tmp_path / f"{team_id}.git").mkdir()

        entry = next(e for e in plan(tmp_path, {team_id: ALICE}).entries if e.name == team_id)

        assert entry.verdict is Verdict.ALREADY_MIGRATED
        assert entry.journal_source == tmp_path / f"{team_id}.git"
        assert "journal is still at" in entry.detail

    def test_a_fully_migrated_pair_is_reported_as_plainly_done(
        self, tmp_path: Path, team_id: str
    ) -> None:
        """The ordinary idempotent case must not acquire the stranded-journal note."""
        _workspace(tmp_path / ALICE, team_id)

        entry = next(e for e in plan(tmp_path, {team_id: ALICE}).entries if e.name == team_id)

        assert entry.verdict is Verdict.ALREADY_MIGRATED
        assert entry.journal_source is None
        assert "journal is still at" not in entry.detail

    def test_a_mapped_team_with_no_directory_anywhere_produces_no_entry(
        self, tmp_path: Path, team_id: str
    ) -> None:
        """A mapping may name every team in the deployment; most have no tree here."""
        assert plan(tmp_path, {team_id: ALICE}).entries == []

    def test_a_file_at_the_root_is_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "README").write_text("not a workspace", encoding="utf-8")

        assert plan(tmp_path, {}).entries == []

    def test_a_root_that_is_not_a_directory_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not a directory"):
            plan(tmp_path / "absent", {})


class TestMappingValidation:
    """The mapping is operator-supplied, so its values go through ``user_segment``."""

    @pytest.mark.parametrize("bad", ["_meta", "", "a/b", ".hidden", "a\\b"])
    def test_a_user_id_that_cannot_be_a_scope_fails_the_plan(
        self, tmp_path: Path, team_id: str, bad: str
    ) -> None:
        _workspace(tmp_path, team_id)

        with pytest.raises(ValueError, match="unusable user ids"):
            plan(tmp_path, {team_id: bad})

    def test_every_bad_value_is_reported_at_once(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError) as excinfo:
            plan(tmp_path, {"a": "_meta", "b": "x/y"})

        assert "_meta" in str(excinfo.value)
        assert "x/y" in str(excinfo.value)


class TestApply:
    """What actually moves, and what is still where it was afterwards."""

    def test_the_tree_and_its_journal_move_together(self, tmp_path: Path, team_id: str) -> None:
        _workspace(tmp_path, team_id, marker="alpha")

        apply(plan(tmp_path, {team_id: ALICE}))

        assert (tmp_path / ALICE / team_id / "note.md").read_text(encoding="utf-8") == "alpha"
        assert (tmp_path / ALICE / f"{team_id}.git" / "HEAD").exists()
        assert not (tmp_path / team_id).exists()
        assert not (tmp_path / f"{team_id}.git").exists()

    def test_a_named_workspace_is_left_exactly_where_it_was(self, tmp_path: Path) -> None:
        _workspace(tmp_path, "notes", marker="named")

        apply(plan(tmp_path, {}))

        assert (tmp_path / "notes" / "note.md").read_text(encoding="utf-8") == "named"
        assert (tmp_path / "notes.git" / "HEAD").exists()

    def test_an_unmapped_team_is_left_exactly_where_it_was(
        self, tmp_path: Path, team_id: str
    ) -> None:
        _workspace(tmp_path, team_id, marker="unmapped")

        apply(plan(tmp_path, {}))

        assert (tmp_path / team_id / "note.md").read_text(encoding="utf-8") == "unmapped"

    def test_one_conflict_refuses_the_whole_plan_and_moves_nothing(self, tmp_path: Path) -> None:
        """The valid candidate must still be at its original path afterwards."""
        blocked, valid = str(uuid4()), str(uuid4())
        _workspace(tmp_path, blocked, marker="blocked")
        _workspace(tmp_path, valid, marker="valid")
        (tmp_path / ALICE / blocked).mkdir(parents=True)

        computed = plan(tmp_path, {blocked: ALICE, valid: ALICE})
        with pytest.raises(MigrationConflictError):
            apply(computed)

        assert (tmp_path / valid / "note.md").read_text(encoding="utf-8") == "valid"
        assert (tmp_path / f"{valid}.git" / "HEAD").exists()
        assert not (tmp_path / ALICE / valid).exists()
        assert (tmp_path / blocked / "note.md").read_text(encoding="utf-8") == "blocked"

    def test_two_principals_reach_two_scopes(self, tmp_path: Path) -> None:
        alice_team, bob_team = str(uuid4()), str(uuid4())
        _workspace(tmp_path, alice_team, marker="a")
        _workspace(tmp_path, bob_team, marker="b")

        apply(plan(tmp_path, {alice_team: ALICE, bob_team: BOB}))

        assert (tmp_path / ALICE / alice_team / "note.md").read_text(encoding="utf-8") == "a"
        assert (tmp_path / BOB / bob_team / "note.md").read_text(encoding="utf-8") == "b"

    def test_a_second_run_over_a_migrated_root_is_a_clean_no_op(
        self, tmp_path: Path, team_id: str
    ) -> None:
        _workspace(tmp_path, team_id, marker="once")
        owners = {team_id: ALICE}
        apply(plan(tmp_path, owners))

        second = plan(tmp_path, owners)
        apply(second)

        assert second.moves == []
        assert second.conflicts == []
        assert (tmp_path / ALICE / team_id / "note.md").read_text(encoding="utf-8") == "once"

    def test_applying_a_conflicted_plan_raises_before_touching_anything(
        self, tmp_path: Path, team_id: str
    ) -> None:
        _workspace(tmp_path, team_id)
        (tmp_path / ALICE / team_id).mkdir(parents=True)

        with pytest.raises(MigrationConflictError, match="nothing was moved"):
            apply(plan(tmp_path, {team_id: ALICE}))


def _break_the_journal_move(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make the journal's move raise, and record every destination attempted.

    The journal is what fails, not the tree, because that is the only ordering
    AC 9's rollback exists for: the tree is already at its new path when the
    error arrives, and putting it back is the whole obligation.

    Returns:
        The destinations ``shutil.move`` was called with, in order.
    """
    real_move = shutil.move
    destinations: list[str] = []

    def failing_move(source: str, destination: str) -> str:
        destinations.append(destination)
        if destination.endswith(".git"):
            raise PermissionError("the journal destination is not writable")
        return str(real_move(source, destination))

    monkeypatch.setattr(shutil, "move", failing_move)
    return destinations


class TestJournalRollback:
    """A candidate that cannot be fully migrated ends the run exactly as it started it."""

    def test_a_failing_journal_move_puts_the_tree_back(
        self, tmp_path: Path, team_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _workspace(tmp_path, team_id, marker="rollback")
        computed = plan(tmp_path, {team_id: ALICE})
        _break_the_journal_move(monkeypatch)

        with pytest.raises(PermissionError):
            apply(computed)

        assert (tmp_path / team_id / "note.md").read_text(encoding="utf-8") == "rollback"
        assert (tmp_path / f"{team_id}.git" / "HEAD").exists()
        assert not (tmp_path / ALICE / team_id).exists()

    def test_the_tree_goes_first_the_journal_second_and_the_restore_third(
        self, tmp_path: Path, team_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _workspace(tmp_path, team_id)
        computed = plan(tmp_path, {team_id: ALICE})
        destinations = _break_the_journal_move(monkeypatch)

        with pytest.raises(PermissionError):
            apply(computed)

        assert [Path(d).name for d in destinations] == [team_id, f"{team_id}.git", team_id]


class TestDestinationAppearsAfterThePlan:
    """The apply-time re-check: a destination that arrives *between* plan and apply.

    No monkeypatch here, deliberately. The rollback specs above simulate the
    failure; these two produce it, and they are the only cover for the one guard
    the plan cannot provide — the plan was computed at an earlier moment, so a
    concurrent writer is exactly what it cannot foresee.

    What makes the guard load-bearing is the shape of the failure without it:
    ``shutil.move`` onto an existing directory moves the source **inside** it, so
    the run *succeeds*, prints ``moved 1 workspace(s)``, exits zero — and leaves
    the tree at ``<scope>/<team>/<team>``, one level below where the resolver will
    ever look. Silent, and reported as success.
    """

    def test_a_tree_destination_that_appears_after_the_plan_refuses(
        self, tmp_path: Path, team_id: str
    ) -> None:
        _workspace(tmp_path, team_id, marker="raced")
        computed = plan(tmp_path, {team_id: ALICE})

        (tmp_path / ALICE / team_id).mkdir(parents=True)  # the concurrent writer

        with pytest.raises(MigrationConflictError, match="appeared since the plan"):
            apply(computed)

        assert (tmp_path / team_id / "note.md").read_text(encoding="utf-8") == "raced"
        assert not (tmp_path / ALICE / team_id / team_id).exists()

    def test_a_journal_destination_that_appears_after_the_plan_puts_the_tree_back(
        self, tmp_path: Path, team_id: str
    ) -> None:
        """The tree has already moved when this one is caught, so the rollback runs."""
        _workspace(tmp_path, team_id, marker="raced")
        computed = plan(tmp_path, {team_id: ALICE})

        (tmp_path / ALICE / f"{team_id}.git").mkdir(parents=True)  # the concurrent writer

        with pytest.raises(MigrationConflictError, match="appeared since the plan"):
            apply(computed)

        assert (tmp_path / team_id / "note.md").read_text(encoding="utf-8") == "raced"
        assert (tmp_path / f"{team_id}.git" / "HEAD").exists()
        assert not (tmp_path / ALICE / team_id).exists()


class TestCli:
    """``main`` — the dry-run default, the two mapping forms, and the exit codes."""

    def test_dry_run_is_the_default_and_moves_nothing(self, tmp_path: Path, team_id: str) -> None:
        _workspace(tmp_path, team_id, marker="untouched")
        owners = tmp_path / "owners.json"
        owners.write_text(json.dumps({team_id: ALICE}), encoding="utf-8")

        code = main(["--root", str(tmp_path), "--owners", str(owners)])

        assert code == 0
        assert (tmp_path / team_id / "note.md").read_text(encoding="utf-8") == "untouched"
        assert (tmp_path / f"{team_id}.git" / "HEAD").exists()
        assert not (tmp_path / ALICE).exists()

    def test_apply_performs_the_moves(self, tmp_path: Path, team_id: str) -> None:
        _workspace(tmp_path, team_id, marker="applied")
        owners = tmp_path / "owners.json"
        owners.write_text(json.dumps({team_id: ALICE}), encoding="utf-8")

        code = main(["--root", str(tmp_path), "--owners", str(owners), "--apply"])

        assert code == 0
        assert (tmp_path / ALICE / team_id / "note.md").read_text(encoding="utf-8") == "applied"
        assert (tmp_path / ALICE / f"{team_id}.git" / "HEAD").exists()

    def test_single_principal_form_needs_no_file(self, tmp_path: Path, team_id: str) -> None:
        _workspace(tmp_path, team_id, marker="anon")

        code = main(["--root", str(tmp_path), "--owner", "anonymous", "--apply"])

        assert code == 0
        assert (tmp_path / "anonymous" / team_id / "note.md").read_text(encoding="utf-8") == "anon"

    def test_single_principal_form_leaves_named_workspaces_alone(self, tmp_path: Path) -> None:
        _workspace(tmp_path, "notes", marker="named")

        code = main(["--root", str(tmp_path), "--owner", "anonymous", "--apply"])

        assert code == 0
        assert (tmp_path / "notes" / "note.md").read_text(encoding="utf-8") == "named"

    def test_an_unmapped_team_id_exits_non_zero_and_moves_nothing(self, tmp_path: Path) -> None:
        mapped, stray = str(uuid4()), str(uuid4())
        _workspace(tmp_path, mapped, marker="mapped")
        _workspace(tmp_path, stray, marker="stray")
        owners = tmp_path / "owners.json"
        owners.write_text(json.dumps({mapped: ALICE}), encoding="utf-8")

        code = main(["--root", str(tmp_path), "--owners", str(owners), "--apply"])

        assert code == 1
        assert (tmp_path / stray / "note.md").read_text(encoding="utf-8") == "stray"
        assert (tmp_path / mapped / "note.md").read_text(encoding="utf-8") == "mapped"
        assert not (tmp_path / ALICE).exists()

    def test_an_unmapped_team_id_is_never_filed_under_anonymous(
        self, tmp_path: Path, team_id: str
    ) -> None:
        """Defaulting would hand one user's tree to the shared anonymous scope."""
        _workspace(tmp_path, team_id)
        owners = tmp_path / "owners.json"
        owners.write_text(json.dumps({}), encoding="utf-8")

        assert main(["--root", str(tmp_path), "--owners", str(owners), "--apply"]) == 1
        assert not (tmp_path / "anonymous").exists()

    def test_a_conflict_exits_non_zero_and_moves_nothing(
        self, tmp_path: Path, team_id: str
    ) -> None:
        _workspace(tmp_path, team_id, marker="blocked")
        (tmp_path / ALICE / team_id).mkdir(parents=True)
        owners = tmp_path / "owners.json"
        owners.write_text(json.dumps({team_id: ALICE}), encoding="utf-8")

        code = main(["--root", str(tmp_path), "--owners", str(owners), "--apply"])

        assert code == 1
        assert (tmp_path / team_id / "note.md").read_text(encoding="utf-8") == "blocked"

    def test_a_bad_mapping_value_exits_two(self, tmp_path: Path, team_id: str) -> None:
        _workspace(tmp_path, team_id)
        owners = tmp_path / "owners.json"
        owners.write_text(json.dumps({team_id: "a/b"}), encoding="utf-8")

        assert main(["--root", str(tmp_path), "--owners", str(owners), "--apply"]) == 2

    def test_a_mapping_that_is_not_an_object_exits_two(self, tmp_path: Path) -> None:
        owners = tmp_path / "owners.json"
        owners.write_text(json.dumps(["a", "b"]), encoding="utf-8")

        assert main(["--root", str(tmp_path), "--owners", str(owners)]) == 2

    def test_a_mapping_with_a_non_string_value_exits_two(self, tmp_path: Path) -> None:
        owners = tmp_path / "owners.json"
        owners.write_text('{"a": 1}', encoding="utf-8")

        assert main(["--root", str(tmp_path), "--owners", str(owners)]) == 2

    def test_a_missing_mapping_file_exits_two(self, tmp_path: Path) -> None:
        assert main(["--root", str(tmp_path), "--owners", str(tmp_path / "absent.json")]) == 2

    def test_a_missing_root_exits_two(self, tmp_path: Path) -> None:
        assert main(["--root", str(tmp_path / "absent"), "--owner", "anonymous"]) == 2

    def test_the_root_defaults_to_the_environment_variable(
        self, tmp_path: Path, team_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _workspace(tmp_path, team_id, marker="env")
        monkeypatch.setenv("AKGENTIC_WORKSPACES_ROOT", str(tmp_path))

        assert main(["--owner", "anonymous", "--apply"]) == 0
        assert (tmp_path / "anonymous" / team_id / "note.md").read_text(encoding="utf-8") == "env"

    def test_naming_both_mapping_forms_is_refused_by_the_parser(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            main(["--root", str(tmp_path), "--owner", "anonymous", "--owners", "o.json"])

    def test_naming_no_mapping_form_is_refused_by_the_parser(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            main(["--root", str(tmp_path)])


class TestRender:
    """The printed plan — an operator's only record of what a run did."""

    def test_every_entry_appears_with_its_verdict(self, tmp_path: Path) -> None:
        mapped, stray = str(uuid4()), str(uuid4())
        _workspace(tmp_path, mapped)
        _workspace(tmp_path, stray, journal=False)
        _workspace(tmp_path, "notes", journal=False)

        text = render(plan(tmp_path, {mapped: ALICE}))

        assert f"{mapped}" in text and Verdict.MOVE.value in text
        assert f"{stray}" in text and Verdict.UNMAPPED.value in text
        assert "notes" in text and Verdict.MANUAL.value in text

    def test_an_empty_root_says_so(self, tmp_path: Path) -> None:
        assert "nothing to do" in render(plan(tmp_path, {}))
