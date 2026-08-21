"""The write gate: every mutation runs on the actor, and every one carries a precondition.

Two tables govern, and they differ deliberately (ADR-036 §3):

- **Whole-file** mutations — ``write``, ``delete`` — replace or remove everything,
  so they demand that the agent has read the whole file and that it has not
  moved since.
- **Anchored** mutations — ``edit``, ``multi_edit``, ``patch`` — are governed by
  their anchor, which is itself a precondition. They are *admitted* on a file
  that changed, with the 7-strategy cascade degraded to exact matching.

``mkdir`` is routed through the actor but gated by neither table: a directory has
no content to clobber.

The property that separates this design from a cache lives in
``test_live_hash.py``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from akgentic.tool.errors import RetriableError
from akgentic.tool.workspace.actor import WorkspaceActor
from akgentic.tool.workspace.edit import EditItem
from akgentic.tool.workspace.models import MutationStatus
from akgentic.tool.workspace.tool import WorkspaceTool
from akgentic.tool.workspace.workspace import Filesystem

from tests.workspace.conftest import (
    HANDSHAKE_TIMEOUT_S,
    WORKSPACE_NAME,
    FakeActorToolObserver,
    FakeOrchestratorProxy,
    card_for,
    mutate,
    outcome_of,
    read,
    tool_named,
)

BODY = "alpha\nbravo\ncharlie\ndelta\n"


@pytest.fixture
def notes(workspace_tree: Path) -> Path:
    """``notes.md``, on disk before any agent has looked at it."""
    path = workspace_tree / "notes.md"
    path.write_text(BODY, encoding="utf-8")
    return path


@pytest.fixture
def bob(
    orchestrator_proxy: FakeOrchestratorProxy,
    wired_card: WorkspaceTool,
) -> tuple[WorkspaceTool, FakeActorToolObserver]:
    """A second agent on the same workspace, sharing ``alice``'s actor."""
    return card_for(orchestrator_proxy, "bob")


# ---------------------------------------------------------------------------
# AC1: every mutation runs on the actor, on the actor's own tree handle
# ---------------------------------------------------------------------------


class TestEveryMutationRunsOnTheActor:
    def test_a_write_lands_in_the_actors_tree_not_the_cards(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        workspace_tree: Path,
    ) -> None:
        # The card's own handle is pointed at a decoy directory while the actor
        # keeps the real one. A closure that still called self.workspace would
        # write into the decoy — nothing else can tell the two apart.
        decoy = Filesystem(str(workspaces_root), "decoy")
        observer = FakeActorToolObserver(orchestrator_proxy)
        with patch("akgentic.tool.workspace.tool.get_workspace", return_value=decoy):
            card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
            card.observer(observer)

        assert mutate(card, "workspace_write", "only.md", "content\n") == "Written: only.md"

        assert (workspace_tree / "only.md").exists()
        assert not (decoy._root / "only.md").exists()

    @pytest.mark.parametrize(
        ("name", "args"),
        [
            ("workspace_write", ("fresh.md", "body\n")),
            ("workspace_mkdir", ("sub",)),
        ],
    )
    def test_a_card_that_was_never_bound_refuses_rather_than_writing_ungated(
        self,
        name: str,
        args: tuple[Any, ...],
        wired_card: WorkspaceTool,
        workspace_tree: Path,
    ) -> None:
        # There is deliberately no ungated fallback: one would be a bypass of
        # the gate reachable from any harness that skipped the binding.
        wired_card._workspace_proxy = None
        with pytest.raises(RuntimeError, match="workspace actor is not bound"):
            tool_named(wired_card, name)(*args)
        assert list(workspace_tree.iterdir()) == []

    def test_two_agents_creating_one_path_produce_one_winner(
        self,
        threaded_orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
    ) -> None:
        # A real mailbox, two real threads. The check and the write happen in one
        # turn, so the loser cannot have observed "absent" and then written over
        # the winner: exactly one create survives.
        alice, _alice_observer = card_for(threaded_orchestrator_proxy, "alice")
        bob, _bob_observer = card_for(threaded_orchestrator_proxy, "bob")

        results: list[str] = []
        refusals: list[str] = []
        start = threading.Barrier(2, timeout=HANDSHAKE_TIMEOUT_S)

        def contend(card: WorkspaceTool, body: str) -> None:
            start.wait()
            try:
                results.append(mutate(card, "workspace_write", "race.md", body))
            except RetriableError as refused:
                refusals.append(str(refused))

        threads = [
            threading.Thread(target=contend, args=(alice, "alice was here\n")),
            threading.Thread(target=contend, args=(bob, "bob was here\n")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=HANDSHAKE_TIMEOUT_S)

        assert results == ["Written: race.md"]
        assert len(refusals) == 1
        assert "read it before overwriting" in refusals[0]
        assert (workspace_tree / "race.md").read_text(encoding="utf-8") in (
            "alice was here\n",
            "bob was here\n",
        )


# ---------------------------------------------------------------------------
# AC2: the whole-file table, one test per row
# ---------------------------------------------------------------------------


class TestWholeFileTableForWrite:
    def test_no_observation_and_no_file_creates(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        assert mutate(wired_card, "workspace_write", "new.md", "body\n") == "Written: new.md"
        assert (workspace_tree / "new.md").read_text(encoding="utf-8") == "body\n"

    def test_no_observation_and_a_live_file_is_refused(
        self, wired_card: WorkspaceTool, notes: Path
    ) -> None:
        with pytest.raises(RetriableError, match="read it before overwriting"):
            mutate(wired_card, "workspace_write", "notes.md", "mine\n")
        assert notes.read_text(encoding="utf-8") == BODY

    def test_a_whole_read_of_an_unchanged_file_overwrites(
        self, wired_card: WorkspaceTool, notes: Path
    ) -> None:
        read(wired_card, "notes.md")
        assert mutate(wired_card, "workspace_write", "notes.md", "mine\n") == "Written: notes.md"
        assert notes.read_text(encoding="utf-8") == "mine\n"

    def test_a_whole_read_of_a_changed_file_is_refused_with_a_diff(
        self,
        wired_card: WorkspaceTool,
        bob: tuple[WorkspaceTool, FakeActorToolObserver],
        notes: Path,
    ) -> None:
        bob_card, _ = bob
        read(wired_card, "notes.md")
        read(bob_card, "notes.md")
        mutate(bob_card, "workspace_write", "notes.md", "bob's version\n")

        with pytest.raises(RetriableError) as refusal:
            mutate(wired_card, "workspace_write", "notes.md", "alice's version\n")

        message = str(refusal.value)
        assert "changed since you read it" in message
        assert "--- live/notes.md" in message
        assert "+++ proposed/notes.md" in message
        assert "-bob's version" in message
        assert "+alice's version" in message
        assert notes.read_text(encoding="utf-8") == "bob's version\n"

    def test_a_paginated_read_of_an_unchanged_file_is_refused(
        self, wired_card: WorkspaceTool, notes: Path
    ) -> None:
        read(wired_card, "notes.md", limit=1)
        with pytest.raises(RetriableError, match="you read only part of it"):
            mutate(wired_card, "workspace_write", "notes.md", "mine\n")
        assert notes.read_text(encoding="utf-8") == BODY

    def test_a_paginated_read_still_licenses_an_edit_on_a_matching_anchor(
        self, wired_card: WorkspaceTool, notes: Path
    ) -> None:
        # The row above and this one are the reason edit exists as a separate
        # capability: a page is a precondition for a region, not for the file.
        read(wired_card, "notes.md", limit=1)
        assert "-bravo" in mutate(wired_card, "workspace_edit", "notes.md", "bravo", "BRAVO")
        assert notes.read_text(encoding="utf-8") == "alpha\nBRAVO\ncharlie\ndelta\n"

    def test_a_file_deleted_since_the_read_is_refused_as_stale(
        self,
        wired_card: WorkspaceTool,
        bob: tuple[WorkspaceTool, FakeActorToolObserver],
        notes: Path,
    ) -> None:
        bob_card, _ = bob
        read(wired_card, "notes.md")
        read(bob_card, "notes.md")
        mutate(bob_card, "workspace_delete", "notes.md")

        with pytest.raises(RetriableError, match="deleted since you read it"):
            mutate(wired_card, "workspace_write", "notes.md", "alice's version\n")
        assert not notes.exists()


class TestWholeFileTableForDelete:
    def test_an_unread_existing_file_is_refused(
        self, wired_card: WorkspaceTool, notes: Path
    ) -> None:
        with pytest.raises(RetriableError, match="read it before overwriting"):
            mutate(wired_card, "workspace_delete", "notes.md")
        assert notes.exists()

    def test_a_fully_read_unchanged_file_is_deleted(
        self, wired_card: WorkspaceTool, notes: Path
    ) -> None:
        read(wired_card, "notes.md")
        assert mutate(wired_card, "workspace_delete", "notes.md") == "Deleted: notes.md"
        assert not notes.exists()

    def test_a_file_changed_since_the_read_is_refused(
        self,
        wired_card: WorkspaceTool,
        bob: tuple[WorkspaceTool, FakeActorToolObserver],
        notes: Path,
    ) -> None:
        bob_card, _ = bob
        read(wired_card, "notes.md")
        read(bob_card, "notes.md")
        mutate(bob_card, "workspace_write", "notes.md", "bob's version\n")

        with pytest.raises(RetriableError, match="changed since you read it"):
            mutate(wired_card, "workspace_delete", "notes.md")
        assert notes.exists()

    def test_a_paginated_read_does_not_license_a_delete(
        self, wired_card: WorkspaceTool, notes: Path
    ) -> None:
        read(wired_card, "notes.md", limit=1)
        with pytest.raises(RetriableError, match="you read only part of it"):
            mutate(wired_card, "workspace_delete", "notes.md")
        assert notes.exists()


# ---------------------------------------------------------------------------
# AC2 / AC6: the anchored table
# ---------------------------------------------------------------------------


class TestAnchoredTable:
    def test_an_unread_existing_file_is_refused(
        self, wired_card: WorkspaceTool, notes: Path
    ) -> None:
        with pytest.raises(RetriableError, match="read it before editing"):
            mutate(wired_card, "workspace_edit", "notes.md", "bravo", "BRAVO")
        assert notes.read_text(encoding="utf-8") == BODY

    def test_an_absent_file_is_still_a_plain_not_found(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        with pytest.raises(RetriableError, match="File not found: missing.md"):
            mutate(wired_card, "workspace_edit", "missing.md", "a", "b")

    def test_a_file_deleted_since_the_read_is_refused_as_stale(
        self, wired_card: WorkspaceTool, notes: Path
    ) -> None:
        read(wired_card, "notes.md")
        notes.unlink()
        with pytest.raises(RetriableError, match="deleted since you read it"):
            mutate(wired_card, "workspace_edit", "notes.md", "bravo", "BRAVO")

    def test_an_unchanged_file_runs_the_full_cascade(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        source = workspace_tree / "main.py"
        source.write_text("def  foo():\n    pass\n", encoding="utf-8")
        read(wired_card, "main.py")

        # A single space where the file has two: only an approximate strategy
        # can place this anchor.
        result = mutate(
            wired_card, "workspace_edit", "main.py", "def foo():\n    pass", "def bar():\n    pass"
        )

        assert not result.startswith("[ERROR]")
        assert "bar" in source.read_text(encoding="utf-8")

    def test_a_changed_file_refuses_the_same_approximate_anchor(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        source = workspace_tree / "main.py"
        source.write_text("def  foo():\n    pass\n", encoding="utf-8")
        read(wired_card, "main.py")
        source.write_text("# banner\ndef  foo():\n    pass\n", encoding="utf-8")

        with pytest.raises(RetriableError) as refusal:
            mutate(
                wired_card,
                "workspace_edit",
                "main.py",
                "def foo():\n    pass",
                "def bar():\n    pass",
            )

        assert "no longer matches it exactly" in str(refusal.value)
        assert "bar" not in source.read_text(encoding="utf-8")

    def test_a_changed_file_still_accepts_an_exact_anchor(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        source = workspace_tree / "main.py"
        source.write_text("def  foo():\n    pass\n", encoding="utf-8")
        read(wired_card, "main.py")
        source.write_text("# banner\ndef  foo():\n    pass\n", encoding="utf-8")

        result = mutate(wired_card, "workspace_edit", "main.py", "def  foo():", "def  bar():")

        assert not result.startswith("[ERROR]")
        assert "def  bar():" in source.read_text(encoding="utf-8")

    def test_an_unmatched_anchor_on_an_unchanged_file_is_a_returned_error(
        self, wired_card: WorkspaceTool, notes: Path
    ) -> None:
        # Not a refusal: nothing moved under the agent, the anchor is simply
        # not there. The distinction matters — a refusal tells the agent to
        # re-read, and re-reading would teach it nothing here.
        read(wired_card, "notes.md")
        result = mutate(wired_card, "workspace_edit", "notes.md", "not present", "x")
        assert result == "[ERROR] old_string not found in notes.md"


# ---------------------------------------------------------------------------
# AC4: an accepted mutation refreshes the writer's own observation, and only its
# ---------------------------------------------------------------------------


class TestAnAcceptedMutationRefreshesItsWriter:
    def test_the_same_agent_writes_twice_with_no_read_between(
        self, wired_card: WorkspaceTool, notes: Path
    ) -> None:
        read(wired_card, "notes.md")
        mutate(wired_card, "workspace_write", "notes.md", "first\n")
        assert mutate(wired_card, "workspace_write", "notes.md", "second\n") == (
            "Written: notes.md"
        )
        assert notes.read_text(encoding="utf-8") == "second\n"

    def test_edit_then_edit_and_write_then_edit(
        self, wired_card: WorkspaceTool, notes: Path
    ) -> None:
        read(wired_card, "notes.md")
        mutate(wired_card, "workspace_edit", "notes.md", "alpha", "ALPHA")
        mutate(wired_card, "workspace_edit", "notes.md", "bravo", "BRAVO")
        mutate(wired_card, "workspace_write", "notes.md", "one\ntwo\n")
        mutate(wired_card, "workspace_edit", "notes.md", "two", "TWO")
        assert notes.read_text(encoding="utf-8") == "one\nTWO\n"

    def test_a_second_agents_older_observation_is_still_refused(
        self,
        wired_card: WorkspaceTool,
        bob: tuple[WorkspaceTool, FakeActorToolObserver],
        notes: Path,
    ) -> None:
        bob_card, _ = bob
        read(wired_card, "notes.md")
        read(bob_card, "notes.md")
        mutate(wired_card, "workspace_write", "notes.md", "alice's version\n")

        with pytest.raises(RetriableError, match="changed since you read it"):
            mutate(bob_card, "workspace_write", "notes.md", "bob's version\n")
        assert notes.read_text(encoding="utf-8") == "alice's version\n"

    def test_the_interleaving_an_operation_order_rule_would_admit(
        self,
        wired_card: WorkspaceTool,
        bob: tuple[WorkspaceTool, FakeActorToolObserver],
        notes: Path,
    ) -> None:
        # read(A) -> write(B) -> write(A). A rule of the form "this agent's last
        # operation on the path was a read" admits this and lets A destroy B's
        # work. The predicate is "the file has not changed", so A is refused.
        bob_card, _ = bob
        read(wired_card, "notes.md")
        read(bob_card, "notes.md")
        mutate(bob_card, "workspace_write", "notes.md", "bob's work\n")

        with pytest.raises(RetriableError):
            mutate(wired_card, "workspace_write", "notes.md", "alice clobbers\n")
        assert notes.read_text(encoding="utf-8") == "bob's work\n"

    def test_an_accepted_delete_turns_the_next_write_into_a_create(
        self, wired_card: WorkspaceTool, notes: Path
    ) -> None:
        read(wired_card, "notes.md")
        mutate(wired_card, "workspace_delete", "notes.md")
        assert mutate(wired_card, "workspace_write", "notes.md", "reborn\n") == (
            "Written: notes.md"
        )
        assert notes.read_text(encoding="utf-8") == "reborn\n"


# ---------------------------------------------------------------------------
# AC5: a rejection carries a diff and names the other writer when known
# ---------------------------------------------------------------------------


class TestRejectionText:
    def test_it_names_the_agent_whose_write_is_on_disk(
        self,
        wired_card: WorkspaceTool,
        bob: tuple[WorkspaceTool, FakeActorToolObserver],
        notes: Path,
    ) -> None:
        bob_card, bob_observer = bob
        read(wired_card, "notes.md")
        read(bob_card, "notes.md")
        mutate(bob_card, "workspace_write", "notes.md", "bob's version\n")

        with pytest.raises(RetriableError) as refusal:
            mutate(wired_card, "workspace_write", "notes.md", "alice's version\n")

        assert f"last written by agent '{bob_observer.myAddress.agent_id}'" in str(refusal.value)

    def test_a_refusal_always_says_what_to_do_next(
        self, wired_card: WorkspaceTool, notes: Path
    ) -> None:
        with pytest.raises(RetriableError) as refusal:
            mutate(wired_card, "workspace_write", "notes.md", "mine\n")
        assert "Read the file again" in str(refusal.value)

    def test_a_delete_refusal_carries_the_live_state_rather_than_a_diff(
        self,
        wired_card: WorkspaceTool,
        bob: tuple[WorkspaceTool, FakeActorToolObserver],
        notes: Path,
    ) -> None:
        # There is no proposed whole-file content to diff against, so the
        # refusal reports what is live instead.
        bob_card, _ = bob
        read(wired_card, "notes.md")
        read(bob_card, "notes.md")
        mutate(bob_card, "workspace_write", "notes.md", "one\ntwo\n")

        with pytest.raises(RetriableError) as refusal:
            mutate(wired_card, "workspace_delete", "notes.md")

        message = str(refusal.value)
        assert "The live file has 2 line(s)" in message
        assert "--- live/" not in message

    def test_a_write_of_identical_content_still_refuses_without_an_empty_diff(
        self,
        wired_card: WorkspaceTool,
        bob: tuple[WorkspaceTool, FakeActorToolObserver],
        notes: Path,
    ) -> None:
        # alice proposes exactly what bob already wrote: the refusal stands (she
        # never saw his version), but there is no diff to show.
        bob_card, _ = bob
        read(wired_card, "notes.md")
        read(bob_card, "notes.md")
        mutate(bob_card, "workspace_write", "notes.md", "agreed\n")

        with pytest.raises(RetriableError) as refusal:
            mutate(wired_card, "workspace_write", "notes.md", "agreed\n")

        assert "would have replaced" not in str(refusal.value)


# ---------------------------------------------------------------------------
# AC7: multi_edit is all-or-nothing across files
# ---------------------------------------------------------------------------


class TestMultiEditIsAtomic:
    @pytest.fixture
    def three_files(self, wired_card: WorkspaceTool, workspace_tree: Path) -> Path:
        for name, body in (("a.py", "x = 1\n"), ("b.py", "y = 2\n"), ("c.py", "z = 3\n")):
            (workspace_tree / name).write_text(body, encoding="utf-8")
            read(wired_card, name)
        return workspace_tree

    def test_a_batch_that_all_succeeds_applies_everything(
        self, wired_card: WorkspaceTool, three_files: Path
    ) -> None:
        mutate(
            wired_card,
            "workspace_multi_edit",
            [
                EditItem(path="a.py", old_string="x = 1", new_string="x = 10"),
                EditItem(path="c.py", old_string="z = 3", new_string="z = 30"),
            ],
        )
        assert (three_files / "a.py").read_text(encoding="utf-8") == "x = 10\n"
        assert (three_files / "c.py").read_text(encoding="utf-8") == "z = 30\n"

    def test_a_missing_anchor_leaves_every_file_untouched(
        self, wired_card: WorkspaceTool, three_files: Path
    ) -> None:
        result = mutate(
            wired_card,
            "workspace_multi_edit",
            [
                EditItem(path="a.py", old_string="x = 1", new_string="x = 10"),
                EditItem(path="b.py", old_string="NOT THERE", new_string="whatever"),
                EditItem(path="c.py", old_string="z = 3", new_string="z = 30"),
            ],
        )
        assert result == "[ERROR] old_string not found in b.py"
        assert (three_files / "a.py").read_text(encoding="utf-8") == "x = 1\n"
        assert (three_files / "c.py").read_text(encoding="utf-8") == "z = 3\n"

    def test_a_gate_refusal_anywhere_leaves_every_file_untouched(
        self, wired_card: WorkspaceTool, three_files: Path
    ) -> None:
        # b.py was never read by this agent, so the batch cannot touch a.py either.
        (three_files / "d.py").write_text("w = 4\n", encoding="utf-8")
        with pytest.raises(RetriableError, match="read it before editing"):
            mutate(
                wired_card,
                "workspace_multi_edit",
                [
                    EditItem(path="a.py", old_string="x = 1", new_string="x = 10"),
                    EditItem(path="d.py", old_string="w = 4", new_string="w = 40"),
                ],
            )
        assert (three_files / "a.py").read_text(encoding="utf-8") == "x = 1\n"
        assert (three_files / "d.py").read_text(encoding="utf-8") == "w = 4\n"

    def test_two_edits_on_one_path_see_each_other(
        self, wired_card: WorkspaceTool, three_files: Path
    ) -> None:
        mutate(
            wired_card,
            "workspace_multi_edit",
            [
                EditItem(path="a.py", old_string="x = 1", new_string="x = 2"),
                EditItem(path="a.py", old_string="x = 2", new_string="x = 3"),
            ],
        )
        assert (three_files / "a.py").read_text(encoding="utf-8") == "x = 3\n"

    def test_a_missing_file_refuses_before_anything_is_written(
        self, wired_card: WorkspaceTool, three_files: Path
    ) -> None:
        with pytest.raises(RetriableError, match="File not found: gone.py"):
            mutate(
                wired_card,
                "workspace_multi_edit",
                [
                    EditItem(path="a.py", old_string="x = 1", new_string="x = 10"),
                    EditItem(path="gone.py", old_string="a", new_string="b"),
                ],
            )
        assert (three_files / "a.py").read_text(encoding="utf-8") == "x = 1\n"

    def test_an_empty_batch_applies_nothing(self, wired_card: WorkspaceTool) -> None:
        assert mutate(wired_card, "workspace_multi_edit", []) == "(no changes applied)"


# ---------------------------------------------------------------------------
# workspace_patch is gated per file, and keeps its partial semantics
# ---------------------------------------------------------------------------


class TestPatchIsGated:
    def test_a_pure_add_over_an_unread_file_is_refused(
        self, wired_card: WorkspaceTool, notes: Path
    ) -> None:
        # A pure-add patch replaces the file wholesale, so it answers to the
        # whole-file table — otherwise patch would be a way around the gate.
        patch_text = "--- /dev/null\n+++ b/notes.md\n@@ -0,0 +1,1 @@\n+replaced\n"
        with pytest.raises(RetriableError, match="read it before overwriting"):
            mutate(wired_card, "workspace_patch", patch_text)
        assert notes.read_text(encoding="utf-8") == BODY

    def test_an_update_over_an_unread_file_is_refused(
        self, wired_card: WorkspaceTool, notes: Path
    ) -> None:
        patch_text = "--- a/notes.md\n+++ b/notes.md\n@@ -1,2 +1,2 @@\n alpha\n-bravo\n+BRAVO\n"
        with pytest.raises(RetriableError, match="read it before editing"):
            mutate(wired_card, "workspace_patch", patch_text)
        assert notes.read_text(encoding="utf-8") == BODY

    def test_a_delete_over_an_unread_file_is_refused(
        self, wired_card: WorkspaceTool, notes: Path
    ) -> None:
        patch_text = "--- a/notes.md\n+++ /dev/null\n@@ -1 +0,0 @@\n-alpha\n"
        with pytest.raises(RetriableError, match="read it before overwriting"):
            mutate(wired_card, "workspace_patch", patch_text)
        assert notes.exists()

    def test_a_read_file_patches_and_refreshes_the_writers_observation(
        self, wired_card: WorkspaceTool, notes: Path
    ) -> None:
        read(wired_card, "notes.md")
        patch_text = "--- a/notes.md\n+++ b/notes.md\n@@ -1,2 +1,2 @@\n alpha\n-bravo\n+BRAVO\n"
        assert mutate(wired_card, "workspace_patch", patch_text) == "updated: notes.md"
        assert "BRAVO" in notes.read_text(encoding="utf-8")
        # The patch refreshed the observation, so a follow-up write is accepted.
        assert mutate(wired_card, "workspace_write", "notes.md", "after\n") == ("Written: notes.md")


# ---------------------------------------------------------------------------
# AC10: mkdir is serialized but not content-gated
# ---------------------------------------------------------------------------


class TestMkdirIsRoutedNotGated:
    def test_a_directory_needs_no_prior_read(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        assert mutate(wired_card, "workspace_mkdir", "src/utils") == "Created: src/utils"
        assert (workspace_tree / "src" / "utils").is_dir()

    def test_it_stays_idempotent(self, wired_card: WorkspaceTool, workspace_tree: Path) -> None:
        mutate(wired_card, "workspace_mkdir", "src")
        assert mutate(wired_card, "workspace_mkdir", "src") == "Created: src"
        assert (workspace_tree / "src").is_dir()

    def test_it_records_no_observation_and_no_writer(
        self,
        wired_card: WorkspaceTool,
        workspace_actor: WorkspaceActor,
        observer: FakeActorToolObserver,
    ) -> None:
        mutate(wired_card, "workspace_mkdir", "src")
        agent_id = str(observer.myAddress.agent_id)
        assert workspace_actor.observation_for(agent_id, "src") is None


# ---------------------------------------------------------------------------
# AC11: the outcome statuses map onto the error contract exactly
# ---------------------------------------------------------------------------


class TestTheErrorContract:
    def test_an_accepted_outcome_carries_the_unchanged_confirmation(
        self, workspace_actor: WorkspaceActor, workspace_tree: Path
    ) -> None:
        outcome = outcome_of(workspace_actor, "apply_write", "solo", "new.md", "body\n")
        assert outcome.status is MutationStatus.ACCEPTED
        assert outcome.message == "Written: new.md"

    def test_a_missing_anchor_is_failed_not_rejected(
        self, wired_card: WorkspaceTool, workspace_actor: WorkspaceActor, notes: Path
    ) -> None:
        read(wired_card, "notes.md")
        agent_id = wired_card._agent_id
        outcome = outcome_of(
            workspace_actor, "apply_edit", agent_id, "notes.md", "absent", "x", False
        )
        assert outcome.status is MutationStatus.FAILED
        assert outcome.message == "[ERROR] old_string not found in notes.md"

    def test_a_gate_refusal_is_rejected(self, workspace_actor: WorkspaceActor, notes: Path) -> None:
        outcome = outcome_of(workspace_actor, "apply_write", "solo", "notes.md", "mine\n")
        assert outcome.status is MutationStatus.REJECTED

    def test_the_outcome_model_round_trips(
        self, workspace_actor: WorkspaceActor, workspace_tree: Path
    ) -> None:
        from akgentic.tool.workspace.models import MutationOutcome

        outcome = outcome_of(workspace_actor, "apply_mkdir", "solo", "sub")
        assert MutationOutcome.model_validate(outcome.model_dump()) == outcome


# ---------------------------------------------------------------------------
# AC12 / AC13: nothing new reaches the LLM, and reads are untouched
# ---------------------------------------------------------------------------


class TestTheToolSurfaceIsUnchanged:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("workspace_write", ["path", "content"]),
            ("workspace_delete", ["path"]),
            ("workspace_edit", ["path", "old_string", "new_string", "replace_all"]),
            ("workspace_multi_edit", ["edits"]),
            ("workspace_patch", ["patch_text"]),
            ("workspace_mkdir", ["path"]),
        ],
    )
    def test_no_mutation_signature_gained_a_parameter(
        self, name: str, expected: list[str], wired_card: WorkspaceTool
    ) -> None:
        import inspect

        assert list(inspect.signature(tool_named(wired_card, name)).parameters) == expected

    def test_no_card_field_can_bypass_the_gate(self) -> None:
        # A single field named force / expected / digest would undo the whole
        # mechanism the first time a rejection was not understood.
        fields = set(WorkspaceTool.model_fields)
        assert not fields & {"force", "expected", "digest", "unsafe", "bypass_gate"}

    def test_the_card_still_serialises(self) -> None:
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
        assert WorkspaceTool.model_validate(card.model_dump()) == card

    def test_a_read_never_becomes_stale(self, wired_card: WorkspaceTool, notes: Path) -> None:
        # Reads answer from the agent's own handle regardless of the gate: no
        # refusal exists on the read side at all.
        assert "alpha" in read(wired_card, "notes.md")
        assert "alpha" in read(wired_card, "notes.md")
