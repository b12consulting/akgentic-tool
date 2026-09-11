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
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from akgentic.tool.errors import RetriableError
from akgentic.tool.workspace.actor import WorkspaceActor
from akgentic.tool.workspace.edit import EditItem
from akgentic.tool.workspace.models import (
    GITIGNORE_NAME,
    MAX_REJECTION_DIFF_LINES,
    MutationStatus,
)
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
    requires_git,
    tool_named,
    workspace_path_for,
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


class TestEveryMutationRunsOnTheCardsResolvedTree:
    """**Premise reversed by decision**: the card's own handle *is* the tree now.

    This class used to point the card's handle at a decoy and assert the write
    landed in the actor's real tree — the invariant that made "every mutation
    runs on the actor" observable. The gate is the card's since 52-5, so that
    sentence is no longer true and the assertion that carried it cannot be kept
    as written. What survives is the half that still matters and is still a live
    hazard: the mutation writes through the handle ``observer()`` **resolved**,
    never one it derives for itself at call time (the lead's checklist, (b): the
    consumer uses the object the card resolved).
    """

    def test_a_write_goes_through_the_handle_the_bind_resolved(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        workspace_tree: Path,
    ) -> None:
        # The handle is swapped *after* the bind, so nothing about path
        # resolution is patched: a gate that re-derived its own tree from the
        # workspace id would ignore the swap and write into the real tree.
        decoy = Filesystem(str(workspaces_root), "decoy")
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
        card.observer(FakeActorToolObserver(orchestrator_proxy))
        card._workspace = decoy

        assert mutate(card, "workspace_write", "only.md", "content\n") == "Written: only.md"

        assert (decoy._root / "only.md").exists()
        assert not (workspace_tree / "only.md").exists()

    def test_the_bind_resolves_the_same_tree_the_actor_owns(
        self,
        wired_card: WorkspaceTool,
        workspace_actor: WorkspaceActor,
        workspace_tree: Path,
    ) -> None:
        """The positive beside it: card and actor are anchored to one directory.

        Two handles over one tree is the arrangement the file lock exists for;
        two handles over *two* trees would be a gate guarding nothing.
        """
        assert wired_card._workspace is not None
        assert wired_card._workspace._root == workspace_tree.resolve()
        assert workspace_actor._workspace._root == wired_card._workspace._root

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
        # the gate reachable from any harness that skipped the binding. The
        # unbound state is card-side now — the ``Filesystem`` the bind resolves.
        #
        # The callable is taken *before* the handle is dropped, because
        # ``get_tools`` builds the read factories too and they capture the same
        # handle: nulling it first would fail in the factory and never reach the
        # gate, which is the thing under test.
        mutation = tool_named(wired_card, name)
        wired_card._workspace = None
        with pytest.raises(RuntimeError, match="workspace is not bound"):
            mutation(*args)
        # The seeded .gitignore is the journal's, written at actor start and
        # before any agent existed; nothing else may have appeared.
        assert [
            entry.name for entry in workspace_tree.iterdir() if entry.name != GITIGNORE_NAME
        ] == []

    def test_a_card_that_never_bound_at_all_refuses_the_same_way(
        self, workspace_tree: Path
    ) -> None:
        """The unconstructed case, reached through the gate rather than a closure.

        A card built and never handed an observer has no ``Filesystem``, no
        metadata directory and no journal. Every one of those is optional to the
        refusal except the first — and the first is what makes an ungated write
        impossible rather than merely unlikely.
        """
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)

        with pytest.raises(RuntimeError, match="workspace is not bound"):
            card.apply_write("fresh.md", "body\n")
        with pytest.raises(RuntimeError, match="workspace is not bound"):
            card.apply_mkdir("sub")

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
    @requires_git
    def test_it_names_the_agent_whose_write_is_on_disk(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        notes: Path,
    ) -> None:
        """The refusal names the agent, not a UUID — and the journal is where it reads it.

        **Re-pointed, not weakened.** The name used to come from a
        ``{path -> last writer}`` map on the actor: one process\'s memory, which
        two workers over one mount held two of and neither could see the
        other\'s. It now comes from the git journal, which is on the tree —
        exactly the move ADR-051 Decision 4 makes. The assertion is unchanged,
        because the *message* is unchanged: it is the product an agent reads,
        and *"last written by agent \'3f2a…\'"* is something a model can read and
        nothing it can act on.
        """
        alice, _alice_observer = card_for(orchestrator_proxy, "alice", git_journal=True)
        bob, bob_observer = card_for(orchestrator_proxy, "bob", git_journal=True)
        read(alice, "notes.md")
        read(bob, "notes.md")
        mutate(bob, "workspace_write", "notes.md", "bob's version\n")

        with pytest.raises(RetriableError) as refusal:
            mutate(alice, "workspace_write", "notes.md", "alice's version\n")

        message = str(refusal.value)
        assert "last written by agent 'bob'" in message
        assert str(bob_observer.myAddress.agent_id) not in message

    def test_with_no_journal_it_names_nobody_and_claims_nothing(
        self,
        wired_card: WorkspaceTool,
        bob: tuple[WorkspaceTool, FakeActorToolObserver],
        notes: Path,
    ) -> None:
        """The second half of ADR-051 Decision 4, and it is not optional.

        Attribution is a best-effort extra that a journal pays for. Without one
        there is no history, so a teammate\'s write and an upload are
        indistinguishable — and claiming either would be a guess stated as a
        fact. The refusal\'s actionable half is unchanged, which is the whole
        point: every configuration gets that.
        """
        bob_card, _bob_observer = bob
        read(wired_card, "notes.md")
        read(bob_card, "notes.md")
        mutate(bob_card, "workspace_write", "notes.md", "bob's version\n")

        with pytest.raises(RetriableError) as refusal:
            mutate(wired_card, "workspace_write", "notes.md", "alice's version\n")

        message = str(refusal.value)
        assert "last written by agent" not in message
        assert "came from outside" not in message
        assert message.startswith("Refused to modify notes.md: it changed since you read it.")
        assert "Read the file again" in message

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

    def test_a_stale_patch_does_not_destroy_the_line_that_shifted_it(
        self, wired_card: WorkspaceTool, notes: Path
    ) -> None:
        """The corruption case, asserted directly. This is why hunks are verified.

        alice reads, bob prepends a line, and alice's patch still carries the old
        line numbers. Spliced at ``old_start`` with no context check, it would
        overwrite ``# banner`` and ``alpha`` — destroying bob's addition and
        reporting ``updated:`` while doing it. Verification finds the hunk's
        context one line lower, applies it *there*, and bob's line survives.
        """
        read(wired_card, "notes.md")
        notes.write_text("# banner\n" + BODY, encoding="utf-8")
        patch_text = "--- a/notes.md\n+++ b/notes.md\n@@ -1,2 +1,2 @@\n alpha\n-bravo\n+BRAVO\n"

        assert mutate(wired_card, "workspace_patch", patch_text) == "updated: notes.md"

        assert notes.read_text(encoding="utf-8") == "# banner\nalpha\nBRAVO\ncharlie\ndelta\n"

    def test_a_changed_file_whose_context_still_verifies_is_admitted(
        self, wired_card: WorkspaceTool, notes: Path
    ) -> None:
        # FR6's degradation, restored: a surgical change survives a concurrent
        # change to an unrelated region of the same file. The hunk's line
        # numbers happen to still hold here; its context is what admits it.
        read(wired_card, "notes.md")
        notes.write_text(BODY + "echo\n", encoding="utf-8")
        patch_text = "--- a/notes.md\n+++ b/notes.md\n@@ -1,2 +1,2 @@\n alpha\n-bravo\n+BRAVO\n"

        assert mutate(wired_card, "workspace_patch", patch_text) == "updated: notes.md"
        assert notes.read_text(encoding="utf-8") == "alpha\nBRAVO\ncharlie\ndelta\necho\n"

    def test_a_changed_file_whose_context_is_gone_is_refused_as_stale(
        self, wired_card: WorkspaceTool, notes: Path
    ) -> None:
        # The other branch: the text the hunk was cut against is not in the file
        # any more, at any offset. There is nothing to anchor to, so the agent is
        # told the file moved under it rather than handed a plausible splice.
        read(wired_card, "notes.md")
        notes.write_text("wholly different content\n", encoding="utf-8")
        patch_text = "--- a/notes.md\n+++ b/notes.md\n@@ -1,2 +1,2 @@\n alpha\n-bravo\n+BRAVO\n"

        with pytest.raises(RetriableError) as refusal:
            mutate(wired_card, "workspace_patch", patch_text)

        assert "line numbers that have since moved" in str(refusal.value)
        assert notes.read_text(encoding="utf-8") == "wholly different content\n"

    def test_the_refused_patch_lands_once_the_agent_re_reads(
        self, wired_card: WorkspaceTool, notes: Path
    ) -> None:
        # The refusal above is recoverable by the step it names.
        read(wired_card, "notes.md")
        notes.write_text("wholly different content\n", encoding="utf-8")
        patch_text = "--- a/notes.md\n+++ b/notes.md\n@@ -1,2 +1,2 @@\n alpha\n-bravo\n+BRAVO\n"
        with pytest.raises(RetriableError):
            mutate(wired_card, "workspace_patch", patch_text)

        read(wired_card, "notes.md")
        notes.write_text(BODY, encoding="utf-8")
        read(wired_card, "notes.md")
        assert mutate(wired_card, "workspace_patch", patch_text) == "updated: notes.md"
        assert notes.read_text(encoding="utf-8") == "alpha\nBRAVO\ncharlie\ndelta\n"

    def test_a_bad_hunk_on_an_unchanged_file_is_a_returned_error_not_a_refusal(
        self, wired_card: WorkspaceTool, notes: Path
    ) -> None:
        # Nothing moved under the agent: the patch is simply wrong about what the
        # file contains. A refusal telling it to re-read would teach it nothing,
        # so this stays the returned [ERROR] string a bad patch has always been.
        read(wired_card, "notes.md")
        patch_text = "--- a/notes.md\n+++ b/notes.md\n@@ -1,2 +1,2 @@\n nope\n-absent\n+NEW\n"

        result = mutate(wired_card, "workspace_patch", patch_text)

        assert result.startswith("[ERROR] notes.md:")
        assert "@@ -1,2 +1,2 @@" in result
        assert notes.read_text(encoding="utf-8") == BODY

    def test_an_ambiguous_hunk_is_not_applied_anywhere(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        # Context occurring twice gives no unambiguous offset, and guessing is
        # exactly the behaviour being removed.
        repeated = workspace_tree / "repeated.md"
        repeated.write_text("x\nmarker\ny\nmarker\n", encoding="utf-8")
        read(wired_card, "repeated.md")
        repeated.write_text("lead\nx\nmarker\ny\nmarker\n", encoding="utf-8")
        patch_text = "--- a/repeated.md\n+++ b/repeated.md\n@@ -2,1 +2,1 @@\n-marker\n+MARKER\n"

        with pytest.raises(RetriableError):
            mutate(wired_card, "workspace_patch", patch_text)

        assert repeated.read_text(encoding="utf-8") == "lead\nx\nmarker\ny\nmarker\n"


# ---------------------------------------------------------------------------
# AC12 / AC14: workspace_patch is all-or-nothing, and a vanished file refuses
# ---------------------------------------------------------------------------


class TestPatchIsAtomic:
    @pytest.fixture
    def two_files(self, wired_card: WorkspaceTool, workspace_tree: Path) -> Path:
        for name, body in (("a.py", "x = 1\n"), ("b.py", "y = 2\n")):
            (workspace_tree / name).write_text(body, encoding="utf-8")
            read(wired_card, name)
        return workspace_tree

    def test_a_refusal_on_the_second_file_leaves_the_first_untouched(
        self, wired_card: WorkspaceTool, two_files: Path
    ) -> None:
        # c.py was never read by this agent, so the gate refuses it — and a.py,
        # which the patch names first, must not have been written.
        (two_files / "c.py").write_text("z = 3\n", encoding="utf-8")
        patch_text = (
            "--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n-x = 1\n+x = 10\n"
            "--- a/c.py\n+++ b/c.py\n@@ -1,1 +1,1 @@\n-z = 3\n+z = 30\n"
        )

        with pytest.raises(RetriableError, match="read it before editing"):
            mutate(wired_card, "workspace_patch", patch_text)

        assert (two_files / "a.py").read_text(encoding="utf-8") == "x = 1\n"
        assert (two_files / "c.py").read_text(encoding="utf-8") == "z = 3\n"

    def test_a_hunk_failure_on_the_second_file_leaves_the_first_untouched(
        self, wired_card: WorkspaceTool, two_files: Path
    ) -> None:
        patch_text = (
            "--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n-x = 1\n+x = 10\n"
            "--- a/b.py\n+++ b/b.py\n@@ -1,1 +1,1 @@\n-not there\n+whatever\n"
        )

        result = mutate(wired_card, "workspace_patch", patch_text)

        assert result.startswith("[ERROR] b.py:")
        # The successfully-rendered a.py line is *not* reported, because nothing
        # was applied. That is the visible consequence of atomicity.
        assert "updated: a.py" not in result
        assert (two_files / "a.py").read_text(encoding="utf-8") == "x = 1\n"

    def test_a_patch_that_all_applies_lands_together(
        self, wired_card: WorkspaceTool, two_files: Path
    ) -> None:
        patch_text = (
            "--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n-x = 1\n+x = 10\n"
            "--- a/b.py\n+++ b/b.py\n@@ -1,1 +1,1 @@\n-y = 2\n+y = 20\n"
        )

        assert mutate(wired_card, "workspace_patch", patch_text) == "updated: a.py\nupdated: b.py"
        assert (two_files / "a.py").read_text(encoding="utf-8") == "x = 10\n"
        assert (two_files / "b.py").read_text(encoding="utf-8") == "y = 20\n"

    def test_a_vanished_file_gives_the_deleted_since_you_read_it_refusal(
        self, wired_card: WorkspaceTool, notes: Path
    ) -> None:
        """Not ``[ERROR] notes.md: notes.md``, which is what render-then-check gave.

        Checking first also means the refusal clears the observation, exactly as
        every other mutation's does — so the agent's next write to the path is
        judged as a create rather than refused forever.
        """
        read(wired_card, "notes.md")
        notes.unlink()
        patch_text = "--- a/notes.md\n+++ b/notes.md\n@@ -1,2 +1,2 @@\n alpha\n-bravo\n+BRAVO\n"

        with pytest.raises(RetriableError, match="deleted since you read it"):
            mutate(wired_card, "workspace_patch", patch_text)

        assert mutate(wired_card, "workspace_write", "notes.md", "rebuilt\n") == (
            "Written: notes.md"
        )

    def test_two_deletion_sections_naming_one_set_delete_it_once(
        self, wired_card: WorkspaceTool, two_files: Path
    ) -> None:
        # deleted_paths reads the whole diff at once, so both sections arrive
        # carrying the same set. Publication is deferred now, so a duplicate
        # would reach delete() a second time on a file already gone.
        patch_text = (
            "--- a/a.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x = 1\n"
            "--- a/b.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-y = 2\n"
        )

        assert mutate(wired_card, "workspace_patch", patch_text) == ("deleted: a.py\ndeleted: b.py")
        assert not (two_files / "a.py").exists()
        assert not (two_files / "b.py").exists()

    def test_an_unread_missing_file_is_still_the_returned_error(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        # No observation to refuse against: the patch simply names a file that is
        # not there, which has always been an [ERROR] line.
        patch_text = "--- a/missing.py\n+++ b/missing.py\n@@ -1,1 +1,1 @@\n-old\n+new\n"
        assert mutate(wired_card, "workspace_patch", patch_text) == "[ERROR] missing.py: missing.py"


# ---------------------------------------------------------------------------
# AC11: losing a staged file is a refusal the agent can act on, not a traceback
# ---------------------------------------------------------------------------


class TestALostStagedFileIsARefusal:
    """The other half of the sweep race, contained where PermissionError already is.

    Two teams sharing a ``workspace_id`` get two actors over one tree, and either
    one's startup sweep can unlink the other's staged file in the sub-millisecond
    window inside ``Filesystem.write``. Nothing is corrupted — the target keeps
    its previous bytes — but ``os.replace`` raises ``FileNotFoundError``, and
    29-3's mutation methods caught only ``PermissionError``.
    """

    def test_a_single_write_refuses_and_says_to_retry(
        self,
        wired_card: WorkspaceTool,
        workspace_actor: WorkspaceActor,
        workspace_tree: Path,
    ) -> None:
        tree = wired_card._workspace
        assert tree is not None
        with patch.object(tree, "write", side_effect=FileNotFoundError("staged file gone")):
            with pytest.raises(RetriableError) as refusal:
                mutate(wired_card, "workspace_write", "fresh.md", "body\n")

        message = str(refusal.value)
        assert "retry exactly the same change" in message
        # It must NOT claim staleness: nothing about the file changed, and
        # sending the agent to re-read would have it redo work already correct.
        assert "changed since you read it" not in message
        assert "Read the file again" not in message

    def test_the_batch_path_refuses_the_same_way(
        self,
        wired_card: WorkspaceTool,
        workspace_actor: WorkspaceActor,
        workspace_tree: Path,
    ) -> None:
        for name, body in (("a.py", "x = 1\n"), ("b.py", "y = 2\n")):
            (workspace_tree / name).write_text(body, encoding="utf-8")
            read(wired_card, name)
        tree = wired_card._workspace
        assert tree is not None

        with patch.object(tree, "write_many", side_effect=FileNotFoundError("gone")):
            with pytest.raises(RetriableError, match="retry exactly the same change"):
                mutate(
                    wired_card,
                    "workspace_multi_edit",
                    [
                        EditItem(path="a.py", old_string="x = 1", new_string="x = 10"),
                        EditItem(path="b.py", old_string="y = 2", new_string="y = 20"),
                    ],
                )

        assert (workspace_tree / "a.py").read_text(encoding="utf-8") == "x = 1\n"

    def test_a_patch_refuses_the_same_way(
        self, wired_card: WorkspaceTool, workspace_actor: WorkspaceActor, notes: Path
    ) -> None:
        read(wired_card, "notes.md")
        tree = wired_card._workspace
        assert tree is not None
        patch_text = "--- a/notes.md\n+++ b/notes.md\n@@ -1,2 +1,2 @@\n alpha\n-bravo\n+BRAVO\n"

        with patch.object(tree, "write_many", side_effect=FileNotFoundError("gone")):
            with pytest.raises(RetriableError, match="retry exactly the same change"):
                mutate(wired_card, "workspace_patch", patch_text)

        assert notes.read_text(encoding="utf-8") == BODY


# ---------------------------------------------------------------------------
# A refusal has to be recoverable, and it must not attribute what it cannot know
# ---------------------------------------------------------------------------


class TestAVanishedFileIsRecoverable:
    def test_the_agent_can_recreate_a_file_deleted_under_it(
        self, wired_card: WorkspaceTool, notes: Path
    ) -> None:
        """The stale-because-gone refusal is a one-time warning, not a dead end.

        Its stated next step is "read the file again", which a missing file
        cannot satisfy — ``workspace_read`` raises and records nothing. If the
        observation survived the refusal, every later write *and* delete of the
        path would be refused for the life of the team and the agent could never
        recreate what a teammate removed.
        """
        read(wired_card, "notes.md")
        notes.unlink()

        with pytest.raises(RetriableError, match="deleted since you read it"):
            mutate(wired_card, "workspace_write", "notes.md", "rebuilt\n")

        assert mutate(wired_card, "workspace_write", "notes.md", "rebuilt\n") == (
            "Written: notes.md"
        )
        assert notes.read_text(encoding="utf-8") == "rebuilt\n"

    def test_a_file_that_came_back_is_still_protected(
        self,
        wired_card: WorkspaceTool,
        bob: tuple[WorkspaceTool, FakeActorToolObserver],
        notes: Path,
    ) -> None:
        # Clearing the observation does not license a clobber: the retry is a
        # create, and the whole-file table judges it against what is on disk now.
        bob_card, _ = bob
        read(wired_card, "notes.md")
        notes.unlink()
        with pytest.raises(RetriableError, match="deleted since you read it"):
            mutate(wired_card, "workspace_write", "notes.md", "alice's version\n")

        mutate(bob_card, "workspace_write", "notes.md", "bob rebuilt it\n")

        with pytest.raises(RetriableError, match="read it before overwriting"):
            mutate(wired_card, "workspace_write", "notes.md", "alice's version\n")
        assert notes.read_text(encoding="utf-8") == "bob rebuilt it\n"

    def test_a_teammates_delete_is_not_reported_as_an_outside_change(
        self,
        wired_card: WorkspaceTool,
        bob: tuple[WorkspaceTool, FakeActorToolObserver],
        notes: Path,
    ) -> None:
        # bob deleted it through the tool. A deleted file has no live bytes to
        # attribute, so the refusal must not claim the change came from outside
        # the team — that misattributes a teammate exactly as naming the wrong
        # writer would.
        bob_card, _ = bob
        read(wired_card, "notes.md")
        read(bob_card, "notes.md")
        mutate(bob_card, "workspace_delete", "notes.md")

        with pytest.raises(RetriableError) as refusal:
            mutate(wired_card, "workspace_write", "notes.md", "alice's version\n")

        message = str(refusal.value)
        assert "deleted since you read it" in message
        assert "came from outside" not in message
        assert "last written by agent" not in message


# ---------------------------------------------------------------------------
# The refusal travels back into the model's turn, so its diff is bounded
# ---------------------------------------------------------------------------


class TestTheRefusalDiffIsBounded:
    def test_a_large_stale_write_does_not_carry_the_whole_file(
        self,
        wired_card: WorkspaceTool,
        bob: tuple[WorkspaceTool, FakeActorToolObserver],
        workspace_tree: Path,
    ) -> None:
        bob_card, _ = bob
        big = workspace_tree / "big.md"
        big.write_text("\n".join(f"line {n}" for n in range(5_000)) + "\n", encoding="utf-8")
        read(wired_card, "big.md", limit=10_000)
        read(bob_card, "big.md", limit=10_000)
        mutate(bob_card, "workspace_write", "big.md", "bob replaced it\n")

        with pytest.raises(RetriableError) as refusal:
            mutate(
                wired_card,
                "workspace_write",
                "big.md",
                "\n".join(f"mine {n}" for n in range(5_000)) + "\n",
            )

        message = str(refusal.value)
        assert message.count("\n") < MAX_REJECTION_DIFF_LINES + 20
        assert "more diff line(s) not shown" in message

    def test_a_small_diff_is_shown_whole(
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

        assert "not shown" not in str(refusal.value)


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

    def test_it_records_no_observation_and_touches_nothing(
        self,
        wired_card: WorkspaceTool,
    ) -> None:
        mutate(wired_card, "workspace_mkdir", "src")
        assert wired_card.observation_for("src") is None
        # And nothing in the write set either, so the journal makes no commit of
        # its own: git does not track empty directories.
        assert wired_card._touched == []


# ---------------------------------------------------------------------------
# AC11: the outcome statuses map onto the error contract exactly
# ---------------------------------------------------------------------------


class TestTheErrorContract:
    def test_an_accepted_outcome_carries_the_unchanged_confirmation(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        outcome = outcome_of(wired_card, "apply_write", "new.md", "body\n")
        assert outcome.status is MutationStatus.ACCEPTED
        assert outcome.message == "Written: new.md"

    def test_a_missing_anchor_is_failed_not_rejected(
        self, wired_card: WorkspaceTool, notes: Path
    ) -> None:
        read(wired_card, "notes.md")
        outcome = outcome_of(wired_card, "apply_edit", "notes.md", "absent", "x", False)
        assert outcome.status is MutationStatus.FAILED
        assert outcome.message == "[ERROR] old_string not found in notes.md"

    def test_a_gate_refusal_is_rejected(self, wired_card: WorkspaceTool, notes: Path) -> None:
        outcome = outcome_of(wired_card, "apply_write", "notes.md", "mine\n")
        assert outcome.status is MutationStatus.REJECTED

    def test_the_outcome_model_round_trips(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        from akgentic.tool.workspace.models import MutationOutcome

        outcome = outcome_of(wired_card, "apply_mkdir", "sub")
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


# ---------------------------------------------------------------------------
# Story 52-5, AC 6: exactly one convergence point, and all six pass through it
# ---------------------------------------------------------------------------


class TestEveryMutationConvergesOnOnePoint:
    """``_gated`` is the sole place the busy check, the commits and the lock live.

    The reason this is structural rather than a comment: a *seventh* mutation
    added later must not be able to forget any of them, and the one that fails
    silently is the ``flock`` — a forgotten lock costs nothing in any test, and
    then loses an update under load on a shared mount.

    The enumeration is by **introspection**, so a mutation added tomorrow joins
    it without anybody remembering to. A public ``apply_*`` that bypassed
    ``_gated`` would appear in the list and fail.
    """

    CONVERGENCE = "_gated"

    def _mutations(self) -> list[str]:
        """Every public mutation the card exposes, found rather than listed."""
        return sorted(
            name
            for name in dir(WorkspaceTool)
            if name.startswith("apply_") and callable(getattr(WorkspaceTool, name))
        )

    def test_the_enumeration_finds_the_six_that_exist_today(self) -> None:
        """The sweep is floored: one that found nothing would pass every check below."""
        assert self._mutations() == [
            "apply_delete",
            "apply_edit",
            "apply_mkdir",
            "apply_multi_edit",
            "apply_patch",
            "apply_write",
        ]

    def test_every_mutation_reaches_the_convergence_point(
        self, wired_card: WorkspaceTool, notes: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Called through the card, with the convergence point watched.

        Each mutation is driven with arguments that reach the gate at all — the
        refusals below are fine, because what is asserted is that the call
        *arrived*, not what it decided.
        """
        from akgentic.tool.workspace.edit import EditItem

        seen: list[str] = []
        real = WorkspaceTool._gated

        def watched(card: WorkspaceTool, capability: str, paths: Any, run: Any) -> Any:
            seen.append(capability)
            return real(card, capability, paths, run)

        monkeypatch.setattr(WorkspaceTool, "_gated", watched)
        arguments: dict[str, tuple[Any, ...]] = {
            "apply_write": ("notes.md", "mine\n"),
            "apply_delete": ("notes.md",),
            "apply_edit": ("notes.md", "alpha", "ALPHA"),
            "apply_multi_edit": ([EditItem(path="notes.md", old_string="a", new_string="b")],),
            "apply_patch": (
                "--- a/notes.md\n+++ b/notes.md\n@@ -1,2 +1,2 @@\n-alpha\n+A\n bravo\n",
            ),
            "apply_mkdir": ("sub",),
        }
        assert set(arguments) == set(self._mutations()), (
            "a mutation was added without a case here — add one rather than "
            "narrowing the enumeration to the ones that still have one"
        )

        for name in self._mutations():
            getattr(wired_card, name)(*arguments[name])

        assert sorted(seen) == sorted(["write", "delete", "edit", "multi_edit", "patch", "mkdir"])
        assert len(seen) == len(self._mutations())

    def test_the_convergence_point_is_one_method_and_it_holds_all_five_duties(self) -> None:
        """The busy check, the two commits, the lock and the stale-mark, in one body.

        Asserted on the **source of one method**, because the defect this
        prevents is a duty quietly moving into the five callers where a sixth
        would then be written without it.
        """
        import inspect

        from akgentic.tool.workspace.card import gate as gate_module

        body = inspect.getsource(gate_module.CardGate._gated)
        for duty in ("_busy_refusal", "commit_out_of_band", "_hold", "commit_paths", "_mark_stale"):
            assert duty in body, f"{duty} no longer converges in _gated"
        # And no ``apply_*`` performs any of them for itself.
        for name in self._mutations():
            source = inspect.getsource(getattr(gate_module.CardGate, name))
            assert "_busy_refusal" not in source
            assert "commit_" not in source
            assert "_hold(" not in source


# ---------------------------------------------------------------------------
# Story 52-5, AC 15: the stale-mark carries exactly the paths that changed
# ---------------------------------------------------------------------------


class _StaleRecorder:
    """A tell proxy that records the stale-mark sets and forwards nothing else."""

    def __init__(self) -> None:
        self.sets: list[list[str]] = []

    def mark_paths_stale(self, paths: list[str]) -> None:
        self.sets.append(list(paths))

    def __getattr__(self, name: str) -> Any:
        def swallow(*args: Any, **kwargs: Any) -> None:
            return None

        return swallow


class TestTheStaleMarkReachesTheIndex:
    """Every accepted mutation signals the index with the paths it touched.

    Deletes included — ``_forget`` appends to the write set for the same reason
    ``_accept`` does, so a seventh mutation gets the signal for free.
    """

    def _card_with(
        self, orchestrator_proxy: FakeOrchestratorProxy, recorder: _StaleRecorder
    ) -> WorkspaceTool:
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
        card.observer(
            FakeActorToolObserver(orchestrator_proxy, name="alice", workspace_tell_proxy=recorder)
        )
        return card

    def test_a_write_signals_its_one_path(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        recorder = _StaleRecorder()
        card = self._card_with(orchestrator_proxy, recorder)

        card.apply_write("fresh.md", "body\n")

        assert recorder.sets == [["fresh.md"]]

    def test_a_delete_signals_the_path_it_removed(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        (workspace_tree / "notes.md").write_text("alpha\n", encoding="utf-8")
        recorder = _StaleRecorder()
        card = self._card_with(orchestrator_proxy, recorder)
        read(card, "notes.md")

        card.apply_delete("notes.md")

        assert recorder.sets == [["notes.md"]]

    def test_a_multi_edit_signals_every_file_it_wrote(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        from akgentic.tool.workspace.edit import EditItem

        for name, body in (("a.py", "x = 1\n"), ("b.py", "y = 2\n")):
            (workspace_tree / name).write_text(body, encoding="utf-8")
        recorder = _StaleRecorder()
        card = self._card_with(orchestrator_proxy, recorder)
        read(card, "a.py")
        read(card, "b.py")

        card.apply_multi_edit(
            [
                EditItem(path="a.py", old_string="x = 1", new_string="x = 10"),
                EditItem(path="b.py", old_string="y = 2", new_string="y = 20"),
            ]
        )

        assert [sorted(paths) for paths in recorder.sets] == [["a.py", "b.py"]]

    def test_a_patch_signals_what_it_created_and_what_it_deleted(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        (workspace_tree / "gone.md").write_text("bye\n", encoding="utf-8")
        recorder = _StaleRecorder()
        card = self._card_with(orchestrator_proxy, recorder)
        read(card, "gone.md")

        outcome = card.apply_patch(
            "--- /dev/null\n+++ b/made.md\n@@ -0,0 +1 @@\n+hello\n"
            "--- a/gone.md\n+++ /dev/null\n@@ -1 +0,0 @@\n-bye\n"
        )

        assert outcome.status is MutationStatus.ACCEPTED, outcome.message
        assert [sorted(paths) for paths in recorder.sets] == [["gone.md", "made.md"]]

    def test_a_refused_mutation_signals_nothing(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        (workspace_tree / "notes.md").write_text("alpha\n", encoding="utf-8")
        recorder = _StaleRecorder()
        card = self._card_with(orchestrator_proxy, recorder)

        outcome = card.apply_write("notes.md", "mine\n")

        assert outcome.status is MutationStatus.REJECTED
        assert recorder.sets == []

    def test_a_lost_signal_never_fails_the_mutation(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """It degrades to a stale index row, which is recoverable; a raise is not."""

        class Dead:
            def mark_paths_stale(self, paths: list[str]) -> None:
                raise RuntimeError("the actor is gone")

            def __getattr__(self, name: str) -> Any:
                return lambda *a, **k: None

        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
        card.observer(
            FakeActorToolObserver(orchestrator_proxy, name="alice", workspace_tell_proxy=Dead())
        )

        assert card.apply_write("fresh.md", "body\n").message == "Written: fresh.md"
        assert (workspace_tree / "fresh.md").read_text(encoding="utf-8") == "body\n"


# ---------------------------------------------------------------------------
# Story 52-5, AC 13: nothing in the gate depends on the journal being enabled
# ---------------------------------------------------------------------------


class TestBothTablesAreIdenticalWithoutAJournal:
    """Every verdict, both tables, with ``git_journal=False`` — byte for byte.

    Cheap, and it catches a whole class in one place: the gate acquired a journal
    object in this story, and the failure mode worth fearing is a verdict that
    quietly starts depending on one. The journal is an **extra** that buys
    history and attribution; with it off the refusal loses its middle line and
    nothing else.

    Driven as a table so the two configurations are compared rather than merely
    both exercised: a row that answered differently with the journal on would
    fail here even if both answers looked reasonable on their own.
    """

    ROWS: list[tuple[str, str]] = [
        ("unread create", "create"),
        ("unread overwrite", "read it before overwriting"),
        ("whole-read overwrite", "accept"),
        ("changed since read", "it changed since you read it"),
        ("paginated overwrite", "a page is not a licence"),
        ("deleted since read", "it was deleted since you read it"),
        ("unread edit", "read it before editing"),
        ("edit on a changed file", "accept"),
        ("stale anchor", "no longer matches it exactly"),
        ("missing anchor", "[ERROR] old_string not found"),
        ("mkdir", "Created"),
    ]

    @staticmethod
    def _told(call: Callable[[], Any]) -> str:
        """What the agent is told — the returned string, or the refusal it raised."""
        try:
            return str(call())
        except RetriableError as refused:
            return str(refused)

    def _verdict(self, card: WorkspaceTool, row: str, tree: Path) -> str:
        """Drive one row and return what the agent is told, refusal or not."""
        notes = tree / "notes.md"
        if row == "unread create":
            return self._told(lambda: mutate(card, "workspace_write", "brand-new.md", "body\n"))
        if row == "mkdir":
            return self._told(lambda: mutate(card, "workspace_mkdir", "sub"))
        notes.write_text(BODY, encoding="utf-8")
        if row in ("unread overwrite", "unread edit", "paginated overwrite"):
            return self._unread_row(card, row)
        read(card, "notes.md")
        return self._observed_row(card, row, notes)

    def _unread_row(self, card: WorkspaceTool, row: str) -> str:
        """The three rows whose agent has not read the file whole."""
        if row == "unread overwrite":
            return self._told(lambda: mutate(card, "workspace_write", "notes.md", "mine\n"))
        if row == "unread edit":
            return self._told(lambda: mutate(card, "workspace_edit", "notes.md", "alpha", "A"))
        assert row == "paginated overwrite"
        read(card, "notes.md", limit=1)
        return self._told(lambda: mutate(card, "workspace_write", "notes.md", "mine\n"))

    def _observed_row(self, card: WorkspaceTool, row: str, notes: Path) -> str:
        """The rows whose agent has read the file whole; some then lose it."""
        if row == "whole-read overwrite":
            return self._told(lambda: mutate(card, "workspace_write", "notes.md", "mine\n"))
        if row == "missing anchor":
            return self._told(lambda: mutate(card, "workspace_edit", "notes.md", "absent", "x"))
        if row == "deleted since read":
            notes.unlink()
            return self._told(lambda: mutate(card, "workspace_write", "notes.md", "mine\n"))
        # The remaining three all need the file to have moved behind the gate.
        notes.write_text("wholly different\n", encoding="utf-8")
        if row == "changed since read":
            return self._told(lambda: mutate(card, "workspace_write", "notes.md", "mine\n"))
        if row == "stale anchor":
            return self._told(lambda: mutate(card, "workspace_edit", "notes.md", "alpha", "A"))
        assert row == "edit on a changed file"
        return self._told(lambda: mutate(card, "workspace_edit", "notes.md", "wholly", "W"))

    @requires_git
    @pytest.mark.parametrize(("row", "expected"), ROWS, ids=[row for row, _ in ROWS])
    def test_the_verdict_is_the_same_with_the_journal_on_and_off(
        self,
        row: str,
        expected: str,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
    ) -> None:
        """Two trees, two cards, one row — and the actionable text must match.

        The **provenance line is the one thing allowed to differ**, and it is
        excluded by comparing the first and last lines rather than the whole
        message: that line is the best-effort extra, and asserting it identical
        would be asserting the opposite of ADR-051 Decision 4.
        """
        verdicts = {}
        for journal in (True, False):
            leaf = f"sweep-{row.replace(' ', '-')}-{'on' if journal else 'off'}"
            tree = workspaces_root / workspace_path_for(leaf)
            tree.mkdir(parents=True, exist_ok=True)
            card, _observer = card_for(
                orchestrator_proxy, f"alice-{leaf}", workspace_id=leaf, git_journal=journal
            )
            verdicts[journal] = self._verdict(card, row, tree)

        on, off = verdicts[True], verdicts[False]
        if expected in ("create", "accept"):
            assert not on.startswith("Refused")
            assert not off.startswith("Refused")
        else:
            assert expected in on, on
            assert expected in off, off
        on_lines, off_lines = on.splitlines(), off.splitlines()
        assert on_lines[0] == off_lines[0]
        assert on_lines[-1] == off_lines[-1]
        # And the journal-off message is never the longer of the two: the extra
        # a journal buys is a line, never a missing one.
        assert len(off_lines) <= len(on_lines)

    def test_a_card_with_no_journal_still_has_one_object_that_is_simply_off(
        self, wired_card: WorkspaceTool
    ) -> None:
        """The degradation is a disabled journal, never a ``None`` nobody guarded.

        Every ``GitJournal`` method is a no-op once the journal is off, which is
        what lets the convergence point call it unconditionally.
        """
        assert wired_card._journal is not None
        assert wired_card._journal.enabled is False
        assert wired_card._journal.last_author("notes.md") is None
