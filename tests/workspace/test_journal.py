"""The git journal: one mutation, one commit, linear history, and nothing required.

Three separable claims are tested here, and they fail independently:

- **the journal records what happened** — one accepted mutation is one commit,
  authored by the agent that made it, containing that mutation's own paths and
  nothing else;
- **the history stays linear and the repository stays out of reach** — no branch
  is ever created, and ``.git`` is a sibling of the tree rather than a member of
  it, so no read capability can see it;
- **none of it is required** — with ``git`` unresolvable or ``git_journal``
  off, every gate behaviour from story 29-3 is identical, character for
  character, and one warning is logged rather than one per mutation.

Assertions are made against parsed commit fields (see ``conftest.journal_log``),
never against a formatted log string, and the absence path patches the resolver
rather than mutating the session's ``PATH``.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from akgentic.tool.errors import RetriableError
from akgentic.tool.workspace.actor import (
    WORKSPACE_ACTOR_ROLE,
    WorkspaceActor,
    workspace_actor_name,
)
from akgentic.tool.workspace.edit import EditItem
from akgentic.tool.workspace.journal import (
    IDENTITY_DOMAIN,
    IDENTITY_FALLBACK,
    GitJournal,
    Identity,
    git_dir_for,
)
from akgentic.tool.workspace.models import (
    GITIGNORE_NAME,
    OUT_OF_BAND_AUTHOR,
    WorkspaceConfig,
)
from akgentic.tool.workspace.readers import DocumentReader
from akgentic.tool.workspace.tool import WorkspaceRead, WorkspaceTool, WorkspaceView

from tests.workspace.conftest import (
    WORKSPACE_NAME,
    FakeActorToolObserver,
    FakeOrchestratorProxy,
    card_for,
    git_show,
    journal_branches,
    journal_log,
    mutate,
    outcome_of,
    read,
    requires_git,
    tool_named,
    working_tree_is_clean,
)
from tests.workspace.test_workspace_actor import staging_name

pytestmark = requires_git

BODY = "alpha\nbravo\ncharlie\ndelta\n"


@pytest.fixture
def wired_card(
    observer: FakeActorToolObserver,
    workspace_tree: Path,
) -> WorkspaceTool:
    """The shared fixture, with the journal turned on.

    The card's default is off, so a file about the journal has to opt in. The
    override is module-level rather than per-test because every test here is
    about journal behaviour; a test that wants it off says so explicitly.
    """
    card = WorkspaceTool(workspace_id=WORKSPACE_NAME, git_journal=True)
    card.observer(observer)
    return card


@pytest.fixture
def notes(workspace_tree: Path) -> Path:
    """``notes.md``, on disk before any agent has looked at it."""
    path = workspace_tree / "notes.md"
    path.write_text(BODY, encoding="utf-8")
    return path


def alice_email(card: WorkspaceTool) -> str:
    """The author email the journal gives *card*'s agent: its id, then the domain."""
    return f"{card._agent_id}@{IDENTITY_DOMAIN}"


def commits_by(tree: Path, author: str) -> list[Any]:
    """Every commit *author* is on record for, oldest first.

    Counting an agent's own commits, rather than the log's total length, keeps
    these assertions independent of whatever a fixture happened to leave in the
    tree before the actor started — which the journal correctly records as
    ``out-of-band`` and which is not what any of them are about.
    """
    return [commit for commit in journal_log(tree) if commit.author_name == author]


# ---------------------------------------------------------------------------
# AC1: the repository exists at actor start, outside the tree, on master
# ---------------------------------------------------------------------------


class TestTheRepository:
    def test_it_is_created_at_actor_start(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        assert git_dir_for(workspace_tree).is_dir()

    def test_it_is_a_sibling_of_the_tree_and_never_inside_it(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        # Structural, not filtered: Filesystem._validate_path refuses anything
        # that is not relative to the root, so a repository placed here simply
        # does not resolve from inside the workspace.
        git_dir = git_dir_for(workspace_tree)
        assert not git_dir.is_relative_to(workspace_tree)
        assert git_dir.parent == workspace_tree.parent
        assert not (workspace_tree / ".git").exists()

    def test_the_initial_branch_is_master(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        assert journal_branches(workspace_tree) == ["master"]

    def test_it_is_not_bare_because_a_commit_succeeds(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        # A git directory whose name is not ".git" is exactly the shape that
        # comes out bare, and a bare repository refuses every `add`. One landed
        # commit is the behavioural proof that core.bare is false.
        mutate(wired_card, "workspace_write", "fresh.md", "body\n")
        assert [commit.subject for commit in journal_log(workspace_tree)][-1] == ("write: fresh.md")

    def test_restarting_over_an_existing_tree_keeps_the_history(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        mutate(wired_card, "workspace_write", "fresh.md", "body\n")
        before = [commit.sha for commit in journal_log(workspace_tree)]

        second = WorkspaceActor(
            config=WorkspaceConfig(
                name=workspace_actor_name(WORKSPACE_NAME),
                role=WORKSPACE_ACTOR_ROLE,
                workspace_name=WORKSPACE_NAME,
            )
        )
        second.on_start()

        after = [commit.sha for commit in journal_log(workspace_tree)]
        assert after[: len(before)] == before
        assert journal_branches(workspace_tree) == ["master"]

    def test_a_workspace_named_like_a_journal_refuses_and_degrades_off(
        self, workspaces_root: Path, orchestrator_proxy: FakeOrchestratorProxy
    ) -> None:
        """``<name>.git`` would share a directory with workspace ``<name>``'s journal.

        Operator-set rather than agent-set, so not reachable by anything the LLM
        does — but destructive and confusing when it happens, so it takes the
        cheap guard: refuse at init and degrade the journal off, which is the
        degradation FR9 already specifies. The gate must stay fully working.
        """
        tree = workspaces_root / "shared.git"
        tree.mkdir()
        (tree / "unseen.md").write_text("someone else's\n", encoding="utf-8")
        card, _observer = card_for(orchestrator_proxy, "alice", workspace_id="shared.git", git_journal=True)

        # The journal is off — no second repository, no seeded ignore file …
        assert not (workspaces_root / "shared.git.git").exists()
        assert not (tree / GITIGNORE_NAME).exists()
        # … and the gate is untouched: creates land, unread overwrites do not.
        assert mutate(card, "workspace_write", "fresh.md", "body\n") == "Written: fresh.md"
        with pytest.raises(RetriableError, match="read it before overwriting"):
            mutate(card, "workspace_write", "unseen.md", "mine\n")

    def test_a_journal_never_initialises_inside_another_workspaces_tree(
        self, workspaces_root: Path, orchestrator_proxy: FakeOrchestratorProxy
    ) -> None:
        """The same collision from the other side, and the half that destroys data.

        Workspace ``shared`` journals to ``shared.git``, which is *workspace*
        ``shared.git``'s tree. Guarding only the ``.git``-named workspace's own
        journal leaves that tree open: ``git init`` there scatters ``HEAD``,
        ``config``, ``objects/`` and ``refs/`` through another team's workspace,
        where its agents can list, read and overwrite them — and overwriting
        ``config`` or ``HEAD`` takes this journal with it.
        """
        victim = workspaces_root / "shared.git"
        victim.mkdir()
        (victim / "theirs.md").write_text("another team's file\n", encoding="utf-8")
        mine = workspaces_root / "shared"
        mine.mkdir()
        (mine / "unseen.md").write_text("already here\n", encoding="utf-8")

        card, _observer = card_for(orchestrator_proxy, "alice", workspace_id="shared", git_journal=True)

        # The other team's tree is exactly as they left it …
        assert [entry.name for entry in victim.iterdir()] == ["theirs.md"]
        # … and the gate is untouched: only the history is lost.
        assert mutate(card, "workspace_write", "fresh.md", "body\n") == "Written: fresh.md"
        with pytest.raises(RetriableError, match="read it before overwriting"):
            mutate(card, "workspace_write", "unseen.md", "mine\n")

    def test_a_real_repository_is_reused_rather_than_refused(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        # The guard above keys on "exists but has no HEAD". A repository this
        # actor already created has one, so a restart must still reuse it —
        # otherwise the guard would disable the journal on every second start.
        mutate(wired_card, "workspace_write", "fresh.md", "body\n")
        assert (git_dir_for(workspace_tree) / "HEAD").is_file()

        second = WorkspaceActor(
            config=WorkspaceConfig(
                name=workspace_actor_name(WORKSPACE_NAME),
                role=WORKSPACE_ACTOR_ROLE,
                workspace_name=WORKSPACE_NAME,
                git_journal=True,
            )
        )
        second.on_start()

        assert second._journal.enabled


# ---------------------------------------------------------------------------
# AC2 / AC2b: one accepted mutation is exactly one commit, authored by its agent
# ---------------------------------------------------------------------------


class TestOneMutationOneCommit:
    def test_a_write_produces_one_commit_authored_by_the_writer(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        mutate(wired_card, "workspace_write", "fresh.md", "body\n")

        mine = commits_by(workspace_tree, "alice")
        assert len(mine) == 1
        assert mine[0].author_email == alice_email(wired_card)
        assert mine[0].files == ["fresh.md"]
        assert mine[0].subject == "write: fresh.md"

    def test_the_committed_content_is_what_was_written(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        mutate(wired_card, "workspace_write", "fresh.md", "the exact bytes\n")
        sha = commits_by(workspace_tree, "alice")[0].sha
        assert git_show(workspace_tree, f"{sha}:fresh.md") == "the exact bytes\n"

    @pytest.mark.parametrize(
        ("capability", "args", "subject"),
        [
            ("workspace_edit", ("notes.md", "bravo", "BRAVO"), "edit: notes.md"),
            ("workspace_delete", ("notes.md",), "delete: notes.md"),
        ],
    )
    def test_every_mutation_kind_commits_once(
        self,
        capability: str,
        args: tuple[Any, ...],
        subject: str,
        wired_card: WorkspaceTool,
        workspace_tree: Path,
        notes: Path,
    ) -> None:
        read(wired_card, "notes.md")

        mutate(wired_card, capability, *args)

        mine = commits_by(workspace_tree, "alice")
        assert len(mine) == 1
        assert mine[0].subject == subject
        assert mine[0].files == ["notes.md"]

    def test_a_patch_commits_once(
        self, wired_card: WorkspaceTool, workspace_tree: Path, notes: Path
    ) -> None:
        read(wired_card, "notes.md")
        patch_text = "--- a/notes.md\n+++ b/notes.md\n@@ -1,2 +1,2 @@\n alpha\n-bravo\n+BRAVO\n"

        mutate(wired_card, "workspace_patch", patch_text)

        mine = commits_by(workspace_tree, "alice")
        assert len(mine) == 1
        assert mine[0].subject == "patch: notes.md"
        assert mine[0].files == ["notes.md"]

    def test_a_multi_edit_across_three_files_is_one_commit(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        for name, body in (("a.py", "x = 1\n"), ("b.py", "y = 2\n"), ("c.py", "z = 3\n")):
            (workspace_tree / name).write_text(body, encoding="utf-8")
            read(wired_card, name)

        mutate(
            wired_card,
            "workspace_multi_edit",
            [
                EditItem(path="a.py", old_string="x = 1", new_string="x = 10"),
                EditItem(path="b.py", old_string="y = 2", new_string="y = 20"),
                EditItem(path="c.py", old_string="z = 3", new_string="z = 30"),
            ],
        )

        # Three files, one commit — which is the whole point of staging the batch.
        mine = commits_by(workspace_tree, "alice")
        assert len(mine) == 1
        assert sorted(mine[0].files) == ["a.py", "b.py", "c.py"]
        assert mine[0].subject == "multi_edit: 3 files"

    def test_a_mkdir_commits_nothing_and_raises_nothing(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        # git does not track empty directories, so there is nothing to record —
        # and a journal that treated that as a failure would log a warning per
        # mkdir for the life of the team.
        before = [commit.sha for commit in journal_log(workspace_tree)]

        assert mutate(wired_card, "workspace_mkdir", "src/utils") == "Created: src/utils"

        assert [commit.sha for commit in journal_log(workspace_tree)] == before
        assert commits_by(workspace_tree, "alice") == []

    def test_a_commit_carries_only_its_own_write_set(
        self,
        wired_card: WorkspaceTool,
        workspace_tree: Path,
        notes: Path,
        monkeypatch: pytest.MonkeyPatch,
        workspace_actor: WorkspaceActor,
    ) -> None:
        """AC2b — staging is by explicit pathspec, never a bare ``git add -A``.

        A write landing between an agent's publication and its commit would
        otherwise be swept into that agent's commit and attributed to it. A gated
        mutation knows exactly which paths it wrote, which is precisely the
        difference from exec (29-5), which has to discover its write set.
        """
        read(wired_card, "notes.md")
        journal = workspace_actor._journal
        original = journal.commit_paths

        def intrude(paths: Any, identity: Any, capability: str) -> None:
            (workspace_tree / "intruder.md").write_text("from nowhere\n", encoding="utf-8")
            original(paths, identity, capability)

        monkeypatch.setattr(journal, "commit_paths", intrude)

        mutate(wired_card, "workspace_write", "notes.md", "alice's version\n")

        log = journal_log(workspace_tree)
        assert log[-1].files == ["notes.md"]
        assert all("intruder.md" not in commit.files for commit in log)


# ---------------------------------------------------------------------------
# AC3: a refused mutation is not in the journal
# ---------------------------------------------------------------------------


class TestARefusalIsNotRecorded:
    @pytest.fixture
    def head(
        self,
        wired_card: WorkspaceTool,
        notes: Path,
        workspace_actor: WorkspaceActor,
        workspace_tree: Path,
    ) -> str:
        """The tip once the fixture's own file has been absorbed as ``out-of-band``.

        Settling first is what makes "HEAD did not move" mean *the refusal added
        nothing*, rather than *the tree happened to be clean*.
        """
        workspace_actor._journal.commit_out_of_band()
        return journal_log(workspace_tree)[-1].sha

    def test_an_unread_whole_file_write(
        self, wired_card: WorkspaceTool, workspace_tree: Path, notes: Path, head: str
    ) -> None:
        with pytest.raises(RetriableError):
            mutate(wired_card, "workspace_write", "notes.md", "mine\n")
        assert journal_log(workspace_tree)[-1].sha == head

    def test_a_changed_file(
        self, wired_card: WorkspaceTool, workspace_tree: Path, notes: Path, head: str
    ) -> None:
        read(wired_card, "notes.md")
        notes.write_text("someone else\n", encoding="utf-8")
        with pytest.raises(RetriableError):
            mutate(wired_card, "workspace_write", "notes.md", "mine\n")
        # The out-of-band write is committed as such — but not as alice's, and
        # her refused write adds nothing of its own.
        log = journal_log(workspace_tree)
        assert log[-1].author_name == OUT_OF_BAND_AUTHOR
        assert commits_by(workspace_tree, "alice") == []

    def test_a_paginated_read_then_a_whole_file_write(
        self, wired_card: WorkspaceTool, workspace_tree: Path, notes: Path, head: str
    ) -> None:
        read(wired_card, "notes.md", limit=1)
        with pytest.raises(RetriableError):
            mutate(wired_card, "workspace_write", "notes.md", "mine\n")
        assert journal_log(workspace_tree)[-1].sha == head

    def test_a_vanished_file(
        self, wired_card: WorkspaceTool, workspace_tree: Path, notes: Path, head: str
    ) -> None:
        read(wired_card, "notes.md")
        notes.unlink()
        with pytest.raises(RetriableError, match="deleted since you read it"):
            mutate(wired_card, "workspace_write", "notes.md", "mine\n")
        # One out-of-band commit for the deletion nobody claimed, and nothing else.
        log = journal_log(workspace_tree)
        assert log[-1].author_name == OUT_OF_BAND_AUTHOR
        assert commits_by(workspace_tree, "alice") == []

    def test_an_exact_anchor_miss_on_a_changed_file(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        source = workspace_tree / "main.py"
        source.write_text("def  foo():\n    pass\n", encoding="utf-8")
        read(wired_card, "main.py")
        source.write_text("# banner\ndef  foo():\n    pass\n", encoding="utf-8")
        with pytest.raises(RetriableError):
            mutate(wired_card, "workspace_edit", "main.py", "def foo():\n    pass", "x")
        assert commits_by(workspace_tree, "alice") == []

    def test_a_refused_patch(
        self, wired_card: WorkspaceTool, workspace_tree: Path, notes: Path, head: str
    ) -> None:
        patch_text = "--- a/notes.md\n+++ b/notes.md\n@@ -1,2 +1,2 @@\n alpha\n-bravo\n+BRAVO\n"
        with pytest.raises(RetriableError):
            mutate(wired_card, "workspace_patch", patch_text)
        assert journal_log(workspace_tree)[-1].sha == head


# ---------------------------------------------------------------------------
# AC4: history is linear
# ---------------------------------------------------------------------------


class TestHistoryIsLinear:
    def test_eight_mutations_from_three_agents_stay_on_one_line(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        alice, _a = card_for(orchestrator_proxy, "alice", git_journal=True)
        bob, _b = card_for(orchestrator_proxy, "bob", git_journal=True)
        carol, _c = card_for(orchestrator_proxy, "carol", git_journal=True)

        mutate(alice, "workspace_write", "one.md", "1\n")
        mutate(bob, "workspace_write", "two.md", "2\n")
        mutate(carol, "workspace_mkdir", "sub")
        mutate(carol, "workspace_write", "sub/three.md", "3\n")
        read(alice, "one.md")
        mutate(alice, "workspace_edit", "one.md", "1", "one")
        read(bob, "two.md")
        mutate(bob, "workspace_delete", "two.md")
        read(carol, "sub/three.md")
        patch_text = "--- a/sub/three.md\n+++ b/sub/three.md\n@@ -1,1 +1,1 @@\n-3\n+three\n"
        mutate(carol, "workspace_patch", patch_text)
        mutate(alice, "workspace_write", "four.md", "4\n")
        mutate(bob, "workspace_write", "five.md", "5\n")

        log = journal_log(workspace_tree)
        assert all(len(commit.parents) <= 1 for commit in log)
        assert len([commit for commit in log if not commit.parents]) == 1
        assert journal_branches(workspace_tree) == ["master"]
        assert {"alice", "bob", "carol"} <= {commit.author_name for commit in log}


# ---------------------------------------------------------------------------
# AC5: a dirty tree commits first, as out-of-band
# ---------------------------------------------------------------------------


class TestOutOfBandCommits:
    def test_a_write_behind_the_actors_back_is_committed_first_and_separately(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        (workspace_tree / "uploaded.md").write_text("from the frontend\n", encoding="utf-8")

        mutate(wired_card, "workspace_write", "mine.md", "alice wrote this\n")

        log = journal_log(workspace_tree)
        out_of_band, agents = log[-2], log[-1]
        assert out_of_band.author_name == OUT_OF_BAND_AUTHOR
        assert out_of_band.files == ["uploaded.md"]
        assert agents.author_name == "alice"
        assert agents.files == ["mine.md"]

    def test_it_happens_even_when_the_two_touch_different_paths(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        (workspace_tree / "elsewhere.md").write_text("unrelated\n", encoding="utf-8")
        mutate(wired_card, "workspace_write", "mine.md", "alice\n")
        assert journal_log(workspace_tree)[-2].author_name == OUT_OF_BAND_AUTHOR

    def test_the_agents_commit_does_not_contain_the_out_of_band_change(
        self, wired_card: WorkspaceTool, workspace_tree: Path, notes: Path
    ) -> None:
        read(wired_card, "notes.md")
        (workspace_tree / "uploaded.md").write_text("from the frontend\n", encoding="utf-8")

        mutate(wired_card, "workspace_write", "notes.md", "alice's version\n")

        assert journal_log(workspace_tree)[-1].files == ["notes.md"]

    def test_a_clean_tree_produces_no_out_of_band_commit(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        mutate(wired_card, "workspace_write", "one.md", "1\n")
        before = len(journal_log(workspace_tree))
        mutate(wired_card, "workspace_write", "two.md", "2\n")
        assert len(journal_log(workspace_tree)) == before + 1


# ---------------------------------------------------------------------------
# AC15: a failure part-way through publication leaves nothing part-published
# ---------------------------------------------------------------------------


class TestAPartPublishedBatchIsNeverCommitted:
    def test_a_staging_failure_on_the_second_file_leaves_the_first_and_the_log_alone(
        self,
        wired_card: WorkspaceTool,
        workspace_actor: WorkspaceActor,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The filesystem half of atomicity, asserted end to end.

        ``multi_edit`` was already atomic against the gate; what it was not
        atomic against was an OS failure *during* publication. Staging every file
        before publishing any moves that failure to a point where nothing is
        visible — and therefore nothing is committed.
        """
        for name, body in (("a.py", "x = 1\n"), ("b.py", "y = 2\n")):
            (workspace_tree / name).write_text(body, encoding="utf-8")
            read(wired_card, name)
        workspace_actor._journal.commit_out_of_band()
        head = journal_log(workspace_tree)[-1].sha

        tree = workspace_actor._workspace
        real_stage = tree._stage
        calls: list[int] = []

        def fail_second(path: str, data: bytes) -> Any:
            calls.append(1)
            if len(calls) == 2:
                raise PermissionError("no room at the inn")
            return real_stage(path, data)

        monkeypatch.setattr(tree, "_stage", fail_second)

        # A bare PermissionError from the write point is the OS refusing the
        # write, not a path leaving the root, and since 29-5 the two say
        # different things (AC12). Which refusal it is does not matter here —
        # what matters is that it refuses, and that nothing landed.
        with pytest.raises(RetriableError, match="was not published"):
            mutate(
                wired_card,
                "workspace_multi_edit",
                [
                    EditItem(path="a.py", old_string="x = 1", new_string="x = 10"),
                    EditItem(path="b.py", old_string="y = 2", new_string="y = 20"),
                ],
            )

        monkeypatch.undo()
        assert (workspace_tree / "a.py").read_text(encoding="utf-8") == "x = 1\n"
        assert journal_log(workspace_tree)[-1].sha == head


# ---------------------------------------------------------------------------
# AC6: .gitignore is seeded once and is effective
# ---------------------------------------------------------------------------


class _StubDocumentReader(DocumentReader):
    """Extraction without markitdown, so the sidecar assertion needs no optional dep."""

    llm_client: None = None

    def extract_text(self, content: bytes, path: str) -> str:
        return "# extracted\n" + "body text " * 20


class TestTheSeededIgnoreFile:
    def test_it_is_written_at_init(self, wired_card: WorkspaceTool, workspace_tree: Path) -> None:
        assert (workspace_tree / GITIGNORE_NAME).exists()

    def test_an_existing_one_is_never_overwritten(
        self, workspaces_root: Path, orchestrator_proxy: FakeOrchestratorProxy
    ) -> None:
        tree = workspaces_root / "preexisting"
        tree.mkdir()
        mine = tree / GITIGNORE_NAME
        mine.write_text("# mine, thanks\n", encoding="utf-8")

        card_for(orchestrator_proxy, "alice", workspace_id="preexisting", git_journal=True)

        assert mine.read_text(encoding="utf-8") == "# mine, thanks\n"

    def test_a_document_read_leaves_the_tree_clean(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        # Since 45-4 a document read writes nothing at all — its extraction lives
        # in ``#Workspace``'s state — so the tree is clean for a stronger reason
        # than "the sidecar is ignored". The ignore file still earns its place on
        # the image-view path below.
        (workspace_tree / "report.pdf").write_bytes(b"%PDF-1.4 fake")
        observer = FakeActorToolObserver(orchestrator_proxy, name="alice")
        card = WorkspaceTool(
            workspace_id=WORKSPACE_NAME,
            git_journal=True,
            workspace_read=WorkspaceRead(document_reader=_StubDocumentReader()),
        )
        card.observer(observer)
        mutate(card, "workspace_write", "seed.md", "x\n")  # absorb the pre-existing pdf
        before_files = sorted(p.name for p in workspace_tree.iterdir())

        read(card, "report.pdf")

        assert sorted(p.name for p in workspace_tree.iterdir()) == before_files
        assert not (workspace_tree / ".report.pdf.md").exists()
        assert working_tree_is_clean(workspace_tree)
        before = len(journal_log(workspace_tree))
        mutate(card, "workspace_write", "after.md", "y\n")
        log = journal_log(workspace_tree)
        assert len(log) == before + 1  # no out-of-band commit provoked by the read
        assert all(".report.pdf.md" not in commit.files for commit in log)

    def test_an_image_view_sidecar_leaves_the_tree_clean(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        # The ignore file's remaining purpose: ``workspace_view`` still writes a
        # resized sidecar beside its source, so a view dirties the tree and every
        # agent's commit would be preceded by an out-of-band commit of
        # regenerable noise. That sidecar is out of scope for 45-4.
        pytest.importorskip("PIL")
        from PIL import Image

        Image.new("RGB", (64, 64), "red").save(workspace_tree / "photo.png")
        observer = FakeActorToolObserver(orchestrator_proxy, name="alice")
        card = WorkspaceTool(
            workspace_id=WORKSPACE_NAME,
            git_journal=True,
            workspace_view=WorkspaceView(max_dimension=16),
        )
        card.observer(observer)
        mutate(card, "workspace_write", "seed.md", "x\n")  # absorb the pre-existing image

        tool_named(card, "workspace_view")("photo.png")

        assert (workspace_tree / ".photo.png.16.png").exists()
        assert working_tree_is_clean(workspace_tree)
        before = len(journal_log(workspace_tree))
        mutate(card, "workspace_write", "after.md", "y\n")
        log = journal_log(workspace_tree)
        assert len(log) == before + 1  # no out-of-band commit for the sidecar
        assert all(".photo.png.16.png" not in commit.files for commit in log)

    def test_a_live_staging_file_leaves_the_tree_clean(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        # An interrupted write leaves one of these behind, and a second team's
        # in-flight write is one right now. Neither belongs in the history.
        in_flight = workspace_tree / staging_name("notes.md")
        in_flight.write_bytes(b"half a file")

        assert working_tree_is_clean(workspace_tree)
        mutate(wired_card, "workspace_write", "fresh.md", "body\n")
        assert all(in_flight.name not in commit.files for commit in journal_log(workspace_tree))

    def test_exec_debris_is_committed_by_nothing(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        cache = workspace_tree / "src" / "__pycache__"
        cache.mkdir(parents=True)
        (cache / "module.cpython-312.pyc").write_bytes(b"\x00compiled")

        assert working_tree_is_clean(workspace_tree)
        mutate(wired_card, "workspace_write", "fresh.md", "body\n")
        assert all("__pycache__" not in " ".join(c.files) for c in journal_log(workspace_tree))


# ---------------------------------------------------------------------------
# AC7: .git is unreachable through every read capability
# ---------------------------------------------------------------------------


class TestTheRepositoryIsUnreachable:
    @pytest.fixture
    def marker(self, wired_card: WorkspaceTool, workspace_tree: Path) -> str:
        """A distinctive string that exists only inside the repository."""
        mutate(wired_card, "workspace_write", "fresh.md", "body\n")
        assert git_dir_for(workspace_tree).is_dir()
        return git_dir_for(workspace_tree).name

    @pytest.mark.parametrize("depth", [1, 0, 2, 5])
    def test_list_cannot_see_it_at_any_depth(
        self, depth: int, wired_card: WorkspaceTool, marker: str
    ) -> None:
        assert marker not in mutate(wired_card, "workspace_list", "", depth)

    @pytest.mark.parametrize("pattern", ["**/*", "**/HEAD", "*", "**/config"])
    def test_glob_cannot_match_it(
        self, pattern: str, wired_card: WorkspaceTool, marker: str
    ) -> None:
        assert "HEAD" not in mutate(wired_card, "workspace_glob", pattern)
        assert marker not in mutate(wired_card, "workspace_glob", pattern)

    def test_grep_cannot_search_it(self, wired_card: WorkspaceTool, marker: str) -> None:
        assert mutate(wired_card, "workspace_grep", "ref: refs/heads") == "No matches found."

    @pytest.mark.parametrize("suffix", ["config", "HEAD"])
    def test_read_cannot_reach_it_by_traversal(
        self, suffix: str, wired_card: WorkspaceTool, marker: str
    ) -> None:
        with pytest.raises(RetriableError, match="escapes workspace root"):
            read(wired_card, f"../{marker}/{suffix}")


# ---------------------------------------------------------------------------
# AC10: the author is the agent's name, and identity reaches git as data
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_a_registered_name_becomes_the_author(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        mutate(wired_card, "workspace_write", "fresh.md", "body\n")
        commit = journal_log(workspace_tree)[-1]
        assert commit.author_name == "alice"
        assert commit.author_email == alice_email(wired_card)

    def test_an_unregistered_agent_falls_back_to_its_id(
        self, workspace_actor: WorkspaceActor, workspace_tree: Path
    ) -> None:
        outcome_of(workspace_actor, "apply_write", "never-registered", "fresh.md", "body\n")
        commit = journal_log(workspace_tree)[-1]
        assert commit.author_name == "never-registered"
        assert commit.author_email == f"never-registered@{IDENTITY_DOMAIN}"

    def test_a_hostile_name_still_commits_sanitised(
        self, workspace_actor: WorkspaceActor, workspace_tree: Path
    ) -> None:
        # Angle brackets would open the email field and a newline would end the
        # identity line — either lets an agent id say something other than a name.
        workspace_actor.register_agent("hostile-id", "Ali<ce>\nGIT_AUTHOR_NAME=root\x07")

        outcome_of(workspace_actor, "apply_write", "hostile-id", "fresh.md", "body\n")

        commit = journal_log(workspace_tree)[-1]
        assert "<" not in commit.author_name
        assert ">" not in commit.author_name
        assert "\n" not in commit.author_name
        assert "\x07" not in commit.author_name
        assert commit.author_email == f"hostile-id@{IDENTITY_DOMAIN}"

    def test_an_identity_that_sanitises_to_nothing_is_never_empty(self) -> None:
        identity = Identity("\x00\x01", "<<<")
        assert identity.name == IDENTITY_FALLBACK
        assert identity.email == f"{IDENTITY_FALLBACK}@{IDENTITY_DOMAIN}"

    def test_an_ambient_git_author_name_does_not_reach_the_commit(
        self,
        wired_card: WorkspaceTool,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A developer running the suite inside a repository, or CI, may export
        # these. Explicit flags beat GIT_DIR; the identity variables have no flag
        # equivalent, so the child environment is scrubbed instead.
        monkeypatch.setenv("GIT_AUTHOR_NAME", "the ambient environment")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "ambient@example.com")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "the ambient environment")

        mutate(wired_card, "workspace_write", "fresh.md", "body\n")

        commit = journal_log(workspace_tree)[-1]
        assert commit.author_name == "alice"
        assert commit.author_email == alice_email(wired_card)

    def test_a_global_git_config_cannot_change_what_is_recorded(
        self,
        wired_card: WorkspaceTool,
        workspace_tree: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``~/.gitconfig`` is the file every machine has, and two of its settings bite.

        ``core.excludesFile`` drops the agent's own file out of the agent's own
        commit — silently, and with a warning per mutation for the rest of the
        team's life — and ``core.autocrlf`` rewrites the bytes the commit holds.
        Scrubbing ``GIT_*`` does not reach either: git finds the file through
        ``HOME``. Both configuration files are switched off in the child instead.

        ``excludesFile`` is the half asserted here, because it is the half this
        harness can see: ``subprocess`` reads git's output with universal
        newlines, so an ``autocrlf`` rewrite is invisible through any helper that
        shells out. It is carried in the config anyway — one setting reaching the
        child and the other not is not a shape this fix can produce.
        """
        home = tmp_path / "hostile-home"
        home.mkdir()
        (home / "ignore").write_text("*.md\n", encoding="utf-8")
        (home / ".gitconfig").write_text(
            f"[core]\n\texcludesFile = {home / 'ignore'}\n\tautocrlf = true\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

        mutate(wired_card, "workspace_write", "fresh.md", "the exact bytes\n")

        commit = journal_log(workspace_tree)[-1]
        assert commit.author_name == "alice"
        assert commit.files == ["fresh.md"]  # excludesFile did not reach it
        assert git_show(workspace_tree, f"{commit.sha}:fresh.md") == "the exact bytes\n"


# ---------------------------------------------------------------------------
# AC9: no journal failure can fail a mutation
# ---------------------------------------------------------------------------


def _git_command(cmd: Any) -> bool:
    """Whether an argv is one of ours, so a stand-in leaves ``rg`` and friends alone."""
    return bool(cmd) and str(cmd[0]).endswith("git")


class TestNoJournalFailureFailsAMutation:
    def test_a_non_zero_exit_leaves_the_mutation_accepted_and_the_journal_on(
        self,
        wired_card: WorkspaceTool,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        real_run = subprocess.run
        calls: list[int] = []

        def failing(cmd: Any, *args: Any, **kwargs: Any) -> Any:
            if _git_command(cmd) and "add" in cmd:
                calls.append(1)
                return subprocess.CompletedProcess(cmd, 128, "", "fatal: something went wrong")
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr(subprocess, "run", failing)
        with caplog.at_level(logging.WARNING):
            assert mutate(wired_card, "workspace_write", "fresh.md", "body\n") == (
                "Written: fresh.md"
            )
            assert mutate(wired_card, "workspace_write", "second.md", "body\n") == (
                "Written: second.md"
            )

        assert (workspace_tree / "fresh.md").read_text(encoding="utf-8") == "body\n"
        assert len(calls) >= 2  # still trying: a bad exit does not disable the journal
        assert any("exited 128" in record.getMessage() for record in caplog.records)

    def test_a_timeout_disables_the_journal_for_the_life_of_the_actor(
        self,
        wired_card: WorkspaceTool,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A hung git must not cost every later mutation the same wall clock.

        Every mutation in the team runs on one thread. One timeout is a lost
        commit; a timeout per mutation is a wedged team. Driven by patching the
        invocation, never by running something slow (NFR4).
        """
        real_run = subprocess.run
        calls: list[Any] = []

        def hang(cmd: Any, *args: Any, **kwargs: Any) -> Any:
            if _git_command(cmd):
                calls.append(cmd)
                raise subprocess.TimeoutExpired(cmd, 1)
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr(subprocess, "run", hang)
        with caplog.at_level(logging.WARNING):
            assert mutate(wired_card, "workspace_write", "fresh.md", "body\n") == (
                "Written: fresh.md"
            )
            after_first = len(calls)
            assert mutate(wired_card, "workspace_write", "second.md", "body\n") == (
                "Written: second.md"
            )

        assert len(calls) == after_first  # no further forks: the journal is off
        assert (workspace_tree / "second.md").read_text(encoding="utf-8") == "body\n"

    def test_git_that_cannot_be_spawned_disables_the_journal(
        self, wired_card: WorkspaceTool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_run = subprocess.run

        def refuse(cmd: Any, *args: Any, **kwargs: Any) -> Any:
            if _git_command(cmd):
                raise OSError("no fork for you")
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr(subprocess, "run", refuse)
        assert mutate(wired_card, "workspace_write", "fresh.md", "body\n") == "Written: fresh.md"

    def test_an_edit_that_changes_nothing_is_a_no_op_not_a_warning(
        self,
        wired_card: WorkspaceTool,
        workspace_tree: Path,
        notes: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # git exits non-zero with nothing staged. Treating that as a failure
        # would log a warning every time an agent replaced text with itself.
        read(wired_card, "notes.md")
        with caplog.at_level(logging.WARNING, logger="akgentic.tool.workspace.journal"):
            result = mutate(wired_card, "workspace_edit", "notes.md", "bravo", "bravo")

        assert result == "(no change) notes.md"
        assert commits_by(workspace_tree, "alice") == []
        assert [record for record in caplog.records if record.name.endswith("journal")] == []

    def test_an_init_that_fails_degrades_rather_than_raising(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # `init -b` needs git >= 2.28. An older git makes the init fail, which is
        # already a degradation path rather than a crash.
        real_run = subprocess.run

        def refuse_init(cmd: Any, *args: Any, **kwargs: Any) -> Any:
            if _git_command(cmd) and "init" in cmd:
                return subprocess.CompletedProcess(cmd, 129, "", "error: unknown switch `b'")
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr(subprocess, "run", refuse_init)
        card, _observer = card_for(orchestrator_proxy, "alice", git_journal=True)

        assert mutate(card, "workspace_write", "fresh.md", "body\n") == "Written: fresh.md"

    def test_a_failing_status_leaves_the_mutation_accepted(
        self, wired_card: WorkspaceTool, workspace_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_run = subprocess.run

        def break_status(cmd: Any, *args: Any, **kwargs: Any) -> Any:
            if _git_command(cmd) and "status" in cmd:
                return subprocess.CompletedProcess(cmd, 128, "", "fatal: not a git repository")
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr(subprocess, "run", break_status)

        assert mutate(wired_card, "workspace_write", "fresh.md", "body\n") == "Written: fresh.md"
        assert (workspace_tree / "fresh.md").read_text(encoding="utf-8") == "body\n"

    def test_a_failing_config_at_init_degrades(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A repository that came out bare fails every later `add` at runtime, on
        # a user's machine. Better to notice it here and record nothing.
        real_run = subprocess.run

        def refuse_config(cmd: Any, *args: Any, **kwargs: Any) -> Any:
            if _git_command(cmd) and "config" in cmd:
                return subprocess.CompletedProcess(cmd, 4, "", "error: could not lock config file")
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr(subprocess, "run", refuse_config)
        card, _observer = card_for(orchestrator_proxy, "alice", git_journal=True)

        assert mutate(card, "workspace_write", "fresh.md", "body\n") == "Written: fresh.md"

    def test_every_invocation_carries_a_timeout(
        self, workspace_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An unbounded fork on the actor's single thread is the failure the
        # timeout-disables-the-journal rule exists to make survivable; it only
        # works if no invocation is missing one.
        real_run = subprocess.run
        timeouts: list[Any] = []

        def record(cmd: Any, *args: Any, **kwargs: Any) -> Any:
            if _git_command(cmd):
                timeouts.append(kwargs.get("timeout"))
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr(subprocess, "run", record)
        journal = GitJournal(workspace_tree, enabled=True, timeout_s=7.5)
        assert journal.initialise()
        journal.commit_paths(["nothing.md"], Identity.out_of_band(), "write")
        journal.is_dirty()

        assert timeouts
        assert all(value == 7.5 for value in timeouts)


# ---------------------------------------------------------------------------
# AC17: git_journal is a card field, not a tool
# ---------------------------------------------------------------------------


class TestTheCardField:
    def test_it_defaults_to_off(self) -> None:
        assert WorkspaceTool().git_journal is False

    def test_it_appears_in_no_tool_signature(self, wired_card: WorkspaceTool) -> None:
        import inspect

        for tool in wired_card.get_tools():
            assert "git_journal" not in inspect.signature(tool).parameters
        assert "git_journal" not in [tool.__name__ for tool in wired_card.get_tools()]

    def test_the_card_round_trips_with_the_field_intact(self) -> None:
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME, git_journal=False)
        restored = WorkspaceTool.model_validate(card.model_dump())
        assert restored.git_journal is False
        assert restored == card

    def test_the_card_survives_a_json_round_trip(self) -> None:
        # What Golden Rule #1b is actually protecting, asserted as behaviour: a
        # non-serializable type leaking into a field breaks here, whatever the
        # model config says. The journal itself is a plain actor attribute and
        # reaches no card field at all.
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME, git_journal=False)
        restored = WorkspaceTool.model_validate_json(card.model_dump_json())
        assert restored.git_journal is False
        assert restored == card

    def test_the_actor_config_round_trips_with_the_field(self) -> None:
        config = WorkspaceConfig(
            name=workspace_actor_name(WORKSPACE_NAME),
            role=WORKSPACE_ACTOR_ROLE,
            workspace_name=WORKSPACE_NAME,
            git_journal=False,
        )
        assert WorkspaceConfig.model_validate(config.model_dump()).git_journal is False

    def test_it_reaches_the_actor_config(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        observer = FakeActorToolObserver(orchestrator_proxy, name="alice")
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME, git_journal=False)
        card.observer(observer)
        _actor_class, config = orchestrator_proxy.create_calls[-1]
        assert isinstance(config, WorkspaceConfig)
        assert config.git_journal is False


# ---------------------------------------------------------------------------
# AC8: git is optional; the gate is not
# ---------------------------------------------------------------------------


@pytest.fixture(params=["card-disabled", "git-absent"])
def journal_off(
    request: pytest.FixtureRequest,
    orchestrator_proxy: FakeOrchestratorProxy,
    workspace_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> WorkspaceTool:
    """A card whose journal is off, one way in each parametrisation.

    The absence path patches the **resolver** rather than the session's ``PATH``:
    mutating ``PATH`` for a test would reach ``rg`` and every other binary the
    suite shells out to.
    """
    if request.param == "git-absent":
        real_which = shutil.which

        def without_git(cmd: str, *args: Any, **kwargs: Any) -> str | None:
            return None if cmd == "git" else real_which(cmd, *args, **kwargs)

        monkeypatch.setattr(shutil, "which", without_git)
    observer = FakeActorToolObserver(orchestrator_proxy, name="alice")
    card = WorkspaceTool(
        workspace_id=WORKSPACE_NAME,
        git_journal=request.param != "card-disabled",
    )
    card.observer(observer)
    return card


@pytest.fixture
def off_notes(workspace_tree: Path) -> Path:
    path = workspace_tree / "notes.md"
    path.write_text(BODY, encoding="utf-8")
    return path


class TestTheGateSurvivesWithoutGit:
    """A representative slice of 29-3's own assertions, run with the journal off.

    A real port, not a weaker set: the accept strings, the refusal wordings and
    the disk state are asserted exactly as the gate's own suite asserts them.
    """

    def test_no_repository_is_created_and_no_ignore_file_is_seeded(
        self, journal_off: WorkspaceTool, workspace_tree: Path
    ) -> None:
        mutate(journal_off, "workspace_write", "fresh.md", "body\n")
        assert not git_dir_for(workspace_tree).exists()
        assert not (workspace_tree / GITIGNORE_NAME).exists()

    def test_no_subprocess_is_ever_spawned(
        self, journal_off: WorkspaceTool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def forbidden(cmd: Any, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError(f"the journal is off and yet it forked: {cmd}")

        monkeypatch.setattr(subprocess, "run", forbidden)
        assert mutate(journal_off, "workspace_write", "fresh.md", "body\n") == "Written: fresh.md"

    @pytest.mark.parametrize("mode", ["git-absent", "card-disabled"])
    def test_exactly_one_warning_is_logged_not_one_per_mutation(
        self,
        mode: str,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Both ways of turning the journal off, because AC8 asks for one warning
        # from each and they leave `initialise` down different branches.
        if mode == "git-absent":
            real_which = shutil.which
            monkeypatch.setattr(
                shutil,
                "which",
                lambda cmd, *a, **k: None if cmd == "git" else real_which(cmd, *a, **k),
            )
        with caplog.at_level(logging.WARNING, logger="akgentic.tool.workspace.journal"):
            observer = FakeActorToolObserver(orchestrator_proxy, name="alice")
            card = WorkspaceTool(
                workspace_id=WORKSPACE_NAME,
                git_journal=mode != "card-disabled",
            )
            card.observer(observer)
            for index in range(5):
                mutate(card, "workspace_write", f"file{index}.md", "body\n")

        journal_warnings = [
            record for record in caplog.records if record.name == "akgentic.tool.workspace.journal"
        ]
        assert len(journal_warnings) == 1

    def test_the_whole_file_table_is_unchanged(
        self, journal_off: WorkspaceTool, off_notes: Path
    ) -> None:
        with pytest.raises(RetriableError, match="read it before overwriting"):
            mutate(journal_off, "workspace_write", "notes.md", "mine\n")
        assert off_notes.read_text(encoding="utf-8") == BODY

        read(journal_off, "notes.md")
        assert mutate(journal_off, "workspace_write", "notes.md", "mine\n") == "Written: notes.md"
        assert off_notes.read_text(encoding="utf-8") == "mine\n"

    def test_a_paginated_read_still_refuses_a_whole_file_write(
        self, journal_off: WorkspaceTool, off_notes: Path
    ) -> None:
        read(journal_off, "notes.md", limit=1)
        with pytest.raises(RetriableError, match="you read only part of it"):
            mutate(journal_off, "workspace_write", "notes.md", "mine\n")
        assert off_notes.read_text(encoding="utf-8") == BODY

    def test_the_anchored_table_is_unchanged(
        self, journal_off: WorkspaceTool, workspace_tree: Path
    ) -> None:
        source = workspace_tree / "main.py"
        source.write_text("def  foo():\n    pass\n", encoding="utf-8")
        read(journal_off, "main.py")
        source.write_text("# banner\ndef  foo():\n    pass\n", encoding="utf-8")

        with pytest.raises(RetriableError) as refusal:
            mutate(
                journal_off,
                "workspace_edit",
                "main.py",
                "def foo():\n    pass",
                "def bar():\n    pass",
            )
        assert "no longer matches it exactly" in str(refusal.value)

        result = mutate(journal_off, "workspace_edit", "main.py", "def  foo():", "def  bar():")
        assert not result.startswith("[ERROR]")
        assert "def  bar():" in source.read_text(encoding="utf-8")

    def test_a_write_behind_the_actors_back_is_still_caught(
        self, journal_off: WorkspaceTool, off_notes: Path
    ) -> None:
        # The property the whole epic rests on. The gate hashes the live file, so
        # losing git loses out-of-band *detection* in the history and nothing at
        # all in the gate.
        read(journal_off, "notes.md")
        off_notes.write_text("someone else's version\n", encoding="utf-8")

        with pytest.raises(RetriableError, match="changed since you read it"):
            mutate(journal_off, "workspace_write", "notes.md", "my version\n")
        assert off_notes.read_text(encoding="utf-8") == "someone else's version\n"

    def test_multi_edit_is_still_all_or_nothing(
        self, journal_off: WorkspaceTool, workspace_tree: Path
    ) -> None:
        for name, body in (("a.py", "x = 1\n"), ("b.py", "y = 2\n")):
            (workspace_tree / name).write_text(body, encoding="utf-8")
            read(journal_off, name)

        result = mutate(
            journal_off,
            "workspace_multi_edit",
            [
                EditItem(path="a.py", old_string="x = 1", new_string="x = 10"),
                EditItem(path="b.py", old_string="NOT THERE", new_string="whatever"),
            ],
        )

        assert result == "[ERROR] old_string not found in b.py"
        assert (workspace_tree / "a.py").read_text(encoding="utf-8") == "x = 1\n"

    def test_a_refusal_still_names_the_other_writer_by_name(
        self, journal_off: WorkspaceTool, orchestrator_proxy: FakeOrchestratorProxy, off_notes: Path
    ) -> None:
        bob, _observer = card_for(orchestrator_proxy, "bob", git_journal=True)
        read(journal_off, "notes.md")
        read(bob, "notes.md")
        mutate(bob, "workspace_write", "notes.md", "bob's version\n")

        with pytest.raises(RetriableError) as refusal:
            mutate(journal_off, "workspace_write", "notes.md", "alice's version\n")
        assert "last written by agent 'bob'" in str(refusal.value)


def test_a_stale_actor_state_never_leaks_a_repository_outside_the_tmp_tree(
    workspaces_root: Path, wired_card: WorkspaceTool
) -> None:
    """Guard against the pleasing hazard: a journal initialised somewhere real.

    Every repository this suite creates must live under the per-test
    ``AKGENTIC_WORKSPACES_ROOT``. A test that leaked the environment variable
    would otherwise have the actor run ``git init`` in the developer's own
    directory — which would look like nothing at all until it did.
    """
    assert git_dir_for(workspaces_root / WORKSPACE_NAME).parent == workspaces_root
    assert list(workspaces_root.glob("*.git"))
    for candidate in workspaces_root.glob("*.git"):
        assert candidate.is_relative_to(workspaces_root)


def test_the_journal_is_a_no_op_before_it_is_initialised(workspace_tree: Path) -> None:
    # Every method is guarded so the actor never has to ask whether git is there.
    journal = GitJournal(workspace_tree, enabled=False, timeout_s=1.0)
    assert not journal.enabled
    assert not journal.is_dirty()
    journal.commit_out_of_band()
    journal.commit_paths(["a.md"], Identity.out_of_band(), "write")
    journal.seed_gitignore(lambda path, data: pytest.fail("seeded with the journal off"))
    assert not git_dir_for(workspace_tree).exists()


def test_the_derived_directory_is_a_sibling_named_after_the_tree() -> None:
    assert git_dir_for(Path("/tmp/workspaces/team-1")) == Path("/tmp/workspaces/team-1.git")


def test_a_uuid_shaped_agent_id_survives_the_email_sanitiser() -> None:
    agent_id = str(uuid4())
    assert Identity("alice", agent_id).email == f"{agent_id}@{IDENTITY_DOMAIN}"
