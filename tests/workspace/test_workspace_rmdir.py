"""``workspace_rmdir`` — the delete strategy, generalised to a directory.

It runs on the card's thread, directly on the filesystem, inside ``_gated``. It
never reaches the sandbox. A recursive removal behaves like a ``workspace_delete``
of every file under the directory, all-or-nothing, locking only the paths this
agent holds a view of.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest
from pydantic_ai import Tool

from akgentic.tool.errors import RetriableError
from akgentic.tool.workspace import card as card_package
from akgentic.tool.workspace.card import params as card_params
from akgentic.tool.workspace.lock import LockTicket, mutation_busy
from akgentic.tool.workspace.locks import LOCKS_DIR_NAME
from akgentic.tool.workspace.tool import WorkspaceTool
from akgentic.tool.workspace.workspace import meta_dir_for
from akgentic.tool.workspace.write.params import WorkspaceRmdir
from tests.workspace.conftest import (
    WORKSPACE_NAME,
    WORKSPACE_PATH,
    FakeActorToolObserver,
    FakeOrchestratorProxy,
    card_for,
    journal_log,
    mutate,
    read,
    requires_git,
    tool_named,
)


def rmdir(card: WorkspaceTool, path: str, **kwargs: bool) -> str:
    """Call ``workspace_rmdir`` exactly as the agent would."""
    return mutate(card, "workspace_rmdir", path, **kwargs)


def populate(tree: Path, files: dict[str, str]) -> None:
    """Write *files* (relative path → body) under *tree*."""
    for relative, body in files.items():
        target = tree / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")


def lock_files() -> set[str]:
    """Every path-lock file under the suite tree's metadata directory."""
    locks = meta_dir_for(WORKSPACE_PATH) / LOCKS_DIR_NAME
    return {entry.name for entry in locks.iterdir()} if locks.exists() else set()


class _RecordingCache:
    """Stands in for the card's ``DocumentCache`` — records every stale-mark."""

    def __init__(self) -> None:
        self.marked: list[list[str]] = []

    def mark_paths_stale(self, paths: list[str]) -> None:
        self.marked.append(list(paths))


def hold_the_tree(card: WorkspaceTool) -> None:
    """Take the exec hold on *card*'s tree, as a run from another agent would."""
    backend = card._lock_backend
    assert backend is not None
    grant = backend.acquire(
        card._workspace_path,
        LockTicket(agent_id="agent-9", agent_name="builder", cmd="make", budget_s=60.0),
    )
    assert grant.run_id


# ---------------------------------------------------------------------------
# A1 — capability wiring
# ---------------------------------------------------------------------------


class TestCapabilityWiring:
    def test_it_is_exposed_by_default(self, wired_card: WorkspaceTool) -> None:
        assert "workspace_rmdir" in {tool.__name__ for tool in wired_card.get_tools()}

    def test_read_only_withholds_it(self, observer: FakeActorToolObserver) -> None:
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME, read_only=True)
        card.observer(observer)
        assert "workspace_rmdir" not in {tool.__name__ for tool in card.get_tools()}

    def test_the_field_set_to_false_withholds_it(self, observer: FakeActorToolObserver) -> None:
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME, workspace_rmdir=False)
        card.observer(observer)
        assert "workspace_rmdir" not in {tool.__name__ for tool in card.get_tools()}

    def test_a_persisted_card_without_the_key_loads_and_exposes_it(
        self, observer: FakeActorToolObserver, workspace_tree: Path
    ) -> None:
        stored = WorkspaceTool(workspace_id=WORKSPACE_NAME).model_dump()
        del stored["workspace_rmdir"]
        card = WorkspaceTool.model_validate(stored)
        assert card.workspace_rmdir is True
        card.observer(observer)
        assert "workspace_rmdir" in {tool.__name__ for tool in card.get_tools()}

    def test_it_is_re_exported_where_stored_records_resolve_it(self) -> None:
        assert card_params.WorkspaceRmdir is WorkspaceRmdir
        assert card_package.WorkspaceRmdir is WorkspaceRmdir
        assert "WorkspaceRmdir" in card_params.__all__
        assert "WorkspaceRmdir" in card_package.__all__


# ---------------------------------------------------------------------------
# A2, A3 — empty, and non-empty only on request
# ---------------------------------------------------------------------------


class TestRemoval:
    def test_an_empty_directory_is_removed(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        (workspace_tree / "a" / "empty").mkdir(parents=True)
        assert rmdir(wired_card, "a/empty") == "Removed: a/empty"
        assert not (workspace_tree / "a" / "empty").exists()
        assert (workspace_tree / "a").is_dir()

    def test_a_non_empty_directory_is_refused_without_recursive(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {"docs/a.md": "a\n", "docs/b.md": "b\n", "docs/sub/c.md": "c\n"})
        with pytest.raises(RetriableError) as refusal:
            rmdir(wired_card, "docs")
        assert str(refusal.value) == (
            "Directory not empty: docs (3 entries) — pass recursive=true only if the user "
            "explicitly asked to remove it and everything under it"
        )
        assert (workspace_tree / "docs" / "sub" / "c.md").exists()

    def test_recursive_removes_the_whole_subtree(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {"docs/a.md": "a\n", "docs/b.md": "b\n", "docs/sub/c.md": "c\n"})
        assert rmdir(wired_card, "docs", recursive=True) == "Removed: docs (3 files)"
        assert not (workspace_tree / "docs").exists()


# ---------------------------------------------------------------------------
# A4 — refusals, each deleting nothing
# ---------------------------------------------------------------------------


class TestRefusals:
    @pytest.mark.parametrize("root", ["", ".", "./", "a/.."])
    def test_the_workspace_root_is_refused_however_it_is_spelled(
        self, wired_card: WorkspaceTool, workspace_tree: Path, root: str
    ) -> None:
        populate(workspace_tree, {"a/keep.md": "k\n"})
        with pytest.raises(RetriableError, match="workspace root"):
            rmdir(wired_card, root, recursive=True)
        assert (workspace_tree / "a" / "keep.md").exists()

    def test_a_file_is_refused(self, wired_card: WorkspaceTool, workspace_tree: Path) -> None:
        populate(workspace_tree, {"notes.md": "n\n"})
        with pytest.raises(RetriableError) as refusal:
            rmdir(wired_card, "notes.md", recursive=True)
        assert str(refusal.value) == "Not a directory: notes.md — use workspace_delete"
        assert (workspace_tree / "notes.md").exists()

    def test_a_missing_path_is_refused(self, wired_card: WorkspaceTool) -> None:
        with pytest.raises(RetriableError) as refusal:
            rmdir(wired_card, "nowhere")
        assert str(refusal.value) == "Directory not found: nowhere"

    def test_an_escaping_path_is_refused(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        outside = workspace_tree.parent / "neighbour"
        outside.mkdir()
        with pytest.raises(RetriableError, match="escapes"):
            rmdir(wired_card, "../neighbour", recursive=True)
        assert outside.is_dir()

    def test_a_symlink_to_a_directory_is_refused_and_its_target_survives(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {"real/keep.md": "k\n"})
        os.symlink(workspace_tree / "real", workspace_tree / "link")
        with pytest.raises(RetriableError) as refusal:
            rmdir(wired_card, "link", recursive=True, force=True)
        assert str(refusal.value) == "link is a symlink — use workspace_delete"
        assert (workspace_tree / "real" / "keep.md").exists()
        assert (workspace_tree / "link").is_symlink()


# ---------------------------------------------------------------------------
# A5, A6 — the view gate over the subtree, and the locks it takes
# ---------------------------------------------------------------------------


class TestTheViewGate:
    def test_one_stale_view_refuses_the_whole_call_and_deletes_nothing(
        self,
        wired_card: WorkspaceTool,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
    ) -> None:
        populate(workspace_tree, {"docs/a.md": "a\n", "docs/b.md": "b\n"})
        read(wired_card, "docs/a.md")
        read(wired_card, "docs/b.md")
        bob, _bob_observer = card_for(orchestrator_proxy, "bob")
        read(bob, "docs/b.md")
        mutate(bob, "workspace_write", "docs/b.md", "bob's version\n")

        with pytest.raises(RetriableError) as refusal:
            rmdir(wired_card, "docs", recursive=True)

        message = str(refusal.value)
        assert message.startswith("Refused to modify docs/b.md: it changed since you read it.")
        assert message.endswith(
            "— re-read the files, or pass force=true only if the user explicitly asked to "
            "discard those changes"
        )
        assert (workspace_tree / "docs" / "a.md").exists()
        assert (workspace_tree / "docs" / "b.md").exists()
        assert (workspace_tree / "docs").is_dir()

    def test_a_vanished_file_under_the_directory_refuses(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {"docs/a.md": "a\n", "docs/b.md": "b\n"})
        read(wired_card, "docs/a.md")
        (workspace_tree / "docs" / "a.md").unlink()
        with pytest.raises(RetriableError, match="it was deleted since you read it"):
            rmdir(wired_card, "docs", recursive=True)
        assert (workspace_tree / "docs" / "b.md").exists()

    def test_a_paginated_read_under_the_directory_refuses(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {"docs/a.md": "1\n2\n3\n"})
        read(wired_card, "docs/a.md", limit=1)
        with pytest.raises(RetriableError, match="you read only part of it"):
            rmdir(wired_card, "docs", recursive=True)
        assert (workspace_tree / "docs" / "a.md").exists()

    def test_a_sibling_sharing_the_name_prefix_is_not_under_the_directory(
        self,
        wired_card: WorkspaceTool,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
    ) -> None:
        # ``build2/x`` is not under ``build``. A bare ``startswith`` would count
        # it, and its stale view would then refuse a removal it has nothing to
        # do with.
        populate(workspace_tree, {"build/out.o": "o\n", "build2/x": "x\n"})
        read(wired_card, "build2/x")
        bob, _bob_observer = card_for(orchestrator_proxy, "bob")
        read(bob, "build2/x")
        mutate(bob, "workspace_write", "build2/x", "changed\n")

        assert rmdir(wired_card, "build", recursive=True) == "Removed: build (1 files)"
        assert (workspace_tree / "build2" / "x").exists()

    def test_locks_cover_the_observed_paths_only(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {f"big/f{index:03}.txt": f"{index}\n" for index in range(200)})
        read(wired_card, "big/f007.txt")
        before = lock_files()

        assert rmdir(wired_card, "big", recursive=True) == "Removed: big (200 files)"

        assert len(lock_files() - before) == 1


# ---------------------------------------------------------------------------
# A7 — journal, index, observations
# ---------------------------------------------------------------------------


class TestJournalIndexAndObservations:
    @requires_git
    def test_one_commit_as_this_agent_naming_the_directory(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {"out/a.txt": "a\n", "out/deep/b.txt": "b\n"})
        alice, _observer = card_for(orchestrator_proxy, "alice", git_journal=True)
        mutate(alice, "workspace_write", "keep.md", "k\n")  # commits the seeded tree first
        before = journal_log(workspace_tree)

        rmdir(alice, "out", recursive=True)

        after = journal_log(workspace_tree)
        assert len(after) == len(before) + 1
        head = after[-1]
        assert head.subject == "rmdir: out"
        assert head.author_name == "alice"
        assert sorted(head.files) == ["out/a.txt", "out/deep/b.txt"]

    def test_every_removed_file_is_marked_stale_and_never_the_directory(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {"out/a.txt": "a\n", "out/deep/b.txt": "b\n"})
        cache = _RecordingCache()
        wired_card._document_cache = cache  # type: ignore[assignment]

        rmdir(wired_card, "out", recursive=True)

        assert cache.marked == [["out/a.txt", "out/deep/b.txt"]]

    def test_observations_under_the_directory_are_dropped(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {"out/a.txt": "a\n"})
        read(wired_card, "out/a.txt")

        rmdir(wired_card, "out", recursive=True)

        assert wired_card.observation_for("out/a.txt") is None
        assert mutate(wired_card, "workspace_write", "out/a.txt", "new\n") == "Written: out/a.txt"


# ---------------------------------------------------------------------------
# A8, A9 — the busy tree, and links inside the tree
# ---------------------------------------------------------------------------


class TestBusyTreeAndLinks:
    def test_a_held_tree_refuses_with_the_busy_text(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {"out/a.txt": "a\n"})
        hold_the_tree(wired_card)
        held = wired_card._lock_backend.holder(  # type: ignore[union-attr]
            wired_card._workspace_path, wired_card._exec_budget_s
        )
        assert held is not None

        with pytest.raises(RetriableError) as refusal:
            rmdir(wired_card, "out", recursive=True)

        assert str(refusal.value) == mutation_busy(held.run_id, "builder")
        assert (workspace_tree / "out" / "a.txt").exists()

    def test_links_inside_the_tree_are_unlinked_and_their_targets_survive(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        outside = workspace_tree.parent / "outside.txt"
        outside.write_text("precious\n", encoding="utf-8")
        populate(workspace_tree, {"sibling/keep.md": "k\n", "doomed/own.txt": "o\n"})
        os.symlink(outside, workspace_tree / "doomed" / "to-file")
        os.symlink(workspace_tree / "sibling", workspace_tree / "doomed" / "to-dir")

        assert rmdir(wired_card, "doomed", recursive=True) == "Removed: doomed (3 files)"

        assert not (workspace_tree / "doomed").exists()
        assert outside.read_text(encoding="utf-8") == "precious\n"
        assert (workspace_tree / "sibling" / "keep.md").exists()


# ---------------------------------------------------------------------------
# A10 — allow_recursive is a deployment setting
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_only_card(observer: FakeActorToolObserver, workspace_tree: Path) -> WorkspaceTool:
    """A card whose deployment allows empty directories only."""
    card = WorkspaceTool(
        workspace_id=WORKSPACE_NAME, workspace_rmdir=WorkspaceRmdir(allow_recursive=False)
    )
    card.observer(observer)
    return card


class TestAllowRecursive:
    def test_it_defaults_to_true_and_a_stored_param_without_the_key_loads_true(self) -> None:
        assert WorkspaceRmdir().allow_recursive is True
        stored = WorkspaceRmdir().model_dump()
        del stored["allow_recursive"]
        assert WorkspaceRmdir.model_validate(stored).allow_recursive is True

    def test_off_the_schema_has_neither_recursive_nor_force(
        self, empty_only_card: WorkspaceTool
    ) -> None:
        tool = tool_named(empty_only_card, "workspace_rmdir")
        assert tool.__name__ == "workspace_rmdir"
        assert list(inspect.signature(tool).parameters) == ["path"]
        properties = Tool(tool).function_schema.json_schema["properties"]
        assert set(properties) == {"path"}

    def test_on_the_schema_has_both(self, wired_card: WorkspaceTool) -> None:
        tool = tool_named(wired_card, "workspace_rmdir")
        properties = Tool(tool).function_schema.json_schema["properties"]
        assert set(properties) == {"path", "recursive", "force"}

    def test_off_a_non_empty_directory_is_refused_as_empty_only(
        self, empty_only_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {"docs/a.md": "a\n"})
        with pytest.raises(RetriableError) as refusal:
            rmdir(empty_only_card, "docs")
        assert str(refusal.value) == (
            "Directory not empty: docs (1 entries) — this workspace removes empty directories only"
        )
        assert (workspace_tree / "docs" / "a.md").exists()

    def test_off_an_empty_directory_is_still_removed(
        self, empty_only_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        (workspace_tree / "empty").mkdir()
        assert rmdir(empty_only_card, "empty") == "Removed: empty"


# ---------------------------------------------------------------------------
# A11 — force skips the view gate, and nothing else
# ---------------------------------------------------------------------------


class TestForce:
    def test_force_without_recursive_is_refused_before_any_filesystem_check(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {"docs/a.md": "a\n"})
        for target in ("docs", "nowhere", "../escape"):
            with pytest.raises(RetriableError) as refusal:
                rmdir(wired_card, target, force=True)
            assert str(refusal.value) == "force requires recursive=true"
        assert (workspace_tree / "docs" / "a.md").exists()

    def test_force_removes_a_tree_a_stale_view_would_have_protected(
        self,
        wired_card: WorkspaceTool,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
    ) -> None:
        populate(workspace_tree, {"docs/a.md": "a\n", "docs/b.md": "b\n"})
        read(wired_card, "docs/a.md")
        read(wired_card, "docs/b.md")
        bob, _bob_observer = card_for(orchestrator_proxy, "bob")
        read(bob, "docs/b.md")
        mutate(bob, "workspace_write", "docs/b.md", "bob's version\n")

        result = rmdir(wired_card, "docs", recursive=True, force=True)

        assert result == "Removed: docs (2 files, forced)"
        assert not (workspace_tree / "docs").exists()
        assert wired_card.observation_for("docs/a.md") is None

    def test_a_forced_call_takes_no_path_lock(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {"docs/a.md": "a\n"})
        read(wired_card, "docs/a.md")
        before = lock_files()
        rmdir(wired_card, "docs", recursive=True, force=True)
        assert lock_files() == before

    def test_a_forced_call_is_still_refused_on_a_held_tree(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {"docs/a.md": "a\n"})
        hold_the_tree(wired_card)
        with pytest.raises(RetriableError, match="builder"):
            rmdir(wired_card, "docs", recursive=True, force=True)
        assert (workspace_tree / "docs" / "a.md").exists()

    def test_a_forced_call_still_refuses_the_root(
        self, wired_card: WorkspaceTool, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {"a/keep.md": "k\n"})
        with pytest.raises(RetriableError, match="workspace root"):
            rmdir(wired_card, ".", recursive=True, force=True)
        assert (workspace_tree / "a" / "keep.md").exists()

    @requires_git
    def test_a_forced_call_is_journaled_and_marks_stale(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        populate(workspace_tree, {"docs/a.md": "a\n"})
        alice, _observer = card_for(orchestrator_proxy, "alice", git_journal=True)
        cache = _RecordingCache()
        alice._document_cache = cache  # type: ignore[assignment]

        rmdir(alice, "docs", recursive=True, force=True)

        head = journal_log(workspace_tree)[-1]
        assert head.subject == "rmdir: docs"
        assert head.author_name == "alice"
        assert cache.marked == [["docs/a.md"]]


# ---------------------------------------------------------------------------
# A12 — the docstring tells the model to ask
# ---------------------------------------------------------------------------


class TestTheDocstringTellsTheModelToAsk:
    # Compared with whitespace collapsed: Python 3.13+ dedents docstrings at compile time.
    RECURSIVE_RULE = (
        "Set recursive=true ONLY when the user has explicitly asked to remove the"
        " directory together with its contents."
    )
    FORCE_RULE = (
        "Set force=true ONLY when the user"
        " has explicitly asked to discard changes made since your last read."
        " Never set it to get past a refusal on your own initiative."
    )

    def test_allowed_the_docstring_carries_both_rules(self, wired_card: WorkspaceTool) -> None:
        doc = " ".join(tool_named(wired_card, "workspace_rmdir").__doc__.split())
        assert self.RECURSIVE_RULE in doc
        assert self.FORCE_RULE in doc

    def test_empty_only_the_docstring_carries_neither(self, empty_only_card: WorkspaceTool) -> None:
        doc = tool_named(empty_only_card, "workspace_rmdir").__doc__
        assert "recursive" not in doc
        assert "force" not in doc
