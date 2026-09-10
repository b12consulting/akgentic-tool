"""The metadata directory is a sibling of the tree, and nothing in the tree can name it.

Guards for :func:`meta_dir_for` (ADR-051 Decision 9). The placement is the whole
subject: ``Filesystem._validate_path`` refuses every path that does not resolve
*inside* the root, so a directory **beside** the tree is one no read capability
can name and no sandboxed run can delete. Inside the tree it would be listable,
globbable, greppable, readable — and removable by an ``rm -rf`` from the very run
whose exec lock it holds.

Two habits run through every spec here:

- the resolver **creates nothing**, so a guard about reaching the directory has
  to create it first. A path that cannot be reached because nothing is there
  would prove nothing at all;
- neither the workspaces root nor the metadata root is ever spelled twice. What
  a test compares against is what :func:`get_workspace` itself resolved, because
  two derivations that drift give two metadata directories over one tree — and a
  test carrying the second one would agree with the defect.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from akgentic.tool.errors import RetriableError
from akgentic.tool.workspace.journal import git_dir_for
from akgentic.tool.workspace.models import GIT_DIR_SUFFIX, META_DIR_SUFFIX
from akgentic.tool.workspace.workspace import (
    PathEscapeError,
    get_workspace,
    leaf_segment,
    meta_dir_for,
    user_segment,
)
from tests.workspace.conftest import (
    WORKSPACE_NAME,
    WORKSPACE_PATH,
    FakeOrchestratorProxy,
    card_for,
    tool_named,
    workspace_root_for,
)

META_ROOT_VAR = "AKGENTIC_WORKSPACE_META_ROOT"
"""The variable that relocates the metadata *parent*, and only that."""

WORKSPACES_ROOT_VAR = "AKGENTIC_WORKSPACES_ROOT"
"""The variable it falls back to — the one the trees themselves hang off."""

LOCK_NAME = "exec.lock"
"""A stand-in for what 52-2 will put in the directory. Any name would do."""

META_LEAF = f"{WORKSPACE_NAME}{META_DIR_SUFFIX}"
"""The metadata directory's own leaf, beside ``WORKSPACE_NAME`` in one scope."""


def _tree_root(workspace_path: str = WORKSPACE_PATH) -> Path:
    """The tree :func:`get_workspace` opens, read from the backend it returns.

    ``Filesystem`` exposes no public root, and every assertion in this module is
    about where the metadata directory sits **relative to that exact path**. A
    second derivation spelled out in the test would be a second rule to keep in
    step — which is the defect one resolver exists to remove.
    """
    return get_workspace(workspace_path)._root


def _seeded_meta(workspace_path: str = WORKSPACE_PATH) -> Path:
    """Create the directory the resolver names and put a file in it.

    Always the test's own doing: production creates nothing here until 52-2.
    """
    meta = meta_dir_for(workspace_path)
    meta.mkdir(parents=True, exist_ok=True)
    (meta / LOCK_NAME).write_text("held", encoding="utf-8")
    return meta


def _escape_forms(meta: Path, root: Path) -> list[str]:
    """Every shape an agent could write to name *meta* from inside *root*.

    The traversal form is spelled literally because that is what an agent types;
    the ``relpath`` form is the same thing computed, which keeps the list correct
    when ``AKGENTIC_WORKSPACE_META_ROOT`` has moved the directory somewhere a
    single ``..`` no longer reaches. The absolute form is the one an agent
    reaches for when the relative ones are refused.
    """
    return [
        f"../{META_LEAF}",
        f"../{META_LEAF}/{LOCK_NAME}",
        os.path.relpath(meta, root),
        os.path.relpath(meta / LOCK_NAME, root),
        str(meta),
        str(meta / LOCK_NAME),
    ]


def _hands_back_nothing(closure: Callable[..., object], *args: str) -> bool:
    """Whether a read closure gives the agent nothing naming the metadata directory.

    A refusal counts, and so does an empty result: from the agent's side they are
    one answer — the directory is not reachable — and which of the two a given
    path produces depends only on whether it escapes the root or merely matches
    nothing inside it.
    """
    try:
        answer = str(closure(*args))
    except RetriableError:
        return True
    return META_DIR_SUFFIX not in answer and LOCK_NAME not in answer


##
## AC 2 and AC 3 — one resolver, it is a sibling, and it writes nothing
##


class TestTheResolverPlacesItBesideTheTree:
    """``meta_dir_for`` derives ``<root>.akgentic``, and derives it only."""

    def test_it_is_a_sibling_of_the_tree(self, workspaces_root: Path) -> None:
        """Same parent, the tree's name plus the suffix — and outside the tree.

        The third assertion is made **directly** rather than inferred from the
        first two: those hold for a path that is spelled correctly and still
        resolves inside the root through a symlink, and it is the containment
        property, not the spelling, that keeps the exec lock out of the sandbox.
        """
        root = _tree_root()
        meta = meta_dir_for(WORKSPACE_PATH)

        assert meta.parent == root.parent
        assert meta.name == f"{root.name}{META_DIR_SUFFIX}"
        assert not meta.is_relative_to(root)

    def test_it_is_absolute(self, workspaces_root: Path) -> None:
        """A relative answer would resolve against whatever the caller's cwd is."""
        assert meta_dir_for(WORKSPACE_PATH).is_absolute()

    def test_the_same_path_resolves_the_same_directory(self, workspaces_root: Path) -> None:
        """One derivation. Two that drifted would give one tree two metadata directories."""
        assert meta_dir_for(WORKSPACE_PATH) == meta_dir_for(WORKSPACE_PATH)

    def test_it_creates_nothing(self, workspaces_root: Path) -> None:
        """No ``mkdir``, no touch, no side effect — creation belongs to a later story."""
        meta = meta_dir_for(WORKSPACE_PATH)

        assert not meta.exists()
        assert not list(workspaces_root.rglob(f"*{META_DIR_SUFFIX}"))

    def test_it_still_creates_nothing_once_the_tree_exists(self, workspace_tree: Path) -> None:
        """The tree being there is what makes an accidental ``exist_ok=True`` invisible."""
        meta = meta_dir_for(WORKSPACE_PATH)

        assert not meta.exists()

    def test_two_workspaces_get_two_directories(self, workspaces_root: Path) -> None:
        """The suffix hangs off the leaf, so neither can name the other's."""
        first = meta_dir_for(WORKSPACE_PATH)
        second = meta_dir_for(f"u-bob/{WORKSPACE_NAME}")

        assert first != second
        assert not first.is_relative_to(second)
        assert not second.is_relative_to(first)


##
## AC 4 — no read capability can reach it
##


class TestNoReadCapabilityCanReachIt:
    """The backend refuses every path that names it, and no listing mentions it."""

    def test_the_backend_refuses_every_path_that_names_it(self, workspaces_root: Path) -> None:
        """``read``, ``list`` and ``exists`` all validate first, so all three refuse.

        ``PathEscapeError`` rather than a bare ``PermissionError``: the agent has
        to be told the path escaped rather than that the file was not writable,
        or it rewrites a correct path for ever.
        """
        workspace = get_workspace(WORKSPACE_PATH)
        meta = _seeded_meta()
        root = _tree_root()

        for path in _escape_forms(meta, root):
            for call in (workspace.read, workspace.list, workspace.exists):
                with pytest.raises(PathEscapeError):
                    call(path)

    def test_listing_the_tree_never_mentions_it(self, workspaces_root: Path) -> None:
        """It is a sibling, so it is not a child — asserted as the whole listing."""
        workspace = get_workspace(WORKSPACE_PATH)
        _seeded_meta()
        workspace.write("notes.md", b"hello")

        assert [entry.name for entry in workspace.list("")] == ["notes.md"]

    def test_no_read_closure_of_a_wired_card_names_it(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """What an agent actually calls: the card's own ``list`` and ``glob``.

        The backend's refusal above is the mechanism; this is the surface. A
        recursive glob is the pattern an agent writes when it is looking around,
        and the two targeted ones are what it writes once it knows the name.
        """
        card, _observer = card_for(orchestrator_proxy, "alice")
        _seeded_meta()
        (workspace_tree / "notes.md").write_text("hello", encoding="utf-8")

        listing = str(tool_named(card, "workspace_list")(""))
        assert "notes.md" in listing
        assert META_DIR_SUFFIX not in listing

        recursive = str(tool_named(card, "workspace_glob")("**/*"))
        assert "notes.md" in recursive
        assert META_DIR_SUFFIX not in recursive

        glob = tool_named(card, "workspace_glob")
        assert _hands_back_nothing(glob, f"*{META_DIR_SUFFIX}/*")
        assert _hands_back_nothing(glob, f"{META_LEAF}/**/*")
        assert _hands_back_nothing(glob, "*", f"../{META_LEAF}")
        assert _hands_back_nothing(tool_named(card, "workspace_list"), f"../{META_LEAF}")

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "workspace_glob validates its `path` argument but not its `pattern`, so a "
            "pattern carrying `..` enumerates outside the tree. Pre-existing and not "
            "this story's to fix — `workspace/card/` is out of scope for 52-1 — but the "
            "guard is written now so the day it is fixed this spec goes green and "
            "strict=True forces its removal."
        ),
    )
    def test_a_traversal_pattern_reaches_it_through_glob(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The one read closure that names something outside the root, today.

        ``workspace_glob("../<leaf>.akgentic/*")`` hands the agent back
        ``../<leaf>.akgentic/exec.lock``: the closure resolves its ``path``
        argument against the root and refuses an escape, but feeds ``pattern``
        straight to ``Path.glob``, and ``Path.relative_to`` is lexical, so a
        match above the root renders as a ``../`` path rather than raising.

        **Names only, never content.** ``workspace_read``, ``workspace_view``
        and ``workspace_grep`` all go through ``_validate_path`` and refuse the
        same string, so nothing inside the directory can be opened this way.

        It is **not** a hole this story opens: the journal's ``<leaf>.git`` has
        been enumerable the same way since it shipped, and the metadata
        directory only inherits it. The remedy belongs to a story that can touch
        ``workspace/card/read.py`` and cover the journal, the tree and the
        ``path``/``pattern`` interaction together.
        """
        card, _observer = card_for(orchestrator_proxy, "alice")
        _seeded_meta()

        assert _hands_back_nothing(tool_named(card, "workspace_glob"), f"../{META_LEAF}/*")


##
## AC 5 and AC 6 — the parent, relocated and defaulted
##


class TestTheMetaRootRelocatesTheParent:
    """``AKGENTIC_WORKSPACE_META_ROOT`` moves the parent and carries the scope."""

    def test_it_moves_the_parent_and_keeps_the_scope(
        self, workspaces_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``<meta-root>/<scope>/<leaf>.akgentic`` — the scope survives the move.

        It is why the resolver takes the two-segment path rather than a resolved
        root: the scope cannot be recovered from an absolute path without also
        knowing which workspaces root it came from.
        """
        elsewhere = tmp_path / "meta-volume"
        monkeypatch.setenv(META_ROOT_VAR, str(elsewhere))

        meta = meta_dir_for(WORKSPACE_PATH)

        assert meta == workspace_root_for(elsewhere, META_LEAF).resolve()

    def test_the_tree_does_not_move_with_it(
        self, workspaces_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The variable is about metadata. A tree that followed it would be a data loss."""
        elsewhere = tmp_path / "meta-volume"
        monkeypatch.setenv(META_ROOT_VAR, str(elsewhere))

        root = _tree_root()

        assert root == workspace_root_for(workspaces_root, WORKSPACE_NAME).resolve()
        assert not root.is_relative_to(elsewhere.resolve())

    def test_unreachability_survives_the_relocation(
        self, workspaces_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Further away is not weaker: every path naming it is still refused."""
        monkeypatch.setenv(META_ROOT_VAR, str(tmp_path / "meta-volume"))
        workspace = get_workspace(WORKSPACE_PATH)
        meta = _seeded_meta()
        root = _tree_root()

        assert not meta.is_relative_to(root)
        for path in _escape_forms(meta, root):
            with pytest.raises(PathEscapeError):
                workspace.read(path)

    def test_unset_it_falls_back_to_the_workspaces_root(
        self, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With only ``AKGENTIC_WORKSPACES_ROOT`` set, ``<meta>`` lands beside the tree."""
        monkeypatch.delenv(META_ROOT_VAR, raising=False)

        meta = meta_dir_for(WORKSPACE_PATH)

        assert meta == workspace_root_for(workspaces_root, META_LEAF).resolve()
        assert meta.parent == _tree_root().parent

    def test_with_both_unset_the_parent_is_the_one_get_workspace_uses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Compared against ``get_workspace``'s own root, never a second literal.

        A ``"./workspaces"`` spelled here would agree with a resolver that had
        drifted from :func:`get_workspace`, which is precisely the failure the
        shared ``_workspaces_root`` exists to prevent.

        ``chdir`` is not decoration: with both variables unset the default is
        relative to the working directory, and ``get_workspace`` **creates** the
        tree it resolves — without this the spec would write into the developer's
        own checkout.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv(META_ROOT_VAR, raising=False)
        monkeypatch.delenv(WORKSPACES_ROOT_VAR, raising=False)

        root = _tree_root()
        meta = meta_dir_for(WORKSPACE_PATH)

        assert meta.parent == root.parent
        assert meta.name == f"{root.name}{META_DIR_SUFFIX}"
        assert not meta.is_relative_to(root)

    def test_the_journal_is_not_moved_by_the_variable(
        self, workspaces_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The journal has its own placement rule, and this variable is not part of it.

        Both directories are siblings of the tree, which is exactly why somebody
        reading the new variable would expect it to move them together.
        """
        elsewhere = tmp_path / "meta-volume"
        monkeypatch.setenv(META_ROOT_VAR, str(elsewhere))
        root = _tree_root()

        git_dir = git_dir_for(root)

        assert git_dir.parent == root.parent
        assert git_dir.name == f"{root.name}{GIT_DIR_SUFFIX}"
        assert not git_dir.is_relative_to(elsewhere.resolve())


##
## AC 8 — a leaf may not end in the new suffix
##


class TestALeafMayNotEndInTheMetadataSuffix:
    """``workspace_id="notes.akgentic"`` would root a tree on another workspace's metadata."""

    def test_a_leaf_ending_in_the_suffix_is_refused(self) -> None:
        """This story is what creates the collision, so this story closes it.

        Once ``<leaf>.akgentic`` is a real sibling directory, a second card
        declaring it as a ``workspace_id`` roots its **tree** there and reads,
        writes and deletes another workspace's exec lock, document cache and
        index as ordinary in-tree activity. Nothing raises: ``_validate_path``
        refuses only what resolves outside the root, and that root is a real
        directory.
        """
        with pytest.raises(ValueError, match="metadata directory"):
            leaf_segment(f"notes{META_DIR_SUFFIX}")

    @pytest.mark.parametrize("spelling", ["notes.AKGENTIC", "notes.Akgentic", "notes.aKgEnTiC"])
    def test_the_match_is_case_insensitive(self, spelling: str) -> None:
        """macOS and Windows are case-insensitive by default, so these are one directory."""
        with pytest.raises(ValueError, match="metadata directory"):
            leaf_segment(spelling)

    def test_the_message_names_the_collision_and_the_value(self) -> None:
        """The admin who caused it has to see which name to change, and to what end."""
        with pytest.raises(ValueError) as excinfo:
            leaf_segment(f"notes{META_DIR_SUFFIX}")

        assert META_DIR_SUFFIX in str(excinfo.value)
        assert f"notes{META_DIR_SUFFIX}" in str(excinfo.value)

    @pytest.mark.parametrize(
        "accepted",
        [
            "akgentic",  # the suffix's letters, in the wrong place
            "akgentic-notes",
            "notes.akgentic.md",  # the suffix mid-name, which names no directory
            "notes.akgent",  # a shorter ending that merely leads the same way
            "notes.akgenticx",
        ],
    )
    def test_only_the_suffix_is_refused_never_the_substring(self, accepted: str) -> None:
        """Over-refusing costs a real workspace its name, which is not a safer failure."""
        assert leaf_segment(accepted) == accepted

    def test_the_rule_is_spelled_once_against_the_resolvers_own_constant(self) -> None:
        """``meta_dir_for`` builds the directory from the constant; this guard refuses it.

        A second ``".akgentic"`` literal in the guard is a rule that has to agree
        with another one, and the two drift silently.
        """
        assert leaf_segment(f"notes{META_DIR_SUFFIX}x") == f"notes{META_DIR_SUFFIX}x"
        with pytest.raises(ValueError):
            leaf_segment(f"x{META_DIR_SUFFIX}")

    def test_a_principal_may_still_end_in_the_suffix(self) -> None:
        """``user_segment`` is unchanged: the collision is between siblings in one scope.

        There is no metadata directory beside a *scope* — the suffix hangs off
        the leaf — so refusing a principal here would turn a containment guard
        into a rejected login at team creation.
        """
        assert user_segment(f"alice{META_DIR_SUFFIX}") == f"alice{META_DIR_SUFFIX}"

    def test_the_two_suffixes_are_refused_with_different_reasons(self) -> None:
        """One shared guard, two collisions — and the admin is told which one they hit.

        The suffixes are checked by one loop over the two derivations' own
        constants. A loop that collapsed the messages would leave somebody
        renaming a workspace without knowing what it collided with.
        """
        with pytest.raises(ValueError, match="journal directory"):
            leaf_segment(f"notes{GIT_DIR_SUFFIX}")
        with pytest.raises(ValueError, match="metadata directory"):
            leaf_segment(f"notes{META_DIR_SUFFIX}")
