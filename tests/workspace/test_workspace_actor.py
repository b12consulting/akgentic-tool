"""Behavioural tests for the ``#Workspace`` actor.

The observation map left this actor with the gate in story 52-5 — one card, one
agent, one map — so what it asserts now is the actor's name, the staging
predicate and the startup sweep of orphaned staging files. The map's own
behaviour, LRU included, is pinned card-side in ``test_observation_recording.py``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from akgentic.core.agent_state import BaseState

from akgentic.tool.workspace.actor import (
    WORKSPACE_ACTOR_NAME,
    WORKSPACE_ACTOR_ROLE,
    WorkspaceActor,
    workspace_actor_name,
)
from akgentic.tool.workspace.models import (
    STAGING_SWEEP_GRACE_S,
    WorkspaceConfig,
)
from akgentic.tool.workspace.workspace import Filesystem, is_staging_name
from tests.workspace.conftest import WORKSPACE_PATH


def start_actor(workspace_path: str = WORKSPACE_PATH) -> WorkspaceActor:
    """Build and start an actor over *workspace_path*, without an actor thread.

    Takes the **resolved** three-segment path — what a card hands the actor — so
    the actor opens the tree the ``workspace_tree`` fixture created.
    """
    actor = WorkspaceActor(
        config=WorkspaceConfig(
            name=workspace_actor_name(workspace_path),
            role=WORKSPACE_ACTOR_ROLE,
            workspace_path=workspace_path,
        )
    )
    actor.on_start()
    return actor


# ---------------------------------------------------------------------------
# AC2: the actor's name
# ---------------------------------------------------------------------------


class TestActorName:
    def test_name_starts_with_the_tool_actor_prefix(self) -> None:
        # The '#' prefix is what the orchestrator's two-phase stop classifies on.
        assert workspace_actor_name("anything").startswith("#")

    def test_name_carries_the_workspace_so_two_workspaces_get_two_actors(self) -> None:
        assert workspace_actor_name("shared") != workspace_actor_name("team-1")

    def test_base_name_is_the_prefix_of_every_derived_name(self) -> None:
        assert workspace_actor_name("team-1").startswith(WORKSPACE_ACTOR_NAME)


# ---------------------------------------------------------------------------
# AC9: the startup sweep
# ---------------------------------------------------------------------------


def staging_name(target: str) -> str:
    """A staging name of the exact shape ``Filesystem.write`` publishes from."""
    return f".{target}.{uuid4().hex}.tmp"


class TestStagingPredicate:
    @pytest.mark.parametrize("target", ["a.md", "notes", "deeply.named.file.py"])
    def test_recognises_what_write_produces(self, target: str) -> None:
        assert is_staging_name(staging_name(target))

    @pytest.mark.parametrize(
        "name",
        [
            ".notes.tmp",  # an agent's own dotfile
            "notes.tmp",  # an agent's own plain temp file
            ".report.pdf.md",  # a read-path sidecar
            "report.md",
            ".a.0123456789abcdef.tmp",  # 16 hex digits, not 32
            ".a.0123456789ABCDEF0123456789ABCDEF.tmp",  # not lowercase hex
            f".{uuid4().hex}.tmp",  # no target-name segment
        ],
    )
    def test_leaves_everything_else_alone(self, name: str) -> None:
        assert not is_staging_name(name)

    def test_the_writer_and_the_predicate_agree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Guard the guard: the predicate is asserted against a name a real write
        # actually staged, so a staging name that stopped going through the shared
        # template — and therefore stopped being swept — goes red here.
        backend = Filesystem(str(tmp_path), "ws")
        staged: list[str] = []
        real_replace = os.replace

        def spy(src: Any, dst: Any) -> None:
            staged.append(Path(src).name)
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", spy)
        backend.write("a.md", b"hello")

        assert staged and all(is_staging_name(name) for name in staged)
        assert (tmp_path / "ws" / "a.md").read_bytes() == b"hello"


def abandon(path: Path) -> None:
    """Age *path* past the sweep's grace window, as an interrupted write would.

    The window exists so a second team's in-flight staging file is not unlinked
    mid-publish; an orphan is by definition older than that. Backdating the mtime
    is how a test gets an orphan without waiting for one.
    """
    stale = time.time() - STAGING_SWEEP_GRACE_S - 60
    os.utime(path, (stale, stale))


class TestStartupSweep:
    def test_removes_orphaned_staging_files_at_any_depth(self, workspace_tree: Path) -> None:
        root_orphan = workspace_tree / staging_name("a.md")
        nested = workspace_tree / "sub"
        nested.mkdir()
        nested_orphan = nested / staging_name("b.md")
        root_orphan.write_bytes(b"partial")
        nested_orphan.write_bytes(b"partial")
        abandon(root_orphan)
        abandon(nested_orphan)

        start_actor()

        assert not root_orphan.exists()
        assert not nested_orphan.exists()

    def test_a_staging_file_being_published_right_now_survives(self, workspace_tree: Path) -> None:
        """The other half of the sweep race, and the reason the window exists.

        ``WorkspaceTool(workspace_id="shared")`` is a supported configuration and
        a tool actor's unicity domain is the team, so two teams over one tree
        means two actors, each sweeping the whole tree at start. Team B's sweep
        must not unlink team A's staged file in the window between ``os.open``
        and ``os.replace`` — nothing is corrupted if it does, but A's healthy
        write turns into a refusal it did nothing to deserve.
        """
        in_flight = workspace_tree / staging_name("a.md")
        in_flight.write_bytes(b"being published right now")

        start_actor()

        assert in_flight.exists()

    def test_leaves_every_other_name_untouched(self, workspace_tree: Path) -> None:
        survivors = [
            workspace_tree / ".notes.tmp",
            workspace_tree / "notes.tmp",
            workspace_tree / ".report.pdf.md",
            workspace_tree / "report.md",
        ]
        for path in survivors:
            path.write_text("keep me", encoding="utf-8")

        start_actor()

        assert all(path.exists() for path in survivors)

    def test_a_directory_shaped_like_a_staging_file_is_left_alone(
        self, workspace_tree: Path
    ) -> None:
        # The sweep matches on a name, so the only thing keeping it off a
        # directory that happens to carry that name — and off everything the
        # user put inside it — is the is_file() conjunct. Assert it, because
        # the name check short-circuits ahead of it in every other test.
        masquerading = workspace_tree / staging_name("a.md")
        masquerading.mkdir()
        (masquerading / "kept.md").write_text("keep me", encoding="utf-8")
        abandon(masquerading)  # old enough that only is_file() can save it

        start_actor()

        assert masquerading.is_dir()
        assert (masquerading / "kept.md").exists()

    def test_an_unremovable_staging_file_does_not_stop_the_actor(
        self, workspace_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        orphan = workspace_tree / staging_name("a.md")
        orphan.write_bytes(b"partial")
        abandon(orphan)  # so the unlink is actually attempted, and actually fails

        def refuse(self: Path, missing_ok: bool = False) -> None:
            raise PermissionError("read-only directory")

        monkeypatch.setattr(Path, "unlink", refuse)
        actor = start_actor()

        assert orphan.exists()
        assert actor.config.workspace_path == WORKSPACE_PATH  # the actor started regardless

    def test_a_tree_with_nothing_to_sweep_starts_cleanly(self, workspace_tree: Path) -> None:
        (workspace_tree / "report.md").write_text("hello", encoding="utf-8")
        actor = start_actor()
        assert (workspace_tree / "report.md").read_text(encoding="utf-8") == "hello"
        assert actor.state.model_dump() == BaseState().model_dump()
