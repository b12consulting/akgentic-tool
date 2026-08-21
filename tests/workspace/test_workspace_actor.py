"""Behavioural tests for the ``#Workspace`` actor (story 29-2).

The actor is wired and observing in this story, and gates nothing — so what is
asserted here is the observation map, its LRU bound, the fact that recording is
not persisted state, and the startup sweep of orphaned staging files.
"""

from __future__ import annotations

import os
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
from akgentic.tool.workspace.models import Observation, WorkspaceConfig, WorkspaceState, content_sha
from akgentic.tool.workspace.workspace import Filesystem, is_staging_name

from tests.workspace.conftest import WORKSPACE_NAME

ALICE = "alice-id"
BOB = "bob-id"


def start_actor(workspace_name: str = WORKSPACE_NAME, cap: int = 256) -> WorkspaceActor:
    """Build and start an actor over *workspace_name*, without an actor thread."""
    actor = WorkspaceActor(
        config=WorkspaceConfig(
            name=workspace_actor_name(workspace_name),
            role=WORKSPACE_ACTOR_ROLE,
            workspace_name=workspace_name,
            max_observations_per_agent=cap,
        )
    )
    actor.on_start()
    return actor


def observation(text: str, full: bool = True) -> Observation:
    """An observation of *text*, hashed exactly as the read path hashes it."""
    return Observation(sha=content_sha(text.encode()), full=full)


class _StateSpy:
    """Records every state-change notification the actor's state emits."""

    def __init__(self) -> None:
        self.notifications: list[BaseState] = []

    def notify_state_change(self, state: BaseState) -> None:
        self.notifications.append(state)


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
# AC7: the map, and the per-agent LRU cap
# ---------------------------------------------------------------------------


class TestObservationMap:
    def test_records_and_reads_back(self, workspaces_root: Path) -> None:
        actor = start_actor()
        obs = observation("hello")
        actor.record_observation(ALICE, "a.md", obs)
        assert actor.observation_for(ALICE, "a.md") == obs

    def test_unknown_agent_and_unknown_path_are_none(self, workspaces_root: Path) -> None:
        actor = start_actor()
        assert actor.observation_for(ALICE, "a.md") is None
        actor.record_observation(ALICE, "a.md", observation("hello"))
        assert actor.observation_for(ALICE, "other.md") is None
        assert actor.observation_for(BOB, "a.md") is None

    def test_two_agents_hold_independent_observations(self, workspaces_root: Path) -> None:
        actor = start_actor()
        actor.record_observation(ALICE, "a.md", observation("alice version"))
        actor.record_observation(BOB, "a.md", observation("bob version"))
        alice = actor.observation_for(ALICE, "a.md")
        bob = actor.observation_for(BOB, "a.md")
        assert alice is not None and bob is not None
        assert alice.sha != bob.sha

    def test_re_recording_replaces_rather_than_grows(self, workspaces_root: Path) -> None:
        actor = start_actor(cap=3)
        actor.record_observation(ALICE, "a.md", observation("v1"))
        actor.record_observation(ALICE, "a.md", observation("v2"))
        current = actor.observation_for(ALICE, "a.md")
        assert current is not None
        assert current.sha == content_sha(b"v2")

    def test_cap_evicts_the_least_recently_used_path(self, workspaces_root: Path) -> None:
        actor = start_actor(cap=3)
        for name in ("a.md", "b.md", "c.md", "d.md"):
            actor.record_observation(ALICE, name, observation(name))
        assert actor.observation_for(ALICE, "a.md") is None
        assert all(actor.observation_for(ALICE, n) is not None for n in ("b.md", "c.md", "d.md"))

    def test_re_recording_refreshes_recency_rather_than_insertion_order(
        self, workspaces_root: Path
    ) -> None:
        # Insertion order would evict "a.md"; recency must evict "b.md" instead.
        actor = start_actor(cap=3)
        for name in ("a.md", "b.md", "c.md"):
            actor.record_observation(ALICE, name, observation(name))
        actor.record_observation(ALICE, "a.md", observation("a.md refreshed"))
        actor.record_observation(ALICE, "d.md", observation("d.md"))
        assert actor.observation_for(ALICE, "b.md") is None
        assert actor.observation_for(ALICE, "a.md") is not None

    def test_a_lookup_does_not_refresh_recency(self, workspaces_root: Path) -> None:
        actor = start_actor(cap=3)
        for name in ("a.md", "b.md", "c.md"):
            actor.record_observation(ALICE, name, observation(name))
        actor.observation_for(ALICE, "a.md")
        actor.record_observation(ALICE, "d.md", observation("d.md"))
        assert actor.observation_for(ALICE, "a.md") is None

    def test_the_cap_is_per_agent_not_global(self, workspaces_root: Path) -> None:
        actor = start_actor(cap=2)
        for name in ("a.md", "b.md"):
            actor.record_observation(ALICE, name, observation(name))
            actor.record_observation(BOB, name, observation(name))
        assert actor.observation_for(ALICE, "a.md") is not None
        assert actor.observation_for(BOB, "a.md") is not None


# ---------------------------------------------------------------------------
# AC8: recording is not persisted state
# ---------------------------------------------------------------------------


class TestRecordingIsNotPersistedState:
    def test_recording_emits_no_state_change_notification(self, workspaces_root: Path) -> None:
        actor = start_actor()
        spy = _StateSpy()
        actor.state.observer(spy)
        spy.notifications.clear()  # attaching an observer notifies once, by design
        actor.record_observation(ALICE, "a.md", observation("hello"))
        actor.observation_for(ALICE, "a.md")
        assert spy.notifications == []

    def test_recording_leaves_the_serialisable_state_untouched(self, workspaces_root: Path) -> None:
        actor = start_actor()
        actor.record_observation(ALICE, "a.md", observation("hello"))
        assert actor.state.model_dump() == WorkspaceState().model_dump()

    def test_state_round_trips_with_no_observation_data(self, workspaces_root: Path) -> None:
        actor = start_actor()
        actor.record_observation(ALICE, "a.md", observation("hello"))
        restored = WorkspaceState.model_validate(actor.state.model_dump())
        assert restored.model_dump() == WorkspaceState().model_dump()


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


class TestStartupSweep:
    def test_removes_orphaned_staging_files_at_any_depth(self, workspace_tree: Path) -> None:
        root_orphan = workspace_tree / staging_name("a.md")
        nested = workspace_tree / "sub"
        nested.mkdir()
        nested_orphan = nested / staging_name("b.md")
        root_orphan.write_bytes(b"partial")
        nested_orphan.write_bytes(b"partial")

        start_actor()

        assert not root_orphan.exists()
        assert not nested_orphan.exists()

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

        start_actor()

        assert masquerading.is_dir()
        assert (masquerading / "kept.md").exists()

    def test_an_unremovable_staging_file_does_not_stop_the_actor(
        self, workspace_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        orphan = workspace_tree / staging_name("a.md")
        orphan.write_bytes(b"partial")

        def refuse(self: Path, missing_ok: bool = False) -> None:
            raise PermissionError("read-only directory")

        monkeypatch.setattr(Path, "unlink", refuse)
        actor = start_actor()

        assert orphan.exists()
        assert actor.observation_for(ALICE, "a.md") is None  # the actor started regardless

    def test_a_tree_with_nothing_to_sweep_starts_cleanly(self, workspace_tree: Path) -> None:
        (workspace_tree / "report.md").write_text("hello", encoding="utf-8")
        actor = start_actor()
        actor.record_observation(ALICE, "report.md", observation("hello"))
        assert actor.observation_for(ALICE, "report.md") is not None
