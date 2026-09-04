"""The read path reports what it observed to ``#Workspace`` (story 29-2).

One O(1) call per tool invocation, on the plain-text read branch only, and
fail-open: a lost observation is a lost precondition, never a lost read.
"""

from __future__ import annotations

import gc
import inspect
import threading
import weakref
from pathlib import Path
from typing import Any

import pytest
from akgentic.tool.core.observer import ActorToolObserver
from akgentic.tool.errors import RetriableError
from akgentic.tool.workspace.actor import WorkspaceActor, workspace_actor_name
from akgentic.tool.workspace.models import Observation, WorkspaceConfig, content_sha
from akgentic.tool.workspace.card.read import _paginate
from akgentic.tool.workspace.tool import WorkspaceTool
from akgentic.tool.workspace.workspace import Filesystem

from tests.workspace.conftest import (
    HANDSHAKE_TIMEOUT_S,
    WORKSPACE_NAME,
    AskOnlyProxy,
    BusyProxy,
    CountingProxy,
    FailingProxy,
    FakeActorToolObserver,
    FakeOrchestratorProxy,
    RecordingTellProxy,
    card_for,
    tool_named,
)

BODY = "alpha\nbravo\ncharlie\ndelta\necho\n"


@pytest.fixture
def seeded_tree(workspace_tree: Path) -> Path:
    """A tree holding ``notes.md`` plus a couple of siblings."""
    (workspace_tree / "notes.md").write_text(BODY, encoding="utf-8")
    (workspace_tree / "other.md").write_text("other\n", encoding="utf-8")
    sub = workspace_tree / "sub"
    sub.mkdir()
    (sub / "nested.md").write_text("nested\n", encoding="utf-8")
    return workspace_tree


def agent_id_of(observer: FakeActorToolObserver) -> str:
    """The identity the card captured for *observer*, as the actor sees it."""
    return str(observer.myAddress.agent_id)


# ---------------------------------------------------------------------------
# AC1 / AC2: one actor per workspace, reached with a single get-or-create
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_two_cards_share_one_actor(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        wired_card: WorkspaceTool,
        workspace_actor: WorkspaceActor,
        observer: FakeActorToolObserver,
        seeded_tree: Path,
    ) -> None:
        second_observer = FakeActorToolObserver(orchestrator_proxy, name="bob")
        second_card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
        second_card.observer(second_observer)

        tool_named(wired_card, "workspace_read")("notes.md")

        # Recorded through card A's proxy, readable through card B's.
        second_proxy = second_card._workspace_proxy
        assert second_proxy is not None
        assert second_proxy.observation_for(agent_id_of(observer), "notes.md") is not None

    def test_the_actor_is_obtained_with_one_get_or_create_call(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        wired_card: WorkspaceTool,
    ) -> None:
        # Never a check-then-create pair: one message, per ADR-025.
        assert len(orchestrator_proxy.create_calls) == 1

    def test_the_config_name_carries_the_tool_actor_prefix(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        wired_card: WorkspaceTool,
    ) -> None:
        _, config = orchestrator_proxy.create_calls[0]
        assert config.name.startswith("#")

    def test_two_workspaces_in_one_team_get_two_actors(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        wired_card: WorkspaceTool,
        workspaces_root: Path,
    ) -> None:
        # The card's workspace is the actor's unicity domain: a second card on a
        # different tree must not be handed the first tree's actor.
        shared_observer = FakeActorToolObserver(orchestrator_proxy, name="bob")
        shared_card = WorkspaceTool(workspace_id="shared")
        shared_card.observer(shared_observer)

        assert workspace_actor_name("shared") in orchestrator_proxy.children
        assert workspace_actor_name(WORKSPACE_NAME) in orchestrator_proxy.children
        assert shared_card._workspace_proxy is not wired_card._workspace_proxy

    def test_the_actor_owns_the_tree_its_card_is_anchored_to(
        self,
        workspace_actor: WorkspaceActor,
        workspace_tree: Path,
    ) -> None:
        assert workspace_actor.config.workspace_name == WORKSPACE_NAME
        assert workspace_actor._workspace._root == workspace_tree.resolve()


# ---------------------------------------------------------------------------
# AC3: full vs paginated
# ---------------------------------------------------------------------------


class TestWhatAReadRecords:
    def test_a_whole_file_read_records_full_true(
        self,
        wired_card: WorkspaceTool,
        workspace_actor: WorkspaceActor,
        observer: FakeActorToolObserver,
        seeded_tree: Path,
    ) -> None:
        tool_named(wired_card, "workspace_read")("notes.md")
        recorded = workspace_actor.observation_for(agent_id_of(observer), "notes.md")
        assert recorded is not None
        assert recorded.full is True
        assert recorded.sha == content_sha(BODY.encode())

    def test_a_limit_truncated_read_records_full_false_with_the_same_sha(
        self,
        wired_card: WorkspaceTool,
        workspace_actor: WorkspaceActor,
        observer: FakeActorToolObserver,
        seeded_tree: Path,
    ) -> None:
        tool_named(wired_card, "workspace_read")("notes.md", limit=2)
        recorded = workspace_actor.observation_for(agent_id_of(observer), "notes.md")
        assert recorded is not None
        assert recorded.full is False
        assert recorded.sha == content_sha(BODY.encode())

    def test_an_offset_shifted_read_records_full_false(
        self,
        wired_card: WorkspaceTool,
        workspace_actor: WorkspaceActor,
        observer: FakeActorToolObserver,
        seeded_tree: Path,
    ) -> None:
        tool_named(wired_card, "workspace_read")("notes.md", offset=2)
        recorded = workspace_actor.observation_for(agent_id_of(observer), "notes.md")
        assert recorded is not None
        assert recorded.full is False
        assert recorded.sha == content_sha(BODY.encode())

    def test_an_empty_file_read_records_full_true(
        self,
        wired_card: WorkspaceTool,
        workspace_actor: WorkspaceActor,
        observer: FakeActorToolObserver,
        workspace_tree: Path,
    ) -> None:
        (workspace_tree / "empty.md").write_bytes(b"")
        tool_named(wired_card, "workspace_read")("empty.md")
        recorded = workspace_actor.observation_for(agent_id_of(observer), "empty.md")
        assert recorded is not None
        assert recorded.full is True

    def test_a_failed_read_records_nothing(
        self,
        wired_card: WorkspaceTool,
        workspace_actor: WorkspaceActor,
        observer: FakeActorToolObserver,
        seeded_tree: Path,
    ) -> None:
        with pytest.raises(RetriableError):
            tool_named(wired_card, "workspace_read")("missing.md")
        assert workspace_actor.observation_for(agent_id_of(observer), "missing.md") is None


class TestPaginateIsTheSingleSourceOfFull:
    """``_paginate`` decides what every observation's ``full`` flag says.

    It is reached through ``workspace_read`` above, but only over the two or
    three windows those tests happen to use. The flag is a precondition the
    write gate will key on, so the edges it is wrong at are worth pinning
    directly — an exact-fit window, a file with no lines at all, and a window
    that starts past the end.
    """

    @pytest.mark.parametrize(
        ("raw", "offset", "limit", "expected_full"),
        [
            ("a\nb\nc\n", 1, 100, True),  # window wider than the file
            ("a\nb\nc\n", 1, 3, True),  # window exactly the file
            ("a\nb\nc\n", 1, 2, False),  # stops one line short
            ("a\nb\nc\n", 2, 100, False),  # starts one line late
            ("a\nb\nc\n", 0, 100, True),  # offset below 1 clamps to the start
            ("a\nb\nc\n", 9, 100, False),  # window begins past the end
            ("", 1, 100, True),  # no lines at all is a whole file
            ("only\n", 1, 100, True),  # single line
        ],
    )
    def test_full_tracks_the_window_the_reader_was_shown(
        self, raw: str, offset: int, limit: int, expected_full: bool
    ) -> None:
        _, full = _paginate(raw, offset, limit)
        assert full is expected_full

    def test_the_text_and_the_flag_come_from_the_same_bounds(self) -> None:
        # Whenever the flag says "not whole", the text must carry the truncation
        # notice or start below line 1 — the two cannot disagree.
        numbered, full = _paginate("a\nb\nc\n", offset=1, limit=2)
        assert full is False
        assert "truncated: 3 lines total, showing 1-2" in numbered
        assert numbered.startswith("1     a")

    def test_a_window_past_the_end_shows_nothing_and_claims_nothing(self) -> None:
        numbered, full = _paginate("a\nb\nc\n", offset=9, limit=100)
        assert numbered == ""
        assert full is False


# ---------------------------------------------------------------------------
# AC4: the silent capabilities
# ---------------------------------------------------------------------------


class TestSilentCapabilities:
    @pytest.mark.parametrize(
        ("name", "args"),
        [
            ("workspace_list", ()),
            ("workspace_glob", ("*.md",)),
            ("workspace_grep", ("alpha",)),
        ],
    )
    def test_they_record_nothing(
        self,
        name: str,
        args: tuple[Any, ...],
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_actor: WorkspaceActor,
        observer: FakeActorToolObserver,
        seeded_tree: Path,
    ) -> None:
        counting = CountingProxy(workspace_actor)
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
        card.observer(FakeActorToolObserver(orchestrator_proxy, workspace_proxy=counting))

        tool_named(card, name)(*args)

        assert counting.calls == []

    def test_view_records_nothing(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_actor: WorkspaceActor,
        workspace_tree: Path,
    ) -> None:
        pytest.importorskip("PIL")
        from PIL import Image

        image_path = workspace_tree / "logo.png"
        Image.new("RGB", (4, 4), "red").save(image_path)

        counting = CountingProxy(workspace_actor)
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
        card.observer(FakeActorToolObserver(orchestrator_proxy, workspace_proxy=counting))

        tool_named(card, "workspace_view")("logo.png")

        assert counting.calls == []

    def test_a_cached_document_read_records_nothing(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_actor: WorkspaceActor,
        workspace_tree: Path,
    ) -> None:
        # On a sidecar cache hit the source bytes are never read, so hashing them
        # would put a full file read back onto the path NFR1 keeps free.
        (workspace_tree / "report.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
        (workspace_tree / ".report.pdf.md").write_text("extracted", encoding="utf-8")

        counting = CountingProxy(workspace_actor)
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
        card.observer(FakeActorToolObserver(orchestrator_proxy, workspace_proxy=counting))

        result = tool_named(card, "workspace_read")("report.pdf")

        assert "extracted" in result
        assert counting.calls == []


# ---------------------------------------------------------------------------
# AC5: exactly one recording call per invocation
# ---------------------------------------------------------------------------


class TestOneCallPerInvocation:
    def test_a_large_file_read_makes_exactly_one_call(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_actor: WorkspaceActor,
        workspace_tree: Path,
    ) -> None:
        big = "\n".join(f"line {n}" for n in range(5000))
        (workspace_tree / "big.md").write_text(big, encoding="utf-8")

        counting = CountingProxy(workspace_actor)
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
        card.observer(FakeActorToolObserver(orchestrator_proxy, workspace_proxy=counting))

        tool_named(card, "workspace_read")("big.md", limit=10_000)

        assert len(counting.calls) == 1


# ---------------------------------------------------------------------------
# AC6: content never travels through the actor, and recording is fail-open
# ---------------------------------------------------------------------------


class TestFailOpen:
    def test_a_raising_recording_still_returns_the_whole_file(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        seeded_tree: Path,
    ) -> None:
        failing = FailingProxy()
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
        card.observer(FakeActorToolObserver(orchestrator_proxy, workspace_proxy=failing))

        result = tool_named(card, "workspace_read")("notes.md")

        assert failing.calls == 1
        for line in BODY.splitlines():
            assert line in result

    def test_a_read_completes_while_the_actor_is_occupied(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        seeded_tree: Path,
    ) -> None:
        busy = BusyProxy()
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
        card.observer(FakeActorToolObserver(orchestrator_proxy, workspace_proxy=busy))
        read = tool_named(card, "workspace_read")

        results: list[str] = []
        occupier = threading.Thread(target=busy.occupy)
        reader = threading.Thread(target=lambda: results.append(read("notes.md")))

        occupier.start()
        assert busy.occupied.wait(timeout=HANDSHAKE_TIMEOUT_S)
        reader.start()
        # Wait until the read has actually reached the recording call and is
        # queueing behind the occupier. Releasing on ``reader.start()`` alone
        # would let a scheduler run the occupier to completion first, and the
        # test would pass having exercised no contention whatsoever.
        assert busy.queued.wait(timeout=HANDSHAKE_TIMEOUT_S)
        assert results == []  # nothing was returned early, and nothing was lost
        busy.release.set()
        occupier.join(timeout=HANDSHAKE_TIMEOUT_S)
        reader.join(timeout=HANDSHAKE_TIMEOUT_S)

        assert not reader.is_alive()
        assert len(results) == 1
        for line in BODY.splitlines():
            assert line in results[0]
        assert busy.calls == ["notes.md"]

    def test_a_card_with_no_bound_actor_still_reads(
        self,
        wired_card: WorkspaceTool,
        seeded_tree: Path,
    ) -> None:
        # The harness shapes that hand a card a bare observer never bind an actor.
        wired_card._workspace_proxy = None
        result = tool_named(wired_card, "workspace_read")("notes.md")
        assert "alpha" in result


# ---------------------------------------------------------------------------
# AC12: the owning agent is still reclaimed
# ---------------------------------------------------------------------------


class TestNoRetention:
    def test_the_owning_agent_is_collected_after_a_read(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        seeded_tree: Path,
    ) -> None:
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
        owner = FakeActorToolObserver(orchestrator_proxy, name="carol")
        card.observer(owner)
        tool_named(card, "workspace_read")("notes.md")

        ref = weakref.ref(owner)
        del owner
        gc.collect()

        assert ref() is None


# ---------------------------------------------------------------------------
# AC11: nothing an agent can do behaves differently
# ---------------------------------------------------------------------------


class TestMutationsAreUnchanged:
    def test_a_write_to_a_new_path_still_needs_no_read(
        self,
        wired_card: WorkspaceTool,
        workspace_actor: WorkspaceActor,
        observer: FakeActorToolObserver,
        workspace_tree: Path,
    ) -> None:
        # Story 29-2 asserted here that a write recorded nothing, because the
        # actor did not yet gate. It does now: creating a file the agent has not
        # read is still accepted — there is nothing to clobber — but the write
        # refreshes the writer's own observation so its next write is not
        # refused with a diff against its own content.
        tool_named(wired_card, "workspace_write")("fresh.md", "content")

        assert (workspace_tree / "fresh.md").read_text(encoding="utf-8") == "content"
        recorded = workspace_actor.observation_for(agent_id_of(observer), "fresh.md")
        assert recorded is not None
        assert recorded == Observation(sha=content_sha(b"content"), full=True)

    def test_the_read_signature_gained_no_parameter(self, wired_card: WorkspaceTool) -> None:
        params = list(inspect.signature(tool_named(wired_card, "workspace_read")).parameters)
        assert params == ["path", "offset", "limit", "force_document_regeneration"]


# ---------------------------------------------------------------------------
# Story 29-3, AC8 / AC9: the observation is a fire-and-forget tell
# ---------------------------------------------------------------------------


class TestTheObservationIsATell:
    """29-2 shipped the record as a blocking ``proxy_ask``; this story converts it.

    The hazard became real here rather than earlier: from this story the actor
    hashes files on its ask path, so a read's observation would queue behind
    another agent's mutation reading a large file — and ``ask_wrapper`` does
    ``future.get(timeout=None)``. The recorder's fail-open ``except`` covers a
    raising actor and a dead one; it can never cover a hung one, which is the
    single failure mode that loses the *read* instead of refusing a *write*.
    """

    def test_the_recorder_uses_the_tell_proxy_and_not_the_ask_proxy(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_actor: WorkspaceActor,
        seeded_tree: Path,
    ) -> None:
        # The ask proxy refuses to carry an observation, so a read that still
        # asked would fail loudly rather than pass while holding the wrong
        # invariant.
        telling = RecordingTellProxy(workspace_actor)
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
        card.observer(
            FakeActorToolObserver(
                orchestrator_proxy,
                workspace_proxy=AskOnlyProxy(workspace_actor),
                workspace_tell_proxy=telling,
            )
        )

        result = tool_named(card, "workspace_read")("notes.md")

        assert "alpha" in result
        assert [path for _, path, _ in telling.calls] == ["notes.md"]

    def test_the_card_binds_both_proxies_over_the_one_address(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
    ) -> None:
        observer = FakeActorToolObserver(orchestrator_proxy)
        WorkspaceTool(workspace_id=WORKSPACE_NAME).observer(observer)

        assert len(observer.ask_targets) == 1
        assert observer.tell_targets == observer.ask_targets

    def test_a_mutation_still_asks(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_actor: WorkspaceActor,
        seeded_tree: Path,
    ) -> None:
        # The split is not "everything becomes a tell": a mutation needs the
        # verdict, so it must ask.
        telling = RecordingTellProxy(workspace_actor)
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
        card.observer(FakeActorToolObserver(orchestrator_proxy, workspace_tell_proxy=telling))

        assert tool_named(card, "workspace_write")("fresh.md", "body\n") == "Written: fresh.md"
        assert telling.calls == []

    def test_the_widened_protocol_is_satisfied_by_the_suites_observer(
        self, observer: FakeActorToolObserver
    ) -> None:
        assert isinstance(observer, ActorToolObserver)

    def test_a_read_completes_while_the_actor_is_busy_hashing_for_someone_else(
        self,
        threaded_orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The NFR1 property, against a real mailbox rather than a stand-in.

        One agent's mutation is held inside the actor's own thread, exactly
        where the live hash reads the file. A reader arriving during that window
        must return its file regardless — driven by an event handshake with an
        upper-bound failure budget, never a wall-clock sleep.
        """
        (workspace_tree / "notes.md").write_text(BODY, encoding="utf-8")
        alice, _alice_observer = card_for(threaded_orchestrator_proxy, "alice")
        bob, _bob_observer = card_for(threaded_orchestrator_proxy, "bob")

        holding = threading.Event()
        release = threading.Event()
        real_read = Filesystem.read

        def slow_read(self: Filesystem, path: str) -> bytes:
            if path == "big.md":
                holding.set()
                release.wait(timeout=HANDSHAKE_TIMEOUT_S)
            return real_read(self, path)

        monkeypatch.setattr(Filesystem, "read", slow_read)

        mutations: list[str] = []
        mutator = threading.Thread(
            target=lambda: mutations.append(
                tool_named(bob, "workspace_write")("big.md", "payload\n")
            )
        )
        mutator.start()
        assert holding.wait(timeout=HANDSHAKE_TIMEOUT_S), "the actor never reached the hash"

        reads: list[str] = []
        reader = threading.Thread(
            target=lambda: reads.append(tool_named(alice, "workspace_read")("notes.md"))
        )
        reader.start()
        reader.join(timeout=HANDSHAKE_TIMEOUT_S)

        assert not reader.is_alive(), "the read waited on a busy actor — it must not ask"
        assert reads and "alpha" in reads[0]

        release.set()
        mutator.join(timeout=HANDSHAKE_TIMEOUT_S)
        assert mutations == ["Written: big.md"]

    def test_a_read_immediately_followed_by_a_write_is_accepted(
        self,
        threaded_orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
    ) -> None:
        # Ordering survives the conversion: the read's tell and the write's ask
        # are two messages from one thread to one mailbox, delivered in order.
        notes = workspace_tree / "notes.md"
        notes.write_text(BODY, encoding="utf-8")
        alice, _observer = card_for(threaded_orchestrator_proxy, "alice")

        tool_named(alice, "workspace_read")("notes.md")
        assert tool_named(alice, "workspace_write")("notes.md", "mine\n") == "Written: notes.md"
        assert notes.read_text(encoding="utf-8") == "mine\n"


def test_the_actor_config_is_fully_serialisable(workspaces_root: Path) -> None:
    config = WorkspaceConfig(name="#Workspace-x", role="ToolActor", workspace_name="x")
    assert WorkspaceConfig.model_validate(config.model_dump()) == config
