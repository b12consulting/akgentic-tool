"""The read path records what it observed, in the card's own map (ADR-051 Decision 2).

One O(1) call per tool invocation, on the plain-text read branch only, and
fail-open: a lost observation is a lost precondition, never a lost read.

**Story 52-5 moved the map off the actor and onto the card.** Three premises
here were reversed by that decision rather than by a failure, and each is
replaced by the guard for the new invariant instead of being deleted: two cards
no longer share one map (they hold two, and neither can read the other's), the
recording is no longer a ``tell`` (there is no message at all, which is a
stronger version of the property the tell bought), and two teams on one tree no
longer share one actor (the tree is the unicity domain, and nothing shared lives
on the actor any more).
"""

from __future__ import annotations

import gc
import inspect
import threading
import uuid
import weakref
from pathlib import Path
from typing import Any

import pytest
from akgentic.core.utils import SerializableBaseModel

from akgentic.tool.core.observer import ActorToolObserver
from akgentic.tool.errors import RetriableError
from akgentic.tool.workspace.actor import (
    WORKSPACE_ACTOR_ROLE,
    WorkspaceActor,
    workspace_actor_name,
)
from akgentic.tool.workspace.card.params import WorkspaceExec
from akgentic.tool.workspace.card.read import _paginate
from akgentic.tool.workspace.documents.models import EXTRACTOR_VERSION
from akgentic.tool.workspace.event import WorkspaceAttached
from akgentic.tool.workspace.models import Observation, WorkspaceConfig, content_sha
from akgentic.tool.workspace.tool import WorkspaceTool
from akgentic.tool.workspace.workspace import Filesystem
from tests.conftest import MockActorAddress
from tests.workspace.conftest import (
    DEFAULT_TEST_PRINCIPAL,
    HANDSHAKE_TIMEOUT_S,
    WORKSPACE_NAME,
    WORKSPACE_PATH,
    FakeActorToolObserver,
    FakeOrchestratorProxy,
    card_for,
    tool_named,
    workspace_path_for,
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


def spy_on_recording(card: WorkspaceTool, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every path *card* records an observation for, and keep recording it.

    It **forwards** to the real method rather than standing in for it: a stub
    would pass every "records nothing" assertion below while also breaking every
    gate that depends on the map, and the map going empty is precisely the
    failure these specs exist to catch.

    Patched on the **class**, because a ``ToolCard`` is a Pydantic model and
    refuses an attribute that is not a field. The read closure looks the method
    up on ``self`` at call time, so the class is where the seam is; *card* is
    still taken so the spy records only this instance\'s calls.
    """
    paths: list[str] = []
    real = WorkspaceTool.record_observation

    def spy(this: WorkspaceTool, path: str, observation: Observation) -> None:
        if this is card:
            paths.append(path)
        real(this, path, observation)

    monkeypatch.setattr(WorkspaceTool, "record_observation", spy)
    return paths


# ---------------------------------------------------------------------------
# AC1 / AC2: one actor per workspace, reached with a single get-or-create
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_two_cards_on_one_tree_hold_two_independent_observation_maps(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        wired_card: WorkspaceTool,
        seeded_tree: Path,
    ) -> None:
        """**Premise reversed by decision** (ADR-051 Decision 2), and inverted here.

        The actor\'s map was keyed ``agent_id -> path``, so a recording made
        through one card was readable through another — which this spec used to
        assert. The map is now the card\'s own slice, and the invariant worth
        pinning is the opposite one: alice reading a file tells bob\'s card
        nothing, and bob is still refused the overwrite he has not earned.
        """
        bob_card, _bob_observer = card_for(orchestrator_proxy, "bob")

        tool_named(wired_card, "workspace_read")("notes.md")

        assert wired_card.observation_for("notes.md") is not None
        assert bob_card.observation_for("notes.md") is None
        with pytest.raises(RetriableError, match="read it before overwriting"):
            tool_named(bob_card, "workspace_write")("notes.md", "bob was here\n")

    def test_the_actor_is_obtained_with_one_get_or_create_call(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        wired_card: WorkspaceTool,
    ) -> None:
        # Never a check-then-create pair: one message, per ADR-025 — the team\'s
        # own child path again, and never the host forward beside it. The empty
        # ``resource_calls`` is the whole of "this process needs no host".
        assert len(orchestrator_proxy.create_calls) == 1
        assert orchestrator_proxy.resource_calls == []

    def test_the_config_name_carries_the_tool_actor_prefix(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        wired_card: WorkspaceTool,
    ) -> None:
        # The prefix is what the orchestrator\'s two-phase stop classifies on.
        [(_cls, config)] = orchestrator_proxy.create_calls
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

        assert workspace_actor_name(workspace_path_for("shared")) in orchestrator_proxy.children
        assert workspace_actor_name(WORKSPACE_PATH) in orchestrator_proxy.children
        assert shared_card._workspace_proxy is not wired_card._workspace_proxy

    def test_the_actor_owns_the_tree_its_card_is_anchored_to(
        self,
        workspace_actor: WorkspaceActor,
        workspace_tree: Path,
    ) -> None:
        assert workspace_actor.config.workspace_path == WORKSPACE_PATH
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
        recorded = wired_card.observation_for("notes.md")
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
        recorded = wired_card.observation_for("notes.md")
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
        recorded = wired_card.observation_for("notes.md")
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
        recorded = wired_card.observation_for("empty.md")
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
        assert wired_card.observation_for("missing.md") is None


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
        wired_card: WorkspaceTool,
        monkeypatch: pytest.MonkeyPatch,
        seeded_tree: Path,
    ) -> None:
        calls = spy_on_recording(wired_card, monkeypatch)

        tool_named(wired_card, name)(*args)

        assert calls == []

    def test_view_records_nothing(
        self,
        wired_card: WorkspaceTool,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("PIL")
        from PIL import Image

        image_path = workspace_tree / "logo.png"
        Image.new("RGB", (4, 4), "red").save(image_path)
        calls = spy_on_recording(wired_card, monkeypatch)

        tool_named(wired_card, "workspace_view")("logo.png")

        assert calls == []

    def test_a_cached_document_read_records_nothing(
        self,
        wired_card: WorkspaceTool,
        workspace_actor: WorkspaceActor,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A document read hashes the source for the *cache*, and records nothing:
        # the agent is shown derived Markdown, so a digest of bytes it never saw
        # would be a false observation, whatever the read cost to produce.
        source = b"%PDF-1.4 not really a pdf"
        (workspace_tree / "report.pdf").write_bytes(source)
        workspace_actor.cache_document(
            "report.pdf", content_sha(source), EXTRACTOR_VERSION, "extracted"
        )

        calls = spy_on_recording(wired_card, monkeypatch)

        result = tool_named(wired_card, "workspace_read")("report.pdf")

        assert "extracted" in result
        assert calls == []


# ---------------------------------------------------------------------------
# AC5: exactly one recording call per invocation
# ---------------------------------------------------------------------------


class TestOneCallPerInvocation:
    def test_a_large_file_read_makes_exactly_one_call(
        self,
        wired_card: WorkspaceTool,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The property is *counted*, never inferred from the resulting map.

        A per-line recorder would leave the map looking identical, so the spy
        wraps the card\'s own ``record_observation`` and forwards to it rather
        than replacing it — a stub would also pass while recording nothing.
        """
        big = "\n".join(f"line {n}" for n in range(5000))
        (workspace_tree / "big.md").write_text(big, encoding="utf-8")
        calls = spy_on_recording(wired_card, monkeypatch)

        tool_named(wired_card, "workspace_read")("big.md", limit=10_000)

        assert calls == ["big.md"]
        assert wired_card.observation_for("big.md") is not None


# ---------------------------------------------------------------------------
# AC6: content never travels through the actor, and recording is fail-open
# ---------------------------------------------------------------------------


class TestFailOpen:
    def test_a_raising_recording_still_returns_the_whole_file(
        self,
        wired_card: WorkspaceTool,
        seeded_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A lost precondition, never a lost read — the recorder swallows anything."""
        calls: list[str] = []

        def explode(this: WorkspaceTool, path: str, observation: Observation) -> None:
            calls.append(path)
            raise RuntimeError("the map is on fire")

        monkeypatch.setattr(WorkspaceTool, "record_observation", explode)

        result = tool_named(wired_card, "workspace_read")("notes.md")

        assert calls == ["notes.md"]
        for line in BODY.splitlines():
            assert line in result
        # And the degradation is towards refusing a write, never accepting one.
        with pytest.raises(RetriableError, match="read it before overwriting"):
            tool_named(wired_card, "workspace_write")("notes.md", "mine\n")

    def test_a_read_reaches_no_actor_at_all(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        seeded_tree: Path,
    ) -> None:
        """**Premise reversed by decision**, and the replacement is stronger.

        This used to hold a stand-in actor\'s lock and assert that a read
        completed anyway — the property a fire-and-forget ``tell`` bought, and
        the best a spec could do while the map lived on another object. The map
        is the card\'s now, so the honest guard is structural: give the card two
        proxies that raise on **any** attribute access, and read. A read that
        still touched either — to record, to look anything up — cannot be slow
        or lost here, it simply fails.
        """

        class Landmine:
            def __getattr__(self, name: str) -> Any:
                raise AssertionError(f"the read path reached the actor: {name}")

        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
        card.observer(FakeActorToolObserver(orchestrator_proxy, name="alice"))
        card._workspace_proxy = Landmine()  # type: ignore[assignment]
        card._workspace_tell = Landmine()  # type: ignore[assignment]

        result = tool_named(card, "workspace_read")("notes.md")

        assert "alpha" in result
        assert card.observation_for("notes.md") is not None

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
        recorded = wired_card.observation_for("fresh.md")
        assert recorded is not None
        assert recorded == Observation(sha=content_sha(b"content"), full=True)

    def test_the_read_signature_gained_no_parameter(self, wired_card: WorkspaceTool) -> None:
        params = list(inspect.signature(tool_named(wired_card, "workspace_read")).parameters)
        assert params == ["path", "offset", "limit", "force_document_regeneration"]


# ---------------------------------------------------------------------------
# Story 29-3, AC8 / AC9: the observation is a fire-and-forget tell
# ---------------------------------------------------------------------------


class TestTheObservationIsATell:
    """29-2 shipped the record as a blocking ``proxy_ask``; 29-3 made it a ``tell``.

    **Story 52-5 removed the message.** The map is the card\'s own, so a
    recording is a dict write on the calling thread and the whole family of
    hazards this class was written against — a read queueing behind another
    agent\'s hash, a hung mailbox the fail-open ``except`` cannot cover — no
    longer has anything to happen in. What is left here are the properties that
    still have a message behind them: the two proxies the exec surface binds,
    the stale-mark a mutation tells, and the ordering a read-then-write depends
    on. ``TestFailOpen`` above carries the structural replacement.
    """

    def test_the_card_binds_both_proxies_over_the_one_address(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
    ) -> None:
        observer = FakeActorToolObserver(orchestrator_proxy)
        WorkspaceTool(workspace_id=WORKSPACE_NAME).observer(observer)

        assert len(observer.ask_targets) == 1
        assert observer.tell_targets == observer.ask_targets

    def test_an_accepted_mutation_tells_the_index_what_it_touched(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspace_actor: WorkspaceActor,
        seeded_tree: Path,
    ) -> None:
        """The one message a mutation still sends, and it is a **tell**.

        The verdict is the card\'s own now, so nothing is asked; what the actor
        still has to hear is which paths went stale for retrieval, and that
        needs no answer.
        """
        telling = _TellRecorder(workspace_actor)
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
        card.observer(FakeActorToolObserver(orchestrator_proxy, workspace_tell_proxy=telling))

        assert tool_named(card, "workspace_write")("fresh.md", "body\n") == "Written: fresh.md"
        assert "mark_paths_stale" in telling.names

    def test_the_widened_protocol_is_satisfied_by_the_suites_observer(
        self, observer: FakeActorToolObserver
    ) -> None:
        assert isinstance(observer, ActorToolObserver)

    def test_a_read_completes_while_another_agent_is_mid_mutation(
        self,
        threaded_orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The NFR1 property, against real threads rather than a stand-in.

        One agent\'s mutation is held open exactly where the live hash reads the
        file — and, since 52-5, while it holds that path\'s ``flock``. A reader
        arriving during that window must return its file regardless: reads are
        not gated and take no lock, so the hold must be invisible to them.
        Driven by an event handshake with an upper-bound failure budget, never a
        wall-clock sleep.
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
        # Ordering is now trivial rather than earned: the read records into the
        # same object the write then gates against, on the same thread, so there
        # is no delivery for anything to be reordered by.
        notes = workspace_tree / "notes.md"
        notes.write_text(BODY, encoding="utf-8")
        alice, _observer = card_for(threaded_orchestrator_proxy, "alice")

        tool_named(alice, "workspace_read")("notes.md")
        assert tool_named(alice, "workspace_write")("notes.md", "mine\n") == "Written: notes.md"
        assert notes.read_text(encoding="utf-8") == "mine\n"


def test_the_actor_config_is_fully_serialisable(workspaces_root: Path) -> None:
    config = WorkspaceConfig(name="#Workspace-x", role="ToolActor", workspace_path="x")
    assert WorkspaceConfig.model_validate(config.model_dump()) == config


# ---------------------------------------------------------------------------
# Story 51-2: the card binds through the host's forward, then attaches
# ---------------------------------------------------------------------------


class _CaseMetadata(SerializableBaseModel):
    """Stand-in team metadata — the resolver reads declared keys by attribute."""

    customer_id: str | None = None
    case_id: str | None = None


def _card_shape(
    shape: str,
    orchestrator_proxy: FakeOrchestratorProxy,
    request: pytest.FixtureRequest,
) -> tuple[WorkspaceTool, str | None]:
    """One of the five card shapes, and the path it resolves to (``None``: the team's own).

    The bare card's path carries the observer's team id, which only exists once
    the observer does, so the caller fills it in.
    """
    if shape == "bare":
        return WorkspaceTool(), None
    if shape == "named":
        return WorkspaceTool(workspace_id=WORKSPACE_NAME), WORKSPACE_PATH
    if shape == "metadata":
        orchestrator_proxy.metadata = _CaseMetadata(customer_id="ACME", case_id="42")
        card = WorkspaceTool(workspace_metadata_keys=["customer_id", "case_id"])
        return card, f"{DEFAULT_TEST_PRINCIPAL}/_meta/customer_id-ACME__case_id-42"
    if shape == "exec":
        request.getfixturevalue("sandbox_script")
        card = WorkspaceTool(
            workspace_id=WORKSPACE_NAME, workspace_exec=WorkspaceExec(mode="local")
        )
        return card, WORKSPACE_PATH
    assert shape == "rag"
    pytest.importorskip("numpy", reason="the [vector_search] extra is not installed")
    # No ``vector_store``: a card that names the in-actor backend is refused at
    # bind, and one that names none resolves to the file-backed one.
    card = WorkspaceTool(workspace_id=WORKSPACE_NAME, workspace_rag_index=True)
    return card, WORKSPACE_PATH


class TestTheCardBindsAsATeamChild:
    """One get-or-create per bind, as a child, with **no host anywhere in the process**.

    **Premise reversed by decision** (ADR-051, *Ruling A*): the bind was a
    forward to a ``WorkspaceHost`` and is a team child again. The reason is not
    that hosting failed but that it became pointless — every piece of shared
    state the host existed to keep single has moved onto the tree, so there is
    nothing left for two actors over one tree to disagree about.

    The two negatives are the point of the class: nothing forwards to a host
    (``resource_calls`` stays empty — the fake orchestrator's forward records the
    attempt and then raises, since 52-6 deleted the host it would have needed),
    and nothing under ``workspace/card/`` so much as imports one. That pair is
    ``akgentic-infra``\'s acceptance guard read from this side of the seam.
    """

    @pytest.mark.parametrize("shape", ["bare", "named", "metadata", "exec", "rag"])
    def test_every_card_shape_binds_once_as_a_child_and_emits_one_event(
        self,
        shape: str,
        orchestrator_proxy: FakeOrchestratorProxy,
        workspaces_root: Path,
        request: pytest.FixtureRequest,
    ) -> None:
        card, path = _card_shape(shape, orchestrator_proxy, request)
        observer = FakeActorToolObserver(orchestrator_proxy)
        card.observer(observer)
        if path is None:
            path = f"{DEFAULT_TEST_PRINCIPAL}/_team/{observer.team_id}"

        workspace_creates = [
            config
            for cls, config in orchestrator_proxy.create_calls
            if cls is WorkspaceActor
        ]
        [config] = workspace_creates
        assert config.name == workspace_actor_name(path)
        assert isinstance(config, WorkspaceConfig)
        assert config.workspace_path == path
        # The negative beside the positive: no host was forwarded to, on any
        # card shape, and this process runs none.
        assert orchestrator_proxy.resource_calls == []

        [event] = [e for e in observer.events if isinstance(e, WorkspaceAttached)]
        assert event.agent_id == observer.myAddress.agent_id
        # The in-process type, which the wire cannot show: a string id
        # serialises to the same string a UUID does.
        assert isinstance(event.agent_id, uuid.UUID)
        assert event.agent_id != observer.team_id
        assert event.workspace_path == path
        # Exactly one per bind, and it is a bare payload rather than an
        # ``EventMessage`` the card built for itself — ``notify_event`` wraps it.
        assert len(observer.events) == 1
        assert type(event) is WorkspaceAttached


class _AskRecorder:
    """An ask stand-in that records ``attach`` and forwards everything to the real actor."""

    def __init__(self, target: WorkspaceActor) -> None:
        self.target = target
        self.attach_calls: list[tuple[object, str]] = []

    def attach(self, agent: object, agent_name: str) -> None:
        self.attach_calls.append((agent, agent_name))
        self.target.attach(agent, agent_name)  # type: ignore[arg-type]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.target, name)


class _TellRecorder:
    """A tell stand-in that records the name of every method reached through it."""

    def __init__(self, target: WorkspaceActor) -> None:
        self.target = target
        self.names: list[str] = []

    def __getattr__(self, name: str) -> Any:
        self.names.append(name)
        return getattr(self.target, name)


def _created_ahead_of_the_card(
    orchestrator_proxy: FakeOrchestratorProxy, max_tracked_writers: int | None = None
) -> WorkspaceActor:
    """Create the test workspace's actor before any card binds, so a bind is a hit on it."""
    config = WorkspaceConfig(
        name=workspace_actor_name(WORKSPACE_PATH),
        role=WORKSPACE_ACTOR_ROLE,
        workspace_path=WORKSPACE_PATH,
    )
    if max_tracked_writers is not None:
        config = config.model_copy(update={"max_tracked_writers": max_tracked_writers})
    orchestrator_proxy.getChildrenOrCreate(WorkspaceActor, config)
    _, actor = orchestrator_proxy.children[config.name]
    assert isinstance(actor, WorkspaceActor)
    return actor


class TestAttachAbsorbsRegisterAgent:
    """``attach`` records the name; ``register_agent`` is gone, and so is the holder half.

    52-5 folded ``register_agent`` into ``attach``, which then did two jobs: it
    recorded the agent as a *holder* for the liveness sweep to count, and it
    recorded the name a refusal prints. 52-6 deleted the sweep, so only the
    second job is left — and it is the load-bearing one. The holder assertions
    are removed rather than weakened; the name assertions are untouched.
    """

    def test_the_old_registration_is_gone_and_attach_is_there(self) -> None:
        assert hasattr(WorkspaceActor, "attach")
        assert not hasattr(WorkspaceActor, "register_agent")
        assert not hasattr(WorkspaceTool, "_register_agent_name")

    def test_attach_records_the_name_under_the_agent_id(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        actor = _created_ahead_of_the_card(orchestrator_proxy)
        address = MockActorAddress("builder")
        key = str(address.agent_id)

        actor.attach(address, "builder")

        assert actor._name_of(key) == "builder"

        # The same agent again overwrites: the newer name wins.
        actor.attach(address, "builder-2")
        assert actor._name_of(key) == "builder-2"

        # A second agent gets its own entry, and the first keeps its own.
        other = MockActorAddress("reviewer")
        actor.attach(other, "reviewer")
        assert actor._name_of(str(other.agent_id)) == "reviewer"
        assert actor._name_of(key) == "builder-2"

    def test_the_name_map_keeps_its_cap(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """LRU on names, by ``move_to_end``. The holder map it was paired with is gone."""
        actor = _created_ahead_of_the_card(orchestrator_proxy, max_tracked_writers=2)
        ann, bert, carl = (MockActorAddress(name) for name in ("ann", "bert", "carl"))

        actor.attach(ann, "ann")
        actor.attach(bert, "bert")
        actor.attach(ann, "ann")  # refreshes ann's recency, so bert is now the oldest
        actor.attach(carl, "carl")

        assert actor._name_of(str(ann.agent_id)) == "ann"
        assert actor._name_of(str(carl.agent_id)) == "carl"
        assert actor._name_of(str(bert.agent_id)) == str(bert.agent_id)  # evicted: id fallback

    def test_the_card_attaches_once_over_the_ask_proxy_and_never_over_the_tell(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        actor = _created_ahead_of_the_card(orchestrator_proxy)
        ask = _AskRecorder(actor)
        tell = _TellRecorder(actor)
        observer = FakeActorToolObserver(
            orchestrator_proxy, workspace_proxy=ask, workspace_tell_proxy=tell
        )
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)

        card.observer(observer)

        assert ask.attach_calls == [(observer.myAddress, str(observer.myAddress.name))]
        assert ask.attach_calls[0][0] is observer.myAddress
        assert actor._name_of(str(observer.myAddress.agent_id)) == str(observer.myAddress.name)
        assert "attach" not in tell.names
        assert "register_agent" not in tell.names
        # The tell recorder is genuinely wired: an accepted mutation signals the
        # index through it. (A *read* no longer sends anything at all — it
        # records into the card's own map.)
        (workspace_tree / "notes.md").write_text(BODY, encoding="utf-8")
        tool_named(card, "workspace_write")("fresh.md", "body\n")
        assert "mark_paths_stale" in tell.names


class TestAFailedAttachFailsTheBind:
    """An unguarded ask: a lost ``attach`` would leave the tree with no name for this agent."""

    def test_the_attach_failure_reaches_the_caller_unchanged(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        error = RuntimeError("actor is dead")

        class DeadAtBind:
            def attach(self, agent: object, agent_name: str) -> None:
                raise error

        observer = FakeActorToolObserver(orchestrator_proxy, workspace_proxy=DeadAtBind())

        with pytest.raises(RuntimeError) as raised:
            WorkspaceTool(workspace_id=WORKSPACE_NAME).observer(observer)

        assert raised.value is error

    def test_a_stand_in_alive_at_bind_binds(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """The positive beside it: the same shape, an ``attach`` that returns, binds."""
        attached: list[str] = []

        class AliveAtBind:
            def attach(self, agent: object, agent_name: str) -> None:
                attached.append(agent_name)

        stand_in = AliveAtBind()
        observer = FakeActorToolObserver(orchestrator_proxy, workspace_proxy=stand_in)
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME)

        card.observer(observer)

        assert attached == [str(observer.myAddress.name)]
        assert card._workspace_proxy is stand_in


class TestTwoTeamsOnOneTree:
    """**Premise reversed by decision.** Two teams shared one actor; they get two.

    The old spec asserted that two teams over one tree were handed one hosted
    actor, because the actor held the exec lease, the document cache, the index
    and the write gate — state two instances could not have held consistently.
    Every one of those is now a file under ``<meta>``, serialised by the
    filesystem across processes as well as teams, so the actor holds nothing two
    of it could disagree about and is an ordinary team child again (*Ruling A*).

    What has to stay true is what these specs assert: the **tree** stays one
    tree, each team gets its own event, and each actor knows its own holder.
    """

    def test_two_teams_get_two_actors_over_one_tree(self, workspace_tree: Path) -> None:
        first_team = FakeOrchestratorProxy()
        second_team = FakeOrchestratorProxy()
        try:
            alice = FakeActorToolObserver(first_team, name="alice")
            bob = FakeActorToolObserver(second_team, name="bob")
            assert alice.team_id != bob.team_id
            assert alice.user_id == bob.user_id
            alice_card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
            alice_card.observer(alice)
            bob_card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
            bob_card.observer(bob)

            name = workspace_actor_name(WORKSPACE_PATH)
            assert list(first_team.children) == [name]
            assert list(second_team.children) == [name]
            first_actor = first_team.children[name][1]
            second_actor = second_team.children[name][1]
            assert first_actor is not second_actor
            # Each team's own stream, each naming its own agent.
            assert alice.events == [
                WorkspaceAttached(agent_id=alice.myAddress.agent_id, workspace_path=WORKSPACE_PATH)
            ]
            assert bob.events == [
                WorkspaceAttached(agent_id=bob.myAddress.agent_id, workspace_path=WORKSPACE_PATH)
            ]
            # Each actor knows exactly its own binder's name, and not the other's.
            assert first_actor._name_of(str(alice.myAddress.agent_id)) == "alice"
            assert second_actor._name_of(str(bob.myAddress.agent_id)) == "bob"
        finally:
            first_team.stop_all()
            second_team.stop_all()

    def test_the_two_actors_still_gate_each_other_through_the_tree(
        self, workspace_tree: Path
    ) -> None:
        """Two actors, one tree, and the gate still holds — which is why this is safe.

        The gate reads the live file on every mutation, so a writer that never
        passed through *this* team\'s actor is caught anyway. That is what makes
        two actors correct rather than a regression.
        """
        first_team = FakeOrchestratorProxy()
        second_team = FakeOrchestratorProxy()
        try:
            alice_card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
            alice_card.observer(FakeActorToolObserver(first_team, name="alice"))
            bob_card = WorkspaceTool(workspace_id=WORKSPACE_NAME)
            bob_card.observer(FakeActorToolObserver(second_team, name="bob"))

            assert tool_named(alice_card, "workspace_write")("shared.md", "alice\n") == (
                "Written: shared.md"
            )

            with pytest.raises(RetriableError, match="read it before overwriting"):
                tool_named(bob_card, "workspace_write")("shared.md", "bob\n")
            assert (workspace_tree / "shared.md").read_text(encoding="utf-8") == "alice\n"
        finally:
            first_team.stop_all()
            second_team.stop_all()


# ---------------------------------------------------------------------------
# Story 52-5, AC3: the map is the card's, holds one agent's slice, keeps its LRU
# ---------------------------------------------------------------------------


def _observation(text: str, full: bool = True) -> Observation:
    """An observation of *text*, hashed exactly as the read path hashes it."""
    return Observation(sha=content_sha(text.encode()), full=full)


def _capped_card(orchestrator_proxy: FakeOrchestratorProxy, cap: int) -> WorkspaceTool:
    """A bound card whose observation map holds at most *cap* paths.

    The cap is a declared **field**, so a spec sets it the way a catalog would
    rather than by patching a constant — and the value it sets is the value the
    LRU below actually enforces.
    """
    card = WorkspaceTool(workspace_id=WORKSPACE_NAME, max_observations_per_agent=cap)
    card.observer(FakeActorToolObserver(orchestrator_proxy, name="alice"))
    return card


class TestTheObservationMapIsTheCards:
    """Moved from the actor with its rules intact — minus the agent dimension.

    The actor's map was ``agent_id -> path -> Observation``; the card's is
    ``path -> Observation``, because one card belongs to one agent. Every LRU
    rule is unchanged and unchanged **for the same reasons**: recording moves a
    path to the end, a lookup does not, and only the path dimension is capped.
    """

    def test_records_and_reads_back(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        card = _capped_card(orchestrator_proxy, 256)
        obs = _observation("hello")
        card.record_observation("a.md", obs)
        assert card.observation_for("a.md") == obs

    def test_an_unknown_path_is_none(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        card = _capped_card(orchestrator_proxy, 256)
        assert card.observation_for("a.md") is None
        card.record_observation("a.md", _observation("hello"))
        assert card.observation_for("other.md") is None

    def test_two_cards_hold_independent_maps_and_neither_reads_the_others(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        alice = _capped_card(orchestrator_proxy, 256)
        bob, _observer = card_for(orchestrator_proxy, "bob")
        alice.record_observation("a.md", _observation("alice version"))
        bob.record_observation("a.md", _observation("bob version"))

        seen_by_alice = alice.observation_for("a.md")
        seen_by_bob = bob.observation_for("a.md")
        assert seen_by_alice is not None and seen_by_bob is not None
        assert seen_by_alice.sha != seen_by_bob.sha
        assert alice._observations is not bob._observations

    def test_re_recording_replaces_rather_than_grows(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        card = _capped_card(orchestrator_proxy, 3)
        card.record_observation("a.md", _observation("v1"))
        card.record_observation("a.md", _observation("v2"))
        current = card.observation_for("a.md")
        assert current is not None
        assert current.sha == content_sha(b"v2")

    def test_cap_evicts_the_least_recently_used_path(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        card = _capped_card(orchestrator_proxy, 3)
        for name in ("a.md", "b.md", "c.md", "d.md"):
            card.record_observation(name, _observation(name))
        assert card.observation_for("a.md") is None
        assert all(card.observation_for(n) is not None for n in ("b.md", "c.md", "d.md"))

    def test_re_recording_refreshes_recency_rather_than_insertion_order(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        # Insertion order would evict "a.md"; recency must evict "b.md" instead.
        card = _capped_card(orchestrator_proxy, 3)
        for name in ("a.md", "b.md", "c.md"):
            card.record_observation(name, _observation(name))
        card.record_observation("a.md", _observation("a.md refreshed"))
        card.record_observation("d.md", _observation("d.md"))
        assert card.observation_for("b.md") is None
        assert card.observation_for("a.md") is not None

    def test_a_lookup_does_not_refresh_recency(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        card = _capped_card(orchestrator_proxy, 3)
        for name in ("a.md", "b.md", "c.md"):
            card.record_observation(name, _observation(name))
        card.observation_for("a.md")
        card.record_observation("d.md", _observation("d.md"))
        assert card.observation_for("a.md") is None

    def test_the_cap_is_per_card_not_shared(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        alice = _capped_card(orchestrator_proxy, 2)
        bob, _observer = card_for(orchestrator_proxy, "bob")
        for name in ("a.md", "b.md"):
            alice.record_observation(name, _observation(name))
            bob.record_observation(name, _observation(name))
        assert alice.observation_for("a.md") is not None
        assert bob.observation_for("a.md") is not None

    def test_the_declared_cap_is_what_the_lru_enforces(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        """Guard against a field that is declared and then ignored.

        Two different caps, two different eviction points: a map that read the
        default constant instead would keep every path in both.
        """
        tight = _capped_card(orchestrator_proxy, 2)
        for name in ("a.md", "b.md", "c.md"):
            tight.record_observation(name, _observation(name))
        assert len(tight._observations) == 2

        roomy = WorkspaceTool(workspace_id=WORKSPACE_NAME, max_observations_per_agent=5)
        roomy.observer(FakeActorToolObserver(orchestrator_proxy, name="bob"))
        for name in ("a.md", "b.md", "c.md"):
            roomy.record_observation(name, _observation(name))
        assert len(roomy._observations) == 3


class TestRecordingIsNotSerialisedState:
    """Recording is runtime state: it must never reach what a catalog stores."""

    def test_recording_leaves_the_serialised_card_untouched(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        card = _capped_card(orchestrator_proxy, 256)
        before = card.model_dump()
        card.record_observation("a.md", _observation("hello"))
        assert card.observation_for("a.md") is not None
        assert card.model_dump() == before

    def test_a_card_that_has_observed_still_round_trips(
        self, orchestrator_proxy: FakeOrchestratorProxy, workspace_tree: Path
    ) -> None:
        card = _capped_card(orchestrator_proxy, 256)
        card.record_observation("a.md", _observation("hello"))

        restored = WorkspaceTool.model_validate(card.model_dump())

        assert restored.max_observations_per_agent == 256
        assert restored._observations == {}
        assert "_observations" not in card.model_dump()
