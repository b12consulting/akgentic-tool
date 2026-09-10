"""Story 51-4: the hosted workspace persists by member-keyed delta, and restores from one.

Every change to ``WorkspaceState.documents`` and ``rag_index`` is reported to the
process's ``WorkspaceHost`` as one ``StateDelta`` per persist point, keyed
``documents.<path>`` / ``rag_index.<path>``. The inert specs replace the one seam
the actor sends through (``_send_delta``) with :func:`delta_recorder`, which folds
every delta into a :class:`DeltaStore` with team 37-1's rules — so "the store holds
what the actor wrote" is asserted by **restoring** from it, never by reading the
deltas back and agreeing with them.

The live specs at the bottom never replace the seam: a real ``ActorSystem``, a
real ``WorkspaceHost`` with a ``DeltaStore`` registered, a real reap, and a
second actor restored from what the first one sent.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from akgentic.core import ActorRegistry
from akgentic.core.actor_address import ActorAddress
from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.agent_config import BaseConfig
from akgentic.core.resource_host import ResourceHost, StateDelta, resolve_state_type

from akgentic.tool.vector_store.protocol import VectorStoreParam
from akgentic.tool.workspace.actor import (
    WORKSPACE_ACTOR_ROLE,
    WorkspaceActor,
    workspace_actor_name,
)
from akgentic.tool.workspace.documents.models import (
    EXTRACTOR_VERSION,
    DocumentExtract,
    NewFileMessage,
    RagChunk,
    RagFile,
    RagStatus,
)
from akgentic.tool.workspace.documents.worker import (
    EMBED_BATCH_SIZE,
    MAX_CONCURRENT_INDEX_WORKERS,
)
from akgentic.tool.workspace.host import WorkspaceHost, workspace_host_address
from akgentic.tool.workspace.models import (
    Observation,
    WorkspaceConfig,
    WorkspaceState,
    content_sha,
)
from tests.workspace.conftest import (
    HANDSHAKE_TIMEOUT_S,
    WORKSPACE_PATH,
    DeltaStore,
    delta_recorder,
    factory_for,
    fast_config,
    wait_until,
    workspace_root_for,
)
from tests.workspace.test_document_cache import _ExtractWithExtraField
from tests.workspace.test_rag_models import _RagFileWithExtraField
from tests.workspace.test_rag_pipeline import RagHarness, write

_A_MARKDOWN = "# A\n\nbody\n"
"""The body the worker extracts for ``a.md`` — ten characters, under every cap below."""


def _capped_actor(max_documents: int = 3, max_document_chars: int = 100) -> WorkspaceActor:
    """An inert actor over the test workspace with caps small enough to evict on purpose."""
    actor = WorkspaceActor(
        config=WorkspaceConfig(
            name=workspace_actor_name(WORKSPACE_PATH),
            role=WORKSPACE_ACTOR_ROLE,
            workspace_path=WORKSPACE_PATH,
            max_documents=max_documents,
            max_document_chars=max_document_chars,
        )
    )
    actor.on_start()
    return actor


@pytest.fixture
def harness(workspace_tree: Path, monkeypatch: pytest.MonkeyPatch) -> RagHarness:
    """A harness installed on a capped inert actor, with retrieval not yet on."""
    built = RagHarness(_capped_actor())
    built.install(monkeypatch)
    return built


def _assert_clean(actor: WorkspaceActor, step: str) -> None:
    """Every write this step made was flushed: both dirty sets are empty."""
    assert actor._dirty_rows == set(), f"{step}: rows written and never persisted"
    assert actor._dirty_documents == set(), f"{step}: documents written and never persisted"


def _restored(store: DeltaStore, actor: WorkspaceActor) -> WorkspaceState:
    """What a fresh actor would be handed for *actor*'s scope."""
    restored = store.load(WorkspaceActor, actor.config.name)
    assert isinstance(restored, WorkspaceState), f"nothing stored for {actor.config.name}"
    return restored


def _drive_every_write_site(harness: RagHarness, tree: Path, checkpoint: object) -> None:
    """Upload, enable, index, embed in three batches, fail, mark stale, and evict by both caps.

    *checkpoint* is called with the step's name after every step except a
    non-final embedding batch, which is the one step that writes a row and
    deliberately does not persist it.
    """
    assert callable(checkpoint)
    actor = harness.actor
    write(tree, "a.md", "# A\n\nbody\n")
    write(tree, "b.md", "# B\n\nother\n")

    actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["a.md", "b.md"], source="upload"))
    assert actor.state.rag_index["a.md"].status is RagStatus.PENDING
    checkpoint("an upload with retrieval off")

    harness.enable()
    checkpoint("enable_rag on the in-memory backend")

    actor.index_paths("")
    assert actor.state.rag_index["a.md"].status is RagStatus.EXTRACTION
    checkpoint("index_paths")

    harness.report("a.md", chunks=EMBED_BATCH_SIZE * 2 + 1, markdown=_A_MARKDOWN, extracted=True)
    assert actor.state.rag_index["a.md"].batches_expected == 3, "not a three-batch file"
    checkpoint("an index result with three batches")

    harness.result("a.md")
    harness.result("a.md")
    assert actor.state.rag_index["a.md"].batches_landed == 2
    assert actor._dirty_rows == {"a.md"}, "a non-final batch is dirty and unpersisted"

    harness.result("a.md")
    assert actor.state.rag_index["a.md"].status is RagStatus.EMBEDDED
    checkpoint("the final batch")

    harness.fail("b.md")
    assert actor.state.rag_index["b.md"].status is RagStatus.FAILED
    checkpoint("an index failure")

    target = tree / "a.md"
    actor.record_observation(
        "alice", "a.md", Observation(sha=content_sha(target.read_bytes()), full=True)
    )
    actor.apply_write("alice", "a.md", "# Rewritten\n")
    assert actor.state.rag_index["a.md"].status is RagStatus.STALE
    checkpoint("a gate write")

    actor.cache_document("c.md", content_sha(b"c"), EXTRACTOR_VERSION, "c" * 60)
    checkpoint("a fill under both caps")
    actor.cache_document("d.md", content_sha(b"d"), EXTRACTOR_VERSION, "d" * 50)
    assert actor.state.documents["a.md"].markdown is None, "the char cap dropped no body"
    assert actor.state.documents["c.md"].markdown is None, "the char cap dropped one body"
    checkpoint("a fill that drops two bodies by the char cap")
    actor.cache_document("e.md", content_sha(b"e"), EXTRACTOR_VERSION, "e" * 10)
    assert "a.md" not in actor.state.documents, "the row cap removed no entry"
    checkpoint("a fill that removes an entry by the row cap")


##
## The restore route itself: core can name the state class a store rebuilds into
##
class TestAStoreCanNameTheWorkspaceState:
    """A store rebuilds a delta-assembled document through ``resolve_state_type``.

    Core's walk reads a direct ``Akgent[Config, State]`` binding only. The
    workspace binds its state through ``DeferredResultActor[...]``, whose own
    ``Akgent`` arguments are type variables, so without a direct binding on the
    class the answer is ``None`` and every store's ``load`` restores nothing.
    """

    def test_resolve_state_type_names_workspace_state(self) -> None:
        assert resolve_state_type(WorkspaceActor) is WorkspaceState


##
## AC 3 — THE FALSIFIER: a restore that lacks an entry the actor wrote
##
class TestARestoreHoldsEveryEntryTheActorWrote:
    """Every write site, one scenario, and a restore compared member for member."""

    def test_the_restore_equals_the_live_state_after_every_write_site(
        self, harness: RagHarness, workspace_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = delta_recorder(harness.actor, monkeypatch)
        steps: list[str] = []

        def checkpoint(step: str) -> None:
            _assert_clean(harness.actor, step)
            steps.append(step)

        _drive_every_write_site(harness, workspace_tree, checkpoint)

        assert len(steps) == 10, f"the scenario skipped a step: {steps}"
        restored = _restored(store, harness.actor)
        assert restored.rag_index == harness.actor.state.rag_index
        assert dict(restored.documents) == dict(harness.actor.state.documents)
        assert {row.status for row in restored.rag_index.values()} == {
            RagStatus.STALE,
            RagStatus.FAILED,
        }
        assert set(restored.documents) == {"c.md", "d.md", "e.md"}


##
## AC 8 — a restore re-queues every row a dead worker was carrying
##
def _row(status: RagStatus, updated_at: datetime) -> RagFile:
    """One row per status, carrying chunks and superseded ids a re-queue must not touch."""
    path = f"{status.value}.md"
    return RagFile(
        path=path,
        status=status,
        indexed_sha=content_sha(path.encode()),
        chunks=[RagChunk(chunk_id=f"{path}-0", ordinal=0, start=0, end=4)],
        chunk_count=1,
        batches_expected=3,
        batches_landed=1,
        superseded_chunk_ids=[f"{path}-old"],
        reason="kept" if status is RagStatus.FAILED else None,
        updated_at=updated_at,
    )


_IN_FLIGHT_ON_RESTORE = (RagStatus.EXTRACTION, RagStatus.SPLITTING, RagStatus.EMBEDDING)


class TestARestoreRequeuesWhatADeadWorkerCarried:
    """No worker survives its parent, so at restore every in-flight row is abandoned."""

    def test_every_in_flight_row_goes_back_to_pending_whatever_its_age(
        self, harness: RagHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # One second old: well inside the reaper's bound, so the running-actor
        # reaper would leave the EMBEDDING row where it is. Only the restore
        # re-queue can move it.
        recent = datetime.now(UTC) - timedelta(seconds=1)
        rows = {f"{status.value}.md": _row(status, recent) for status in RagStatus}
        store = delta_recorder(harness.actor, monkeypatch)

        harness.actor.init_state(WorkspaceState(rag_index=dict(rows)))

        live = harness.actor.state.rag_index
        for status in _IN_FLIGHT_ON_RESTORE:
            path = f"{status.value}.md"
            assert live[path].status is RagStatus.PENDING, f"{path} is still {live[path].status}"
            assert (live[path].batches_expected, live[path].batches_landed) == (0, 0)
            assert live[path].chunks == rows[path].chunks
            assert live[path].superseded_chunk_ids == rows[path].superseded_chunk_ids
        for status in (RagStatus.PENDING, RagStatus.EMBEDDED, RagStatus.STALE, RagStatus.FAILED):
            path = f"{status.value}.md"
            assert live[path] == rows[path], f"{path} moved on restore"
        assert len(store.applied) == 1
        assert store.keys_applied() == {
            f"rag_index.{status.value}.md" for status in _IN_FLIGHT_ON_RESTORE
        }

    def test_the_requeued_rows_are_drained_by_the_next_index_pass(
        self, harness: RagHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Four ``PENDING`` rows, four workers — the stuck ones are really unstuck.

        A cluster backend, so the in-memory re-mark of ``EMBEDDED`` (51-1's, at
        child creation) does not add a fifth.
        """
        recent = datetime.now(UTC) - timedelta(seconds=1)
        rows = {f"{status.value}.md": _row(status, recent) for status in RagStatus}
        harness.actor.init_state(WorkspaceState(rag_index=dict(rows)))

        with factory_for("weaviate", lambda _context: harness.vs):
            harness.enable(collection=VectorStoreParam(backend="weaviate"))
        harness.actor.index_paths("")

        assert harness.actor._index_active == {
            "pending.md",
            "extraction.md",
            "splitting.md",
            "embedding.md",
        }


##
## Lead ruling 5 — the restored LRU is recency order, not first-insertion order
##
class TestTheRestoredCacheIsInRecencyOrder:
    """A store keeps a re-filled key where it first landed; the restore re-sorts."""

    def test_the_restored_documents_are_ordered_by_extracted_at(
        self, workspaces_root: Path
    ) -> None:
        actor = _capped_actor(max_documents=3, max_document_chars=10_000)
        base = datetime.now(UTC)

        def extract(path: str, minutes: int) -> DocumentExtract:
            return DocumentExtract(
                path=path,
                source_sha=content_sha(path.encode()),
                extractor_version=EXTRACTOR_VERSION,
                markdown=path,
                char_count=len(path),
                extracted_at=base + timedelta(minutes=minutes),
            )

        # First-insertion order b, a, c — as a store hands it back after b was
        # filled first and re-filled last.
        actor.init_state(
            WorkspaceState(
                documents={
                    "b.md": extract("b.md", 2),
                    "a.md": extract("a.md", 1),
                    "c.md": extract("c.md", 3),
                }
            )
        )

        assert list(actor.state.documents) == ["a.md", "b.md", "c.md"]
        actor.cache_document("d.md", content_sha(b"d"), EXTRACTOR_VERSION, "d")
        assert "a.md" not in actor.state.documents, "the LRU evicted by insertion order"
        assert list(actor.state.documents) == ["b.md", "c.md", "d.md"]


##
## AC 1 — one delta per persist point, member-keyed, told under the actor's name
##
_MEMBERS = ("documents", "rag_index")


def _checking_recorder(
    actor: WorkspaceActor, monkeypatch: pytest.MonkeyPatch
) -> tuple[DeltaStore, list[str]]:
    """A recorder that checks every delta against the live state **at the moment it is sent**.

    Returns the store it feeds and the list of violations it found, which a
    spec asserts empty. Checking at send time is the only way to compare a
    value with the member it was dumped from: by the end of a scenario the
    member has moved on.
    """
    store = DeltaStore()
    violations: list[str] = []

    def _send(scope: str, delta: StateDelta) -> None:
        if scope != actor.config.name:
            violations.append(f"scope {scope!r} is not {actor.config.name!r}")
        violations.extend(_shape_violations(delta))
        violations.extend(_value_violations(actor, delta))
        store.apply(WorkspaceActor, scope, delta)

    monkeypatch.setattr(actor, "_send_delta", _send)
    return store, violations


def _shape_violations(delta: StateDelta) -> list[str]:
    """AC 1 (b), (c), (e): non-empty, disjoint, and every key a ``<member>.<path>``."""
    found: list[str] = []
    if not delta.set and not delta.unset:
        found.append("an empty delta was sent")
    if set(delta.set) & set(delta.unset):
        found.append(f"set and unset overlap: {set(delta.set) & set(delta.unset)}")
    for key in [*delta.set, *delta.unset]:
        member, dot, path = key.partition(".")
        if member not in _MEMBERS or not dot or not path:
            found.append(f"{key!r} is not a member key")
    return found


def _value_violations(actor: WorkspaceActor, delta: StateDelta) -> list[str]:
    """AC 1 (d): every ``set`` is the live member's dump, every ``unset`` really gone."""
    found: list[str] = []
    for key, value in delta.set.items():
        member, _, path = key.partition(".")
        live = getattr(actor.state, member).get(path)
        if live is None or value != live.model_dump():
            found.append(f"{key!r} is not the live member's dump")
        elif not isinstance(value, dict) or value.get("path") != path:
            found.append(f"{key!r} carries a member for another path")
    for key in delta.unset:
        member, _, path = key.partition(".")
        if path in getattr(actor.state, member):
            found.append(f"{key!r} is unset but still present")
    return found


class TestEveryDeltaIsMemberKeyed:
    def test_every_delta_of_the_whole_scenario_obeys_the_shape(
        self, harness: RagHarness, workspace_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store, violations = _checking_recorder(harness.actor, monkeypatch)
        counts: list[int] = []

        _drive_every_write_site(
            harness, workspace_tree, lambda _step: counts.append(len(store.applied))
        )

        assert violations == []
        per_step = [after - before for before, after in zip([0, *counts[:-1]], counts, strict=True)]
        # One delta per persist point the step ran, never more, and none for a
        # step that wrote nothing. The enable re-marks nothing (no ``EMBEDDED``
        # row yet). The index result runs two persist points — the worker's
        # extraction is a fill, then the file's own transition — and the final
        # batch carries the two non-final batches' counters in its one delta.
        assert per_step == [1, 0, 1, 2, 1, 1, 1, 1, 1, 1]
        assert {key.partition(".")[0] for key in store.keys_applied()} == set(_MEMBERS)

    def test_the_whole_state_channel_is_silent(
        self, harness: RagHarness, workspace_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No write site calls ``notify_state_change()``: the observer sees nothing."""
        delta_recorder(harness.actor, monkeypatch)
        seen: list[object] = []

        class _Observer:
            def notify_state_change(self, state: object) -> None:
                seen.append(state)

        harness.actor.state.observer(_Observer())
        seen.clear()  # attaching an observer notifies once, by design

        _drive_every_write_site(harness, workspace_tree, lambda _step: None)

        assert seen == []

    def test_a_persist_with_nothing_dirty_sends_nothing(
        self, harness: RagHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = delta_recorder(harness.actor, monkeypatch)

        harness.actor._persist()

        assert store.applied == []


##
## AC 2 — every write site persists, one spec per row of the inventory
##
def _live_set(store: DeltaStore, actor: WorkspaceActor, path: str) -> dict[str, object]:
    """What the **last** delta naming ``rag_index.<path>`` set, checked against the live row."""
    key = f"rag_index.{path}"
    values = [delta.set[key] for _, _, delta in store.applied if key in delta.set]
    assert values, f"no delta named {key}"
    assert values[-1] == actor.state.rag_index[path].model_dump()
    return values[-1]


def _fill_the_worker_slots(actor: WorkspaceActor) -> None:
    """Occupy every index-worker slot, so a queued row stays ``PENDING``."""
    actor._index_active.update(f"busy-{n}.md" for n in range(MAX_CONCURRENT_INDEX_WORKERS))


class TestEveryWriteSitePersists:
    def test_enqueue_new_through_index_paths(
        self, harness: RagHarness, workspace_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness.enable()
        write(workspace_tree, "a.md")
        _fill_the_worker_slots(harness.actor)
        store = delta_recorder(harness.actor, monkeypatch)

        harness.actor.index_paths("")

        assert _live_set(store, harness.actor, "a.md")["status"] == RagStatus.PENDING.value

    def test_enqueue_requeue_through_index_paths(
        self, harness: RagHarness, workspace_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness.enable()
        write(workspace_tree, "a.md")
        harness.actor.index_paths("")
        harness.report("a.md")
        harness.result("a.md")
        write(workspace_tree, "a.md", "# Changed\n\nnew bytes\n")
        _fill_the_worker_slots(harness.actor)
        store = delta_recorder(harness.actor, monkeypatch)

        harness.actor.index_paths("")

        value = _live_set(store, harness.actor, "a.md")
        assert value["status"] == RagStatus.PENDING.value
        assert value["superseded_chunk_ids"], "the old chunk set was not carried to supersede"

    def test_enqueue_new_and_requeue_through_an_upload_with_retrieval_off(
        self, harness: RagHarness, workspace_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write(workspace_tree, "a.md")
        store = delta_recorder(harness.actor, monkeypatch)

        harness.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["a.md"]))
        first = _live_set(store, harness.actor, "a.md")
        write(workspace_tree, "a.md", "# Changed\n")
        harness.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["a.md"]))
        second = _live_set(store, harness.actor, "a.md")

        assert len(store.applied) == 2
        assert first["indexed_sha"] != second["indexed_sha"]

    def test_spawn_to_extraction_and_to_splitting(
        self, harness: RagHarness, workspace_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Queued and persisted first, so on the spawning pass only ``_spawn`` writes the rows.

        Queued and spawned in one pass, ``_enqueue`` would already have marked
        the path dirty and a spawn that bypassed the setter would ride along.
        """
        harness.enable()
        write(workspace_tree, "raw.md")
        cached_sha = write(workspace_tree, "cached.md")
        harness.actor.cache_document("cached.md", cached_sha, EXTRACTOR_VERSION, "# Cached\n")
        _fill_the_worker_slots(harness.actor)
        harness.actor.index_paths("")
        assert {row.status for row in harness.actor.state.rag_index.values()} == {RagStatus.PENDING}
        harness.actor._index_active.clear()
        store = delta_recorder(harness.actor, monkeypatch)

        harness.actor.index_paths("")

        assert len(store.applied) == 1
        assert _live_set(store, harness.actor, "raw.md")["status"] == RagStatus.EXTRACTION.value
        assert _live_set(store, harness.actor, "cached.md")["status"] == RagStatus.SPLITTING.value

    def test_an_index_result(
        self, harness: RagHarness, workspace_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness.enable()
        write(workspace_tree, "a.md")
        harness.actor.index_paths("")
        store = delta_recorder(harness.actor, monkeypatch)

        harness.report("a.md", chunks=2)

        value = _live_set(store, harness.actor, "a.md")
        assert value["status"] == RagStatus.EMBEDDING.value
        assert value["chunk_count"] == 2

    def test_the_final_batch_and_the_superseded_clear(
        self, harness: RagHarness, workspace_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness.enable()
        write(workspace_tree, "a.md")
        harness.actor.index_paths("")
        harness.report("a.md")
        harness.result("a.md")
        write(workspace_tree, "a.md", "# Changed\n")
        harness.actor.index_paths("")
        harness.report("a.md")
        assert harness.actor.state.rag_index["a.md"].superseded_chunk_ids
        store = delta_recorder(harness.actor, monkeypatch)

        harness.result("a.md")

        value = _live_set(store, harness.actor, "a.md")
        assert value["status"] == RagStatus.EMBEDDED.value
        assert value["superseded_chunk_ids"] == []
        assert harness.vs.of("remove"), "the superseded chunks were never removed"

    def test_fail_through_an_index_failure(
        self, harness: RagHarness, workspace_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness.enable()
        write(workspace_tree, "a.md")
        harness.actor.index_paths("")
        store = delta_recorder(harness.actor, monkeypatch)

        harness.fail("a.md", reason="unreadable")

        assert _live_set(store, harness.actor, "a.md")["status"] == RagStatus.FAILED.value

    def test_fail_through_an_embedding_error_and_a_failed_add(
        self, harness: RagHarness, workspace_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from akgentic.tool.errors import RetriableError

        harness.enable()
        write(workspace_tree, "a.md")
        write(workspace_tree, "b.md")
        harness.actor.index_paths("")
        harness.report("a.md")
        harness.report("b.md")
        store = delta_recorder(harness.actor, monkeypatch)

        harness.error("a.md", reason="rate limited")
        harness.vs.add_error = RetriableError("dead cluster")
        harness.result("b.md")

        assert _live_set(store, harness.actor, "a.md")["reason"] == "rate limited"
        assert "dead cluster" in str(_live_set(store, harness.actor, "b.md")["reason"])

    def test_reap_stale_embedding_through_index_paths(
        self, harness: RagHarness, workspace_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness.enable()
        harness.actor.state.rag_index["old.md"] = RagFile(
            path="old.md",
            status=RagStatus.EMBEDDING,
            indexed_sha="old",
            updated_at=datetime.now(UTC) - timedelta(hours=1),
        )
        _fill_the_worker_slots(harness.actor)
        store = delta_recorder(harness.actor, monkeypatch)

        harness.actor.index_paths("")

        assert _live_set(store, harness.actor, "old.md")["status"] == RagStatus.PENDING.value

    def test_requeue_embedded_rows_through_enable_rag_on_the_in_memory_engine(
        self, harness: RagHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness.actor.init_state(
            WorkspaceState(rag_index={"embedded.md": _row(RagStatus.EMBEDDED, datetime.now(UTC))})
        )
        store = delta_recorder(harness.actor, monkeypatch)

        harness.enable()

        assert len(store.applied) == 1
        assert _live_set(store, harness.actor, "embedded.md")["status"] == RagStatus.PENDING.value

    def test_mark_paths_stale(
        self, harness: RagHarness, workspace_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness.enable()
        write(workspace_tree, "a.md")
        harness.actor.index_paths("")
        harness.report("a.md")
        harness.result("a.md")
        store = delta_recorder(harness.actor, monkeypatch)

        harness.actor.mark_paths_stale(["a.md"])

        assert _live_set(store, harness.actor, "a.md")["status"] == RagStatus.STALE.value


class TestTheTwoGapsTheNotifyCallsHad:
    """Two writes the whole-state notify never carried, each closed by construction."""

    def test_a_report_whose_row_moved_on_still_persists_the_next_spawn(
        self, harness: RagHarness, workspace_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The early return frees a slot, ``_drain`` spawns the waiting file, and that persists."""
        harness.enable()
        for n in range(MAX_CONCURRENT_INDEX_WORKERS + 1):
            write(workspace_tree, f"f{n}.md")
        harness.actor.index_paths("")
        rows = harness.actor.state.rag_index
        [waiting] = [path for path, row in rows.items() if row.status is RagStatus.PENDING]
        running = next(path for path, row in rows.items() if row.status is RagStatus.EXTRACTION)
        store = delta_recorder(harness.actor, monkeypatch)

        harness.report(running, source_sha="bytes-the-row-has-moved-on-from")

        assert rows[waiting].status is RagStatus.EXTRACTION
        assert _live_set(store, harness.actor, waiting)["status"] == RagStatus.EXTRACTION.value

    def test_a_spawn_that_fails_inside_drain_persists_the_failed_row(
        self, harness: RagHarness, workspace_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing queued, the first spawn fails: ``_drain`` moved a row and said nothing moved."""
        write(workspace_tree, "a.md")
        harness.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["a.md"]))
        harness.enable()
        harness.spawn_error = RuntimeError("can't start new thread")
        store = delta_recorder(harness.actor, monkeypatch)

        answer = harness.actor.index_paths("")

        assert answer.startswith("0 file(s) queued, 1 already current")
        assert _live_set(store, harness.actor, "a.md")["status"] == RagStatus.FAILED.value


##
## AC 4 — the read side stays silent (the converted specs live beside their suites)
##
class TestTheSnapshotIsARead:
    def test_rag_snapshot_sends_nothing(
        self, harness: RagHarness, workspace_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness.enable()
        write(workspace_tree, "a.md")
        harness.actor.index_paths("")
        store = delta_recorder(harness.actor, monkeypatch)

        snapshot = harness.actor.rag_snapshot(max_pending_shown=20)

        assert [row.path for row in snapshot.rows] == ["a.md"]
        assert store.applied == []


##
## AC 5 — eviction: a set for a dropped body, an unset for a removed entry, one delta
##
class TestEvictionInTheFillsOneDelta:
    def test_a_row_cap_eviction_is_an_unset(
        self, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        actor = _capped_actor(max_documents=2, max_document_chars=10_000)
        store = delta_recorder(actor, monkeypatch)
        for name in ("a.md", "b.md", "c.md"):
            actor.cache_document(name, content_sha(name.encode()), EXTRACTOR_VERSION, name)

        _, _, last = store.applied[-1]
        assert set(last.set) == {"documents.c.md"}
        assert last.unset == ["documents.a.md"]
        assert dict(_restored(store, actor).documents) == dict(actor.state.documents)

    def test_a_char_cap_eviction_is_a_set_of_the_victim_without_its_body(
        self, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        actor = _capped_actor(max_documents=10, max_document_chars=100)
        store = delta_recorder(actor, monkeypatch)
        actor.cache_document("a.md", content_sha(b"a"), EXTRACTOR_VERSION, "a" * 60)
        actor.cache_document("b.md", content_sha(b"b"), EXTRACTOR_VERSION, "b" * 50)

        _, _, last = store.applied[-1]
        assert set(last.set) == {"documents.a.md", "documents.b.md"}
        assert last.unset == []
        assert last.set["documents.a.md"]["markdown"] is None  # type: ignore[index]
        assert dict(_restored(store, actor).documents) == dict(actor.state.documents)

    def test_a_document_over_the_char_cap_drops_its_own_body_in_its_one_set(
        self, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        actor = _capped_actor(max_documents=10, max_document_chars=100)
        store = delta_recorder(actor, monkeypatch)
        actor.cache_document("huge.md", content_sha(b"h"), EXTRACTOR_VERSION, "h" * 150)

        [(_, _, delta)] = store.applied
        assert set(delta.set) == {"documents.huge.md"}
        assert delta.set["documents.huge.md"]["markdown"] is None  # type: ignore[index]
        assert dict(_restored(store, actor).documents) == dict(actor.state.documents)


##
## AC 6 — Golden Rule 12 through the store, and the write-site canary
##
class TestAMemberSurvivesTheStoreWhole:
    def test_a_row_subclass_and_its_unknown_field_round_trip(
        self, harness: RagHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        harness.actor.state.rag_index["a.md"] = _RagFileWithExtraField(
            path="a.md", status=RagStatus.EMBEDDED, updated_at=datetime.now(UTC)
        )
        store = delta_recorder(harness.actor, monkeypatch)

        harness.actor.mark_paths_stale(["a.md"])

        [(_, _, delta)] = store.applied
        value = delta.set["rag_index.a.md"]
        assert isinstance(value, dict)
        assert value["extra_field"] == "sentinel"
        assert str(value["__model__"]).endswith("._RagFileWithExtraField")
        row = _restored(store, harness.actor).rag_index["a.md"]
        assert isinstance(row, _RagFileWithExtraField)
        assert row.extra_field == "sentinel"
        assert row.status is RagStatus.STALE

    def test_a_document_subclass_whose_body_a_fill_dropped_round_trips(
        self, workspaces_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        actor = _capped_actor(max_documents=10, max_document_chars=100)
        actor.state.documents["a.md"] = _ExtractWithExtraField(
            path="a.md",
            source_sha=content_sha(b"a"),
            extractor_version=EXTRACTOR_VERSION,
            markdown="a" * 60,
            char_count=60,
            extracted_at=datetime.now(UTC),
        )
        store = delta_recorder(actor, monkeypatch)

        actor.cache_document("b.md", content_sha(b"b"), EXTRACTOR_VERSION, "b" * 50)

        value = store.applied[-1][2].set["documents.a.md"]
        assert isinstance(value, dict)
        assert value["extra_field"] == "sentinel"
        assert value["markdown"] is None
        restored = _restored(store, actor).documents["a.md"]
        assert isinstance(restored, _ExtractWithExtraField)
        assert restored.extra_field == "sentinel"


_STATE_MAPS = frozenset({"documents", "rag_index"})
_READ_METHODS = frozenset({"get", "items", "keys", "values"})


def _map_writers(source: str) -> list[str]:
    """The function enclosing every expression that may write ``state.documents`` / ``rag_index``.

    Strict by construction: a use of the map counts as a write unless it is one
    of the known reads — ``.get`` / ``.items`` / ``.keys`` / ``.values`` called,
    a subscript load, a membership test, or iteration. A subscript store or
    delete, a mutating method, passing the map as an argument, binding it to a
    name or putting it in a container all count, so an alias cannot launder a
    write past this walk.
    """
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    writers: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and node.attr in _STATE_MAPS):
            continue
        owner = node.value
        if not (
            (isinstance(owner, ast.Name) and owner.id == "state")
            or (isinstance(owner, ast.Attribute) and owner.attr == "state")
        ):
            continue
        if _is_a_read(node, parents):
            continue
        enclosing: ast.AST | None = node
        while enclosing is not None and not isinstance(
            enclosing, ast.FunctionDef | ast.AsyncFunctionDef
        ):
            enclosing = parents.get(enclosing)
        writers.append(enclosing.name if enclosing is not None else "<module>")
    return writers


def _is_a_read(node: ast.Attribute, parents: dict[ast.AST, ast.AST]) -> bool:
    """Whether this use of a state map is one of the four known read shapes."""
    if not isinstance(node.ctx, ast.Load):
        return False
    parent = parents.get(node)
    if isinstance(parent, ast.Attribute) and parent.attr in _READ_METHODS:
        return isinstance(parents.get(parent), ast.Call)
    if isinstance(parent, ast.Subscript) and parent.value is node:
        return isinstance(parent.ctx, ast.Load)
    if isinstance(parent, ast.Compare) and node in parent.comparators:
        return True
    return isinstance(parent, ast.For | ast.comprehension) and parent.iter is node


class TestOnlyThreeFunctionsWriteTheStateMaps:
    """The persistence inventory is structural: a write outside these three is unpersisted."""

    def test_the_writers_under_workspace_actor_are_exactly_three(self) -> None:
        from akgentic.tool.workspace import actor as actor_package

        package = Path(str(actor_package.__file__)).parent
        modules = sorted(package.glob("*.py"))
        assert {module.name for module in modules} >= {"__init__.py", "documents.py"}

        writers = {
            writer
            for module in modules
            for writer in _map_writers(module.read_text(encoding="utf-8"))
        }

        assert writers == {"_put_row", "cache_document", "document_extract"}

    def test_the_walk_sees_every_shape_of_write_and_no_read(self) -> None:
        source = (
            "def reads(self, state):\n"
            "    a = self.state.rag_index.get('p')\n"
            "    b = self.state.rag_index['p']\n"
            "    c = 'p' in self.state.documents\n"
            "    for p in self.state.documents: pass\n"
            "    d = [x for x in state.rag_index.items()]\n"
            "def store(self): self.state.rag_index['p'] = 1\n"
            "def delete(self): del self.state.documents['p']\n"
            "def pop(self): self.state.documents.pop('p')\n"
            "def update(self): self.state.rag_index.update({})\n"
            "def argument(self): evict(self.state.documents)\n"
            "def alias(self): m = self.state.rag_index\n"
            "def rebind(self): self.state.documents = {}\n"
            "def augmented(self): self.state.rag_index['p'] += 1\n"
            "'''self.state.rag_index['p'] = 1 in a docstring is not a write'''\n"
        )

        assert sorted(_map_writers(source)) == sorted(
            ["store", "delete", "pop", "update", "argument", "alias", "rebind", "augmented"]
        )


##
## AC 7 — the host path, live: fill, reap, restore
##
_LIVE_PATH = "u-alice/restored"
"""The tree the live specs host — one per spec, since each spec has its own system."""


@pytest.fixture
def system(workspaces_root: Path) -> Iterator[ActorSystem]:
    """A real actor system with both hosts, torn down whatever the spec did.

    Both hosts, so a delta sent to the wrong kind's host lands in a registry
    that drops it rather than finding no host at all.
    """
    actor_system = ActorSystem()
    try:
        actor_system.createActor(
            WorkspaceHost, config=BaseConfig(name="#WorkspaceHost", role="ResourceHost")
        )
        actor_system.createActor(
            ResourceHost, config=BaseConfig(name="#ResourceHost", role="ResourceHost")
        )
        yield actor_system
    finally:
        actor_system.shutdown(timeout=10)
        ActorRegistry.stop_all()
        assert ActorSystem.find_by_class(WorkspaceActor) == [], "a workspace outlived its test"


def _host_proxy(system: ActorSystem) -> WorkspaceHost:
    [host] = ActorSystem.find_by_class(WorkspaceHost)
    return system.proxy_ask(host, WorkspaceHost)


def _reaped(address: ActorAddress, config: WorkspaceConfig) -> bool:
    """Wait for the nobody-attached actor at *address* to reap itself, within a bound."""
    budget = config.reap_grace_s + 6 * config.sweep_interval_s + HANDSHAKE_TIMEOUT_S
    return wait_until(lambda: not address.is_alive(), timeout=budget)


def _fill_and_upload(system: ActorSystem, address: ActorAddress, tree: Path) -> str:
    """One cache fill and one upload with retrieval off, through a real proxy."""
    proxy = system.proxy_ask(address, WorkspaceActor)
    sha = content_sha(b"the source bytes")
    proxy.cache_document("doc.md", sha, EXTRACTOR_VERSION, "# Body\n")
    (tree / "up.md").write_text("# Uploaded\n", encoding="utf-8")
    proxy.receiveMsg_NewFileMessage(NewFileMessage(paths=["up.md"], source="upload"))
    return sha


class TestTheHostPathLive:
    def test_a_fill_and_an_upload_survive_the_reap_into_a_new_actor(
        self, system: ActorSystem, workspaces_root: Path
    ) -> None:
        store = DeltaStore()
        _host_proxy(system).register_store(store)
        config = fast_config(_LIVE_PATH)
        address = _host_proxy(system).getResourceOrCreate(WorkspaceActor, config)
        tree = workspace_root_for(workspaces_root, "restored")

        sha = _fill_and_upload(system, address, tree)

        wanted = {"documents.doc.md", "rag_index.up.md"}
        assert wait_until(lambda: wanted <= store.keys_applied(config.name)), (
            f"the host applied only {store.keys_applied(config.name)}"
        )
        assert {cls for cls, _, _ in store.applied} == {WorkspaceActor}
        assert _reaped(address, config), "nobody attached, and the tree never reaped"

        renewed = _host_proxy(system).getResourceOrCreate(WorkspaceActor, config)

        assert renewed.agent_id != address.agent_id, "the host answered the dead actor"
        restored = system.proxy_ask(renewed, WorkspaceActor)
        assert restored.document_extract("doc.md", sha, EXTRACTOR_VERSION) == "# Body\n"
        rows = restored.rag_snapshot(20).rows
        assert [(row.path, row.status) for row in rows] == [("up.md", RagStatus.PENDING.value)]

    def test_a_cold_host_persists_nothing_and_raises_nothing(
        self, system: ActorSystem, workspaces_root: Path
    ) -> None:
        config = fast_config(_LIVE_PATH)
        address = _host_proxy(system).getResourceOrCreate(WorkspaceActor, config)
        tree = workspace_root_for(workspaces_root, "restored")

        sha = _fill_and_upload(system, address, tree)

        assert address.is_alive(), "a fill on a cold host stopped the actor"
        assert _reaped(address, config), "nobody attached, and the tree never reaped"
        renewed = _host_proxy(system).getResourceOrCreate(WorkspaceActor, config)
        assert renewed.agent_id != address.agent_id
        restored = system.proxy_ask(renewed, WorkspaceActor)
        assert restored.document_extract("doc.md", sha, EXTRACTOR_VERSION) is None


class TestNoHostAtAll:
    def test_every_persist_point_is_a_silent_no_op_without_a_host(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """The inert harness runs no ``WorkspaceHost``; the whole scenario must still run."""
        assert workspace_host_address() is None, "a host is running — nothing is tested"

        _drive_every_write_site(harness, workspace_tree, lambda _step: None)

        assert harness.actor.state.rag_index["a.md"].status is RagStatus.STALE
