"""Story 52-3: the workspace persists what it knows about a document to **disk**.

The successor to 51-4's delta suite. Every change to a document's cached
extraction or its retrieval row is one write of one file under ``<meta>/rag/``,
performed on the turn that made it — there is no dirty set, no persist point and
no host in the path at all. What used to be asserted by *restoring* from a
recorded delta stream is asserted here by **reading the record back through a
second store object**, which is a strictly stronger claim: a delta could be
counted without anything reaching a disk, and a file that a second object can
read cannot have stayed in memory.

The live specs at the bottom keep the shape 51-4 gave them — a real
``ActorSystem``, a real stop and a second actor over the same tree — and their
claim is now simply that the second actor reads what the first one wrote, with
**nothing registered anywhere** to carry it.

**What went with the deltas, and why it is not a loss.** The classes that
asserted a delta's *shape* (member-keyed, one per persist point, never the bare
mapping) guarded a channel that no longer exists; they are removed rather than
weakened. The classes that asserted something about the workspace — every write
site lands, the reaper covers what a dead worker carried, recency governs
eviction, a subclass survives the round trip, only the known functions write —
are all re-pointed below, because deleting a test that guarded an invariant takes
the invariant with it.

(Renamed from ``test_state_deltas.py`` in 52-6, the vocabulary sweep 52-3
deferred to it: what it guards has not been a delta since 52-3, and a file named
after a channel that no longer exists sends the next reader looking for one.)
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

from akgentic.tool.vector_store.protocol import VectorStoreParam
from akgentic.tool.workspace.actor import (
    WORKSPACE_ACTOR_ROLE,
    WorkspaceActor,
    workspace_actor_name,
)
from akgentic.tool.workspace.documents.models import (
    EMBEDDING_STALE_AFTER_S,
    EXTRACTOR_VERSION,
    DocumentExtract,
    NewFileMessage,
    RagChunk,
    RagFile,
    RagStatus,
)
from akgentic.tool.workspace.documents.store import DocumentEntry, YamlDocumentStore
from akgentic.tool.workspace.documents.worker import (
    EMBED_BATCH_SIZE,
    MAX_CONCURRENT_INDEX_WORKERS,
)
from akgentic.tool.workspace.models import (
    WorkspaceConfig,
    content_sha,
)
from tests.workspace.conftest import (
    HANDSHAKE_TIMEOUT_S,
    WORKSPACE_PATH,
    attach_store,
    factory_for,
    seed_extract,
    seed_row,
    stored_docs,
    stored_entries,
    stored_rows,
    wait_until,
    watch_store,
    workspace_config,
    workspace_root_for,
)
from tests.workspace.test_document_cache import _ExtractWithExtraField
from tests.workspace.test_rag_models import _RagFileWithExtraField
from tests.workspace.test_rag_pipeline import RagHarness, write


class _EntryWithExtraField(DocumentEntry):
    """A stored record carrying a field the write path has never heard of.

    **Module level, not method level, and that is load-bearing**: the serializer
    stamps ``__model__ = "<module>.<name>"`` and resolves it with
    ``import_module`` plus ``getattr``, so a class defined inside a test function
    is unresolvable and its record comes back as a parse failure rather than as
    itself.
    """

    extra_field: str = "sentinel"


class _CapturingStore:
    """A store that keeps the entry objects it was handed, in memory and whole.

    The only way to ask what a writer *built*, as opposed to what survived being
    written: a YAML round trip normalises a subclass back to ``DocumentEntry``,
    so a read-back can never answer the copy-semantics question.
    """

    def __init__(self) -> None:
        self._held: dict[tuple[str, str], DocumentEntry] = {}
        self.written: list[DocumentEntry] = []

    def seed(self, tree_key: str, entry: DocumentEntry) -> None:
        """Place *entry* as though a previous turn had written it."""
        self._held[(tree_key, entry.path)] = entry

    def get_document(self, tree_key: str, path: str) -> DocumentEntry | None:
        return self._held.get((tree_key, path))

    def put_document(self, tree_key: str, entry: DocumentEntry) -> None:
        self.written.append(entry)
        self._held[(tree_key, entry.path)] = entry

    def evict(self, tree_key: str, path: str) -> None:
        self._held.pop((tree_key, path), None)

    def list_documents(self, tree_key: str) -> list[DocumentEntry]:
        return [entry for (key, _), entry in self._held.items() if key == tree_key]


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
    # The card announces this at bind time; a directly built actor gets none.
    return attach_store(actor)


@pytest.fixture
def harness(workspace_tree: Path, monkeypatch: pytest.MonkeyPatch) -> RagHarness:
    """A harness installed on a capped inert actor, with retrieval not yet on."""
    built = RagHarness(_capped_actor())
    built.install(monkeypatch)
    return built


def _on_disk(actor: WorkspaceActor, path: str) -> DocumentEntry:
    """*path*'s record, read through a **second** store object. Never the actor's own."""
    entry = stored_entries(actor).get(path)
    assert entry is not None, f"nothing on disk for {path}"
    return entry


def _row_on_disk(actor: WorkspaceActor, path: str) -> RagFile:
    """*path*'s index row, read back off the disk."""
    row = _on_disk(actor, path).row
    assert row is not None, f"{path} has a record but no row"
    return row


def _drive_every_write_site(harness: RagHarness, tree: Path, checkpoint: object) -> None:
    """Upload, enable, index, embed in three batches, fail, mark stale, and evict by both caps.

    **The most valuable thing in this module**, and the reason it survives the
    delta machinery it was written for: it is the one place that reaches every
    write site the document pipeline has, in the order a real tree reaches them.

    *checkpoint* is called with the step's name after every step — including the
    non-final embedding batches, which is the one change from 51-4's version.
    Those were the single step that wrote a row and deliberately persisted
    nothing; with no dirty set they now write like every other.
    """
    assert callable(checkpoint)
    actor = harness.actor
    write(tree, "a.md", "# A\n\nbody\n")
    write(tree, "b.md", "# B\n\nother\n")

    actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["a.md", "b.md"], source="upload"))
    assert _row_on_disk(actor, "a.md").status is RagStatus.PENDING
    checkpoint("an upload with retrieval off")

    harness.enable()
    checkpoint("enable_rag on the in-memory backend")

    actor.index_paths("")
    assert _row_on_disk(actor, "a.md").status is RagStatus.EXTRACTION
    checkpoint("index_paths")

    harness.report("a.md", chunks=EMBED_BATCH_SIZE * 2 + 1, markdown=_A_MARKDOWN, extracted=True)
    assert _row_on_disk(actor, "a.md").batches_expected == 3, "not a three-batch file"
    checkpoint("an index result with three batches")

    harness.result("a.md")
    checkpoint("a non-final embedding batch")
    harness.result("a.md")
    assert _row_on_disk(actor, "a.md").batches_landed == 2
    checkpoint("a second non-final embedding batch")

    harness.result("a.md")
    assert _row_on_disk(actor, "a.md").status is RagStatus.EMBEDDED
    checkpoint("the final batch")

    harness.fail("b.md")
    assert _row_on_disk(actor, "b.md").status is RagStatus.FAILED
    checkpoint("an index failure")

    # The gate is the card's since 52-5, and what reaches the actor is the
    # stale-mark rather than the write — which is exactly the seam this spec
    # cares about: the index row has to be on disk by the end of the turn.
    actor.mark_paths_stale(["a.md"])
    assert _row_on_disk(actor, "a.md").status is RagStatus.STALE
    checkpoint("a gate write")

    actor.cache_document("c.md", content_sha(b"c"), EXTRACTOR_VERSION, "c" * 60)
    checkpoint("a fill under both caps")
    actor.cache_document("d.md", content_sha(b"d"), EXTRACTOR_VERSION, "d" * 50)
    assert stored_docs(actor)["a.md"].markdown is None, "the char cap dropped no body"
    assert stored_docs(actor)["c.md"].markdown is None, "the char cap dropped one body"
    checkpoint("a fill that drops two bodies by the char cap")
    actor.cache_document("e.md", content_sha(b"e"), EXTRACTOR_VERSION, "e" * 10)
    # ``a.md`` keeps its record because it keeps its **row** — an eviction may
    # never de-index a file — but its extraction half is gone.
    assert "a.md" not in stored_docs(actor), "the row cap dropped no extraction"
    assert _on_disk(actor, "a.md").row is not None, "the row cap de-indexed a file"
    checkpoint("a fill that removes an extraction by the row cap")


##
## AC 13 — every write site is on disk when the turn that made it ends
##
class TestEveryWriteSiteIsOnDiskWhenTheTurnEnds:
    """The successor to ``TestEveryWriteSitePersists``, asked of the disk.

    Every step of the driver is checked immediately after it runs, through a
    **second** store object: what the write said is what a different reader can
    already see. Under the delta suite this had to wait for a persist point; it
    does not any more, and the checkpoint is what pins that.
    """

    def test_every_step_is_readable_by_another_object_the_moment_it_ends(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        seen: list[str] = []

        def checkpoint(step: str) -> None:
            seen.append(str(step))
            for path, entry in stored_entries(harness.actor).items():
                # Read back whole, through an object the actor has never
                # touched: a record that did not reach the disk is not here, and
                # one that reached it half-written does not parse.
                assert entry.path == path
                assert entry.extract is not None or entry.row is not None, (
                    f"{step}: {path} is an empty record"
                )

        _drive_every_write_site(harness, workspace_tree, checkpoint)

        assert len(seen) == 12, f"the driver stopped reaching every write site: {seen}"

    def test_a_non_final_embedding_batch_is_on_disk_too(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """**Deliberately inverted from 51-4**, which pinned the opposite.

        A batch that landed without settling its file used to write ``batches_landed``
        into memory and send nothing, so a crash before the next persist point lost
        the counter and the file waited out the reaper. It writes now.
        """
        harness.enable()
        write(workspace_tree, "a.md")
        harness.actor.index_paths("")
        harness.report("a.md", chunks=EMBED_BATCH_SIZE * 2 + 1)

        harness.result("a.md")

        assert _row_on_disk(harness.actor, "a.md").batches_landed == 1
        assert _row_on_disk(harness.actor, "a.md").status is RagStatus.EMBEDDING

    def test_enqueue_new_through_index_paths(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        harness.enable()
        write(workspace_tree, "a.md")
        _fill_the_worker_slots(harness.actor)

        harness.actor.index_paths("")

        assert _row_on_disk(harness.actor, "a.md").status is RagStatus.PENDING

    def test_enqueue_requeue_through_index_paths(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        harness.enable()
        write(workspace_tree, "a.md")
        harness.actor.index_paths("")
        harness.report("a.md")
        harness.result("a.md")
        write(workspace_tree, "a.md", "# Changed\n\nnew bytes\n")
        _fill_the_worker_slots(harness.actor)

        harness.actor.index_paths("")

        row = _row_on_disk(harness.actor, "a.md")
        assert row.status is RagStatus.PENDING
        assert row.superseded_chunk_ids, "the old chunk set was not carried to supersede"

    def test_enqueue_new_and_requeue_through_an_upload_with_retrieval_off(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        write(workspace_tree, "a.md")

        harness.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["a.md"]))
        first = _row_on_disk(harness.actor, "a.md").indexed_sha
        write(workspace_tree, "a.md", "# Changed\n")
        harness.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["a.md"]))
        second = _row_on_disk(harness.actor, "a.md").indexed_sha

        assert first != second

    def test_spawn_to_extraction_and_to_splitting(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """The status says which half of the work the worker actually has to do."""
        harness.enable()
        write(workspace_tree, "raw.md")
        cached_sha = write(workspace_tree, "cached.md")
        harness.actor.cache_document("cached.md", cached_sha, EXTRACTOR_VERSION, "# Cached\n")
        _fill_the_worker_slots(harness.actor)
        harness.actor.index_paths("")
        assert {row.status for row in stored_rows(harness.actor).values()} == {RagStatus.PENDING}
        harness.actor._index_active.clear()

        harness.actor.index_paths("")

        assert _row_on_disk(harness.actor, "raw.md").status is RagStatus.EXTRACTION
        assert _row_on_disk(harness.actor, "cached.md").status is RagStatus.SPLITTING

    def test_an_index_result(self, harness: RagHarness, workspace_tree: Path) -> None:
        harness.enable()
        write(workspace_tree, "a.md")
        harness.actor.index_paths("")

        harness.report("a.md", chunks=2)

        row = _row_on_disk(harness.actor, "a.md")
        assert row.status is RagStatus.EMBEDDING
        assert row.chunk_count == 2

    def test_the_final_batch_and_the_superseded_clear(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        harness.enable()
        write(workspace_tree, "a.md")
        harness.actor.index_paths("")
        harness.report("a.md")
        harness.result("a.md")
        write(workspace_tree, "a.md", "# Changed\n")
        harness.actor.index_paths("")
        harness.report("a.md")
        assert _row_on_disk(harness.actor, "a.md").superseded_chunk_ids

        harness.result("a.md")

        row = _row_on_disk(harness.actor, "a.md")
        assert row.status is RagStatus.EMBEDDED
        assert row.superseded_chunk_ids == []
        assert harness.vs.of("remove"), "the superseded chunks were never removed"

    def test_fail_through_an_index_failure(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        harness.enable()
        write(workspace_tree, "a.md")
        harness.actor.index_paths("")

        harness.fail("a.md", reason="unreadable")

        assert _row_on_disk(harness.actor, "a.md").status is RagStatus.FAILED

    def test_fail_through_an_embedding_error_and_a_failed_add(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        from akgentic.tool.errors import RetriableError

        harness.enable()
        write(workspace_tree, "a.md")
        write(workspace_tree, "b.md")
        harness.actor.index_paths("")
        harness.report("a.md")
        harness.report("b.md")

        harness.error("a.md", reason="rate limited")
        harness.vs.add_error = RetriableError("dead cluster")
        harness.result("b.md")

        assert _row_on_disk(harness.actor, "a.md").reason == "rate limited"
        assert "dead cluster" in (_row_on_disk(harness.actor, "b.md").reason or "")

    def test_the_reaper_through_index_paths(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        harness.enable()
        seed_row(harness.actor, "old.md", RagFile(
            path="old.md",
            status=RagStatus.EMBEDDING,
            indexed_sha="old",
            updated_at=datetime.now(UTC) - timedelta(hours=1),
        ))
        _fill_the_worker_slots(harness.actor)

        harness.actor.index_paths("")

        assert _row_on_disk(harness.actor, "old.md").status is RagStatus.PENDING

    def test_mark_paths_stale(self, harness: RagHarness, workspace_tree: Path) -> None:
        harness.enable()
        write(workspace_tree, "a.md")
        harness.actor.index_paths("")
        harness.report("a.md")
        harness.result("a.md")

        harness.actor.mark_paths_stale(["a.md"])

        assert _row_on_disk(harness.actor, "a.md").status is RagStatus.STALE


class TestTheTwoGapsTheNotifyCallsHad:
    """Two writes the whole-state notify never carried, each still closed by construction."""

    def test_a_report_whose_row_moved_on_still_writes_the_next_spawn(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """The early return frees a slot, ``_drain`` spawns the waiting file, and that lands."""
        harness.enable()
        for n in range(MAX_CONCURRENT_INDEX_WORKERS + 1):
            write(workspace_tree, f"f{n}.md")
        harness.actor.index_paths("")
        rows = stored_rows(harness.actor)
        [waiting] = [path for path, row in rows.items() if row.status is RagStatus.PENDING]
        running = next(path for path, row in rows.items() if row.status is RagStatus.EXTRACTION)

        harness.report(running, source_sha="bytes-the-row-has-moved-on-from")

        assert _row_on_disk(harness.actor, waiting).status is RagStatus.EXTRACTION

    def test_a_spawn_that_fails_inside_drain_writes_the_failed_row(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """Nothing queued, the first spawn fails: ``_drain`` moved a row and said nothing moved."""
        write(workspace_tree, "a.md")
        harness.actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["a.md"]))
        harness.enable()
        harness.spawn_error = RuntimeError("can't start new thread")

        answer = harness.actor.index_paths("")

        assert answer.startswith("0 file(s) queued, 1 already current")
        assert _row_on_disk(harness.actor, "a.md").status is RagStatus.FAILED


class TestTheSnapshotIsARead:
    def test_rag_snapshot_writes_nothing(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        harness.enable()
        write(workspace_tree, "a.md")
        harness.actor.index_paths("")
        writes = watch_store(harness.actor)

        snapshot = harness.actor.rag_snapshot(max_pending_shown=20)

        assert [row.path for row in snapshot.rows] == ["a.md"]
        assert writes.puts == []
        assert writes.evicted == []


##
## AC 16 — one reaper covers every abandoned in-flight row, in any process
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


_ABANDONED = (RagStatus.EXTRACTION, RagStatus.SPLITTING, RagStatus.EMBEDDING)
_UNTOUCHED = (RagStatus.PENDING, RagStatus.EMBEDDED, RagStatus.STALE, RagStatus.FAILED)


def _fill_the_worker_slots(actor: WorkspaceActor) -> None:
    """Occupy every index-worker slot, so a queued row stays ``PENDING``."""
    actor._index_active.update(f"busy-{n}.md" for n in range(MAX_CONCURRENT_INDEX_WORKERS))


class TestTheReaperCoversEveryAbandonedRow:
    """The successor to ``TestARestoreRequeuesWhatADeadWorkerCarried``.

    The invariant it guarded is real and survives: **a row a dead worker was
    carrying must be queued again, or nothing will ever move it** — ``_drain``
    spawns ``PENDING`` only and ``_is_accounted_for`` counts in-flight as
    current. What changed is that there is no longer a moment called "restore"
    at which every such row is abandoned by construction, because another
    process may be working on one right now. ``_index_active`` is the
    discriminator and the age bound is the other half.
    """

    def test_every_abandoned_in_flight_row_goes_back_to_pending(
        self, harness: RagHarness
    ) -> None:
        stale = datetime.now(UTC) - timedelta(seconds=EMBEDDING_STALE_AFTER_S + 1)
        rows = {f"{status.value}.md": _row(status, stale) for status in RagStatus}
        for path, row in rows.items():
            seed_row(harness.actor, path, row)

        assert harness.actor.reap_abandoned_rows() is True

        live = stored_rows(harness.actor)
        for status in _ABANDONED:
            path = f"{status.value}.md"
            assert live[path].status is RagStatus.PENDING, f"{path} is still {live[path].status}"
            assert (live[path].batches_expected, live[path].batches_landed) == (0, 0)
            # Kept, both of them: the superseded ids are still owed a removal,
            # and the chunk set keeps the row's heading paths renderable.
            assert live[path].chunks == rows[path].chunks
            assert live[path].superseded_chunk_ids == rows[path].superseded_chunk_ids
        for status in _UNTOUCHED:
            path = f"{status.value}.md"
            assert live[path] == rows[path], f"{path} moved when it should not have"

    def test_a_row_a_live_worker_is_carrying_is_never_reaped(self, harness: RagHarness) -> None:
        """``_index_active`` is per-process by construction: a path in it is live **here**."""
        ancient = datetime.now(UTC) - timedelta(days=7)
        seed_row(harness.actor, "extraction.md", _row(RagStatus.EXTRACTION, ancient))
        harness.actor._index_active.add("extraction.md")

        assert harness.actor.reap_abandoned_rows() is False

        assert _row_on_disk(harness.actor, "extraction.md").status is RagStatus.EXTRACTION

    def test_a_row_inside_the_bound_is_not_reaped(self, harness: RagHarness) -> None:
        """The near side of the bound — a predicate with the comparison inverted fails here."""
        recent = datetime.now(UTC) - timedelta(seconds=1)
        for status in _ABANDONED:
            seed_row(harness.actor, f"{status.value}.md", _row(status, recent))

        assert harness.actor.reap_abandoned_rows() is False

        live = stored_rows(harness.actor)
        for status in _ABANDONED:
            assert live[f"{status.value}.md"].status is status

    def test_a_row_past_the_bound_is_reaped(self, harness: RagHarness) -> None:
        """The far side of the same bound, so neither direction can pass alone."""
        past = datetime.now(UTC) - timedelta(seconds=EMBEDDING_STALE_AFTER_S + 1)
        for status in _ABANDONED:
            seed_row(harness.actor, f"{status.value}.md", _row(status, past))

        assert harness.actor.reap_abandoned_rows() is True

        live = stored_rows(harness.actor)
        for status in _ABANDONED:
            assert live[f"{status.value}.md"].status is RagStatus.PENDING

    def test_the_requeued_rows_are_drained_by_the_next_index_pass(
        self, harness: RagHarness
    ) -> None:
        """Four ``PENDING`` rows, four workers — the stuck ones are really unstuck.

        A cluster backend, so nothing about the in-memory branch is involved.
        """
        past = datetime.now(UTC) - timedelta(seconds=EMBEDDING_STALE_AFTER_S + 1)
        for status in RagStatus:
            seed_row(harness.actor, f"{status.value}.md", _row(status, past))

        with factory_for("weaviate", lambda _context: harness.vs):
            harness.enable(collection=VectorStoreParam(backend="weaviate"))
        harness.actor.index_paths("")

        assert harness.actor._index_active == {
            "pending.md",
            "extraction.md",
            "splitting.md",
            "embedding.md",
        }

    def test_a_subclass_row_survives_the_re_queue_whole(self, harness: RagHarness) -> None:
        """Golden Rule 12: a copy-and-override, never a rebuild naming today's fields."""
        past = datetime.now(UTC) - timedelta(seconds=EMBEDDING_STALE_AFTER_S + 1)
        seed_row(harness.actor, "a.md", _RagFileWithExtraField(
            path="a.md", status=RagStatus.EMBEDDING, updated_at=past
        ))

        assert harness.actor.reap_abandoned_rows() is True

        row = _row_on_disk(harness.actor, "a.md")
        assert isinstance(row, _RagFileWithExtraField)
        assert row.extra_field == "sentinel"
        assert row.status is RagStatus.PENDING


##
## AC 14 — eviction sorts by ``extracted_at``, and never de-indexes a file
##
class TestEvictionOrdersByExtractedAt:
    """The successor to ``TestTheRestoredCacheIsInRecencyOrder``.

    Its claim was that a store hands a cache back in first-insertion order and
    the restore has to re-sort it by ``extracted_at`` or the LRU evicts the wrong
    entry. There is no restore and no insertion order any more — a directory
    glob's order is the file system's — so the same claim now applies at the one
    place recency is consumed: the eviction pass.
    """

    def _base(self) -> datetime:
        """An hour ago, so every seeded record is genuinely older than the next fill.

        Not ``now``: a seed stamped in the future would make the document the
        spec then fills the *oldest*, and the eviction would be correct while
        the spec was wrong.
        """
        return datetime.now(UTC) - timedelta(hours=1)

    def _extract(self, path: str, base: datetime, minutes: int) -> DocumentExtract:
        return DocumentExtract(
            path=path,
            source_sha=content_sha(path.encode()),
            extractor_version=EXTRACTOR_VERSION,
            markdown=path,
            char_count=len(path),
            extracted_at=base + timedelta(minutes=minutes),
        )

    def test_the_row_cap_drops_the_least_recently_extracted(
        self, workspaces_root: Path
    ) -> None:
        # **The names are chosen so that recency and alphabetical order
        # disagree**, and that is the whole point of the spec. With ``a.md``
        # oldest, a pass that sorted by path — or by the digest the file is
        # named after — reaches the same answer by luck, and the guard passes
        # against an implementation that has no idea what recency is. Here the
        # oldest extraction is ``z.md`` and the newest seeded one is ``a.md``.
        actor = _capped_actor(max_documents=3, max_document_chars=10_000)
        base = self._base()
        for name, minutes in (("m.md", 2), ("z.md", 1), ("a.md", 3)):
            seed_extract(actor, name, self._extract(name, base, minutes))

        actor.cache_document("d.md", content_sha(b"d"), EXTRACTOR_VERSION, "d")

        assert "z.md" not in stored_docs(actor), "the oldest extraction survived"
        assert set(stored_docs(actor)) == {"a.md", "m.md", "d.md"}

    def test_the_char_cap_drops_the_least_recently_extracted_body(
        self, workspaces_root: Path
    ) -> None:
        # Same trap, same defence: ``z.md`` is the oldest body and ``a.md`` the
        # newest, so a pass ordering by path or by digest drops the wrong one.
        actor = _capped_actor(max_documents=10, max_document_chars=100)
        base = self._base()
        for name, minutes in (("a.md", 2), ("z.md", 1)):
            seed_extract(actor, name, self._extract(name, base, minutes).model_copy(
                update={"markdown": name[0] * 60, "char_count": 60}
            ))

        actor.cache_document("c.md", content_sha(b"c"), EXTRACTOR_VERSION, "c")

        docs = stored_docs(actor)
        assert docs["z.md"].markdown is None, "the oldest body survived"
        assert docs["z.md"].char_count == 60, "the dropped body lost its metadata"
        assert docs["a.md"].markdown == "a" * 60, "a newer body was dropped instead"

    def test_a_row_cap_eviction_keeps_the_index_row(self, workspaces_root: Path) -> None:
        """An eviction must never de-index a file — the caps bound the cache only."""
        actor = _capped_actor(max_documents=1, max_document_chars=10_000)
        base = self._base()
        seed_extract(actor, "a.md", self._extract("a.md", base, 1))
        seed_row(actor, "a.md", RagFile(
            path="a.md", status=RagStatus.EMBEDDED, indexed_sha="x", updated_at=base
        ))

        actor.cache_document("b.md", content_sha(b"b"), EXTRACTOR_VERSION, "b")

        survivor = _on_disk(actor, "a.md")
        assert survivor.extract is None
        assert survivor.row is not None and survivor.row.status is RagStatus.EMBEDDED

    def test_a_char_cap_eviction_keeps_the_index_row(self, workspaces_root: Path) -> None:
        actor = _capped_actor(max_documents=10, max_document_chars=100)
        base = self._base()
        seed_extract(actor, "a.md", self._extract("a.md", base, 1).model_copy(
            update={"markdown": "a" * 60, "char_count": 60}
        ))
        seed_row(actor, "a.md", RagFile(
            path="a.md", status=RagStatus.EMBEDDED, indexed_sha="x", updated_at=base
        ))

        actor.cache_document("b.md", content_sha(b"b"), EXTRACTOR_VERSION, "b" * 60)

        survivor = _on_disk(actor, "a.md")
        assert survivor.extract is not None and survivor.extract.markdown is None
        assert survivor.row is not None and survivor.row.status is RagStatus.EMBEDDED

    def test_a_document_over_the_char_cap_drops_its_own_body(
        self, workspaces_root: Path
    ) -> None:
        actor = _capped_actor(max_documents=10, max_document_chars=100)

        actor.cache_document("huge.md", content_sha(b"h"), EXTRACTOR_VERSION, "h" * 150)

        assert stored_docs(actor)["huge.md"].markdown is None
        assert stored_docs(actor)["huge.md"].char_count == 150


##
## Golden Rule 12 through the store — a subclass survives the YAML round trip
##
class TestAMemberSurvivesTheStoreWhole:
    """What ``__model__`` bought through a delta, it must still buy through YAML.

    ``SerializableBaseModel`` stamps the concrete class into the dump, so a
    subclass carrying a field the write path has never heard of comes back as
    itself. A dump built by naming today's fields would not — which is the defect
    Golden Rule 12 exists for, one layer below the ``model_copy`` calls.
    """

    def test_a_row_subclass_and_its_unknown_field_round_trip(
        self, harness: RagHarness
    ) -> None:
        seed_row(harness.actor, "a.md", _RagFileWithExtraField(
            path="a.md", status=RagStatus.EMBEDDED, updated_at=datetime.now(UTC)
        ))

        harness.actor.mark_paths_stale(["a.md"])

        row = _row_on_disk(harness.actor, "a.md")
        assert isinstance(row, _RagFileWithExtraField)
        assert row.extra_field == "sentinel"
        assert row.status is RagStatus.STALE

    def test_a_document_subclass_whose_body_a_fill_dropped_round_trips(
        self, workspaces_root: Path
    ) -> None:
        actor = _capped_actor(max_documents=10, max_document_chars=100)
        seed_extract(actor, "a.md", _ExtractWithExtraField(
            path="a.md",
            source_sha=content_sha(b"a"),
            extractor_version=EXTRACTOR_VERSION,
            markdown="a" * 60,
            char_count=60,
            extracted_at=datetime.now(UTC),
        ))

        actor.cache_document("b.md", content_sha(b"b"), EXTRACTOR_VERSION, "b" * 50)

        restored = stored_docs(actor)["a.md"]
        assert isinstance(restored, _ExtractWithExtraField)
        assert restored.extra_field == "sentinel"
        assert restored.markdown is None

    def test_put_row_preserves_a_record_field_the_write_path_never_heard_of(
        self, workspaces_root: Path
    ) -> None:
        """Golden Rule 12 on ``DocumentEntry`` itself, not only on its two halves.

        **Proved necessary by mutation.** Replacing ``_put_row``'s
        ``model_copy(update=...)`` with ``DocumentEntry(path=..., extract=...,
        row=...)`` — which names every field that exists today, and is therefore
        what a developer would actually write — killed no spec at all. The two
        subclass guards above did not catch it: the row and the extract are
        passed through by value and survive either way, so what an enumerated
        rebuild destroys is a field of the **record**, which nothing was
        watching.

        It is asserted on the entry handed to the store rather than on one read
        back, and that is deliberate rather than a convenience: a *subclass* of
        ``DocumentEntry`` does not survive the YAML round trip today — ``_read``
        validates into ``DocumentEntry`` and pydantic refuses the subclass
        instance the tag resolves to — so a read-back formulation would fail for
        a reason that has nothing to do with the rule. Recorded in the backlog;
        the copy semantics are what this spec is for.
        """
        actor = _capped_actor()
        captured = _CapturingStore()
        actor.configure_document_store(captured)
        captured.seed(WORKSPACE_PATH, _EntryWithExtraField(path="a.md"))

        actor._put_row("a.md", RagFile(
            path="a.md", status=RagStatus.PENDING, updated_at=datetime.now(UTC)
        ))

        [written] = captured.written
        assert isinstance(written, _EntryWithExtraField), "an enumerated rebuild dropped the field"
        assert written.extra_field == "sentinel"
        assert written.row is not None and written.row.status is RagStatus.PENDING

    def test_cache_document_preserves_a_record_field_the_write_path_never_heard_of(
        self, workspaces_root: Path
    ) -> None:
        """The same rule at the other writer, for the same reason."""
        actor = _capped_actor()
        captured = _CapturingStore()
        actor.configure_document_store(captured)
        captured.seed(WORKSPACE_PATH, _EntryWithExtraField(path="a.md"))

        actor.cache_document("a.md", content_sha(b"a"), EXTRACTOR_VERSION, "# body")

        written = captured.written[0]
        assert isinstance(written, _EntryWithExtraField), "an enumerated rebuild dropped the field"
        assert written.extra_field == "sentinel"
        assert written.extract is not None and written.extract.markdown == "# body"

    def test_a_whole_record_round_trips_through_the_yaml_encoding(
        self, workspaces_root: Path
    ) -> None:
        """Both halves at once, since one file carries both."""
        actor = _capped_actor()
        seed_row(actor, "a.md", _RagFileWithExtraField(
            path="a.md", status=RagStatus.EMBEDDED, updated_at=datetime.now(UTC)
        ))
        seed_extract(actor, "a.md", _ExtractWithExtraField(
            path="a.md",
            source_sha=content_sha(b"a"),
            extractor_version=EXTRACTOR_VERSION,
            markdown="# body",
            char_count=6,
            extracted_at=datetime.now(UTC),
        ))

        entry = YamlDocumentStore().get_document(WORKSPACE_PATH, "a.md")

        assert entry is not None
        assert isinstance(entry.row, _RagFileWithExtraField)
        assert isinstance(entry.extract, _ExtractWithExtraField)


##
## The write-site canary, one layer down: who is allowed to write a record
##
_WRITE_HELPERS = frozenset({"_save", "put_document", "evict"})
"""Every expression that can put a record on disk from inside the actor package.

``_save`` is the mixin's own one-line wrapper; the other two are the store's own
methods, named so that a site reaching past ``_save`` straight to
``self._document_store`` is caught rather than laundered.
"""


def _record_writers(source: str) -> list[str]:
    """The function enclosing every call that may write a document record.

    The successor to 51-4's ``_map_writers``. That walk asked which functions
    wrote the two state maps; this asks which functions write the file those maps
    became. It is the same structural question — **the write inventory is a
    property of the code, not something a reviewer has to remember** — and it is
    strictly narrower, because a write now has exactly one shape: a call.

    Reads are not writes and are deliberately not counted: ``get_document`` and
    ``list_documents`` may be called from anywhere.
    """
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    writers: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and node.attr in _WRITE_HELPERS):
            continue
        if not isinstance(parents.get(node), ast.Call):
            continue
        enclosing: ast.AST | None = node
        while enclosing is not None and not isinstance(
            enclosing, ast.FunctionDef | ast.AsyncFunctionDef
        ):
            enclosing = parents.get(enclosing)
        writers.append(enclosing.name if enclosing is not None else "<module>")
    return writers


class TestOnlyTheKnownFunctionsWriteARecord:
    """The write inventory is structural: a write outside these four is unaccounted for.

    Two of them are the story's two writers proper — ``_put_row`` for a row and
    ``cache_document`` for an extraction. Two more are the eviction pass and the
    helper it removes through, which write only what ``evict_document_bodies``
    has already decided. The fifth is ``_save`` itself, the one-line wrapper the
    other four go through: it is here because the walk sees the call inside its
    own body, and leaving it out would mean a spec that agreed with the code by
    exception rather than by rule.
    """

    def test_the_record_writers_under_workspace_actor_are_exactly_five(self) -> None:
        from akgentic.tool.workspace import actor as actor_package

        package = Path(str(actor_package.__file__)).parent
        modules = sorted(package.glob("*.py"))
        assert {module.name for module in modules} >= {"__init__.py", "documents.py"}

        writers = {
            writer
            for module in modules
            for writer in _record_writers(module.read_text(encoding="utf-8"))
        }

        assert writers == {
            "_save",  # the one-line wrapper the other four go through
            "_put_row",
            "cache_document",
            "_apply_document_caps",
            "_forget_extract",
        }

    def test_the_walk_sees_a_write_through_the_helper_and_past_it(self) -> None:
        """The positive control, including the shape that bypasses ``_save``."""
        source = (
            "def reads(self):\n"
            "    a = self._document_store.get_document('k', 'p')\n"
            "    b = self._document_store.list_documents('k')\n"
            "def saves(self): self._save(entry)\n"
            "def bypasses(self): self._document_store.put_document('k', entry)\n"
            "def removes(self): self._document_store.evict('k', 'p')\n"
            "def nested(self):\n"
            "    if True:\n"
            "        self._save(entry)\n"
            "'''self._save(entry) in a docstring is not a write'''\n"
        )

        assert sorted(_record_writers(source)) == sorted(
            ["saves", "bypasses", "removes", "nested"]
        )


##
## AC 5, live — a second actor over the same tree reads what the first one wrote
##
_LIVE_PATH = "u-alice/restored"
"""The tree the live specs host — one per spec, since each spec has its own system."""


@pytest.fixture
def system(workspaces_root: Path) -> Iterator[ActorSystem]:
    """A real actor system running **no host of any kind**, torn down whatever the spec did.

    51-4's version created a ``WorkspaceHost`` and core's base ``ResourceHost``
    beside it, because the workspace was created through the first one. There is
    no host in this package any more (52-6), so the actors below are created
    directly and the fixture's only remaining job is the teardown assertion.
    """
    actor_system = ActorSystem()
    try:
        yield actor_system
    finally:
        actor_system.shutdown(timeout=10)
        ActorRegistry.stop_all()
        assert ActorSystem.find_by_class(WorkspaceActor) == [], "a workspace outlived its test"


def _create(system: ActorSystem, config: WorkspaceConfig) -> ActorAddress:
    """Start one workspace actor over *config*'s tree, through the public API."""
    return system.createActor(WorkspaceActor, config=config)


def _stopped(system: ActorSystem, address: ActorAddress) -> bool:
    """Stop the actor at *address* the way its team's teardown would, and wait for it.

    51-4's predecessor waited out a **reap** here — the grace plus six sweep
    ticks — because nothing else could end a hosted actor. The actor is an
    ordinary team child now, so what ends it is a stop, and the claim under test
    is unchanged: the records outlive the actor that wrote them.
    """
    system.proxy_ask(address, WorkspaceActor).stop()
    return wait_until(lambda: not address.is_alive(), timeout=HANDSHAKE_TIMEOUT_S)


def _fill_and_upload(system: ActorSystem, address: ActorAddress, tree: Path) -> str:
    """One cache fill and one upload with retrieval off, through a real proxy."""
    proxy = system.proxy_ask(address, WorkspaceActor)
    proxy.configure_document_store(YamlDocumentStore())
    sha = content_sha(b"the source bytes")
    proxy.cache_document("doc.md", sha, EXTRACTOR_VERSION, "# Body\n")
    (tree / "up.md").write_text("# Uploaded\n", encoding="utf-8")
    proxy.receiveMsg_NewFileMessage(NewFileMessage(paths=["up.md"], source="upload"))
    return sha


class TestTheDiskPathLive:
    def test_a_fill_and_an_upload_survive_the_actor_into_a_new_one(
        self, system: ActorSystem, workspaces_root: Path
    ) -> None:
        """The headline of the story, at full scale: **nothing is registered anywhere**.

        51-4's version of this spec had to register a ``DeltaStore`` on a host
        first, and what it proved was that the host's store carried the two
        records across the reap. Nothing carries them now: the second actor reads
        the same files the first one wrote, and the two share no memory at all —
        the first is stopped before the second exists.
        """
        config = workspace_config(_LIVE_PATH)
        address = _create(system, config)
        tree = workspace_root_for(workspaces_root, "restored")

        sha = _fill_and_upload(system, address, tree)

        assert _stopped(system, address), "the actor never stopped"

        renewed = _create(system, config)

        assert renewed.agent_id != address.agent_id, "the same actor came back"
        restored = system.proxy_ask(renewed, WorkspaceActor)
        restored.configure_document_store(YamlDocumentStore())
        assert restored.document_extract("doc.md", sha, EXTRACTOR_VERSION) == "# Body\n"
        rows = restored.rag_snapshot(20).rows
        assert [(row.path, row.status) for row in rows] == [("up.md", RagStatus.PENDING.value)]

    def test_a_second_actor_reads_them_while_the_first_is_still_alive(
        self, system: ActorSystem, workspaces_root: Path
    ) -> None:
        """Two actors over one tree, at once — the arrangement 52-5 made legal.

        Re-pointed from ``test_the_host_needs_no_store_registered_at_all``, whose
        subject — a host with no store registered on it — is retired. What that
        spec was really asserting is that no in-memory carrier is involved, and
        the strongest form of that is now available and was not then: a **live**
        second actor, sharing nothing with the first, reads the first one's
        records off the tree.
        """
        config = workspace_config(_LIVE_PATH)
        address = _create(system, config)
        tree = workspace_root_for(workspaces_root, "restored")

        sha = _fill_and_upload(system, address, tree)

        assert address.is_alive(), "the fill stopped the actor"
        second = _create(system, config)
        assert second.agent_id != address.agent_id
        assert address.is_alive(), "the first actor was displaced rather than joined"
        reader = system.proxy_ask(second, WorkspaceActor)
        reader.configure_document_store(YamlDocumentStore())
        assert reader.document_extract("doc.md", sha, EXTRACTOR_VERSION) == "# Body\n"


class TestNoStoreAtAll:
    def test_every_write_site_is_a_silent_no_op_without_a_store(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """A lost announcement degrades: the cache misses and the index looks empty.

        Never a raise, and never an ``assert self._document_store is not None``.
        The whole scenario must still run to the end — this actor owns the write
        gate, and a document path that could take it down is the defect the
        degradation exists to prevent.
        """
        harness.actor._document_store = None

        _drive_every_write_site_degraded(harness, workspace_tree)

        assert stored_entries(harness.actor) == {}
        assert harness.actor.rag_snapshot(max_pending_shown=20).rows == []
        assert harness.actor.document_extract("a.md", "any", EXTRACTOR_VERSION) is None


def _drive_every_write_site_degraded(harness: RagHarness, tree: Path) -> None:
    """The driver's steps, without the assertions that read a record back.

    Separate from :func:`_drive_every_write_site` rather than parameterised: with
    no store every one of those assertions is false by design, and a driver with
    a "skip the checks" flag is a driver that can silently stop checking.
    """
    actor = harness.actor
    write(tree, "a.md", "# A\n\nbody\n")
    write(tree, "b.md", "# B\n\nother\n")
    actor.receiveMsg_NewFileMessage(NewFileMessage(paths=["a.md", "b.md"], source="upload"))
    harness.enable()
    actor.index_paths("")
    actor.mark_paths_stale(["a.md"])
    actor.cache_document("c.md", content_sha(b"c"), EXTRACTOR_VERSION, "c" * 60)
    assert actor.reap_abandoned_rows() is False
    assert actor.rag_search("body") in {
        "Retrieval indexing is not available for this workspace.",
        "Nothing in the retrieval index matched that query. "
        "Use workspace_rag_list to see which files are indexed.",
    }
