"""The extraction cache: hit/miss, recency, the two caps, and the no-write-on-read guard.

These specs address the actor **directly**, which is what keeps them a guard on
the cache itself rather than on whoever calls it. ``workspace_read`` became that
caller in 45-4; the read path's own guards live in
``test_document_read_path.py``. What is asserted here is the shape the read path
calls into and the one rule the whole design rests on: **the read path never
persists state**.

The cache is one file per source document under the tree's sibling metadata
directory (story 52-3). Reads are the majority of workspace traffic, so a write
on a read — or on a cache hit, which a read is — would put a disk write on the
path ADR-036's NFR1 exists to keep free. A *fill* is different: it is amortised
against the seconds of extraction that preceded it. The specs count **writes**
through a recording store, which is the same question the delta recorder used to
ask one layer down, and a stronger one: a delta could be counted without anything
reaching a disk, and a put cannot.

**Every read-back goes through a second store object**, never the one the actor
holds, so what a spec asserts is what reached the disk rather than what reached
memory.

The caps are exercised **at** the cap and one past it, so an off-by-one in
either direction is visible. Small caps throughout: the production defaults
would make each spec build 32 entries or two megabytes of text.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from akgentic.tool.workspace.actor import (
    WORKSPACE_ACTOR_ROLE,
    WorkspaceActor,
    workspace_actor_name,
)
from akgentic.tool.workspace.documents.models import (
    DEFAULT_MAX_DOCUMENT_CHARS,
    DEFAULT_MAX_DOCUMENTS,
    EXTRACTOR_VERSION,
    DocumentExtract,
    RagFile,
    RagStatus,
    evict_document_bodies,
)
from akgentic.tool.workspace.documents.store import DocumentEntry, YamlDocumentStore
from akgentic.tool.workspace.models import WorkspaceConfig, content_sha
from akgentic.tool.workspace.tool import WorkspaceTool

from tests.workspace.conftest import (
    WORKSPACE_PATH,
    RecordingDocumentStore,
    attach_store,
    read,
    seed_extract,
    seed_row,
    stored_docs,
    stored_entries,
    watch_store,
)


def start_actor(
    max_documents: int = DEFAULT_MAX_DOCUMENTS,
    max_document_chars: int = DEFAULT_MAX_DOCUMENT_CHARS,
) -> WorkspaceActor:
    """Build and start an actor over the test workspace, without an actor thread."""
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
    # A directly built actor is handed no store — resolving one is the card's
    # job. This stands in for the bind.
    return attach_store(actor)


def sha_of(text: str) -> str:
    """The digest 45-4's caller will hand in, over the source bytes."""
    return content_sha(text.encode())


def fill(
    actor: WorkspaceActor,
    path: str,
    body: str = "body",
    source: str | None = None,
    version: int = EXTRACTOR_VERSION,
) -> None:
    """Cache *body* as the extraction of *path*, whose source is *source*."""
    actor.cache_document(path, sha_of(source if source is not None else path), version, body)


def look_up(
    actor: WorkspaceActor,
    path: str,
    source: str | None = None,
    version: int = EXTRACTOR_VERSION,
) -> str | None:
    """The ask 45-4 will make: the Markdown on a hit, ``None`` on any miss."""
    return actor.document_extract(path, sha_of(source if source is not None else path), version)


def a_row(path: str) -> RagFile:
    """An ``EMBEDDED`` index row, so an eviction spec can prove it survived."""
    return RagFile(
        path=path, status=RagStatus.EMBEDDED, indexed_sha=sha_of(path), updated_at=datetime.now(UTC)
    )


def an_extract(path: str, body: str | None, source: str | None = None) -> DocumentExtract:
    """One cache entry, built the way ``cache_document`` builds it."""
    return DocumentExtract(
        path=path,
        source_sha=sha_of(source if source is not None else path),
        extractor_version=EXTRACTOR_VERSION,
        markdown=body,
        char_count=len(body) if body is not None else 0,
        extracted_at=datetime.now(UTC),
    )


def watch(actor: WorkspaceActor) -> RecordingDocumentStore:
    """Record every write *actor* performs from now on — the one channel that persists."""
    return watch_store(actor)


class _ExtractWithExtraField(DocumentExtract):
    """A persisted extract carrying a field the write path has never heard of.

    Golden Rule #12's guard, in the only formulation that works. A whole-model
    comparison would stay green against an enumerated rebuild naming every field
    that exists *today* — which is exactly the code that silently destroys the
    field added tomorrow.
    """

    extra_field: str = "sentinel"


# ---------------------------------------------------------------------------
# AC1: the model
# ---------------------------------------------------------------------------


class TestDocumentExtract:
    def test_carries_exactly_the_six_fields(self) -> None:
        assert set(DocumentExtract.model_fields) == {
            "path",
            "source_sha",
            "extractor_version",
            "markdown",
            "char_count",
            "extracted_at",
        }

    def test_round_trips_through_pydantic(self) -> None:
        entry = an_extract("notes.docx", "# Notes")
        assert DocumentExtract.model_validate(entry.model_dump()) == entry

    def test_a_dropped_body_round_trips_as_none(self) -> None:
        entry = an_extract("notes.docx", None)
        assert DocumentExtract.model_validate(entry.model_dump()).markdown is None

    def test_it_round_trips_through_json_which_is_the_persisted_encoding(self) -> None:
        # ``model_dump()`` above hands back a live ``datetime``; the snapshot
        # that ``notify_state_change()`` emits is ``model_dump_json()``, where
        # ``extracted_at`` becomes an ISO string. Only this asserts the form
        # that is actually persisted survives the trip back.
        entry = an_extract("notes.docx", "# Notes")
        assert DocumentExtract.model_validate_json(entry.model_dump_json()) == entry


# ---------------------------------------------------------------------------
# AC2 (52-3): the cache is on disk, and nothing about it is in the actor
# ---------------------------------------------------------------------------


class TestTheCacheIsOnDisk:
    """What the state field used to guard, moved to where the cache actually is."""

    def test_a_fresh_tree_has_no_records(self, workspaces_root: Path) -> None:
        actor = start_actor()
        assert stored_docs(actor) == {}

    def test_a_filled_cache_is_readable_through_a_second_store_object(
        self, workspaces_root: Path
    ) -> None:
        # The successor to the state round trip, and a stronger claim: a second
        # object shares nothing with the actor's own, so a body it can read is a
        # body that reached the disk.
        actor = start_actor()
        fill(actor, "notes.docx", body="# Notes")
        assert stored_docs(actor)["notes.docx"].markdown == "# Notes"

    def test_the_record_survives_the_yaml_encoding_it_is_persisted_in(
        self, workspaces_root: Path
    ) -> None:
        # YAML, not a python-mode dump: that — and the ISO string
        # ``extracted_at`` becomes inside it — is the shape a second process
        # reads back.
        actor = start_actor()
        fill(actor, "notes.docx", body="# Notes")
        written = YamlDocumentStore().get_document(WORKSPACE_PATH, "notes.docx")
        assert written is not None
        assert written.extract == stored_docs(actor)["notes.docx"]

    def test_the_actor_carries_no_cache_of_its_own(self, workspaces_root: Path) -> None:
        # The point of the move: the state declares no document field at all, so
        # a second process reading the same mount is not reading a copy of one.
        actor = start_actor()
        fill(actor, "notes.docx", body="# Notes")
        assert "documents" not in type(actor.state).model_fields
        assert "rag_index" not in type(actor.state).model_fields


# ---------------------------------------------------------------------------
# AC6: hit and miss — four distinct miss reasons, four specs
# ---------------------------------------------------------------------------


class TestHitAndMiss:
    def test_a_hit_returns_the_body(self, workspaces_root: Path) -> None:
        actor = start_actor()
        fill(actor, "notes.docx", body="# Notes")
        assert look_up(actor, "notes.docx") == "# Notes"

    def test_an_unknown_path_is_a_miss(self, workspaces_root: Path) -> None:
        actor = start_actor()
        assert look_up(actor, "never-seen.docx") is None

    def test_a_source_sha_mismatch_is_a_miss(self, workspaces_root: Path) -> None:
        # The file changed under the cache: the extract describes bytes that are
        # no longer there.
        actor = start_actor()
        fill(actor, "notes.docx", body="# Notes", source="v1")
        assert look_up(actor, "notes.docx", source="v2") is None

    def test_an_extractor_version_mismatch_is_a_miss(self, workspaces_root: Path) -> None:
        # Bumping the constant invalidates every cached extract everywhere, with
        # no sweep and no migration.
        actor = start_actor()
        fill(actor, "notes.docx", body="# Notes", version=EXTRACTOR_VERSION)
        assert look_up(actor, "notes.docx", version=EXTRACTOR_VERSION + 1) is None

    def test_an_evicted_body_is_a_miss(self, workspaces_root: Path) -> None:
        # Metadata alone is not a hit — the body is what the caller asked for.
        #
        # **This spec no longer has a discriminator, and that is a fact about
        # the code rather than a weakened assertion.** The old one was the LRU
        # reorder: a hit moved the entry and a miss did not, so dropping the
        # ``markdown is None`` clause was visible. Story 52-3 deletes that
        # reorder — recency is ``extracted_at`` now — and with it the only way
        # to tell the two apart, because without the clause the lookup falls
        # through to ``return extract.markdown``, which *is* ``None``. The
        # clause is now a readability guard, not a behavioural one. What is
        # still worth pinning, and is pinned here, is that the record survives
        # the lookup untouched.
        actor = start_actor()
        seed_extract(actor, "a.docx", an_extract("a.docx", None))
        recorder = watch(actor)
        assert look_up(actor, "a.docx") is None
        assert recorder.written == []
        assert stored_docs(actor)["a.docx"].markdown is None

    def test_an_empty_body_is_a_hit(self, workspaces_root: Path) -> None:
        # An extraction that legitimately produced nothing is not a miss: re-running
        # it would produce nothing again.
        actor = start_actor()
        fill(actor, "empty.docx", body="")
        assert look_up(actor, "empty.docx") == ""


# ---------------------------------------------------------------------------
# AC14 (52-3): recency is ``extracted_at``, and a directory has no order
# ---------------------------------------------------------------------------


class TestRecencyIsExtractedAt:
    """The successor to the LRU-order class, on the field that now carries it.

    A directory glob's order is the file system's, so the insertion order the
    old LRU rode on does not exist any more. ``extracted_at`` is stamped at every
    fill and is what the eviction pass sorts on.
    """

    def test_a_re_fill_refreshes_recency_rather_than_duplicating(
        self, workspaces_root: Path
    ) -> None:
        actor = start_actor()
        for name in ("a.docx", "b.docx", "c.docx"):
            fill(actor, name)
        before = stored_docs(actor)["a.docx"].extracted_at
        fill(actor, "a.docx", body="re-extracted")
        after = stored_docs(actor)
        assert set(after) == {"a.docx", "b.docx", "c.docx"}
        assert after["a.docx"].markdown == "re-extracted"
        assert after["a.docx"].extracted_at > before

    def test_a_re_fill_leaves_exactly_one_record(self, workspaces_root: Path) -> None:
        # Keyed by path, not by content: a second fill at other bytes replaces.
        actor = start_actor()
        fill(actor, "a.docx", body="first", source="v1")
        fill(actor, "a.docx", body="second", source="v2")
        assert list(stored_docs(actor)) == ["a.docx"]
        assert stored_docs(actor)["a.docx"].markdown == "second"

    def test_a_hit_does_not_refresh_recency(self, workspaces_root: Path) -> None:
        # **Deliberately inverted from what 45-3 guarded.** The hit used to move
        # the entry to the end of the LRU; story 52-3 deletes that, because
        # recording it would be a write on the read path — the one rule this
        # whole design rests on. The cost is at most one extra re-extraction of
        # a file that was read but not re-extracted, and it is not to be
        # "fixed" back into a write.
        actor = start_actor()
        fill(actor, "a.docx")
        stamped = stored_docs(actor)["a.docx"].extracted_at
        assert look_up(actor, "a.docx") is not None
        assert stored_docs(actor)["a.docx"].extracted_at == stamped

    def test_a_miss_stamps_nothing(self, workspaces_root: Path) -> None:
        actor = start_actor()
        for name in ("a.docx", "b.docx"):
            fill(actor, name)
        before = {path: entry.extracted_at for path, entry in stored_docs(actor).items()}
        look_up(actor, "never-seen.docx")
        look_up(actor, "a.docx", source="a different source")
        assert {path: entry.extracted_at for path, entry in stored_docs(actor).items()} == before


# ---------------------------------------------------------------------------
# AC12/AC13 (52-3): THE HEADLINE CRITERION — a read writes nothing
# ---------------------------------------------------------------------------


class TestTheReadPathWritesNothing:
    """The no-notify matrix, asked of the disk instead of of a delta stream.

    The same four questions, one layer down and one step stronger: a delta could
    be counted without anything being written, and a put cannot.
    """

    def test_a_text_read_writes_nothing(
        self,
        wired_card: WorkspaceTool,
        workspace_actor: WorkspaceActor,
        workspace_tree: Path,
    ) -> None:
        # The property epic 29 shipped and neither story may spend: reads are
        # the majority of workspace traffic and none of them writes.
        (workspace_tree / "plain.txt").write_text("hello\n", encoding="utf-8")
        recorder = watch(workspace_actor)
        assert "hello" in read(wired_card, "plain.txt")
        assert recorder.puts == []
        assert recorder.evicted == []

    def test_a_cache_hit_writes_nothing(self, workspaces_root: Path) -> None:
        # It also performs **one** lookup and no listing: a hit that scanned the
        # directory would put the cost of the whole cache on every read.
        actor = start_actor()
        fill(actor, "notes.docx", body="# Notes")
        fill(actor, "other.docx", body="# Other")
        recorder = watch(actor)
        assert look_up(actor, "notes.docx") == "# Notes"
        assert recorder.puts == []
        assert recorder.gets == [(WORKSPACE_PATH, "notes.docx")]
        assert recorder.listings == []

    def test_a_lookup_miss_writes_nothing(self, workspaces_root: Path) -> None:
        actor = start_actor()
        recorder = watch(actor)
        assert look_up(actor, "never-seen.docx") is None
        assert recorder.puts == []

    def test_a_fill_writes_the_one_path_it_filled(self, workspaces_root: Path) -> None:
        actor = start_actor()
        recorder = watch(actor)
        fill(actor, "notes.docx", body="# Notes")
        assert recorder.written == ["notes.docx"]

    def test_a_fill_that_also_evicts_writes_only_what_it_touched(
        self, workspaces_root: Path
    ) -> None:
        # Never a rewrite of every surviving record: the pass applies the
        # evictor's verdict to the paths it named and to nothing else.
        actor = start_actor(max_documents=1)
        fill(actor, "a.docx")
        recorder = watch(actor)
        fill(actor, "b.docx")
        assert recorder.written == ["b.docx"]
        assert recorder.evicted == [(WORKSPACE_PATH, "a.docx")]
        assert list(stored_docs(actor)) == ["b.docx"]


# ---------------------------------------------------------------------------
# AC7 / AC10: the two caps, at the cap and one past it
# ---------------------------------------------------------------------------


class TestMaxDocumentsCap:
    def test_at_the_cap_nothing_is_evicted(self, workspaces_root: Path) -> None:
        actor = start_actor(max_documents=3)
        for name in ("a.docx", "b.docx", "c.docx"):
            fill(actor, name)
        # A set: the store is a directory, and a glob's order is the file
        # system's, so membership is the only thing worth asserting here.
        assert set(stored_docs(actor)) == {"a.docx", "b.docx", "c.docx"}

    def test_one_past_the_cap_removes_the_lru_oldest_entry_outright(
        self, workspaces_root: Path
    ) -> None:
        # The entry, not just its body: this cap answers the number of rows in
        # the map, and metadata kept forever is the leak it exists to stop.
        actor = start_actor(max_documents=3)
        for name in ("a.docx", "b.docx", "c.docx", "d.docx"):
            fill(actor, name)
        assert "a.docx" not in stored_docs(actor)
        assert set(stored_docs(actor)) == {"b.docx", "c.docx", "d.docx"}

    def test_a_hit_no_longer_protects_an_entry_from_the_next_eviction(
        self, workspaces_root: Path
    ) -> None:
        # **Deliberately inverted from what 45-3 guarded**, and named so nobody
        # reads it as a regression. Recording a hit's recency would be a write on
        # the read path, which is the one thing this cache may not do; the price
        # is that the eviction pass sees only what a *fill* stamped, so the
        # oldest extraction goes even if it was read a moment ago. The cost is
        # one re-extraction.
        actor = start_actor(max_documents=3)
        for name in ("a.docx", "b.docx", "c.docx"):
            fill(actor, name)
        look_up(actor, "a.docx")
        fill(actor, "d.docx")
        assert "a.docx" not in stored_docs(actor)
        assert "b.docx" in stored_docs(actor)

    def test_the_row_cap_never_de_indexes_a_file(self, workspaces_root: Path) -> None:
        # AC 14's other half, and the reason the row cap does not simply unlink:
        # both halves of a document live in one file now, so removing it would
        # take the index row with it. ``evict_document_bodies``' own contract
        # forbids that — a search hit renders from what the vector store holds,
        # so an indexed file stays searchable with no cached body at all.
        actor = start_actor(max_documents=1)
        fill(actor, "indexed.docx")
        seed_row(actor, "indexed.docx", a_row("indexed.docx"))
        fill(actor, "newer.docx")

        survivors = stored_entries(actor)
        assert "indexed.docx" in survivors, "the record went with the body it evicted"
        assert survivors["indexed.docx"].extract is None
        assert survivors["indexed.docx"].row is not None
        assert survivors["indexed.docx"].row.status is RagStatus.EMBEDDED

    def test_a_record_with_no_row_is_removed_outright(self, workspaces_root: Path) -> None:
        # The other side of the same rule: with no index half to keep, the row
        # cap removes the file rather than leaving an empty record behind.
        actor = start_actor(max_documents=1)
        fill(actor, "a.docx")
        fill(actor, "b.docx")
        assert "a.docx" not in stored_entries(actor)


class TestMaxDocumentCharsCap:
    def test_at_the_cap_no_body_is_dropped(self, workspaces_root: Path) -> None:
        actor = start_actor(max_documents=100, max_document_chars=100)
        fill(actor, "a.docx", body="a" * 60)
        fill(actor, "b.docx", body="b" * 40)
        assert stored_docs(actor)["a.docx"].markdown == "a" * 60
        assert stored_docs(actor)["b.docx"].markdown == "b" * 40

    def test_one_past_the_cap_drops_the_lru_oldest_body_and_keeps_its_metadata(
        self, workspaces_root: Path
    ) -> None:
        actor = start_actor(max_documents=100, max_document_chars=100)
        fill(actor, "a.docx", body="a" * 60)
        fill(actor, "b.docx", body="b" * 40)
        before = stored_docs(actor)["a.docx"]
        fill(actor, "c.docx", body="c")

        dropped = stored_docs(actor)["a.docx"]
        assert dropped.markdown is None
        assert dropped.path == before.path
        assert dropped.source_sha == before.source_sha
        assert dropped.extractor_version == before.extractor_version
        assert dropped.char_count == before.char_count
        assert dropped.extracted_at == before.extracted_at
        # It drops one body, not every body: 40 + 1 fits under the cap.
        assert stored_docs(actor)["b.docx"].markdown == "b" * 40
        assert stored_docs(actor)["c.docx"].markdown == "c"

    def test_a_single_over_cap_document_drops_its_own_body(self, workspaces_root: Path) -> None:
        # A permanent miss costing one re-extraction per read — correct, and not
        # a case to special-case. The loop must terminate rather than spin.
        actor = start_actor(max_documents=100, max_document_chars=100)
        fill(actor, "huge.docx", body="x" * 150)
        assert stored_docs(actor)["huge.docx"].markdown is None
        assert stored_docs(actor)["huge.docx"].char_count == 150

    def test_the_char_sum_counts_only_bodied_entries(self, workspaces_root: Path) -> None:
        # A dropped body must not keep pressing on the cap it already relieved.
        actor = start_actor(max_documents=100, max_document_chars=100)
        fill(actor, "a.docx", body="a" * 90)
        fill(actor, "b.docx", body="b" * 90)
        assert stored_docs(actor)["a.docx"].markdown is None
        assert stored_docs(actor)["b.docx"].markdown == "b" * 90


class TestEvictDocumentBodiesIsPure:
    def test_it_reports_every_path_it_touched(self) -> None:
        documents = {
            "a.docx": an_extract("a.docx", "a" * 60),
            "b.docx": an_extract("b.docx", "b" * 60),
        }
        assert evict_document_bodies(documents, max_documents=10, max_document_chars=100) == [
            "a.docx"
        ]

    def test_an_empty_map_is_a_no_op(self) -> None:
        documents: dict[str, DocumentExtract] = {}
        assert evict_document_bodies(documents, max_documents=0, max_document_chars=0) == []
        assert documents == {}

    def test_a_bodiless_map_over_the_char_cap_terminates(self) -> None:
        # No bodied entry remains, so there is nothing left to drop — the loop
        # must exit rather than spin on a cap it can no longer satisfy.
        documents = {"a.docx": an_extract("a.docx", None)}
        assert evict_document_bodies(documents, max_documents=10, max_document_chars=0) == []
        assert documents["a.docx"].markdown is None


# ---------------------------------------------------------------------------
# AC9: Golden Rule #12 — the body drop copies, it does not rebuild
# ---------------------------------------------------------------------------


class TestGoldenRule12BodyDrop:
    def test_a_body_drop_preserves_a_field_the_write_path_never_heard_of(self) -> None:
        # An enumerated rebuild returns a plain ``DocumentExtract`` and fails
        # both assertions. A whole-model comparison would fail neither, which is
        # why this is the guard the rule prescribes.
        documents: dict[str, DocumentExtract] = {
            "a.docx": _ExtractWithExtraField(
                path="a.docx",
                source_sha=sha_of("a.docx"),
                extractor_version=EXTRACTOR_VERSION,
                markdown="a" * 60,
                char_count=60,
                extracted_at=datetime.now(UTC),
            )
        }
        evict_document_bodies(documents, max_documents=10, max_document_chars=10)

        survivor = documents["a.docx"]
        assert survivor.markdown is None
        assert isinstance(survivor, _ExtractWithExtraField)
        assert survivor.extra_field == "sentinel"


# ---------------------------------------------------------------------------
# AC6: neither method raises — a document path degrades, it never propagates
# ---------------------------------------------------------------------------


class TestDegradation:
    def test_a_lookup_of_an_absent_path_returns_none(self, workspaces_root: Path) -> None:
        actor = start_actor()
        assert look_up(actor, "") is None
        assert look_up(actor, "nested/deeply/absent.docx") is None

    def test_a_fill_of_an_empty_body_records_a_zero_char_count(self, workspaces_root: Path) -> None:
        actor = start_actor()
        fill(actor, "empty.docx", body="")
        assert stored_docs(actor)["empty.docx"].char_count == 0

    def test_a_lookup_of_an_already_bodiless_entry_returns_none_twice(
        self, workspaces_root: Path
    ) -> None:
        actor = start_actor()
        stored_docs(actor)["a.docx"] = an_extract("a.docx", None)
        assert look_up(actor, "a.docx") is None
        assert look_up(actor, "a.docx") is None

    def test_char_count_is_computed_from_the_body_not_supplied(self, workspaces_root: Path) -> None:
        # It cannot disagree with the body it describes, because it is never a
        # parameter.
        actor = start_actor()
        fill(actor, "notes.docx", body="12345")
        assert stored_docs(actor)["notes.docx"].char_count == 5
