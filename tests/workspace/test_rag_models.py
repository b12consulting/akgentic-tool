"""The retrieval index's models, its chunk ids, and the caps that follow the backend.

Three properties are asserted here that nothing else in the suite can see:

- **A chunk is a pair of offsets.** ``RagChunk`` carries no text of any kind, and
  a field that appeared later would make every persisted state a copy of every
  document it describes.
- **A chunk id is deterministic across processes, and the scope is inside it.**
  The determinism is pinned against a literal rather than against a second call
  in the same process, which would agree even with a per-process namespace.
- **Every ``RagFile`` transition is a copy-and-override.** The guard is a
  subclass carrying a field the write path has never heard of, because a
  whole-model comparison can only compare the fields that exist today and would
  pass green against a rebuild that enumerates all of them.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from akgentic.tool.workspace.actor import (
    WORKSPACE_ACTOR_ROLE,
    WorkspaceActor,
    workspace_actor_name,
)
from akgentic.tool.workspace.documents.models import (
    CHUNK_ID_NAMESPACE,
    DEFAULT_MAX_DOCUMENT_CHARS,
    DEFAULT_MAX_DOCUMENTS,
    EMBEDDING_STALE_AFTER_S,
    IN_MEMORY_MAX_DOCUMENT_CHARS,
    IN_MEMORY_MAX_DOCUMENTS,
    RAG_COLLECTION,
    RagChunk,
    RagFile,
    RagStatus,
    chunk_id,
    derived_document_caps,
)
from akgentic.tool.workspace.models import WorkspaceConfig

from tests.workspace.conftest import WORKSPACE_NAME

# Recorded once, from a call made in a *different* process. A second call made
# here would agree with a namespace minted per process, which is exactly the
# defect the literal namespace exists to prevent — so the expected value has to
# come from outside this run.
_KNOWN_ID = "879636b4-302d-568f-a5eb-5a2351b4a179"
_KNOWN_ARGS = ("ws", "a.md", "abc", 0)


def _actor() -> WorkspaceActor:
    """A started actor over the test workspace, with no actor thread."""
    actor = WorkspaceActor(
        config=WorkspaceConfig(
            name=workspace_actor_name(WORKSPACE_NAME),
            role=WORKSPACE_ACTOR_ROLE,
            workspace_name=WORKSPACE_NAME,
        )
    )
    actor.on_start()
    return actor


class _RagFileWithExtraField(RagFile):
    """A ``RagFile`` carrying a field every write path is unaware of.

    This is the whole of the Golden Rule #12 guard, and the shape matters. A
    whole-model comparison would pass against a rebuild naming every field that
    exists **today** — it can only compare what exists now, so the field added
    tomorrow sits at its default on both sides and its loss is invisible. A
    subclass makes the loss structural: an enumerated ``RagFile(...)`` returns a
    plain ``RagFile`` and cannot carry ``extra_field`` at all.
    """

    extra_field: str = "sentinel"


class TestChunkIdentity:
    """``chunk_id`` is deterministic, and both digest inputs are load-bearing."""

    def test_the_namespace_is_a_literal_and_not_minted_per_process(self) -> None:
        """A ``uuid4()`` namespace would make every id non-deterministic."""
        assert str(CHUNK_ID_NAMESPACE) == "2f5c1a90-7b64-5c3e-9f21-0d8a4c6b1e73"

    def test_a_fixed_input_produces_the_id_recorded_from_another_process(self) -> None:
        """Idempotent retry depends on two processes agreeing, not on one being stable."""
        assert chunk_id(*_KNOWN_ARGS) == _KNOWN_ID

    def test_changing_only_the_scope_changes_the_id(self) -> None:
        """One collection holds every workspace, and the read path is keyed on ref_id.

        Two trees holding the same file at the same path with identical bytes must
        not mint the same id: ``InMemoryBackend`` resolves ``{ref_id: entry}``
        last-one-wins, so the collision would be silent and cross-workspace.
        """
        assert chunk_id("other-workspace", "a.md", "abc", 0) != _KNOWN_ID

    def test_changing_only_the_source_digest_changes_the_id(self) -> None:
        """Re-index is add-then-remove, which is only safe while ids cannot collide."""
        assert chunk_id("ws", "a.md", "def", 0) != _KNOWN_ID

    def test_changing_only_the_ordinal_changes_the_id(self) -> None:
        """Two chunks of one document are two entries, not one overwritten twice."""
        assert chunk_id("ws", "a.md", "abc", 1) != _KNOWN_ID


class TestChunksAreOffsets:
    """A stored chunk is a coordinate pair, never a copy of the document."""

    def test_rag_chunk_carries_no_text_of_any_kind(self) -> None:
        """A body field would duplicate every document inside the actor's state."""
        assert set(RagChunk.model_fields) == {
            "chunk_id",
            "ordinal",
            "start",
            "end",
            "heading_path",
            "header_start",
            "header_end",
        }

    def test_every_field_survives_a_json_round_trip(self) -> None:
        """Golden Rule #1b, asserted as the property rather than as the config.

        The config itself is not checkable here: ``SerializableBaseModel`` sets
        ``arbitrary_types_allowed`` for the whole hierarchy and Pydantic copies
        ``model_config`` into every subclass's dict, so both spellings of that
        assertion are vacuous — ``DocumentExtract`` would pass them too. What a
        leaking type actually breaks is the round trip, and these rows are
        persisted as JSON inside ``WorkspaceState`` and read back on resume.
        """
        chunk = RagChunk(
            chunk_id="c1", ordinal=3, start=10, end=40, heading_path=["A", "B"], header_start=1
        )
        assert RagChunk.model_validate_json(chunk.model_dump_json()) == chunk

    def test_a_rag_file_round_trips_through_validation(self) -> None:
        """It is persisted inside ``WorkspaceState`` and read back on resume."""
        original = RagFile(
            path="a.md",
            status=RagStatus.EMBEDDED,
            indexed_sha="abc",
            chunks=[RagChunk(chunk_id="c1", ordinal=0, start=0, end=5, heading_path=["A"])],
            chunk_count=1,
            updated_at=datetime.now(UTC),
        )
        assert RagFile.model_validate(original.model_dump()) == original


class TestTheCollectionIsOnePerDeployment:
    """One class for every workspace; the tree is a property, never a class name."""

    def test_the_collection_name_is_frozen(self) -> None:
        """A class per workspace would be a Weaviate schema mutation per team."""
        assert RAG_COLLECTION == "workspace_chunks"


class TestDerivedDocumentCaps:
    """The extraction caps follow the vector backend, and only when vectors exist."""

    def test_in_memory_with_retrieval_on_takes_the_small_pair(self) -> None:
        """In-memory keeps every vector in ``VectorStoreState``, re-serialised per notify."""
        assert derived_document_caps("inmemory", True) == (
            IN_MEMORY_MAX_DOCUMENTS,
            IN_MEMORY_MAX_DOCUMENT_CHARS,
        )

    def test_weaviate_with_retrieval_on_takes_the_large_pair(self) -> None:
        """The vectors live in the cluster, so nothing derives from the document cap."""
        assert derived_document_caps("weaviate", True) == (
            DEFAULT_MAX_DOCUMENTS,
            DEFAULT_MAX_DOCUMENT_CHARS,
        )

    def test_retrieval_off_takes_the_large_pair_whatever_the_backend(self) -> None:
        """No vectors exist, so shrinking the cache would cost for nothing."""
        assert derived_document_caps("inmemory", False) == (
            DEFAULT_MAX_DOCUMENTS,
            DEFAULT_MAX_DOCUMENT_CHARS,
        )
        assert derived_document_caps("weaviate", False) == (
            DEFAULT_MAX_DOCUMENTS,
            DEFAULT_MAX_DOCUMENT_CHARS,
        )


class TestEveryTransitionIsACopy:
    """Golden Rule #12, guarded by a field the write paths have never heard of."""

    def test_marking_stale_preserves_an_unknown_field(self, workspace_tree: object) -> None:
        """``mark_paths_stale`` is a status bump, and one of seven per indexed file."""
        actor = _actor()
        actor.state.rag_index["a.md"] = _RagFileWithExtraField(
            path="a.md", status=RagStatus.EMBEDDED, updated_at=datetime.now(UTC)
        )

        actor.mark_paths_stale(["a.md"])

        result = actor.state.rag_index["a.md"]
        assert result.status is RagStatus.STALE
        assert isinstance(result, _RagFileWithExtraField)
        assert result.extra_field == "sentinel"

    def test_the_reapers_revert_preserves_an_unknown_field(self, workspace_tree: object) -> None:
        """A second, structurally different transition — it moves four fields, not one."""
        actor = _actor()
        stale = datetime.now(UTC) - timedelta(seconds=EMBEDDING_STALE_AFTER_S + 1)
        actor.state.rag_index["a.md"] = _RagFileWithExtraField(
            path="a.md",
            status=RagStatus.EMBEDDING,
            batches_expected=3,
            batches_landed=1,
            updated_at=stale,
        )

        assert actor.reap_stale_embedding() is True

        result = actor.state.rag_index["a.md"]
        assert result.status is RagStatus.PENDING
        assert isinstance(result, _RagFileWithExtraField)
        assert result.extra_field == "sentinel"

    def test_the_guard_is_not_vacuous(self) -> None:
        """The subclass has to actually add a field the base does not declare."""
        assert "extra_field" in _RagFileWithExtraField.model_fields
        assert "extra_field" not in RagFile.model_fields


class TestTheEmbeddingBound:
    """A file cannot sit at ``EMBEDDING`` for ever, because nothing else will free it."""

    def test_a_file_past_the_bound_is_queued_again(self, workspace_tree: object) -> None:
        """After a resume the vector store's requester map is gone with the request."""
        actor = _actor()
        actor.state.rag_index["a.md"] = RagFile(
            path="a.md",
            status=RagStatus.EMBEDDING,
            batches_expected=2,
            batches_landed=1,
            updated_at=datetime.now(UTC) - timedelta(seconds=EMBEDDING_STALE_AFTER_S + 1),
        )

        assert actor.reap_stale_embedding() is True

        entry = actor.state.rag_index["a.md"]
        assert entry.status is RagStatus.PENDING
        assert (entry.batches_expected, entry.batches_landed) == (0, 0)

    def test_a_file_just_inside_the_bound_does_not_move(self, workspace_tree: object) -> None:
        """A live embed must not be restarted underneath itself."""
        actor = _actor()
        actor.state.rag_index["a.md"] = RagFile(
            path="a.md",
            status=RagStatus.EMBEDDING,
            updated_at=datetime.now(UTC) - timedelta(seconds=EMBEDDING_STALE_AFTER_S - 5),
        )

        assert actor.reap_stale_embedding() is False
        assert actor.state.rag_index["a.md"].status is RagStatus.EMBEDDING

    def test_a_file_at_another_status_is_never_reaped(self, workspace_tree: object) -> None:
        """The bound is about a signal that is not coming, not about age."""
        actor = _actor()
        old = datetime.now(UTC) - timedelta(days=7)
        actor.state.rag_index["a.md"] = RagFile(
            path="a.md", status=RagStatus.FAILED, reason="boom", updated_at=old
        )

        assert actor.reap_stale_embedding() is False
        assert actor.state.rag_index["a.md"].status is RagStatus.FAILED
