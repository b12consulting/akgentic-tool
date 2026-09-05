"""The indexing pipeline on the actor: enable, queue, spawn, batch, settle, reap.

These specs address the actor **directly** — the handlers are the actor's public
surface for a worker and for ``#VectorStore``, and calling them is how the other
workspace suites address it. What is stood in for is everything on the far side
of a proxy: the vector store, the orchestrator, and the workers themselves.

The vector store double keeps an **ordered** call log rather than three separate
lists, because one of the properties under test is an ordering — every ``add``
for a path precedes every ``remove`` for it, and two unordered lists cannot say
so.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from akgentic.core.agent_state import BaseState
from akgentic.tool.vector_store.embedding_actor import EmbeddingCompleted
from akgentic.tool.vector_store.protocol import CollectionConfig
from akgentic.tool.workspace.actor import (
    WORKSPACE_ACTOR_ROLE,
    WorkspaceActor,
    workspace_actor_name,
)
from akgentic.tool.workspace.card.params import WorkspaceRagIndex
from akgentic.tool.workspace.documents.models import (
    EMBEDDING_STALE_AFTER_S,
    EXTRACTOR_VERSION,
    RAG_COLLECTION,
    RagChunk,
    RagFile,
    RagStatus,
    chunk_id,
)
from akgentic.tool.workspace.documents.worker import (
    EMBED_BATCH_SIZE,
    MAX_CONCURRENT_INDEX_WORKERS,
    IndexError,
    IndexRequest,
    IndexResult,
    IndexWorker,
)
from akgentic.tool.workspace.models import WorkspaceConfig, WorkspaceState, content_sha
from akgentic.tool.workspace.readers import DocumentReader

from tests.conftest import MockActorAddress
from tests.workspace.conftest import WORKSPACE_NAME, DeadAddress

##
## Doubles
##


class FakeVectorStore:
    """``#VectorStore`` as this actor uses it — three methods and an ordered log."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.create_error: Exception | None = None
        self.remove_error: Exception | None = None

    def create_collection(self, name: str, config: CollectionConfig) -> None:
        self.calls.append(("create", (name, config)))
        if self.create_error is not None:
            raise self.create_error

    def add(
        self,
        collection: str,
        entries: list[Any],
        requester: Any = None,
        request_ref: str | None = None,
    ) -> None:
        self.calls.append(("add", (collection, list(entries), request_ref, requester)))

    def remove(
        self,
        collection: str,
        ref_ids: list[str],
        scope: str | None = None,
        path_prefix: str | None = None,
    ) -> None:
        self.calls.append(("remove", (collection, list(ref_ids), scope)))
        if self.remove_error is not None:
            raise self.remove_error

    def kinds(self) -> list[str]:
        """The call log reduced to its verbs, in order."""
        return [kind for kind, _ in self.calls]

    def of(self, kind: str) -> list[Any]:
        """Every payload recorded under *kind*, in order."""
        return [payload for recorded, payload in self.calls if recorded == kind]


class StateSpy:
    """Records every state-change notification, the shape 45-3's specs use."""

    def __init__(self) -> None:
        self.notifications: list[BaseState] = []

    def notify_state_change(self, state: BaseState) -> None:
        self.notifications.append(state)


class RagHarness:
    """Wires an inert actor to a fake vector store and a fake spawn path.

    Nothing here starts an actor system. ``createActor`` records the worker it was
    asked for and hands back a stand-in address; ``proxy_tell`` on that address
    records the :class:`IndexRequest` instead of running anything. Tests then
    deliver the worker's report themselves, which is what lets a spec exercise a
    report that arrives *late*, or for a file that has since moved on.
    """

    def __init__(self, actor: WorkspaceActor) -> None:
        self.actor = actor
        self.vs = FakeVectorStore()
        # ``DeadAddress`` rather than a plain stand-in: ``notify_state_change``
        # reaches into ``ActorAddressImpl._actor_ref`` for anything it believes is
        # alive, which no stand-in has. Reporting dead is the honest answer — there
        # is no orchestrator behind this address — and it keeps the actor's own
        # state notifications out of the way of what these specs are about.
        self.orchestrator = DeadAddress("orchestrator")
        self.vs_address: MockActorAddress | None = MockActorAddress("#VectorStore")
        self.requests: list[IndexRequest] = []
        self.worker_names: list[str] = []
        self.spawn_error: BaseException | None = None

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Point the actor's orchestrator, proxies and spawn path at this harness."""
        self.actor._orchestrator = self.orchestrator
        monkeypatch.setattr(self.actor, "proxy_ask", self._ask)
        monkeypatch.setattr(self.actor, "proxy_tell", self._tell)
        monkeypatch.setattr(self.actor, "createActor", self._create)

    def enable(
        self,
        params: WorkspaceRagIndex | None = None,
        reader: DocumentReader | None = None,
        collection: CollectionConfig | None = None,
        agent_id: str = "alice",
    ) -> None:
        """Announce retrieval exactly as a bound card does."""
        self.actor.enable_rag(
            agent_id,
            params or WorkspaceRagIndex(),
            reader or DocumentReader(llm_client=None),
            collection or CollectionConfig(backend="inmemory"),
        )

    def watch(self) -> StateSpy:
        """Attach a notification spy, discarding the attach-time notification."""
        spy = StateSpy()
        self.actor.state.observer(spy)
        spy.notifications.clear()
        return spy

    ##
    ## Driving the pipeline
    ##
    def report(
        self,
        path: str,
        chunks: int = 1,
        markdown: str = "# A\n\nbody\n",
        source_sha: str | None = None,
        extracted: bool = False,
        texts: list[str] | None = None,
    ) -> None:
        """Deliver an ``IndexResult`` for *path*, as its worker would."""
        sha = source_sha if source_sha is not None else self._sha_of(path)
        built = [
            RagChunk(
                chunk_id=chunk_id(WORKSPACE_NAME, path, sha, ordinal),
                ordinal=ordinal,
                start=0,
                end=len(markdown),
            )
            for ordinal in range(chunks)
        ]
        self.actor.receiveMsg_IndexResult(
            IndexResult(
                path=path,
                scope=WORKSPACE_NAME,
                source_sha=sha,
                markdown=markdown,
                extracted=extracted,
                chunks=built,
                texts=texts if texts is not None else [f"text {n}" for n in range(chunks)],
            )
        )

    def fail(self, path: str, reason: str = "boom", source_sha: str | None = None) -> None:
        """Deliver an ``IndexError`` for *path*, as its worker would."""
        self.actor.receiveMsg_IndexError(
            IndexError(
                path=path,
                scope=WORKSPACE_NAME,
                source_sha=source_sha if source_sha is not None else self._sha_of(path),
                reason=reason,
            )
        )

    def complete(
        self, path: str, error: str | None = None, collection: str = RAG_COLLECTION
    ) -> None:
        """Deliver one ``EmbeddingCompleted``, as ``#VectorStore`` would."""
        self.actor.receiveMsg_EmbeddingCompleted(
            EmbeddingCompleted(
                request_id="r", request_ref=path, collection=collection, count=1, error=error
            )
        )

    def _sha_of(self, path: str) -> str:
        entry = self.actor.state.rag_index.get(path)
        assert entry is not None and entry.indexed_sha is not None, f"{path} was never queued"
        return entry.indexed_sha

    ##
    ## Proxy plumbing
    ##
    def _ask(self, address: Any, actor_type: Any = None, timeout: int | None = None) -> Any:
        if address is self.orchestrator:
            return SimpleNamespace(get_team_member=lambda name: self.vs_address)
        if address is self.vs_address:
            return self.vs
        raise AssertionError(f"unexpected ask target {address}")

    def _tell(self, address: Any, actor_type: Any = None) -> Any:
        if address is self.vs_address:
            return self.vs
        return SimpleNamespace(receiveMsg_IndexRequest=self.requests.append)

    def _create(self, actor_class: Any, agent_id: Any = None, config: Any = None) -> Any:
        if self.spawn_error is not None:
            raise self.spawn_error
        assert actor_class is IndexWorker
        assert config is not None
        self.worker_names.append(config.name)
        return MockActorAddress(config.name, config.role)


##
## Fixtures
##


@pytest.fixture
def actor(workspace_tree: Path) -> WorkspaceActor:
    """A started actor over the test workspace, with no actor thread."""
    started = WorkspaceActor(
        config=WorkspaceConfig(
            name=workspace_actor_name(WORKSPACE_NAME),
            role=WORKSPACE_ACTOR_ROLE,
            workspace_name=WORKSPACE_NAME,
        )
    )
    started.on_start()
    return started


@pytest.fixture
def harness(actor: WorkspaceActor, monkeypatch: pytest.MonkeyPatch) -> RagHarness:
    """A harness already installed on the actor, with retrieval **not** yet on."""
    built = RagHarness(actor)
    built.install(monkeypatch)
    return built


def write(tree: Path, name: str, body: str = "# Title\n\nSome text.\n") -> str:
    """Write *name* into the tree and return the digest the actor will compute."""
    target = tree / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return content_sha(body.encode("utf-8"))


##
## Group C — enabling retrieval, and the collection created lazily
##


class TestEnableRag:
    """The card tells the actor; the actor never inspects a card."""

    def test_nothing_is_created_before_a_card_enables_retrieval(self, harness: RagHarness) -> None:
        """A workspace with retrieval off must never create a collection."""
        assert harness.vs.calls == []
        assert harness.actor._vs_proxy is None

    def test_enabling_creates_the_one_collection(self, harness: RagHarness) -> None:
        """Lazily, in ``enable_rag`` — never in ``on_start``."""
        harness.enable(collection=CollectionConfig(backend="inmemory", tenant="acme"))

        [(name, config)] = harness.vs.of("create")
        assert name == RAG_COLLECTION
        assert config.tenant == "acme"

    def test_the_card_collection_reaches_create_collection(self, harness: RagHarness) -> None:
        """``rag_collection`` is the card's only lever on the backend and the tenant."""
        harness.enable(collection=CollectionConfig(backend="weaviate", dimension=3072))

        [(_, config)] = harness.vs.of("create")
        assert (config.backend, config.dimension) == ("weaviate", 3072)

    def test_indexing_is_unavailable_until_retrieval_is_enabled(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """Degraded mode is a sentence, not an exception — this actor owns the gate."""
        write(workspace_tree, "a.md")

        assert harness.actor.index_paths("") == (
            "Retrieval indexing is not available for this workspace."
        )

    def test_a_missing_vector_store_degrades_rather_than_raising(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """``PlanActor`` raises here; this actor must not — it owns the write gate."""
        harness.vs_address = None

        harness.enable()  # must not raise

        assert harness.actor._vs_proxy is None
        assert harness.actor.index_paths("").startswith("Retrieval indexing is not available")

    def test_a_failing_create_collection_degrades_rather_than_raising(
        self, harness: RagHarness
    ) -> None:
        """A transient backend fault must not take the workspace down."""
        harness.vs.create_error = RuntimeError("cluster unreachable")

        harness.enable()  # must not raise

        assert harness.actor._vs_proxy is None

    def test_a_broken_proxy_never_raises_out_of_enable_rag(
        self, harness: RagHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Everything in this method is wrapped, including the orchestrator ask."""

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("proxy is gone")

        monkeypatch.setattr(harness.actor, "proxy_ask", boom)

        harness.enable()  # must not raise

        assert harness.actor._vs_proxy is None

    def test_the_first_enable_fixes_the_parameters_for_the_tree(self, harness: RagHarness) -> None:
        """Two agents on one team must not make one file chunk two ways."""
        first = WorkspaceRagIndex(chunk_chars=800)
        harness.enable(params=first)

        harness.enable(params=WorkspaceRagIndex(chunk_chars=1600), agent_id="bob")

        assert harness.actor._rag_params == first
        assert len(harness.vs.of("create")) == 1

    def test_a_second_enable_with_equal_parameters_changes_nothing(
        self, harness: RagHarness
    ) -> None:
        """Idempotent, and silent — the common case is two identical cards."""
        harness.enable(params=WorkspaceRagIndex(chunk_chars=800))
        harness.enable(params=WorkspaceRagIndex(chunk_chars=800))

        assert len(harness.vs.of("create")) == 1


##
## Group D — queueing, candidate discovery, and the spawn side
##


class TestCandidateDiscovery:
    """A candidate is a file the read path can already turn into text."""

    def test_text_and_document_extensions_are_queued_and_the_rest_counted(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """The set arithmetic is the one ``card/read.py`` already draws."""
        harness.enable()
        write(workspace_tree, "notes.md")
        write(workspace_tree, "data.csv", "a,b\n1,2\n")
        (workspace_tree / "photo.png").write_bytes(b"\x89PNG")
        (workspace_tree / "archive.zip").write_bytes(b"PK\x03\x04")

        answer = harness.actor.index_paths("")

        assert answer == "2 file(s) queued, 0 already current, 2 unsupported"
        assert set(harness.actor.state.rag_index) == {"notes.md", "data.csv"}

    def test_images_are_excluded_even_though_the_reader_claims_them(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """An OCR'd photograph is not what this index is for."""
        harness.enable()
        (workspace_tree / "scan.jpg").write_bytes(b"\xff\xd8\xff")

        assert harness.actor.index_paths("") == "0 file(s) queued, 0 already current, 1 unsupported"

    def test_subdirectories_are_walked(self, harness: RagHarness, workspace_tree: Path) -> None:
        """The whole tree, through ``Filesystem.list`` and nothing else."""
        harness.enable()
        write(workspace_tree, "top.md")
        write(workspace_tree, "deep/nested/inner.md")

        harness.actor.index_paths("")

        assert set(harness.actor.state.rag_index) == {"top.md", "deep/nested/inner.md"}

    def test_dot_prefixed_names_are_skipped(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """Staging files and vestigial extraction sidecars both start with a dot."""
        harness.enable()
        write(workspace_tree, "real.md")
        write(workspace_tree, ".report.pdf.md", "# leftover\n")

        harness.actor.index_paths("")

        assert set(harness.actor.state.rag_index) == {"real.md"}

    def test_a_single_file_path_is_a_legal_argument(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """``workspace_rag_index("notes.md")`` indexes that file and nothing else."""
        harness.enable()
        write(workspace_tree, "notes.md")
        write(workspace_tree, "other.md")

        harness.actor.index_paths("notes.md")

        assert set(harness.actor.state.rag_index) == {"notes.md"}

    def test_a_directory_path_indexes_what_is_under_it(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        harness.enable()
        write(workspace_tree, "docs/one.md")
        write(workspace_tree, "elsewhere/two.md")

        harness.actor.index_paths("docs")

        assert set(harness.actor.state.rag_index) == {"docs/one.md"}

    def test_a_path_that_escapes_the_root_is_skipped_not_raised(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """``workspace_rag_index`` is reachable from a model; it must never raise."""
        harness.enable()

        assert harness.actor.index_paths("../..") == (
            "0 file(s) queued, 0 already current, 0 unsupported"
        )
        assert harness.actor.state.rag_index == {}

    def test_a_missing_path_is_skipped_not_raised(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        harness.enable()

        assert harness.actor.index_paths("nowhere") == (
            "0 file(s) queued, 0 already current, 0 unsupported"
        )


class TestIdempotence:
    """A file already current at its live bytes is not re-indexed."""

    def test_an_embedded_file_at_the_same_digest_is_already_current(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        harness.enable()
        write(workspace_tree, "notes.md")
        harness.actor.index_paths("")
        harness.report("notes.md")
        harness.complete("notes.md")
        assert harness.actor.state.rag_index["notes.md"].status is RagStatus.EMBEDDED

        assert harness.actor.index_paths("") == (
            "0 file(s) queued, 1 already current, 0 unsupported"
        )

    def test_force_re_indexes_a_current_file(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """``force`` is exactly the override of the idempotence check."""
        harness.enable()
        write(workspace_tree, "notes.md")
        harness.actor.index_paths("")
        harness.report("notes.md")
        harness.complete("notes.md")

        answer = harness.actor.index_paths("", force=True)

        assert answer == "1 file(s) queued, 0 already current, 0 unsupported"

    def test_changed_bytes_are_queued_again_without_force(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """The cache key is the content, so a replaced file is a new file."""
        harness.enable()
        write(workspace_tree, "notes.md")
        harness.actor.index_paths("")
        harness.report("notes.md")
        harness.complete("notes.md")
        write(workspace_tree, "notes.md", "# Replaced\n\nOther text.\n")

        assert harness.actor.index_paths("") == (
            "1 file(s) queued, 0 already current, 0 unsupported"
        )

    def test_a_run_already_in_flight_over_the_same_bytes_is_not_restarted(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """Re-queueing it would reset a live run and spawn a second worker for it."""
        harness.enable()
        write(workspace_tree, "notes.md")
        harness.actor.index_paths("")
        spawned = len(harness.requests)

        assert harness.actor.index_paths("") == (
            "0 file(s) queued, 1 already current, 0 unsupported"
        )
        assert len(harness.requests) == spawned


class TestTheSpawnSide:
    """What the worker is handed, and how many workers exist at once."""

    def test_the_request_carries_everything_the_worker_needs(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """Notably the extraction configuration, which lives on the card."""
        reader = DocumentReader(llm_client=None, llm_model="test-model")
        params = WorkspaceRagIndex(chunk_chars=900)
        harness.enable(params=params, reader=reader)
        sha = write(workspace_tree, "notes.md")

        harness.actor.index_paths("")

        [request] = harness.requests
        assert request.path == "notes.md"
        assert request.scope == WORKSPACE_NAME
        assert request.source_sha == sha
        assert request.markdown is None
        assert request.params == params
        assert request.reader == reader

    def test_a_cached_body_is_handed_over_and_the_status_says_so(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """The worker then only splits — which is what ``SPLITTING`` records."""
        harness.enable()
        sha = write(workspace_tree, "notes.md")
        harness.actor.cache_document("notes.md", sha, EXTRACTOR_VERSION, "# Cached\n\nBody.\n")

        harness.actor.index_paths("")

        [request] = harness.requests
        assert request.markdown == "# Cached\n\nBody.\n"
        assert harness.actor.state.rag_index["notes.md"].status is RagStatus.SPLITTING

    def test_an_uncached_body_leaves_the_file_at_extraction(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        harness.enable()
        write(workspace_tree, "notes.md")

        harness.actor.index_paths("")

        assert harness.actor.state.rag_index["notes.md"].status is RagStatus.EXTRACTION

    def test_the_worker_name_starts_with_the_teardown_marker(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        harness.enable()
        write(workspace_tree, "notes.md")

        harness.actor.index_paths("")

        assert harness.worker_names[0].startswith("#index-")

    def test_no_more_than_the_cap_may_have_a_worker_at_once(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """500 candidates would otherwise be 500 actors in one mailbox turn."""
        harness.enable()
        for index in range(MAX_CONCURRENT_INDEX_WORKERS + 2):
            write(workspace_tree, f"file{index}.md")

        harness.actor.index_paths("")

        assert len(harness.requests) == MAX_CONCURRENT_INDEX_WORKERS
        pending = [
            path
            for path, entry in harness.actor.state.rag_index.items()
            if entry.status is RagStatus.PENDING
        ]
        assert len(pending) == 2

    def test_a_settling_file_drains_the_next_pending_one(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """Which is what makes ``PENDING`` a queue rather than a dead end."""
        harness.enable()
        for index in range(MAX_CONCURRENT_INDEX_WORKERS + 1):
            write(workspace_tree, f"file{index}.md")
        harness.actor.index_paths("")
        first = harness.requests[0].path

        harness.report(first)

        assert len(harness.requests) == MAX_CONCURRENT_INDEX_WORKERS + 1

    def test_an_index_error_also_drains(self, harness: RagHarness, workspace_tree: Path) -> None:
        harness.enable()
        for index in range(MAX_CONCURRENT_INDEX_WORKERS + 1):
            write(workspace_tree, f"file{index}.md")
        harness.actor.index_paths("")

        harness.fail(harness.requests[0].path)

        assert len(harness.requests) == MAX_CONCURRENT_INDEX_WORKERS + 1

    def test_a_spawn_failure_fails_the_file_rather_than_looping(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """A file left ``PENDING`` with no worker would be drained for ever."""
        harness.enable()
        write(workspace_tree, "notes.md")
        harness.spawn_error = RuntimeError("no thread available")

        harness.actor.index_paths("")

        entry = harness.actor.state.rag_index["notes.md"]
        assert entry.status is RagStatus.FAILED
        assert "no thread available" in (entry.reason or "")


##
## Group D/E — the settle side
##


class TestBatching:
    """``EmbeddingService.embed`` sends every text in one request, so batches matter."""

    def test_a_large_file_is_split_into_ceil_n_over_the_batch_size_calls(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """An 800-page document is one request that fails whole, unless batched."""
        harness.enable()
        write(workspace_tree, "big.md")
        harness.actor.index_paths("")

        harness.report("big.md", chunks=EMBED_BATCH_SIZE * 2 + 1)

        adds = harness.vs.of("add")
        assert len(adds) == 3
        assert [len(entries) for _, entries, _, _ in adds] == [
            EMBED_BATCH_SIZE,
            EMBED_BATCH_SIZE,
            1,
        ]
        assert harness.actor.state.rag_index["big.md"].batches_expected == 3

    def test_every_batch_is_correlated_by_path_and_addressed_to_this_actor(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """``request_ref`` is how a completion finds the row that is counting it."""
        harness.enable()
        write(workspace_tree, "big.md")
        harness.actor.index_paths("")

        harness.report("big.md", chunks=EMBED_BATCH_SIZE + 1)

        for collection, _, request_ref, requester in harness.vs.of("add"):
            assert collection == RAG_COLLECTION
            assert request_ref == "big.md"
            # ``myAddress`` builds a fresh wrapper per call, so identity is the
            # agent id rather than the object.
            assert requester.agent_id == harness.actor.myAddress.agent_id

    def test_each_entry_carries_the_scope_the_path_and_the_ordinal(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """Without them a scoped removal removes nothing and a scoped search finds nothing."""
        harness.enable()
        write(workspace_tree, "notes.md")
        harness.actor.index_paths("")

        harness.report("notes.md", chunks=2, texts=["first", "second"])

        [(_, entries, _, _)] = harness.vs.of("add")
        assert [entry.scope for entry in entries] == [WORKSPACE_NAME, WORKSPACE_NAME]
        assert [entry.path for entry in entries] == ["notes.md", "notes.md"]
        assert [entry.ordinal for entry in entries] == [0, 1]
        assert [entry.text for entry in entries] == ["first", "second"]
        assert all(entry.vector == [] for entry in entries)

    def test_embedded_is_reached_only_after_the_last_batch(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """A file claimed ``EMBEDDED`` early is a file search cannot fully find."""
        harness.enable()
        write(workspace_tree, "big.md")
        harness.actor.index_paths("")
        harness.report("big.md", chunks=EMBED_BATCH_SIZE * 2 + 1)

        harness.complete("big.md")
        assert harness.actor.state.rag_index["big.md"].status is RagStatus.EMBEDDING
        harness.complete("big.md")
        assert harness.actor.state.rag_index["big.md"].status is RagStatus.EMBEDDING
        harness.complete("big.md")
        assert harness.actor.state.rag_index["big.md"].status is RagStatus.EMBEDDED

    def test_a_failing_batch_fails_the_file_and_later_batches_are_ignored(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """Batch 2's error, then batch 3's success — the status must not move again."""
        harness.enable()
        write(workspace_tree, "big.md")
        harness.actor.index_paths("")
        harness.report("big.md", chunks=EMBED_BATCH_SIZE * 2 + 1)

        harness.complete("big.md")  # batch 1 lands
        harness.complete("big.md", error="rate limited")  # batch 2 fails
        failed_at = harness.actor.state.rag_index["big.md"].updated_at
        harness.complete("big.md")  # batch 3 succeeds, and is dropped

        entry = harness.actor.state.rag_index["big.md"]
        assert entry.status is RagStatus.FAILED
        assert entry.reason == "rate limited"
        assert entry.updated_at == failed_at

    def test_a_completion_for_another_collection_is_ignored(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """The actor is a requester for one collection and must not read another's."""
        harness.enable()
        write(workspace_tree, "notes.md")
        harness.actor.index_paths("")
        harness.report("notes.md")

        harness.complete("notes.md", collection="Planning")

        assert harness.actor.state.rag_index["notes.md"].status is RagStatus.EMBEDDING

    def test_a_completion_for_an_unknown_path_is_ignored(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        harness.enable()

        harness.complete("never-seen.md")  # must not raise

        assert harness.actor.state.rag_index == {}

    def test_only_the_final_transition_notifies(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """A 1,900-chunk document must cost one event, not thirty."""
        harness.enable()
        write(workspace_tree, "big.md")
        harness.actor.index_paths("")
        harness.report("big.md", chunks=EMBED_BATCH_SIZE * 2 + 1)
        spy = harness.watch()

        harness.complete("big.md")
        harness.complete("big.md")
        assert spy.notifications == []

        harness.complete("big.md")
        assert len(spy.notifications) == 1


class TestReIndexOrdering:
    """Add-then-remove, never the other way round, and the removal is wrapped."""

    def _reindex(self, harness: RagHarness, tree: Path) -> list[str]:
        """Index, embed, change the file, index again — and return the old ids."""
        harness.enable()
        write(tree, "notes.md")
        harness.actor.index_paths("")
        harness.report("notes.md", chunks=2)
        harness.complete("notes.md")
        old_ids = [c.chunk_id for c in harness.actor.state.rag_index["notes.md"].chunks]

        write(tree, "notes.md", "# Replaced\n\nOther text.\n")
        harness.actor.index_paths("")
        return old_ids

    def test_the_old_ids_are_held_while_the_new_ones_are_produced(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """``chunks`` must hold the **new** set so a landing batch can be attributed."""
        old_ids = self._reindex(harness, workspace_tree)

        entry = harness.actor.state.rag_index["notes.md"]
        assert entry.superseded_chunk_ids == old_ids
        assert entry.chunks == []

    def test_every_add_precedes_every_remove_for_the_path(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """The other order leaves the file absent from search while the list calls it stale."""
        self._reindex(harness, workspace_tree)
        harness.vs.calls.clear()

        harness.report("notes.md", chunks=2)
        harness.complete("notes.md")

        kinds = harness.vs.kinds()
        assert "add" in kinds and "remove" in kinds
        assert max(i for i, k in enumerate(kinds) if k == "add") < kinds.index("remove")

    def test_the_removal_is_scoped_to_this_workspace(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """One collection holds every workspace; an unscoped removal is a cross-tree one."""
        old_ids = self._reindex(harness, workspace_tree)
        harness.report("notes.md", chunks=2)
        harness.complete("notes.md")

        [(collection, ref_ids, scope)] = harness.vs.of("remove")
        assert collection == RAG_COLLECTION
        assert ref_ids == old_ids
        assert scope == WORKSPACE_NAME

    def test_a_successful_removal_clears_the_superseded_ids(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        self._reindex(harness, workspace_tree)
        harness.report("notes.md", chunks=2)
        harness.complete("notes.md")

        assert harness.actor.state.rag_index["notes.md"].superseded_chunk_ids == []

    def test_a_failing_removal_keeps_the_ids_and_does_not_fail_the_file(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """``remove`` re-raises a missing collection as ``RetriableError``."""
        old_ids = self._reindex(harness, workspace_tree)
        harness.vs.remove_error = RuntimeError("collection is gone")

        harness.report("notes.md", chunks=2)
        harness.complete("notes.md")

        entry = harness.actor.state.rag_index["notes.md"]
        assert entry.status is RagStatus.EMBEDDED
        assert entry.superseded_chunk_ids == old_ids

    def test_a_failed_re_index_leaves_the_old_chunks_in_place(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """A stale file stays searchable at its previous content."""
        self._reindex(harness, workspace_tree)
        harness.report("notes.md", chunks=2)

        harness.complete("notes.md", error="rate limited")

        assert harness.actor.state.rag_index["notes.md"].status is RagStatus.FAILED
        assert harness.vs.of("remove") == []


class TestReportAttribution:
    """A report belongs to the run that issued it, and to nothing else."""

    def test_a_report_for_a_digest_that_has_moved_on_is_dropped(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """Applying it would overwrite a live run's chunk set with a stale one."""
        harness.enable()
        write(workspace_tree, "notes.md")
        harness.actor.index_paths("")

        harness.report("notes.md", chunks=5, source_sha="a-digest-nobody-is-waiting-for")

        entry = harness.actor.state.rag_index["notes.md"]
        assert entry.status is RagStatus.EXTRACTION
        assert entry.chunks == []
        assert harness.vs.of("add") == []

    def test_a_report_for_an_unknown_path_is_dropped(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        harness.enable()

        harness.actor.receiveMsg_IndexResult(
            IndexResult(
                path="never-queued.md",
                scope=WORKSPACE_NAME,
                source_sha="x",
                markdown="body",
                extracted=True,
                chunks=[],
                texts=[],
            )
        )

        assert harness.actor.state.rag_index == {}

    def test_a_worker_extracted_body_fills_the_extraction_cache(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """A read that follows the index therefore costs no second extraction."""
        harness.enable()
        sha = write(workspace_tree, "report.md")
        harness.actor.index_paths("")

        harness.report("report.md", markdown="# From the worker\n", extracted=True)

        assert harness.actor.document_extract("report.md", sha, EXTRACTOR_VERSION) == (
            "# From the worker\n"
        )

    def test_a_supplied_body_does_not_refill_the_cache(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """It came from there; writing it back would be a notify for nothing."""
        harness.enable()
        write(workspace_tree, "notes.md")
        harness.actor.index_paths("")

        harness.report("notes.md", extracted=False)

        assert harness.actor.state.documents == {}

    def test_an_empty_document_settles_immediately(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """Nothing to embed is a settled file, not a file waiting for a signal."""
        harness.enable()
        write(workspace_tree, "empty.md", "   \n")
        harness.actor.index_paths("")

        harness.report("empty.md", chunks=0)

        entry = harness.actor.state.rag_index["empty.md"]
        assert entry.status is RagStatus.EMBEDDED
        assert (entry.chunk_count, entry.batches_expected) == (0, 0)
        assert harness.vs.of("add") == []

    def test_mismatched_chunks_and_texts_fail_the_file(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """The strict zip that follows would otherwise raise inside the handler."""
        harness.enable()
        write(workspace_tree, "notes.md")
        harness.actor.index_paths("")

        harness.report("notes.md", chunks=3, texts=["only one"])

        assert harness.actor.state.rag_index["notes.md"].status is RagStatus.FAILED
        assert harness.vs.of("add") == []

    def test_an_index_error_fails_the_file_and_keeps_its_chunks(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        harness.enable()
        write(workspace_tree, "notes.md")
        harness.actor.index_paths("")
        harness.report("notes.md", chunks=2)
        harness.complete("notes.md")
        write(workspace_tree, "notes.md", "# Replaced\n")
        harness.actor.index_paths("")

        harness.fail("notes.md", reason="RuntimeError: extractor died")

        entry = harness.actor.state.rag_index["notes.md"]
        assert entry.status is RagStatus.FAILED
        assert entry.reason == "RuntimeError: extractor died"

    def test_a_handler_never_raises_out_of_the_actor(
        self, harness: RagHarness, workspace_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An exception in a document handler would kill the actor that owns the gate."""
        harness.enable()
        write(workspace_tree, "notes.md")
        harness.actor.index_paths("")

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("state is broken")

        monkeypatch.setattr(harness.actor, "_on_index_result", boom)
        monkeypatch.setattr(harness.actor, "_on_embedding_completed", boom)

        harness.report("notes.md")  # must not raise
        harness.complete("notes.md")  # must not raise


##
## Group E — the gate marks stale, and does not re-index
##


class TestTheGateMarksStale:
    """One direct call on ``self`` from the one point six mutations converge on."""

    def _indexed(self, harness: RagHarness, tree: Path, name: str = "notes.md") -> None:
        harness.enable()
        write(tree, name)
        harness.actor.index_paths("")
        harness.report(name)
        harness.complete(name)
        assert harness.actor.state.rag_index[name].status is RagStatus.EMBEDDED

    def test_an_accepted_write_marks_the_file_stale(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        self._indexed(harness, workspace_tree)
        harness.actor.record_observation(
            "alice", "notes.md", _observation_of(workspace_tree / "notes.md")
        )

        harness.actor.apply_write("alice", "notes.md", "# Rewritten\n")

        assert harness.actor.state.rag_index["notes.md"].status is RagStatus.STALE

    def test_an_accepted_delete_marks_the_file_stale(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """``_forget`` appends to the write set too, so deletes are covered for free."""
        self._indexed(harness, workspace_tree)
        harness.actor.record_observation(
            "alice", "notes.md", _observation_of(workspace_tree / "notes.md")
        )

        harness.actor.apply_delete("alice", "notes.md")

        assert harness.actor.state.rag_index["notes.md"].status is RagStatus.STALE

    def test_it_marks_and_does_not_re_index(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """Gate writes mark stale; uploads index. Auto-indexing every save would pay twice."""
        self._indexed(harness, workspace_tree)
        harness.actor.record_observation(
            "alice", "notes.md", _observation_of(workspace_tree / "notes.md")
        )
        spawned = len(harness.requests)

        harness.actor.apply_write("alice", "notes.md", "# Rewritten\n")

        assert len(harness.requests) == spawned

    def test_a_refused_mutation_marks_nothing(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """``_accept`` never ran, so there is no write set and nothing changed."""
        self._indexed(harness, workspace_tree)

        harness.actor.apply_write("alice", "notes.md", "# Rewritten\n")

        assert harness.actor.state.rag_index["notes.md"].status is RagStatus.EMBEDDED

    def test_a_tree_that_was_never_indexed_pays_no_notify(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """The common case, and it must stay free on the mutation path."""
        spy = harness.watch()

        harness.actor.apply_write("alice", "fresh.md", "# New\n")

        assert spy.notifications == []

    def test_marking_an_already_stale_file_notifies_nothing(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """A no-op must not be an event."""
        self._indexed(harness, workspace_tree)
        harness.actor.mark_paths_stale(["notes.md"])
        spy = harness.watch()

        harness.actor.mark_paths_stale(["notes.md"])

        assert spy.notifications == []

    def test_marking_an_unindexed_path_notifies_nothing(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        spy = harness.watch()

        harness.actor.mark_paths_stale(["never-indexed.md"])

        assert spy.notifications == []

    def test_marking_a_real_change_notifies_exactly_once(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """Once per call, however many paths moved."""
        self._indexed(harness, workspace_tree, "one.md")
        write(workspace_tree, "two.md")
        harness.actor.index_paths("")
        harness.report("two.md")
        harness.complete("two.md")
        spy = harness.watch()

        harness.actor.mark_paths_stale(["one.md", "two.md"])

        assert len(spy.notifications) == 1


##
## Group F — the snapshot the render is built from
##


class TestRagSnapshot:
    """A render, and therefore free — no file access of any kind."""

    def _seed(self, actor: WorkspaceActor, pending: int, embedded: int) -> None:
        now = datetime.now(UTC)
        for index in range(embedded):
            actor.state.rag_index[f"done{index}.md"] = RagFile(
                path=f"done{index}.md",
                status=RagStatus.EMBEDDED,
                chunk_count=3,
                updated_at=now,
            )
        for index in range(pending):
            actor.state.rag_index[f"wait{index}.md"] = RagFile(
                path=f"wait{index}.md", status=RagStatus.PENDING, updated_at=now
            )

    def test_pending_rows_are_capped_and_the_rest_counted(self, actor: WorkspaceActor) -> None:
        """A 10,000-file tree must not flood the context window with identical rows."""
        self._seed(actor, pending=10, embedded=2)

        state = actor.rag_snapshot(max_pending_shown=3)

        assert state.pending_hidden == 7
        assert sum(1 for row in state.rows if row.status == "pending") == 3

    def test_everything_that_is_not_pending_is_always_shown(self, actor: WorkspaceActor) -> None:
        """Those rows each say something different; pending rows all say the same."""
        self._seed(actor, pending=50, embedded=6)

        state = actor.rag_snapshot(max_pending_shown=1)

        assert sum(1 for row in state.rows if row.status == "embedded") == 6

    def test_a_failure_reason_reaches_the_row(self, actor: WorkspaceActor) -> None:
        actor.state.rag_index["a.md"] = RagFile(
            path="a.md",
            status=RagStatus.FAILED,
            reason="rate limited",
            updated_at=datetime.now(UTC),
        )

        [row] = actor.rag_snapshot(max_pending_shown=5).rows
        assert (row.status, row.reason) == ("failed", "rate limited")

    def test_the_snapshot_performs_no_file_access_at_all(
        self, actor: WorkspaceActor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It runs once per turn per agent; a tree walk here is a walk on the hot path.

        Every filesystem entry point is **recorded** rather than made to raise.
        Raising works and was tried first, and it takes pytest's own reporting down
        with it — the traceback formatter reads source files through the very
        methods a blanket patch replaces, so the failure arrives as an
        ``INTERNALERROR`` instead of as a readable assertion. A recorder gives the
        same verdict and names what was touched.
        """
        from akgentic.tool.workspace.workspace import Filesystem

        touched: list[str] = []

        def recorder(name: str) -> Any:
            original = getattr(Filesystem, name)

            def wrapper(self: Filesystem, *args: Any, **kwargs: Any) -> Any:
                touched.append(name)
                return original(self, *args, **kwargs)

            return wrapper

        for name in ("read", "read_bytes", "list", "exists", "write", "delete", "mkdir"):
            monkeypatch.setattr(Filesystem, name, recorder(name))
        monkeypatch.setattr(Filesystem, "_validate_path", recorder("_validate_path"))
        self._seed(actor, pending=2, embedded=2)

        state = actor.rag_snapshot(max_pending_shown=20)

        assert touched == [], f"the snapshot touched the filesystem: {touched}"
        assert len(state.rows) == 4

    def test_the_snapshot_does_not_run_the_reaper(self, actor: WorkspaceActor) -> None:
        """A state mutation from a render would fire on every turn of every agent."""
        actor.state.rag_index["a.md"] = RagFile(
            path="a.md",
            status=RagStatus.EMBEDDING,
            updated_at=datetime.now(UTC) - timedelta(seconds=EMBEDDING_STALE_AFTER_S + 1),
        )

        actor.rag_snapshot(max_pending_shown=20)

        assert actor.state.rag_index["a.md"].status is RagStatus.EMBEDDING

    def test_a_restored_snapshot_is_reaped_on_the_way_in(
        self, actor: WorkspaceActor
    ) -> None:
        """``on_start`` cannot do it: it assigns a fresh state before any restore.

        The resume hook is ``init_state`` — what ``akgentic-team``'s restorer
        calls with the persisted snapshot — so that is where the bound is applied,
        and this spec is what says the criterion's *intent* is met rather than its
        letter.
        """
        restored = WorkspaceState()
        restored.rag_index["a.md"] = RagFile(
            path="a.md",
            status=RagStatus.EMBEDDING,
            indexed_sha="old",
            updated_at=datetime.now(UTC) - timedelta(seconds=EMBEDDING_STALE_AFTER_S + 1),
        )

        actor.init_state(restored)

        assert actor.state.rag_index["a.md"].status is RagStatus.PENDING

    def test_a_restored_snapshot_inside_the_bound_is_left_alone(
        self, actor: WorkspaceActor
    ) -> None:
        """A restart during a live embed still has a signal that may arrive."""
        restored = WorkspaceState()
        restored.rag_index["a.md"] = RagFile(
            path="a.md",
            status=RagStatus.EMBEDDING,
            updated_at=datetime.now(UTC),
        )

        actor.init_state(restored)

        assert actor.state.rag_index["a.md"].status is RagStatus.EMBEDDING

    def test_index_paths_does_run_the_reaper(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """``on_start`` and here — the two places that are not a turn path."""
        harness.enable()
        harness.actor.state.rag_index["a.md"] = RagFile(
            path="a.md",
            status=RagStatus.EMBEDDING,
            indexed_sha="old",
            updated_at=datetime.now(UTC) - timedelta(seconds=EMBEDDING_STALE_AFTER_S + 1),
        )

        harness.actor.index_paths("nowhere")

        assert harness.actor.state.rag_index["a.md"].status is not RagStatus.EMBEDDING


def _observation_of(path: Path) -> Any:
    """The observation an agent that read *path* whole would have recorded."""
    from akgentic.tool.workspace.models import Observation

    return Observation(sha=content_sha(path.read_bytes()), full=True)
