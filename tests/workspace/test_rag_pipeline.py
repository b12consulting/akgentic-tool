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

import logging
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from akgentic.tool.vector_store.actor import VectorStoreActor
from akgentic.tool.vector_store.embedding_actor import (
    EmbeddingError,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingWorker,
)
from akgentic.tool.vector_store.protocol import (
    CollectionStatus,
    SearchHit,
    SearchResult,
    VectorStoreParam,
)
from akgentic.tool.vector_store.vector import VectorEntry
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
    DocumentExtract,
    RagChunk,
    RagFile,
    RagStatus,
    chunk_id,
)
from akgentic.tool.workspace.documents.worker import (
    EMBED_BATCH_SIZE,
    MAX_CONCURRENT_INDEX_WORKERS,
    IndexFailure,
    IndexRequest,
    IndexResult,
    IndexWorker,
)
from akgentic.tool.workspace.models import WorkspaceConfig, content_sha
from akgentic.tool.workspace.readers import DocumentReader
from tests.conftest import MockActorAddress
from tests.workspace.conftest import (
    WORKSPACE_PATH,
    RecordingDocumentStore,
    attach_store,
    factory_for,
    seed_extract,
    seed_row,
    stored_docs,
    stored_rows,
    watch_store,
)

_DOCUMENTS_LOGGER = "akgentic.tool.workspace.actor.documents"
_UNAVAILABLE = "Retrieval indexing is not available for this workspace."
_NO_HITS = (
    "Nothing in the retrieval index matched that query. "
    "Use workspace_rag_list to see which files are indexed."
)

##
## Doubles
##


class FakeVectorStore:
    """The store child as this actor uses it — four methods and an ordered log.

    ``search`` answers from what ``add`` put in and ``remove`` took out, with no
    similarity at all: every held entry in the asked scope is a hit at score
    ``1.0``. That is enough for the restore specs, whose question is whether the
    engine *holds* a chunk, not how well it matches.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.create_error: Exception | None = None
        self.remove_error: Exception | None = None
        self.add_error: Exception | None = None
        self.held: list[VectorEntry] = []

    def create_collection(self, name: str, config: VectorStoreParam) -> None:
        self.calls.append(("create", (name, config)))
        if self.create_error is not None:
            raise self.create_error

    def add(self, collection: str, entries: list[Any]) -> None:
        self.calls.append(("add", (collection, list(entries))))
        if self.add_error is not None:
            raise self.add_error
        self.held.extend(entries)

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
        gone = set(ref_ids)
        self.held = [entry for entry in self.held if entry.ref_id not in gone]

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        scope: str | None = None,
        path_prefix: str | None = None,
    ) -> SearchResult:
        self.calls.append(("search", (collection, top_k, scope, path_prefix)))
        hits = [
            SearchHit(
                ref_type=entry.ref_type,
                ref_id=entry.ref_id,
                text=entry.text,
                score=1.0,
                scope=entry.scope,
                path=entry.path,
                ordinal=entry.ordinal,
            )
            for entry in self.held
            if (scope is None or entry.scope == scope)
            and (not path_prefix or (entry.path or "").startswith(path_prefix))
        ]
        return SearchResult(hits=hits[:top_k], status=CollectionStatus.READY)

    def kinds(self) -> list[str]:
        """The call log reduced to its verbs, in order."""
        return [kind for kind, _ in self.calls]

    def of(self, kind: str) -> list[Any]:
        """Every payload recorded under *kind*, in order."""
        return [payload for recorded, payload in self.calls if recorded == kind]


def _embedded(ref_id: str, path: str = "a.md", ordinal: int = 0) -> VectorEntry:
    """One entry as an ``#embed-`` worker hands it back — vector filled in."""
    return VectorEntry(
        ref_type="workspace_chunk",
        ref_id=ref_id,
        text=f"text {ordinal}",
        vector=[0.1, 0.2],
        scope=WORKSPACE_PATH,
        path=path,
        ordinal=ordinal,
    )


class StaticEmbedder:
    """The query leg's embedder, answering one fixed vector and never a network."""

    def __init__(self) -> None:
        self.embeds: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embeds.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


def _explodes(*args: Any, **kwargs: Any) -> Any:
    """Raise whatever it is called with — a seam a spec asserts is never reached."""
    raise RuntimeError("this seam must not be reached")


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
        self.embedder = StaticEmbedder()
        # The actor is handed **no** orchestrator at all — ``install`` sets the
        # slot to ``None`` — because that is what a hosted actor gets, and an ask
        # to one is the trap ``_ask`` springs rather than a lookup it answers.
        # Nothing resolves a store here any more: the card announces one.
        self.vs_address = MockActorAddress("#VectorStore-child")
        self.store_configs: list[Any] = []
        """Every store actor ``createActor`` was asked for — permanently empty now.

        Kept rather than deleted: it is the negative that "the actor creates no
        store of its own" is asserted on, and a list nothing appends to is only
        worth having while something still *would* append to it if the branch
        came back.
        """
        self.requests: list[IndexRequest] = []
        self.embed_requests: list[EmbeddingRequest] = []
        self.worker_names: list[str] = []
        self.embed_worker_names: list[str] = []
        self.spawn_error: BaseException | None = None
        self.embed_spawn_error: BaseException | None = None
        self.index_root = Path(tempfile.mkdtemp(prefix="rag-index-"))
        """The ``<meta>`` a card would stamp onto the param it announces."""
        self._monkeypatch: pytest.MonkeyPatch | None = None

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Take the actor's orchestrator away and point its proxies and spawns here."""
        self._monkeypatch = monkeypatch
        self.actor._orchestrator = None
        # A card would announce this at bind time; a directly built actor gets
        # none, and every document path would silently degrade to a miss.
        attach_store(self.actor)
        monkeypatch.setattr(self.actor, "proxy_ask", self._ask)
        monkeypatch.setattr(self.actor, "proxy_tell", self._tell)
        monkeypatch.setattr(self.actor, "createActor", self._create)

    def enable(
        self,
        params: WorkspaceRagIndex | None = None,
        reader: DocumentReader | None = None,
        collection: VectorStoreParam | None = None,
        agent_id: str = "alice",
        announce: bool = True,
        store: Any = None,
    ) -> None:
        """Announce the store and then retrieval, exactly as a bound card does.

        **The two announcements are in the card's order**, and the harness keeps
        them that way rather than assigning the slot: the actor enables retrieval
        on the turn ``enable_rag`` arrives, so a store handed over afterwards
        would arrive to an actor that has already decided it has none.

        ``announce=False`` is the lost-announcement case — the only degraded path
        this actor has left, since it resolves nothing of its own.
        """
        if announce:
            self.actor.configure_vector_store(store if store is not None else self.vs)
        self.actor.enable_rag(
            agent_id,
            params or WorkspaceRagIndex(),
            reader or DocumentReader(llm_client=None),
            collection or VectorStoreParam(backend="local", root=str(self.index_root)),
        )
        # ``enable_rag`` builds a real ``EmbeddingService`` from the card's param;
        # a search in these specs must embed through the double, never a network.
        if self.actor._embedder is not None:
            self.actor._embedder = self.embedder

    @property
    def store_names(self) -> list[str]:
        """The name of every store child ``createActor`` was asked for, in order."""
        return [config.name for config in self.store_configs]

    def record_writes(self) -> RecordingDocumentStore:
        """Record every store write the actor performs from now on.

        The successor to ``record_deltas``: the persist points it counted are
        gone, and what a spec asks instead is which records a turn wrote. The
        recorder delegates to a real store, so a later read-back still sees
        exactly what the actor wrote.
        """
        return watch_store(self.actor)

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
                chunk_id=chunk_id(WORKSPACE_PATH, path, sha, ordinal),
                ordinal=ordinal,
                start=0,
                end=len(markdown),
            )
            for ordinal in range(chunks)
        ]
        self.actor.receiveMsg_IndexResult(
            IndexResult(
                path=path,
                scope=WORKSPACE_PATH,
                source_sha=sha,
                markdown=markdown,
                extracted=extracted,
                chunks=built,
                texts=texts if texts is not None else [f"text {n}" for n in range(chunks)],
            )
        )

    def fail(self, path: str, reason: str = "boom", source_sha: str | None = None) -> None:
        """Deliver an ``IndexFailure`` for *path*, as its worker would."""
        self.actor.receiveMsg_IndexFailure(
            IndexFailure(
                path=path,
                scope=WORKSPACE_PATH,
                source_sha=source_sha if source_sha is not None else self._sha_of(path),
                reason=reason,
            )
        )

    def result(
        self,
        path: str,
        entries: list[VectorEntry] | None = None,
        collection: str = RAG_COLLECTION,
    ) -> None:
        """Deliver one ``EmbeddingResult``, as an ``#embed-`` worker would."""
        self.actor.receiveMsg_EmbeddingResult(
            EmbeddingResult(
                collection=collection,
                entries=entries if entries is not None else [_embedded("e1")],
                request_id="r",
                request_ref=path,
            )
        )

    def error(self, path: str, reason: str = "boom", collection: str = RAG_COLLECTION) -> None:
        """Deliver one ``EmbeddingError``, as an ``#embed-`` worker would."""
        self.actor.receiveMsg_EmbeddingError(
            EmbeddingError(collection=collection, error=reason, request_id="r", request_ref=path)
        )

    def _sha_of(self, path: str) -> str:
        row = stored_rows(self.actor).get(path)
        assert row is not None and row.indexed_sha is not None, f"{path} was never queued"
        return row.indexed_sha

    ##
    ## Reading the index back — always off the disk, through a second object
    ##
    @property
    def rows(self) -> dict[str, RagFile]:
        """The retrieval index as the disk holds it, keyed by path."""
        return stored_rows(self.actor)

    @property
    def docs(self) -> dict[str, DocumentExtract]:
        """The extraction cache as the disk holds it, keyed by path."""
        return stored_docs(self.actor)

    ##
    ## Proxy plumbing
    ##
    def _ask(self, address: Any, actor_type: Any = None, timeout: int | None = None) -> Any:
        if address is self.vs_address:
            return self.vs
        raise AssertionError(f"unexpected ask target {address}")

    def _tell(self, address: Any, actor_type: Any = None) -> Any:
        if address is self.vs_address:
            return self.vs
        return SimpleNamespace(
            receiveMsg_IndexRequest=self.requests.append,
            receiveMsg_DeferredPayload=self.embed_requests.append,
        )

    def _create(self, actor_class: Any, agent_id: Any = None, config: Any = None) -> Any:
        assert config is not None
        if actor_class is VectorStoreActor:
            # Recorded rather than refused, so ``store_configs == []`` is a
            # negative about behaviour and not about this branch being missing.
            self.store_configs.append(config)
            return self.vs_address
        if actor_class is EmbeddingWorker:
            if self.embed_spawn_error is not None:
                raise self.embed_spawn_error
            self.embed_worker_names.append(config.name)
            return MockActorAddress(config.name, config.role)
        if self.spawn_error is not None:
            raise self.spawn_error
        assert actor_class is IndexWorker
        self.worker_names.append(config.name)
        return MockActorAddress(config.name, config.role)


##
## Fixtures
##


def _started_actor(workspace_path: str) -> WorkspaceActor:
    """A started actor over *workspace_path*, with no actor thread."""
    started = WorkspaceActor(
        config=WorkspaceConfig(
            name=workspace_actor_name(workspace_path),
            role=WORKSPACE_ACTOR_ROLE,
            workspace_path=workspace_path,
        )
    )
    started.on_start()
    # The card announces this at bind time; a directly built actor gets none.
    return attach_store(started)


@pytest.fixture
def actor(workspace_tree: Path) -> WorkspaceActor:
    """A started actor over the test workspace, with no actor thread."""
    return _started_actor(WORKSPACE_PATH)


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
        harness.enable(collection=VectorStoreParam(backend="inmemory", tenant="acme"))

        [(name, config)] = harness.vs.of("create")
        assert name == RAG_COLLECTION
        assert config.tenant == "acme"

    def test_the_card_collection_reaches_create_collection(self, harness: RagHarness) -> None:
        """``vector_store`` is the card's only lever on the backend and the tenant."""
        harness.enable(collection=VectorStoreParam(backend="inmemory", dimension=3072))

        [(_, config)] = harness.vs.of("create")
        assert (config.backend, config.dimension) == ("inmemory", 3072)

    def test_a_cluster_param_binds_the_announced_store_and_builds_nothing(
        self, harness: RagHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """**Premise reversed (was: the actor calls the factory).**

        Story 51-1 put backend construction on this actor; the card does it now
        (ADR-051 Decision 8), so the registered factory must not be reached from
        here at all — whatever the param names. A factory that raises is the
        falsifier: if this actor still called one, the spec would degrade instead
        of binding.
        """
        built: list[object] = []
        double = FakeVectorStore()

        def _factory(context: object) -> object:
            built.append(context)
            raise AssertionError("the actor built a backend of its own")

        with factory_for("weaviate", _factory):
            harness.enable(
                collection=VectorStoreParam(backend="weaviate", dimension=3072), store=double
            )

        assert built == []
        assert harness.actor._vs_proxy is double
        assert harness.store_names == []
        [(name, config)] = double.of("create")
        assert name == RAG_COLLECTION
        assert (config.backend, config.dimension) == ("weaviate", 3072)

    def test_a_store_that_was_never_announced_degrades_without_raising(
        self,
        harness: RagHarness,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """**Premise reversed (was: a factory that raises).**

        The actor resolves nothing, so a failed connect is now the card's to
        report and the only degraded path left here is a lost announcement. It
        degrades exactly as every earlier cause did: one WARNING naming the
        workspace, no proxy, no traceback, and the sentence from the indexer.
        """
        with caplog.at_level(logging.WARNING, logger=_DOCUMENTS_LOGGER):
            harness.enable(announce=False)  # must not raise

        assert harness.actor._vs_proxy is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert WORKSPACE_PATH in warnings[0].getMessage()
        assert "no vector store was announced" in warnings[0].getMessage()
        assert warnings[0].exc_info is None
        assert harness.actor.index_paths("") == _UNAVAILABLE

    def test_indexing_is_unavailable_until_retrieval_is_enabled(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """Degraded mode is a sentence, not an exception — this actor owns the gate."""
        write(workspace_tree, "a.md")

        assert harness.actor.index_paths("") == (
            "Retrieval indexing is not available for this workspace."
        )

    def test_the_actor_creates_no_store_of_its_own_for_any_param(self, harness: RagHarness) -> None:
        """**Premise reversed (was: the spawn that the environment refused).**

        Both halves of the pair this replaces — an outage degrading and a defect
        propagating — described a ``createActor`` that no longer exists. What
        survives them is the rule they were both about: this actor resolves
        nothing. ``createActor`` is watched rather than removed, so the branch
        coming back turns this red instead of going unnoticed.
        """
        harness.enable(collection=VectorStoreParam(backend="inmemory"))

        assert harness.store_configs == []
        assert harness.actor._vs_proxy is harness.vs

    def test_resolving_the_store_asks_nobody_and_cannot_raise(self, harness: RagHarness) -> None:
        """It returns the announced object, or ``None``, and does nothing else.

        The ask proxy is made to explode: a resolve that still looked anything up
        would raise through it rather than answering.
        """
        harness.actor.proxy_ask = _explodes  # type: ignore[method-assign]
        harness.actor.createActor = _explodes  # type: ignore[method-assign]

        assert harness.actor._resolve_store() is None

        harness.actor.configure_vector_store(harness.vs)
        assert harness.actor._resolve_store() is harness.vs

    def test_a_failing_embedder_never_raises_out_of_enable_rag(
        self, harness: RagHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """**Premise re-pointed (was: a broken proxy).**

        ``_acquire_vs_proxy`` no longer builds a proxy, but it still builds an
        embedder, and everything it does is inside one ``try``. A raise there
        must leave retrieval off rather than taking the actor down.
        """
        from akgentic.tool.vector_store import embedding_actor

        monkeypatch.setattr(embedding_actor, "build_embedding_service", _explodes)

        harness.enable()  # must not raise

        assert harness.actor._vs_proxy is None

    def test_a_failing_create_collection_degrades_rather_than_raising(
        self, harness: RagHarness
    ) -> None:
        """A transient backend fault must not take the workspace down."""
        harness.vs.create_error = RuntimeError("cluster unreachable")

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


class TestTheStoreArrivesByAnnouncement:
    """**Premise reversed.** There is no store child: the card hands one over.

    Story 51-1 made the in-memory store this actor's own child, which core
    ADR-022 Decision 3 required of a *hosted* actor. Nothing is hosted after this
    epic, and ADR-022 Decision 2 records that the prohibition was always on the
    actor rather than on the card — so the binding goes back to the card, which
    is what lets a workspace share the team's one ``#VectorStore`` with whatever
    planning or knowledge-graph card the team also carries.

    What survives 51-1 unchanged is the invariant the class below it still holds:
    the actor asks its orchestrator for nothing.
    """

    def test_a_workspace_with_no_orchestrator_indexes_a_document(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """The hosted-style falsifier: no orchestrator anywhere, and the pipeline runs whole.

        The harness answers no orchestrator ask — there is none to answer — so a
        resolve that still went looking for the team's store would raise inside
        ``enable_rag``, degrade, and leave every assertion below red.
        """
        assert harness.actor.orchestrator is None
        write(workspace_tree, "a.md")

        harness.enable()

        assert harness.actor._vs_proxy is harness.vs
        assert harness.actor.index_paths("") == (
            "1 file(s) queued, 0 already current, 0 unsupported"
        )
        assert len(harness.worker_names) == 1
        harness.report("a.md")
        harness.result("a.md")
        [(collection, entries)] = harness.vs.of("add")
        assert collection == RAG_COLLECTION
        assert [entry.ref_id for entry in entries] == ["e1"]
        assert harness.rows["a.md"].status is RagStatus.EMBEDDED

    def test_enabling_on_a_file_backed_param_creates_no_child(self, harness: RagHarness) -> None:
        """**Inverted.** The card created the store; this actor binds what it was given."""
        harness.enable()

        assert harness.store_configs == []
        assert harness.actor._vs_proxy is harness.vs

    def test_enabling_on_a_cluster_param_creates_no_child_either(self, harness: RagHarness) -> None:
        """Neither branch creates one now — the difference between them left this actor."""
        double = FakeVectorStore()

        harness.enable(
            collection=VectorStoreParam(backend="weaviate", dimension=3072), store=double
        )

        assert harness.actor._vs_proxy is double
        assert harness.store_configs == []

    def test_two_workspaces_bind_their_own_announced_store_and_create_nothing(
        self, workspace_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """**Inverted.** Two trees held two children; they hold two announcements.

        The two stores must still not be the same object — that is what the two
        differently-named children used to guarantee — so the identity of each
        actor's bound store is the assertion the name check becomes.
        """
        first = RagHarness(_started_actor(WORKSPACE_PATH))
        first.install(monkeypatch)
        second = RagHarness(_started_actor("u-bob/other-workspace"))
        second.install(monkeypatch)

        first.enable()
        second.enable()

        assert first.store_configs == []
        assert second.store_configs == []
        assert first.actor._vs_proxy is first.vs
        assert second.actor._vs_proxy is second.vs
        assert first.actor._vs_proxy is not second.actor._vs_proxy

    def test_the_announcement_must_land_before_retrieval_is_enabled(
        self, harness: RagHarness
    ) -> None:
        """The ordering the card establishes, seen from the actor's side.

        A store configured *after* ``enable_rag`` arrives too late: the first call
        wins for the tree, so the actor has already settled into degraded mode and
        a later announcement binds nothing.
        """
        harness.enable(announce=False)
        assert harness.actor._vs_proxy is None

        harness.actor.configure_vector_store(harness.vs)

        assert harness.actor._vs_proxy is None
        assert harness.actor.index_paths("") == _UNAVAILABLE


class TestTheEmbeddedReMarkIsDeleted:
    """AC 15 (52-3): an ``EMBEDDED`` row over a fresh engine stays ``EMBEDDED``.

    **This class is the inverse of the one it replaces, deliberately.**
    ``_requeue_embedded_rows`` put every ``EMBEDDED`` row back to ``PENDING`` the
    moment an in-memory store child was created, because a child of a hosted
    actor has no checkpoint and therefore held nothing the rows claimed
    (story 51-1, ADR-049 Decision 7).

    With the index on disk that premise is gone, and **porting the rule would
    blank a good cache on every process start** — one full re-extraction and
    re-embedding of every indexed file, on every restart, charged to whoever pays
    for embeddings. These specs exist so that a future reader who finds an
    ``EMBEDDED`` row over a fresh engine and thinks "surely this should be
    re-queued" gets a red test instead of a plausible-looking commit.

    **The one-story window is closed.** Between 52-3 and this story an in-memory
    engine on a fresh process was genuinely empty while its rows said
    ``EMBEDDED``. The embeddings are under ``<meta>/index/`` now, so an
    ``EMBEDDED`` row over a fresh process is **honest** — and the case that would
    still have been dishonest, a workspace running on the in-actor index, is
    refused at bind rather than left to be re-marked
    (``TestAWorkspaceMayNotRunOnTheInActorBackend``, ADR-051 Decision 11).
    """

    _BODY = "# A\n\nbody\n"

    def _stored(self, harness: RagHarness, tree: Path) -> RagChunk:
        """One ``EMBEDDED`` row for ``a.md`` on disk at the live digest, body evicted."""
        sha = write(tree, "a.md", self._BODY)
        now = datetime.now(UTC)
        old = RagChunk(
            chunk_id=chunk_id(WORKSPACE_PATH, "a.md", sha, 0),
            ordinal=0,
            start=0,
            end=len(self._BODY),
            heading_path=["A"],
        )
        seed_row(
            harness.actor,
            "a.md",
            RagFile(
                path="a.md",
                status=RagStatus.EMBEDDED,
                indexed_sha=sha,
                chunks=[old],
                chunk_count=1,
                updated_at=now,
            ),
        )
        seed_extract(
            harness.actor,
            "a.md",
            DocumentExtract(
                path="a.md",
                source_sha=sha,
                extractor_version=EXTRACTOR_VERSION,
                markdown=None,
                char_count=len(self._BODY),
                extracted_at=now,
            ),
        )
        return old

    def test_enabling_an_in_memory_engine_leaves_the_row_embedded(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """The headline: nothing is re-marked, and its chunk set is untouched."""
        old = self._stored(harness, workspace_tree)

        harness.enable()

        row = harness.rows["a.md"]
        assert row.status is RagStatus.EMBEDDED
        assert row.chunks == [old]
        assert row.superseded_chunk_ids == []

    def test_no_index_worker_is_spawned_for_it(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """A re-mark would make it ``PENDING``, which ``_drain`` spawns for.

        Asserting the spawn rather than only the status is what makes this
        guard bite: the harness accepts a report for a run nobody issued, so a
        status assertion alone would not notice a worker that started.
        """
        self._stored(harness, workspace_tree)

        harness.enable()

        assert harness.worker_names == []
        assert harness.requests == []
        assert harness.actor.index_paths("") == (
            "0 file(s) queued, 1 already current, 0 unsupported"
        )
        assert harness.worker_names == []

    def test_enabling_writes_nothing_at_all(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """A re-mark is a write, so counting writes is the direct question."""
        self._stored(harness, workspace_tree)
        writes = harness.record_writes()

        harness.enable()

        assert writes.written == []

    def test_the_cluster_branch_is_unchanged_and_binds_the_double(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """The branch that never re-marked still does not, and still binds."""
        self._stored(harness, workspace_tree)
        double = FakeVectorStore()

        harness.enable(
            collection=VectorStoreParam(backend="weaviate", dimension=3072), store=double
        )

        assert harness.actor._vs_proxy is double
        assert harness.store_configs == []
        assert harness.rows["a.md"].status is RagStatus.EMBEDDED
        assert harness.worker_names == []

    def test_a_fresh_index_enables_without_writing(self, harness: RagHarness) -> None:
        """The other half: no row at all, nothing written, and the store is bound."""
        writes = harness.record_writes()

        harness.enable()

        assert harness.actor._vs_proxy is harness.vs
        assert writes.written == []


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
        assert set(harness.rows) == {"notes.md", "data.csv"}

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

        assert set(harness.rows) == {"top.md", "deep/nested/inner.md"}

    def test_dot_prefixed_names_are_skipped(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """Staging files and vestigial extraction sidecars both start with a dot."""
        harness.enable()
        write(workspace_tree, "real.md")
        write(workspace_tree, ".report.pdf.md", "# leftover\n")

        harness.actor.index_paths("")

        assert set(harness.rows) == {"real.md"}

    def test_a_single_file_path_is_a_legal_argument(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """``workspace_rag_index("notes.md")`` indexes that file and nothing else."""
        harness.enable()
        write(workspace_tree, "notes.md")
        write(workspace_tree, "other.md")

        harness.actor.index_paths("notes.md")

        assert set(harness.rows) == {"notes.md"}

    def test_a_directory_path_indexes_what_is_under_it(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        harness.enable()
        write(workspace_tree, "docs/one.md")
        write(workspace_tree, "elsewhere/two.md")

        harness.actor.index_paths("docs")

        assert set(harness.rows) == {"docs/one.md"}

    def test_a_path_that_escapes_the_root_is_skipped_not_raised(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """``workspace_rag_index`` is reachable from a model; it must never raise."""
        harness.enable()

        assert harness.actor.index_paths("../..") == (
            "0 file(s) queued, 0 already current, 0 unsupported"
        )
        assert harness.rows == {}

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
        harness.result("notes.md")
        assert harness.rows["notes.md"].status is RagStatus.EMBEDDED

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
        harness.result("notes.md")

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
        harness.result("notes.md")
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
        assert request.scope == WORKSPACE_PATH
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
        assert harness.rows["notes.md"].status is RagStatus.SPLITTING

    def test_an_uncached_body_leaves_the_file_at_extraction(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        harness.enable()
        write(workspace_tree, "notes.md")

        harness.actor.index_paths("")

        assert harness.rows["notes.md"].status is RagStatus.EXTRACTION

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
            path for path, entry in harness.rows.items() if entry.status is RagStatus.PENDING
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

        entry = harness.rows["notes.md"]
        assert entry.status is RagStatus.FAILED
        assert "no thread available" in (entry.reason or "")


##
## Group D/E — the settle side
##


class TestBatching:
    """``EmbeddingService.embed`` sends every text in one request, so batches matter."""

    def test_a_large_file_spawns_ceil_n_over_the_batch_size_workers(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """An 800-page document is one request that fails whole, unless batched."""
        harness.enable()
        write(workspace_tree, "big.md")
        harness.actor.index_paths("")

        harness.report("big.md", chunks=EMBED_BATCH_SIZE * 2 + 1)

        assert len(harness.embed_requests) == 3
        assert [len(request.entries) for request in harness.embed_requests] == [
            EMBED_BATCH_SIZE,
            EMBED_BATCH_SIZE,
            1,
        ]
        assert len(harness.embed_worker_names) == 3
        assert all(name.startswith("#embed-") for name in harness.embed_worker_names)
        assert harness.rows["big.md"].batches_expected == 3

    def test_no_write_reaches_the_store_on_the_spawn_turn(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """The write now happens when a worker reports, never when one is spawned."""
        harness.enable()
        write(workspace_tree, "big.md")
        harness.actor.index_paths("")

        harness.report("big.md", chunks=EMBED_BATCH_SIZE + 1)

        assert harness.vs.of("add") == []

    def test_every_request_carries_the_path_as_request_ref(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """``request_ref`` is how a report finds the row that is counting it."""
        harness.enable()
        write(workspace_tree, "big.md")
        harness.actor.index_paths("")

        harness.report("big.md", chunks=EMBED_BATCH_SIZE + 1)

        for request in harness.embed_requests:
            assert request.collection == RAG_COLLECTION
            assert request.request_ref == "big.md"
            assert request.deferred_key

    def test_the_cards_own_embedding_model_reaches_the_payload(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """AC 14: the consumer's ``VectorStoreParam`` is what embeds, from this story on."""
        harness.enable(
            collection=VectorStoreParam(
                backend="inmemory",
                embedding_model="text-embedding-3-large",
                dimension=3072,
            )
        )
        write(workspace_tree, "notes.md")
        harness.actor.index_paths("")

        harness.report("notes.md", chunks=2)

        [request] = harness.embed_requests
        assert request.embedding_model == "text-embedding-3-large"
        assert request.embedding_provider == "openai"

    def test_each_entry_carries_the_scope_the_path_and_the_ordinal(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """Without them a scoped removal removes nothing and a scoped search finds nothing."""
        harness.enable()
        write(workspace_tree, "notes.md")
        harness.actor.index_paths("")

        harness.report("notes.md", chunks=2, texts=["first", "second"])

        [request] = harness.embed_requests
        entries = request.entries
        assert [entry.scope for entry in entries] == [WORKSPACE_PATH, WORKSPACE_PATH]
        assert [entry.path for entry in entries] == ["notes.md", "notes.md"]
        assert [entry.ordinal for entry in entries] == [0, 1]
        assert [entry.text for entry in entries] == ["first", "second"]
        assert all(entry.vector == [] for entry in entries)

    def test_a_spawn_failure_fails_the_file_and_stops_issuing(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """A worker that never started can never report, so the row must not wait."""
        harness.enable()
        write(workspace_tree, "big.md")
        harness.actor.index_paths("")
        harness.embed_spawn_error = RuntimeError("no thread")

        harness.report("big.md", chunks=EMBED_BATCH_SIZE * 2 + 1)

        entry = harness.rows["big.md"]
        assert entry.status is RagStatus.FAILED
        assert "no thread" in (entry.reason or "")
        assert harness.embed_requests == []

    def test_a_lost_collection_param_fails_the_file_rather_than_parking_it(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """Every exit from the spawn helper either starts a worker or settles the row.

        ``batches_expected`` is written before the first spawn, so an exit that
        does neither leaves the file at ``EMBEDDING`` waiting on a report nobody
        will send — the reaper livelock the write-side gate exists to remove.
        """
        harness.enable()
        write(workspace_tree, "big.md")
        harness.actor.index_paths("")
        harness.actor._rag_collection = None

        harness.report("big.md", chunks=EMBED_BATCH_SIZE + 1)

        entry = harness.rows["big.md"]
        assert entry.status is RagStatus.FAILED
        assert harness.embed_requests == []

    def test_embedded_is_reached_only_after_the_last_batch(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """A file claimed ``EMBEDDED`` early is a file search cannot fully find."""
        harness.enable()
        write(workspace_tree, "big.md")
        harness.actor.index_paths("")
        harness.report("big.md", chunks=EMBED_BATCH_SIZE * 2 + 1)

        harness.result("big.md")
        assert harness.rows["big.md"].status is RagStatus.EMBEDDING
        assert harness.rows["big.md"].batches_landed == 1
        harness.result("big.md")
        assert harness.rows["big.md"].status is RagStatus.EMBEDDING
        harness.result("big.md")
        assert harness.rows["big.md"].status is RagStatus.EMBEDDED

    def test_each_result_is_written_by_ask_with_its_own_entries(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """AC 15: one ``add`` per report, carrying that report's batch."""
        harness.enable()
        write(workspace_tree, "big.md")
        harness.actor.index_paths("")
        harness.report("big.md", chunks=EMBED_BATCH_SIZE + 1)

        harness.result("big.md", entries=[_embedded("first")])

        [(collection, entries)] = harness.vs.of("add")
        assert collection == RAG_COLLECTION
        assert [entry.ref_id for entry in entries] == ["first"]
        assert harness.rows["big.md"].status is RagStatus.EMBEDDING
        assert harness.rows["big.md"].batches_landed == 1

    def test_a_failing_batch_fails_the_file_and_later_batches_are_ignored(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """Batch 2's error, then batch 3's success — the status must not move again."""
        harness.enable()
        write(workspace_tree, "big.md")
        harness.actor.index_paths("")
        harness.report("big.md", chunks=EMBED_BATCH_SIZE * 2 + 1)

        harness.result("big.md")  # batch 1 lands
        harness.error("big.md", reason="rate limited")  # batch 2 fails
        failed_at = harness.rows["big.md"].updated_at
        harness.result("big.md")  # batch 3 succeeds, and is dropped

        entry = harness.rows["big.md"]
        assert entry.status is RagStatus.FAILED
        assert entry.reason == "rate limited"
        assert entry.updated_at == failed_at

    def test_a_report_for_another_collection_is_ignored(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """The actor counts for one collection and must not read another's."""
        harness.enable()
        write(workspace_tree, "notes.md")
        harness.actor.index_paths("")
        harness.report("notes.md")

        harness.result("notes.md", collection="Planning")

        assert harness.rows["notes.md"].status is RagStatus.EMBEDDING
        assert harness.vs.of("add") == []

    def test_a_report_for_an_unknown_path_is_ignored(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        harness.enable()

        harness.result("never-seen.md")  # must not raise
        harness.error("never-seen.md")  # must not raise

        assert harness.rows == {}
        assert harness.vs.of("add") == []

    def test_every_landing_batch_is_written_including_the_intermediate_ones(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """**Deliberately inverted from what 51-4 guarded**, and named so.

        A batch that landed without settling its file used to write nothing on
        its own turn: the row was marked dirty and rode on the next delta, so a
        1,900-chunk document cost one delta rather than thirty. Story 52-3
        removes the dirty set — that was in-memory write-behind state, which is
        exactly what moving the index to disk exists to remove — so a
        1,900-chunk document now costs ``ceil(chunks / EMBED_BATCH_SIZE)``
        rewrites of one file.

        It is a cost this story accepts rather than engineers around, for a
        second reason as well as the first: the counter has to survive a crash,
        or the file parks at ``EMBEDDING`` until the reaper finds it ten minutes
        later and the whole extraction repeats.
        """
        harness.enable()
        write(workspace_tree, "big.md")
        harness.actor.index_paths("")
        harness.report("big.md", chunks=EMBED_BATCH_SIZE * 2 + 1)
        assert harness.rows["big.md"].batches_expected == 3
        writes = harness.record_writes()

        harness.result("big.md")
        assert harness.rows["big.md"].batches_landed == 1
        harness.result("big.md")
        assert harness.rows["big.md"].batches_landed == 2
        # The counter is on disk after every batch, not only at the settle —
        # which is the property the crash argument turns on.
        assert writes.written == ["big.md", "big.md"]

        harness.result("big.md")
        assert writes.written == ["big.md", "big.md", "big.md"]
        settled = harness.rows["big.md"]
        assert settled.status is RagStatus.EMBEDDED
        assert settled.batches_landed == 3


class TestTheWriteSide:
    """AC 16: the gate moved to the write, and a write that cannot land is visible."""

    def _two_batches(self, harness: RagHarness, workspace_tree: Path) -> None:
        harness.enable()
        write(workspace_tree, "big.md")
        harness.actor.index_paths("")
        harness.report("big.md", chunks=EMBED_BATCH_SIZE + 1)

    def test_a_write_that_cannot_land_fails_the_file_on_that_turn(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        from akgentic.tool.errors import RetriableError

        self._two_batches(harness, workspace_tree)
        harness.vs.add_error = RetriableError("Collection 'workspace_chunks' does not exist")
        writes = harness.record_writes()

        harness.result("big.md")

        entry = harness.rows["big.md"]
        assert entry.status is RagStatus.FAILED
        assert "does not exist" in (entry.reason or "")
        assert entry.batches_landed == 0
        assert writes.written == ["big.md"]
        assert harness.vs.of("remove") == []

    def test_a_second_result_after_a_failed_write_is_dropped(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        from akgentic.tool.errors import RetriableError

        self._two_batches(harness, workspace_tree)
        harness.vs.add_error = RetriableError("dead cluster")
        harness.result("big.md")
        failed_at = harness.rows["big.md"].updated_at
        harness.vs.add_error = None
        writes = harness.record_writes()

        harness.result("big.md")

        entry = harness.rows["big.md"]
        assert entry.status is RagStatus.FAILED
        assert entry.updated_at == failed_at
        assert writes.written == []

    def test_an_embedding_error_fails_the_file_the_same_way(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        self._two_batches(harness, workspace_tree)
        writes = harness.record_writes()

        harness.error("big.md", reason="rate limited")

        entry = harness.rows["big.md"]
        assert entry.status is RagStatus.FAILED
        assert entry.reason == "rate limited"
        assert writes.written == ["big.md"]

    def test_the_tell_proxy_is_gone(self, harness: RagHarness) -> None:
        """AC 16: one proxy, and every call through it is an ask."""
        harness.enable()
        assert not hasattr(harness.actor, "_vs_tell")


class TestReIndexOrdering:
    """Add-then-remove, never the other way round, and the removal is wrapped."""

    def _reindex(self, harness: RagHarness, tree: Path) -> list[str]:
        """Index, embed, change the file, index again — and return the old ids."""
        harness.enable()
        write(tree, "notes.md")
        harness.actor.index_paths("")
        harness.report("notes.md", chunks=2)
        harness.result("notes.md")
        old_ids = [c.chunk_id for c in harness.rows["notes.md"].chunks]

        write(tree, "notes.md", "# Replaced\n\nOther text.\n")
        harness.actor.index_paths("")
        return old_ids

    def test_the_old_ids_are_held_while_the_new_ones_are_produced(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """``chunks`` must hold the **new** set so a landing batch can be attributed."""
        old_ids = self._reindex(harness, workspace_tree)

        entry = harness.rows["notes.md"]
        assert entry.superseded_chunk_ids == old_ids
        assert entry.chunks == []

    def test_every_add_precedes_every_remove_for_the_path(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """The other order leaves the file absent from search while the list calls it stale."""
        self._reindex(harness, workspace_tree)
        harness.vs.calls.clear()

        harness.report("notes.md", chunks=2)
        harness.result("notes.md")

        kinds = harness.vs.kinds()
        assert "add" in kinds and "remove" in kinds
        assert max(i for i, k in enumerate(kinds) if k == "add") < kinds.index("remove")

    def test_the_removal_is_scoped_to_this_workspace(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """One collection holds every workspace; an unscoped removal is a cross-tree one."""
        old_ids = self._reindex(harness, workspace_tree)
        harness.report("notes.md", chunks=2)
        harness.result("notes.md")

        [(collection, ref_ids, scope)] = harness.vs.of("remove")
        assert collection == RAG_COLLECTION
        assert ref_ids == old_ids
        assert scope == WORKSPACE_PATH

    def test_a_successful_removal_clears_the_superseded_ids(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        self._reindex(harness, workspace_tree)
        harness.report("notes.md", chunks=2)
        harness.result("notes.md")

        assert harness.rows["notes.md"].superseded_chunk_ids == []

    def test_a_failing_removal_keeps_the_ids_and_does_not_fail_the_file(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """``remove`` re-raises a missing collection as ``RetriableError``."""
        old_ids = self._reindex(harness, workspace_tree)
        harness.vs.remove_error = RuntimeError("collection is gone")

        harness.report("notes.md", chunks=2)
        harness.result("notes.md")

        entry = harness.rows["notes.md"]
        assert entry.status is RagStatus.EMBEDDED
        assert entry.superseded_chunk_ids == old_ids

    def test_a_failed_re_index_leaves_the_old_chunks_in_place(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """A stale file stays searchable at its previous content."""
        self._reindex(harness, workspace_tree)
        harness.report("notes.md", chunks=2)

        harness.error("notes.md", reason="rate limited")

        assert harness.rows["notes.md"].status is RagStatus.FAILED
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

        entry = harness.rows["notes.md"]
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
                scope=WORKSPACE_PATH,
                source_sha="x",
                markdown="body",
                extracted=True,
                chunks=[],
                texts=[],
            )
        )

        assert harness.rows == {}

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

        assert harness.docs == {}

    def test_an_empty_document_settles_immediately(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """Nothing to embed is a settled file, not a file waiting for a signal."""
        harness.enable()
        write(workspace_tree, "empty.md", "   \n")
        harness.actor.index_paths("")

        harness.report("empty.md", chunks=0)

        entry = harness.rows["empty.md"]
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

        assert harness.rows["notes.md"].status is RagStatus.FAILED
        assert harness.vs.of("add") == []

    def test_an_index_error_fails_the_file_and_keeps_its_chunks(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        harness.enable()
        write(workspace_tree, "notes.md")
        harness.actor.index_paths("")
        harness.report("notes.md", chunks=2)
        harness.result("notes.md")
        write(workspace_tree, "notes.md", "# Replaced\n")
        harness.actor.index_paths("")

        harness.fail("notes.md", reason="RuntimeError: extractor died")

        entry = harness.rows["notes.md"]
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
        monkeypatch.setattr(harness.actor, "_on_embedding_result", boom)
        monkeypatch.setattr(harness.actor, "_on_embedding_error", boom)

        harness.report("notes.md")  # must not raise
        harness.result("notes.md")  # must not raise
        harness.error("notes.md")  # must not raise


##
## Group E — the gate marks stale, and does not re-index
##


class TestTheGateMarksStale:
    """The actor's half of the stale-mark: what ``mark_paths_stale`` does with a set.

    **Re-pointed by decision.** The gate used to call this directly on ``self``
    from ``_journalled``; it is card-side since 52-5 and arrives as a **tell**,
    so this class now drives the receiving end with the sets a mutation would
    send. The sending end — that every accepted mutation sends exactly the paths
    it touched, deletes included, and that a refused one sends nothing — is
    ``TestTheStaleMarkReachesTheIndex`` in ``test_write_gate.py``.
    """

    def _indexed(self, harness: RagHarness, tree: Path, name: str = "notes.md") -> None:
        harness.enable()
        write(tree, name)
        harness.actor.index_paths("")
        harness.report(name)
        harness.result(name)
        assert harness.rows[name].status is RagStatus.EMBEDDED

    def test_an_accepted_write_marks_the_file_stale(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        self._indexed(harness, workspace_tree)

        harness.actor.mark_paths_stale(["notes.md"])

        assert harness.rows["notes.md"].status is RagStatus.STALE

    def test_a_deleted_path_marks_stale_the_same_way(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """A delete joins the write set too, so it arrives here as any write does."""
        self._indexed(harness, workspace_tree)
        (workspace_tree / "notes.md").unlink()

        harness.actor.mark_paths_stale(["notes.md"])

        assert harness.rows["notes.md"].status is RagStatus.STALE

    def test_it_marks_and_does_not_re_index(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """Gate writes mark stale; uploads index. Auto-indexing every save would pay twice."""
        self._indexed(harness, workspace_tree)
        spawned = len(harness.requests)

        harness.actor.mark_paths_stale(["notes.md"])

        assert len(harness.requests) == spawned

    def test_a_tree_that_was_never_indexed_pays_no_delta(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """The common case, and it must stay free on the mutation path."""
        write(workspace_tree, "fresh.md")
        writes = harness.record_writes()

        harness.actor.mark_paths_stale(["fresh.md"])

        assert writes.written == []

    def test_marking_an_already_stale_file_sends_nothing(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """A no-op must not be a delta."""
        self._indexed(harness, workspace_tree)
        harness.actor.mark_paths_stale(["notes.md"])
        assert harness.rows["notes.md"].status is RagStatus.STALE
        writes = harness.record_writes()

        harness.actor.mark_paths_stale(["notes.md"])

        assert writes.written == []

    def test_marking_an_unindexed_path_sends_nothing(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        writes = harness.record_writes()

        harness.actor.mark_paths_stale(["never-indexed.md"])

        assert writes.written == []

    def test_marking_a_real_change_persists_exactly_once(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """Once per call, however many paths moved."""
        self._indexed(harness, workspace_tree, "one.md")
        write(workspace_tree, "two.md")
        harness.actor.index_paths("")
        harness.report("two.md")
        harness.result("two.md")
        writes = harness.record_writes()

        harness.actor.mark_paths_stale(["one.md", "two.md"])

        assert sorted(writes.written) == ["one.md", "two.md"]


##
## Group F — the snapshot the render is built from
##


class TestRagSnapshot:
    """A render, and therefore free — no file access of any kind."""

    def _seed(self, actor: WorkspaceActor, pending: int, embedded: int) -> None:
        now = datetime.now(UTC)
        for index in range(embedded):
            seed_row(
                actor,
                f"done{index}.md",
                RagFile(
                    path=f"done{index}.md",
                    status=RagStatus.EMBEDDED,
                    chunk_count=3,
                    updated_at=now,
                ),
            )
        for index in range(pending):
            seed_row(
                actor,
                f"wait{index}.md",
                RagFile(path=f"wait{index}.md", status=RagStatus.PENDING, updated_at=now),
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
        seed_row(
            actor,
            "a.md",
            RagFile(
                path="a.md",
                status=RagStatus.FAILED,
                reason="rate limited",
                updated_at=datetime.now(UTC),
            ),
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
        """A mutation from a render would fire on every turn of every agent."""
        seed_row(
            actor,
            "a.md",
            RagFile(
                path="a.md",
                status=RagStatus.EMBEDDING,
                updated_at=datetime.now(UTC) - timedelta(seconds=EMBEDDING_STALE_AFTER_S + 1),
            ),
        )

        actor.rag_snapshot(max_pending_shown=20)

        assert stored_rows(actor)["a.md"].status is RagStatus.EMBEDDING

    def test_the_render_is_sorted_by_path(self, actor: WorkspaceActor) -> None:
        """A directory glob's order is the file system's, so the render sorts.

        Without it the same index renders in a different order on two turns,
        which reads to an agent as though the index had changed.
        """
        now = datetime.now(UTC)
        for name in ("zebra.md", "alpha.md", "middle.md"):
            seed_row(actor, name, RagFile(path=name, status=RagStatus.EMBEDDED, updated_at=now))

        state = actor.rag_snapshot(max_pending_shown=20)

        assert [row.path for row in state.rows] == ["alpha.md", "middle.md", "zebra.md"]

    def test_index_paths_does_run_the_reaper(
        self, harness: RagHarness, workspace_tree: Path
    ) -> None:
        """``on_start`` and here — the two places that are not a turn path."""
        harness.enable()
        seed_row(
            harness.actor,
            "a.md",
            RagFile(
                path="a.md",
                status=RagStatus.EMBEDDING,
                indexed_sha="old",
                updated_at=datetime.now(UTC) - timedelta(seconds=EMBEDDING_STALE_AFTER_S + 1),
            ),
        )

        harness.actor.index_paths("nowhere")

        assert harness.rows["a.md"].status is not RagStatus.EMBEDDING


def _observation_of(path: Path) -> Any:
    """The observation an agent that read *path* whole would have recorded."""
    from akgentic.tool.workspace.models import Observation

    return Observation(sha=content_sha(path.read_bytes()), full=True)
