"""A search and ``#Workspace``'s mailbox do not wait for each other, either way.

**Three properties, and the loose form of each is untestable.** None of them is
"the search runs on a particular thread" — a closure runs on whichever thread
calls it, and here that is the test's.

1. *While a search's external call is in flight, the ``#Workspace`` mailbox is not
   held*, so another message on that mailbox is answered. That is what
   :mod:`akgentic.tool.workspace.actor.__init__`'s "and never external" claims,
   and what a query embed plus a ``store.search`` on the actor's own turn
   falsified. Story 55-6.
2. *While the ``#Workspace`` mailbox is held by a turn on the **indexing** path, a
   search still answers* — the converse, and the one story 57-1 is about. Until
   then the keyword leg, the fusion and the render were an ask, so a search
   queued behind every index worker's report.
3. *Two cards of one team search at once rather than serialising*, which one
   mailbox cannot produce.

Property 2 is deliberately **not** property 1 with the arguments swapped: parking
a *vector* call would prove nothing about story 57-1, because the vector leg was
already card-side before it. The mailbox must be held by indexing.

**It lives in its own module rather than in ``test_rag_search.py``** for one
reason: that module drives an inert actor through a harness built for it
(``SearchHarness``, a store announced by hand), and the properties here are only
observable against a **live** actor on a real thread, with a real mailbox, a real
exec lease and a real card closure. Nothing of either harness is usable by the
other.

**Property 1's seam is ``VectorStoreService.search``**, and that is the
load-bearing choice. It is the one seam that existed in **both** shapes of story
55-6 — on the actor's turn before it, on the caller's thread after — so the same
source asserts the same thing on either side of that move. A stand-in that
stalled the actor's embedder instead would have been **vacuous afterwards**,
because the actor no longer embeds. The embed is stalled too, as a second
parameterised case, because the audit's finding names both calls.

Since story 57-1 **no** part of a search is on that mailbox at all — the keyword
leg, the fusion and the render left with the vector leg — so property 1 is now
about a call the mailbox never sees either way, and it stays because a future
change that put one back would redden it.

**Property 2's seam is ``VectorStoreService.add``**, called by
``DocumentsMixin._on_embedding_result`` on the actor's own turn, one ask per
landing batch. It is on the indexing path by construction, which is what property
2 requires and what property 1's seam cannot give it.

**Two sentinels stop each property from passing vacuously**, and both are
mandatory for each:

- ``entered`` is asserted to have been **set**. A stand-in the production path
  never reached would otherwise leave the probe or the search unobstructed and
  the guard green with the defect in place.
- a positive control runs the same probe or search with the seam **unarmed**, and
  it must answer. A mis-wired one that never worked at all would otherwise make
  the guard green for the opposite reason.

Property 3's equivalents are the arrival count reaching **two** and an unarmed
control in which both searches answer.

**Red-before is a bounded failure, not a hang.** Every probe and every search runs
on its own thread and is joined with a timeout; the assertion is
``not thread.is_alive()`` plus the answers it recorded.

Nothing here starts docker: ``sandbox_script`` installs
:class:`~tests.workspace.conftest.FakeBackend` at the ``local`` key, which is what
``test_exec.py`` does and says so in its own first paragraph. Nothing here reaches
a network either — the vector store is a stand-in built through the registry's
own factory seam, and ``build_embedding_service`` is replaced with one that
returns a fixed vector.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from akgentic.tool.vector_store.embedding_actor import EmbeddingResult
from akgentic.tool.vector_store.protocol import (
    CollectionStatus,
    SearchHit,
    SearchResult,
    VectorStoreParam,
)
from akgentic.tool.vector_store.vector import VectorEntry
from akgentic.tool.workspace.actor import workspace_actor_name
from akgentic.tool.workspace.documents.models import (
    EXTRACTOR_VERSION,
    RAG_COLLECTION,
    DocumentExtract,
    RagChunk,
    RagFile,
    RagStatus,
    chunk_id,
)
from akgentic.tool.workspace.documents.store import (
    DOCUMENT_STORE_CLASSES,
    DocumentEntry,
    YamlDocumentStore,
)
from akgentic.tool.workspace.models import content_sha
from akgentic.tool.workspace.rag.search import RagSearchResult
from akgentic.tool.workspace.tool import WorkspaceExec, WorkspaceTool
from tests.workspace.conftest import (
    HANDSHAKE_TIMEOUT_S,
    WORKSPACE_NAME,
    WORKSPACE_PATH,
    FakeActorToolObserver,
    FakeOrchestratorProxy,
    SandboxScript,
    factory_for,
    tool_named,
)

PROBE_JOIN_S = HANDSHAKE_TIMEOUT_S
"""How long the probe thread is given to answer while a search is in flight.

A **budget**, never a delay: the probe joins the instant it is done, and the
timeout exists so that a held mailbox is reported as a failed assertion instead
of hanging the suite.
"""

STALL_BACKSTOP_S = 3 * HANDSHAKE_TIMEOUT_S
"""How long a parked stand-in waits to be released before it gives up.

**Strictly above :data:`PROBE_JOIN_S`, and that ordering is load-bearing.** At
equal budgets a red run races: the stand-in's own wait can expire first, which
releases the seam and turns "the mailbox was held" into an unrelated failure
inside the stand-in. The backstop exists only so a test that never reaches its
``finally`` cannot leak a blocked thread; it must never be the first thing to
fire.
"""

_HIT_TEXT = "Payment terms are net thirty."

_INDEXED_PATH = "invoice.md"
_BODY = f"# Invoice\n\n{_HIT_TEXT}\n"


def _seed_counting_row() -> None:
    """Put one fully indexed record on disk, its row counting two batches.

    ``EMBEDDING`` with ``batches_expected=2`` and ``batches_landed=0`` is what
    ``_counting_row`` needs to accept a report and what makes
    ``_on_embedding_result`` take the "one more batch to come" branch — so the
    handler does not settle the file and the record the keyword leg reads is
    unchanged by the report. The extraction half plus a matching digest is what
    makes that leg hit at all.
    """
    sha = content_sha(_BODY.encode("utf-8"))
    YamlDocumentStore().put_document(
        WORKSPACE_PATH,
        DocumentEntry(
            path=_INDEXED_PATH,
            extract=DocumentExtract(
                path=_INDEXED_PATH,
                source_sha=sha,
                extractor_version=EXTRACTOR_VERSION,
                markdown=_BODY,
                char_count=len(_BODY),
                extracted_at=datetime.now(UTC),
            ),
            row=RagFile(
                path=_INDEXED_PATH,
                status=RagStatus.EMBEDDING,
                indexed_sha=sha,
                chunk_count=1,
                batches_expected=2,
                batches_landed=0,
                chunks=[
                    RagChunk(
                        chunk_id=chunk_id(WORKSPACE_PATH, _INDEXED_PATH, sha, 0),
                        ordinal=0,
                        start=0,
                        end=len(_BODY),
                        heading_path=["Invoice"],
                    )
                ],
                updated_at=datetime.now(UTC),
            ),
        ),
    )


class _Stall:
    """One seam a stand-in can be parked at, and the record that it was reached.

    ``armed`` is what makes the same two stand-ins serve both parameterised cases
    and the positive control: an unarmed stall is a straight pass-through, so the
    no-search-in-flight run and the un-stalled seam cost nothing at all.
    """

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.released = threading.Event()
        self.armed = False

    def hold(self) -> None:
        """Announce arrival and wait to be let go — or pass straight through."""
        if not self.armed:
            return
        self.entered.set()
        assert self.released.wait(timeout=STALL_BACKSTOP_S), (
            "the stalled call was never released — the test leaked a blocked thread"
        )


class _StallingStore:
    """A cluster client with no actor behind it, stallable inside ``search``.

    Deliberately **not** a proxy over a ``#VectorStore`` child: ``_acquire_vs_proxy``'s
    own docstring records that the slot may hold either, and the client shape is
    the one where ``search`` is a second HTTP round trip on the same turn as the
    embed. A guard written against the actor-backed shape would prove the cheaper
    half.

    ``add`` carries its own stall rather than refusing outright, and that is
    property 2's seam: ``DocumentsMixin._on_embedding_result`` calls it **on the
    actor's own mailbox turn**, one ask per landing batch, which is the cheapest
    genuinely-on-the-indexing-path place to hold that mailbox open. A search must
    never reach it, so the ``adds`` list is what says whether one did.
    """

    def __init__(self, stall: _Stall, add_stall: _Stall | None = None) -> None:
        self.stall = stall
        self.add_stall = add_stall if add_stall is not None else _Stall()
        self.searches: list[tuple[str, int, str | None, str | None]] = []
        self.collections: list[str] = []
        self.adds: list[str] = []

    def create_collection(self, name: str, config: VectorStoreParam) -> None:
        self.collections.append(name)

    def add(self, collection: str, entries: list[VectorEntry]) -> None:
        self.adds.append(collection)
        self.add_stall.hold()

    def remove(
        self,
        collection: str,
        ref_ids: list[str],
        scope: str | None = None,
        path_prefix: str | None = None,
    ) -> None:
        raise AssertionError("a search must remove nothing")

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        scope: str | None = None,
        path_prefix: str | None = None,
        query: Any = None,
    ) -> SearchResult:
        self.searches.append((collection, top_k, scope, path_prefix))
        self.stall.hold()
        return SearchResult(
            status=CollectionStatus.READY,
            hits=[
                SearchHit(
                    ref_type="workspace_chunk",
                    ref_id="chunk-0",
                    text=_HIT_TEXT,
                    score=0.9,
                    scope=scope,
                    path="invoice.md",
                    ordinal=0,
                )
            ],
        )


class _StallingEmbedder:
    """The query embedder, stallable — the audit's *other* call on that turn."""

    def __init__(self, stall: _Stall) -> None:
        self.stall = stall
        self.embeds: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embeds.append(list(texts))
        self.stall.hold()
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class _Probe:
    """Two messages on ``#Workspace``'s mailbox, run on a thread of their own.

    Exactly the two calls the audit names — ``exec_status`` and ``request_exec``
    — reached through the card's own closures rather than by poking the actor, so
    what is proved is the path an agent actually takes.
    """

    def __init__(self, card: WorkspaceTool) -> None:
        self.card = card
        self.answers: list[str] = []
        self.error: BaseException | None = None

    def __call__(self) -> None:
        try:
            self.answers.append(str(tool_named(self.card, "workspace_exec_result")("no-such-run")))
            self.answers.append(str(tool_named(self.card, "workspace_exec")("echo hi")))
        except BaseException as exc:  # noqa: BLE001 — reported, never swallowed
            self.error = exc

    def run(self) -> threading.Thread:
        """Start the probe and return its thread, joined by the caller."""
        thread = threading.Thread(target=self, name="mailbox-probe", daemon=True)
        thread.start()
        return thread


class _Wired:
    """A live ``#Workspace``, a card with retrieval and exec on, and the two stalls."""

    def __init__(
        self,
        card: WorkspaceTool,
        observer: FakeActorToolObserver,
        store: _StallingStore,
        embedder: _StallingEmbedder,
        actor: Any,
    ) -> None:
        self.card = card
        self.observer = observer
        """Held because the card holds its observer weakly."""
        self.store = store
        self.embedder = embedder
        self.actor = actor
        """The live actor's Pykka proxy — how a message is put on its mailbox."""

    def search_on_its_own_thread(self) -> tuple[threading.Thread, list[RagSearchResult]]:
        """Call the card's real ``workspace_rag_search`` on a thread, capturing its answer.

        **The model is captured, never ``str()``-ed.** The assertions below used
        to read ``_HIT_TEXT in str(...)``, and ``str()`` of a Pydantic model is
        its repr — which contains the chunk text. Under a model return those
        assertions would have stayed green while testing nothing at all.
        """
        answers: list[RagSearchResult] = []

        def run() -> None:
            answers.append(tool_named(self.card, "workspace_rag_search")("payment"))

        thread = threading.Thread(target=run, name="rag-search", daemon=True)
        thread.start()
        return thread, answers

    def start_a_landing_batch(self) -> None:
        """Put one ``EmbeddingResult`` on the actor's mailbox, and do not wait for it.

        This is the **indexing** path: ``_on_embedding_result`` resolves the
        counting row, then calls ``store.add`` on this same turn — where the
        ``add`` stall parks it, holding the mailbox open until the test releases
        it. The row is seeded ``EMBEDDING`` with two batches expected, so the
        handler takes the counting branch and never settles the file.
        """
        _seed_counting_row()
        self.actor.receiveMsg_EmbeddingResult(
            EmbeddingResult(
                collection=RAG_COLLECTION,
                entries=[],
                request_id="batch-0",
                request_ref=_INDEXED_PATH,
            )
        )


@pytest.fixture
def stalls() -> dict[str, _Stall]:
    """One stall per seam, keyed by the name the parametrisation uses.

    ``add`` is the indexing-path seam property 2 parks at; it is never one of the
    parametrised search seams.
    """
    return {"search": _Stall(), "embed": _Stall(), "add": _Stall()}


@pytest.fixture
def wired(
    threaded_orchestrator_proxy: FakeOrchestratorProxy,
    workspace_tree: Path,
    sandbox_script: SandboxScript,
    stalls: dict[str, _Stall],
    monkeypatch: pytest.MonkeyPatch,
) -> _Wired:
    """Bind a retrieval-and-exec card onto a **live** ``#Workspace``.

    The store arrives through the registry's own factory seam, which is the
    branch a cluster backend takes — no ``#VectorStore`` actor is created, so
    the object the card announces is the object both shapes of the search reach.

    ``build_embedding_service`` is replaced at its **source module**, which is
    where both the actor's function-level import and the capability's find it.
    """
    store = _StallingStore(stalls["search"], stalls["add"])
    embedder = _StallingEmbedder(stalls["embed"])
    sandbox_script.gate.set()  # a run that simply completes; the probe is not about exec
    monkeypatch.setenv("AKGENTIC_WEAVIATE_URL", "https://cluster.example")
    monkeypatch.setattr(
        "akgentic.tool.vector_store.embedding_actor.build_embedding_service",
        lambda model, provider: embedder,
    )
    observer = FakeActorToolObserver(threaded_orchestrator_proxy, name="alice")
    card = WorkspaceTool(
        workspace_id=WORKSPACE_NAME,
        workspace_rag_search=True,
        workspace_exec=WorkspaceExec(poll_attempts=1, poll_delay_seconds=0.0),
        vector_store=VectorStoreParam(backend="weaviate"),
    )
    with factory_for("weaviate", lambda _context: store):
        card.observer(observer)
    # The two announcements are tells. One round trip on the same mailbox is what
    # proves they have been applied — ``enable_rag`` included, and with it the
    # ``create_collection`` that leaves the actor un-degraded.
    pykka_proxy = threaded_orchestrator_proxy.children[workspace_actor_name(WORKSPACE_PATH)][1]
    pykka_proxy.state.get(timeout=HANDSHAKE_TIMEOUT_S)
    assert store.collections == [RAG_COLLECTION], "retrieval never came up on the actor"
    return _Wired(card, observer, store, embedder, pykka_proxy)


class TestTheProbeItself:
    """The positive control. A probe that never worked would make the guard green."""

    def test_with_no_search_in_flight_both_calls_answer(self, wired: _Wired) -> None:
        """Neither seam is armed, so nothing stalls and the mailbox is plainly free."""
        probe = _Probe(wired.card)
        thread = probe.run()
        thread.join(timeout=PROBE_JOIN_S)

        assert probe.error is None
        assert not thread.is_alive()
        assert len(probe.answers) == 2


@pytest.mark.parametrize("seam", ["search", "embed"])
class TestTheMailboxAnswersWhileASearchIsInFlight:
    """The audit's two calls, against a search parked at an external seam."""

    def test_exec_status_and_request_exec_answer(
        self, wired: _Wired, stalls: dict[str, _Stall], seam: str
    ) -> None:
        """The whole property, in one sequence and with no timing in the assertion.

        The order is fixed: arm the seam, start the search, wait for the stand-in
        to announce arrival, probe, join with a budget, release, join the search.
        Every wait is on an event or a thread, never on a clock.
        """
        stall = stalls[seam]
        stall.armed = True
        search_thread, answers = wired.search_on_its_own_thread()
        try:
            assert stall.entered.wait(timeout=HANDSHAKE_TIMEOUT_S), (
                f"the {seam} stand-in was never reached — the guard would be vacuous"
            )
            assert not stall.released.is_set()

            probe = _Probe(wired.card)
            probe_thread = probe.run()
            probe_thread.join(timeout=PROBE_JOIN_S)
        finally:
            stall.released.set()

        assert probe.error is None
        assert not probe_thread.is_alive(), (
            "#Workspace did not answer while a search's external call was in flight — "
            "the mailbox is held for the duration of the round trip"
        )
        assert len(probe.answers) == 2

        search_thread.join(timeout=HANDSHAKE_TIMEOUT_S)
        assert not search_thread.is_alive()
        assert answers[0].hits, "the search answered no hits at all"
        assert any(_HIT_TEXT in hit.text for hit in answers[0].hits), (
            "the search did not answer with the hits it was handed"
        )

    def test_the_stand_in_was_actually_reached(
        self, wired: _Wired, stalls: dict[str, _Stall], seam: str
    ) -> None:
        """55-4's lesson, as its own row: ``entered`` set is the non-vacuity.

        Stated separately from the property above so that a stand-in pointed at a
        seam the production path never takes reddens *here*, naming the cause,
        rather than showing up as an unexplained timeout in the spec beside it.
        """
        stall = stalls[seam]
        stall.armed = True
        search_thread, _answers = wired.search_on_its_own_thread()
        try:
            entered = stall.entered.wait(timeout=HANDSHAKE_TIMEOUT_S)
        finally:
            stall.released.set()
        search_thread.join(timeout=HANDSHAKE_TIMEOUT_S)

        assert entered, f"the {seam} seam is not on the search's path"
        # Both seams, whichever was armed: the query is embedded once and the
        # collection is searched once, so neither stand-in can be parked
        # somewhere the production path merely happens to pass near.
        assert wired.embedder.embeds == [["payment"]]
        assert [collection for collection, *_rest in wired.store.searches] == [RAG_COLLECTION]


class TestASearchAnswersWhileTheActorIsIndexing:
    """Property 2 — the converse, and the one story 57-1 is about.

    The mailbox is held by a turn on the **indexing** path, and a search through
    the card's own callable completes and returns its hits anyway. Before the
    move the search made a blocking ask for its keyword leg, so its thread would
    still be alive at the join.

    Nothing here is timed. The sequence is fixed: seed the counting row, put one
    ``EmbeddingResult`` on the mailbox, wait for the ``add`` stand-in to announce
    arrival, run the search on its own thread, join with a budget, release.
    """

    def test_the_search_answers_while_a_batch_is_landing(
        self, wired: _Wired, stalls: dict[str, _Stall]
    ) -> None:
        """The whole property, in one sequence and with no clock in the assertion."""
        landing = stalls["add"]
        landing.armed = True
        wired.start_a_landing_batch()
        try:
            assert landing.entered.wait(timeout=HANDSHAKE_TIMEOUT_S), (
                "the add stand-in was never reached — the mailbox is not held and "
                "the guard would be vacuous"
            )
            assert not landing.released.is_set()

            search_thread, answers = wired.search_on_its_own_thread()
            search_thread.join(timeout=PROBE_JOIN_S)
        finally:
            landing.released.set()

        assert not search_thread.is_alive(), (
            "the search did not answer while #Workspace was indexing — its keyword "
            "leg is queued behind the held mailbox again"
        )
        assert answers[0].hits, "the search answered, but carried no hits"
        assert any(_HIT_TEXT in hit.text for hit in answers[0].hits)

    def test_the_add_stand_in_was_actually_reached(
        self, wired: _Wired, stalls: dict[str, _Stall]
    ) -> None:
        """55-4's lesson as its own row: a seam nobody reached holds no mailbox.

        Stated separately so that a report the production path drops — a row that
        is not ``EMBEDDING``, a mismatched ``request_ref`` — reddens *here*,
        naming the cause, instead of showing up as an unexplained pass next door.
        """
        landing = stalls["add"]
        landing.armed = True
        wired.start_a_landing_batch()
        try:
            entered = landing.entered.wait(timeout=HANDSHAKE_TIMEOUT_S)
        finally:
            landing.released.set()

        assert entered, "the indexing path never reached store.add"
        assert wired.store.adds == [RAG_COLLECTION]

    def test_with_the_seam_unarmed_the_search_answers_too(self, wired: _Wired) -> None:
        """The positive control. A search that never worked would pass the row above.

        The ``adds`` list is the second half of it: a search must reach ``add``
        not at all, so a guard that had accidentally armed the *search* path would
        show up here rather than as a green run.
        """
        wired.start_a_landing_batch()

        search_thread, answers = wired.search_on_its_own_thread()
        search_thread.join(timeout=PROBE_JOIN_S)

        assert not search_thread.is_alive()
        assert any(_HIT_TEXT in hit.text for hit in answers[0].hits)
        assert wired.store.adds == [RAG_COLLECTION], "the search itself wrote to the store"


class _Crowd:
    """How many searches are inside the record listing at once, and the gate they wait at.

    The counter is the whole property. "Two searches both answered" proves
    nothing — one mailbox answers two asks in sequence and both callers get their
    answer. *Both inside one seam at the same instant* is what a mailbox cannot
    produce, and it is a fact about positions rather than about a clock.
    """

    def __init__(self) -> None:
        self.armed = False
        self.lock = threading.Lock()
        self.arrivals = 0
        self.both_inside = threading.Event()
        self.release = threading.Event()

    def enter(self) -> None:
        """Count one arrival, announce a full house, and wait to be let go."""
        if not self.armed:
            return
        with self.lock:
            self.arrivals += 1
            if self.arrivals >= 2:
                self.both_inside.set()
        assert self.release.wait(timeout=STALL_BACKSTOP_S), (
            "a blocked listing was never released — the test leaked a blocked thread"
        )


_CROWD = _Crowd()
"""The live counter. Module level because the registry constructs the store itself.

``resolve_document_store()`` calls the registered class with no arguments — that
is the production path and the reason this seam is reached at all — so the state
cannot be a constructor argument. The fixture below replaces it per spec.
"""

_STORE_KEY = "blocking-yaml"
"""The registry key the two cards below resolve their document store through."""


class _BlockingDocumentStore(YamlDocumentStore):
    """A real store that parks inside ``list_documents`` while :data:`_CROWD` is armed.

    Registered through ``DOCUMENT_STORE_CLASSES``, which the store module
    documents as a mutable injection window — the shape ``factory_for`` already
    uses for vector backends, and the reason no private attribute is reached for
    here. Every other method is the real one's, so both cards read the records
    they were seeded.
    """

    def list_documents(self, tree_key: str) -> list[DocumentEntry]:
        _CROWD.enter()
        return super().list_documents(tree_key)


@pytest.fixture
def crowd() -> Iterator[_Crowd]:
    """Install the blocking store for one spec, and never leave a thread parked."""
    global _CROWD  # noqa: PLW0603 — the registry builds the store with no arguments
    _CROWD = _Crowd()
    DOCUMENT_STORE_CLASSES[_STORE_KEY] = _BlockingDocumentStore
    try:
        yield _CROWD
    finally:
        _CROWD.release.set()
        DOCUMENT_STORE_CLASSES.pop(_STORE_KEY, None)


class TestTwoCardsOfOneTeamSearchAtOnce:
    """Property 3 — concurrency, stated as positions rather than as timing.

    Two cards of one team, two searches, two threads, and **both inside the
    record listing at the same instant**. One mailbox cannot produce that: the
    second search would be queued behind the first, the count would stay at one,
    and ``both_inside`` would never be set. So the count staying at one is exactly
    the red-before, and it arrives as a bounded assertion rather than as a hang.

    This is *not* a cross-process property — two cards of one team in one process
    is its whole subject — so no child-process harness is owed here.
    """

    @pytest.fixture
    def cards(
        self,
        threaded_orchestrator_proxy: FakeOrchestratorProxy,
        workspace_tree: Path,
        crowd: _Crowd,
        monkeypatch: pytest.MonkeyPatch,
    ) -> list[WorkspaceTool]:
        """Two search cards of one team over one tree, both reading the blocking store.

        The vector leg is degraded by an embedder that raises, so each search is
        the keyword leg alone — which is the leg that reads the records and
        therefore the one the seam sits in. ``_retrieval_bound()`` is asserted so
        that a bind which resolved nothing cannot make the guard pass at the gate.
        """
        _seed_counting_row()
        monkeypatch.setenv("AKGENTIC_DOCUMENT_STORE", _STORE_KEY)
        monkeypatch.setenv("AKGENTIC_WEAVIATE_URL", "https://cluster.example")

        def _explodes(model: str, provider: str) -> object:
            raise RuntimeError("no embedder in this spec")

        monkeypatch.setattr(
            "akgentic.tool.vector_store.embedding_actor.build_embedding_service", _explodes
        )
        built: list[WorkspaceTool] = []
        self._observers = []
        for name in ("alice", "bob"):
            observer = FakeActorToolObserver(threaded_orchestrator_proxy, name=name)
            card = WorkspaceTool(
                workspace_id=WORKSPACE_NAME,
                workspace_rag_search=True,
                vector_store=VectorStoreParam(backend="weaviate"),
            )
            with factory_for("weaviate", lambda _context: _StallingStore(_Stall())):
                card.observer(observer)
            assert card._retrieval_bound(), f"{name}'s card resolved no store"
            self._observers.append(observer)  # a card holds its observer weakly
            built.append(card)
        return built

    def test_both_searches_are_inside_the_listing_at_once(
        self, cards: list[WorkspaceTool], crowd: _Crowd
    ) -> None:
        """The count reaching two is the property; one mailbox cannot reach it."""
        crowd.armed = True
        answers: list[list[RagSearchResult]] = [[], []]
        threads = [
            threading.Thread(
                target=lambda index=index: answers[index].append(  # type: ignore[misc]
                    tool_named(cards[index], "workspace_rag_search")("payment")
                ),
                name=f"rag-search-{index}",
                daemon=True,
            )
            for index in range(2)
        ]
        for thread in threads:
            thread.start()
        try:
            reached_two = crowd.both_inside.wait(timeout=HANDSHAKE_TIMEOUT_S)
        finally:
            crowd.release.set()
        for thread in threads:
            thread.join(timeout=PROBE_JOIN_S)

        assert reached_two, (
            "only one search was ever inside the record listing — the two are "
            "serialising, which is what a shared mailbox does and a shared file does not"
        )
        assert crowd.arrivals == 2
        assert all(not thread.is_alive() for thread in threads)
        assert all(
            any(_HIT_TEXT in hit.text for hit in answer[0].hits) for answer in answers
        )

    def test_with_the_seam_unarmed_both_searches_answer(
        self, cards: list[WorkspaceTool], crowd: _Crowd
    ) -> None:
        """The positive control: two searches that never worked would pass the row above."""
        assert crowd.armed is False

        answers = [tool_named(card, "workspace_rag_search")("payment") for card in cards]

        assert all(any(_HIT_TEXT in hit.text for hit in answer.hits) for answer in answers)
        assert crowd.arrivals == 0
