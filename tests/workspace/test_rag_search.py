"""``workspace_rag_search``: two legs, fusion, the result model, degradation.

The vector store double here wraps a **real** :class:`InMemoryBackend` rather than
returning a canned hit list, and that is what makes the scope-isolation spec worth
anything: ``scope`` and ``path_prefix`` are honoured by the code that will honour
them in a deployment, including ``_map_search_hits``'s last-one-wins resolution of
``{ref_id: entry}`` — the mechanism epic row 1 records. A double that filtered in
the test would have proved that the test filters.

Embeddings are a fixed four-word bag rather than a network call, so a cosine
ordering is deterministic and a spec can say which hit comes first.

**Both legs run in one place now, so every spec is driven through one callable.**
``tool_named(card, "workspace_rag_search")`` is what an agent holds, and it is
what these specs call. Story 57-1 took the keyword leg, the fusion and the answer
off ``#Workspace`` — they read the document records, which the card's own
:class:`~akgentic.tool.workspace.documents.cache.DocumentCache` reaches without a
mailbox — so there is no ``actor.rag_search`` left to drive and no split to
reproduce here.

:class:`SearchHarness` still binds both halves, and the actor is still what the
seeding helpers write through: one inert actor, seeded through the document
store, and a real :class:`WorkspaceTool` whose resolved vector store **is** the
double and whose own cache reads the very records ``seed_row`` / ``seed_extract``
put on disk — :meth:`SearchHarness.bind_card` asserts the two resolved the same
tree, which is what makes that true. So :meth:`SearchHarness.run` exercises the
whole production path end to end, with no leg re-implemented here. The card takes
the ``weaviate`` branch because that is the one that resolves a client with no
store actor behind it — the shape where ``search`` is a second round trip — and
its factory is swapped through the registry's own seam.

**Where a spec calls a moved function directly, that is deliberate and says so**:
the fusion module's own default for ``alpha`` is reachable from no production
caller, because the closure always sends a float.

**Nothing here asserts on ``str()`` of the answer.** A search answers a
:class:`RagSearchResult` since story 58-3, and ``str()`` of a Pydantic model is
its repr — which carries the chunk text, so a substring assertion over it would
stay green while testing nothing. Every assertion below reads a field.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from akgentic.core.agent_state import BaseState

from akgentic.tool.vector_store.backends.inmemory import InMemoryBackend
from akgentic.tool.vector_store.hybrid import DEFAULT_ALPHA, OVERFETCH
from akgentic.tool.vector_store.protocol import (
    PATH_PREFIX_REJECTED,
    SearchResult,
    VectorStoreParam,
)
from akgentic.tool.vector_store.vector import VectorEntry
from akgentic.tool.workspace.actor import (
    WORKSPACE_ACTOR_ROLE,
    WorkspaceActor,
    workspace_actor_name,
)
from akgentic.tool.workspace.documents.models import (
    EXTRACTOR_VERSION,
    RAG_COLLECTION,
    DocumentExtract,
    RagChunk,
    RagFile,
    RagStatus,
    chunk_id,
)
from akgentic.tool.workspace.documents.store import DocumentEntry, YamlDocumentStore
from akgentic.tool.workspace.models import WorkspaceConfig, content_sha
from akgentic.tool.workspace.rag.params import WorkspaceRagIndex, WorkspaceRagSearch
from akgentic.tool.workspace.rag.search import (
    MatchKind,
    RagSearchHit,
    RagSearchResult,
    search_documents,
)
from akgentic.tool.workspace.readers import DocumentReader
from akgentic.tool.workspace.tool import WorkspaceTool
from tests.conftest import MockActorAddress
from tests.workspace.conftest import (
    WORKSPACE_NAME,
    WORKSPACE_PATH,
    FakeActorToolObserver,
    FakeOrchestratorProxy,
    attach_store,
    drop_row,
    factory_for,
    seed_extract,
    seed_row,
    stored_docs,
    stored_rows,
    tool_named,
    watch_store,
    workspace_path_for,
)

_UNAVAILABLE = "Retrieval indexing is not available for this workspace."
_NO_HITS = (
    "Nothing in the retrieval index matched that query. "
    "Use workspace_rag_list to see which files are indexed."
)
_REJECTED_PREFIX = PATH_PREFIX_REJECTED
"""Imported rather than copied: the closure returns the protocol's own constant."""

_VOCABULARY = ("invoice", "payment", "holiday", "refund")
"""The whole of the embedding model, so a cosine ordering is a fact of the test."""


def vector_for(text: str) -> list[float]:
    """Return a bag-of-words vector over :data:`_VOCABULARY`.

    A text carrying none of the four words gets the all-ones vector rather than
    the zero vector: ``search_cosine`` clamps a zero norm, and a spec should not
    depend on what that clamp happens to do.
    """
    lowered = text.lower()
    counts = [float(lowered.count(word)) for word in _VOCABULARY]
    return counts if any(counts) else [1.0] * len(_VOCABULARY)


class SearchStore:
    """``#VectorStore`` as the search path uses it, over a real backend."""

    def __init__(self) -> None:
        self.backend = InMemoryBackend()
        self.backend.create_collection(RAG_COLLECTION, VectorStoreParam(backend="inmemory"))
        self.searches: list[tuple[str, int, str | None, str | None]] = []
        self.search_error: Exception | None = None

    def create_collection(self, name: str, config: VectorStoreParam) -> None:
        self.backend.create_collection(name, config)

    def add(self, collection: str, entries: list[VectorEntry]) -> None:
        self.backend.add(collection, entries)

    def remove(
        self,
        collection: str,
        ref_ids: list[str],
        scope: str | None = None,
        path_prefix: str | None = None,
    ) -> None:
        self.backend.remove(collection, ref_ids, scope=scope, path_prefix=path_prefix)

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        scope: str | None = None,
        path_prefix: str | None = None,
    ) -> SearchResult:
        self.searches.append((collection, top_k, scope, path_prefix))
        if self.search_error is not None:
            raise self.search_error
        return self.backend.search(
            collection, query_vector, top_k, scope=scope, path_prefix=path_prefix
        )

    def store_chunk(self, ref_id: str, scope: str, path: str, ordinal: int, text: str) -> None:
        """Put one embedded chunk in the collection, as the indexing path would."""
        self.backend.add(
            RAG_COLLECTION,
            [
                VectorEntry(
                    ref_type="workspace_chunk",
                    ref_id=ref_id,
                    text=text,
                    vector=vector_for(text),
                    scope=scope,
                    path=path,
                    ordinal=ordinal,
                )
            ],
        )


class SearchEmbedder:
    """The consumer's own embedding service — the query leg no longer goes to the store.

    The vector store embeds nothing after story 49-3, so the double that used to
    carry ``embed`` beside ``search`` is split in two and this half stands in for
    ``build_embedding_service`` on the **card's** side. The actor has no embedder
    slot at all — it had one, unread, until story 55-9 deleted it.
    """

    def __init__(self) -> None:
        self.embeds: list[list[str]] = []
        self.embed_error: Exception | None = None
        self.embed_returns: list[list[float]] | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embeds.append(list(texts))
        if self.embed_error is not None:
            raise self.embed_error
        if self.embed_returns is not None:
            return self.embed_returns
        return [vector_for(text) for text in texts]


class SearchHarness:
    """An inert actor, and a real card whose proxy is that actor.

    The actor is handed **no** orchestrator, and resolves no store of its own:
    the card announces one at bind time, so this harness announces the double
    through ``configure_vector_store`` exactly as a card would. ``createActor``
    is kept and pointed at a trap — a spawn from here is a regression, not a
    path.

    :meth:`bind_card` adds the other half. The vector leg of a search is the
    card's now, so a spec about it has to go through a **real**
    :class:`WorkspaceTool`: one bound with this harness's actor as its
    ``_workspace_proxy`` and this harness's double as its resolved vector store.
    :meth:`run` is then the callable an agent actually holds, and nothing about
    either leg is re-implemented here.
    """

    def __init__(self, actor: WorkspaceActor, store: SearchStore) -> None:
        self.actor = actor
        self.store = store
        self.embedder = SearchEmbedder()
        self.vs_address = MockActorAddress("#VectorStore-child")
        self.card: WorkspaceTool | None = None
        self._observer: FakeActorToolObserver | None = None
        """Held: a card holds its observer weakly, and a dropped one is collected."""
        self._orchestrator_proxy: FakeOrchestratorProxy | None = None
        self._monkeypatch: pytest.MonkeyPatch | None = None
        self._workspace_id = WORKSPACE_NAME

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.actor._orchestrator = None
        monkeypatch.setattr(self.actor, "proxy_ask", self._ask)
        monkeypatch.setattr(self.actor, "proxy_tell", self._tell)
        monkeypatch.setattr(self.actor, "createActor", self._create)

    def enable(self, announce: bool = True) -> None:
        """Announce the store, then retrieval — the card's order, always.

        ``announce=False`` is the lost-announcement case: parameters set, no
        proxy, which is the half-enabled state every degradation spec here needs.
        """
        if announce:
            self.actor.configure_vector_store(self.store)
        self.actor.enable_rag(
            "alice",
            WorkspaceRagIndex(),
            DocumentReader(llm_client=None),
            VectorStoreParam(backend="weaviate"),
        )

    def bind_card(
        self,
        orchestrator_proxy: FakeOrchestratorProxy,
        monkeypatch: pytest.MonkeyPatch,
        workspace_id: str = WORKSPACE_NAME,
        **search_params: Any,
    ) -> WorkspaceTool:
        """Bind a real card over this harness's actor and store, and return it.

        The ``weaviate`` branch is taken deliberately: it is the one where
        ``needs_store_actor`` is false and the card resolves a **client** through
        the registry's factory, which is the shape the story's search leg is
        about. The factory is swapped through ``factory_for``, the registry's own
        seam, so the card's resolution code runs unmodified.

        ``build_embedding_service`` is replaced at its **source module**, which is
        where the capability's function-level import finds it; the double records
        every text it is handed.
        """
        monkeypatch.setenv("AKGENTIC_WEAVIATE_URL", "https://cluster.example")
        monkeypatch.setattr(
            "akgentic.tool.vector_store.embedding_actor.build_embedding_service",
            lambda model, provider: self.embedder,
        )
        observer = FakeActorToolObserver(orchestrator_proxy, workspace_proxy=self.actor)
        card = WorkspaceTool(
            workspace_id=workspace_id,
            workspace_rag_search=WorkspaceRagSearch(**search_params),
            vector_store=VectorStoreParam(backend="weaviate"),
        )
        with factory_for("weaviate", lambda _context: self.store):
            card.observer(observer)
        assert card._workspace_path == self.actor.config.workspace_path, (
            "the card and its actor resolved different trees — the scope predicate "
            "this harness exists to exercise would filter everything out"
        )
        self.card = card
        self._observer = observer
        self._orchestrator_proxy = orchestrator_proxy
        self._monkeypatch = monkeypatch
        self._workspace_id = workspace_id
        return card

    def run(self, query: str, **kwargs: Any) -> RagSearchResult:
        """Search through the card's own callable — both legs, production path.

        The model is returned as the callable answers it. **Never ``str()``-ed**:
        the repr of a hit contains its text, so a substring assertion over it
        would pass on a result that had lost every field this epic added.
        """
        assert self.card is not None, "bind_card first"
        answer = tool_named(self.card, "workspace_rag_search")(query, **kwargs)
        assert isinstance(answer, RagSearchResult)
        return answer

    def run_with(self, query: str, **search_params: Any) -> RagSearchResult:
        """Search through a card configured with *search_params*.

        The two knobs a caller cannot pass per call — ``alpha`` and
        ``score_threshold`` — are card configuration, so a spec about either
        rebinds rather than reaching past the closure.
        """
        assert self._orchestrator_proxy is not None and self._monkeypatch is not None
        card = self.bind_card(
            self._orchestrator_proxy, self._monkeypatch, self._workspace_id, **search_params
        )
        answer = tool_named(card, "workspace_rag_search")(query)
        assert isinstance(answer, RagSearchResult)
        return answer

    def index(
        self,
        path: str,
        body: str,
        spans: list[tuple[int, int, list[str]]],
        *,
        scope: str | None = None,
        cache: bool = True,
        embedded: bool = True,
        lines: list[tuple[int, int] | None] | None = None,
    ) -> str:
        """Index *path* into both actor maps and into the collection.

        Args:
            path: Workspace-relative path.
            body: The extracted Markdown the offsets index into.
            spans: ``(start, end, heading_path)`` per chunk, in ordinal order.
            scope: The workspace the chunks belong to. Defaults to this actor's.
            cache: Whether the extraction body is held — ``False`` stands in for
                an evicted body.
            embedded: Whether the chunks reach the vector store at all.
            lines: ``(start_line, end_line)`` per chunk, positional by ordinal,
                or ``None`` for a chunk that records no range. Defaults to
                **all-None**, so every spec that does not ask for coordinates
                keeps exercising the row-predates-58-2 path.

                The pairs are **literal and hand-chosen**, never re-derived here
                from ``body``: the derivation is pinned by ``test_splitter.py``'s
                ``TestTheLineRange``, and a harness that re-derived would make a
                coordinate spec assert its own arithmetic instead of transport.

        Returns:
            The digest both maps agree on.
        """
        owner = scope or self.actor.config.workspace_path
        sha = content_sha(body.encode("utf-8"))
        chunks: list[RagChunk] = []
        for ordinal, (start, end, heading) in enumerate(spans):
            identity = chunk_id(owner, path, sha, ordinal)
            pair = lines[ordinal] if lines is not None and ordinal < len(lines) else None
            chunks.append(
                RagChunk(
                    chunk_id=identity,
                    ordinal=ordinal,
                    start=start,
                    end=end,
                    heading_path=heading,
                    start_line=None if pair is None else pair[0],
                    end_line=None if pair is None else pair[1],
                )
            )
            if embedded:
                self.store.store_chunk(identity, owner, path, ordinal, body[start:end])
        if owner == self.actor.config.workspace_path:
            seed_row(self.actor, path, RagFile(
                path=path,
                status=RagStatus.EMBEDDED,
                indexed_sha=sha,
                # Stamped, so every keyword spec below exercises the *matching*
                # branch of the extractor guard rather than passing for ever
                # through the ``None`` escape hatch a legacy row takes.
                indexed_extractor_version=EXTRACTOR_VERSION,
                chunks=chunks,
                chunk_count=len(chunks),
                updated_at=datetime.now(UTC),
            ))
            seed_extract(self.actor, path, DocumentExtract(
                path=path,
                source_sha=sha,
                extractor_version=EXTRACTOR_VERSION,
                markdown=body if cache else None,
                char_count=len(body),
                extracted_at=datetime.now(UTC),
            ))
        return sha

    def _ask(self, address: Any, actor_type: Any = None, timeout: int | None = None) -> Any:
        if address is self.vs_address:
            return self.store
        raise AssertionError(f"unexpected ask target {address}")

    def _tell(self, address: Any, actor_type: Any = None) -> Any:
        return self.store

    def _create(self, actor_class: Any, agent_id: Any = None, config: Any = None) -> Any:
        raise AssertionError(f"the actor spawned {actor_class}; the card resolves the store")


def build_actor(workspace_path: str = WORKSPACE_PATH) -> WorkspaceActor:
    """A started actor over *workspace_path*, with no actor thread.

    Takes the **resolved** three-segment path, which is what an actor is
    configured with — the leaf alone would put it on a tree no card reaches.
    """
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
def store() -> SearchStore:
    return SearchStore()


@pytest.fixture
def search(
    workspace_tree: Path,
    store: SearchStore,
    orchestrator_proxy: FakeOrchestratorProxy,
    monkeypatch: pytest.MonkeyPatch,
) -> SearchHarness:
    """A harness with retrieval enabled and a card bound — what a search assumes."""
    built = SearchHarness(build_actor(), store)
    built.install(monkeypatch)
    built.enable()
    built.bind_card(orchestrator_proxy, monkeypatch)
    return built


##
## The document the specs search
##

_INVOICE = "# Invoice\n\nPayment terms are net thirty.\n\nA refund is issued on request.\n"
_SPLIT = _INVOICE.index("A refund")
_FIRST = (0, _SPLIT, ["Invoice", "Payment terms"])
_SECOND = (_SPLIT, len(_INVOICE), ["Invoice", "Refunds"])


def paths(result: RagSearchResult) -> list[str]:
    """The path of every hit, in order — the shape most specs here assert on.

    ``hit_count`` stood here until story 58-3 and counted score labels, because a
    chunk's own text routinely contains blank lines and splitting the rendered
    answer on them measured the document rather than the search. There is no
    string to split any more: a count is ``len(result.hits)`` and the hazard the
    helper existed for is gone with the render.
    """
    return [hit.path for hit in result.hits]


class TestDegradation:
    """Every failure mode answers a ``note`` and none of them raises.

    The wording of all three sentences is unchanged by story 58-3; what changed
    is that they arrive as ``RagSearchResult.note`` with an empty ``hits`` list
    rather than as the bare return value.
    """

    def test_a_card_that_resolved_no_engine_answers_the_sentence(
        self, search: SearchHarness
    ) -> None:
        """**Cause re-pointed, invariant unchanged** (fourth time — see Trap 1).

        It was "the team's ``#VectorStore`` was not found", then "the child could
        not be spawned", then "the card announced no store", and it is now the
        card's own :meth:`RagFactories._retrieval_bound` answering ``False``
        because the engine is missing. What the spec guards has survived all four:
        a tree with nothing to search answers the unavailable sentence.

        This is one of the two terms of that predicate. The actor's gate it
        reproduces was ``self._vs_proxy is None`` — "no store was announced" — and
        the card is what announces it.
        """
        assert search.card is not None
        search.card._vector_store = None

        assert search.card._retrieval_bound() is False
        answer = search.run("payment")
        assert (answer.note, answer.hits) == (_UNAVAILABLE, [])

    def test_a_card_with_an_engine_but_no_collection_param_answers_the_sentence(
        self, search: SearchHarness
    ) -> None:
        """Both halves of enablement are required; a half-enabled tree is degraded.

        The other term of :meth:`RagFactories._retrieval_bound`, reproducing the
        actor's ``self._rag_params is None`` — no card ever announced
        ``enable_rag``. ``_resolved_store`` is derived exactly when
        ``_rag_enabled()`` holds and is the value ``_announce_rag`` sends, so its
        absence is that absence.
        """
        assert search.card is not None
        search.card._resolved_store = None

        assert search.card._retrieval_bound() is False
        answer = search.run("payment")
        assert (answer.note, answer.hits) == (_UNAVAILABLE, [])

    def test_a_card_that_never_bound_a_tree_answers_the_sentence(self) -> None:
        """The first gate, in the position the ``None`` proxy used to hold.

        ``_build_document_cache`` runs in ``observer()``, so a card that never
        bound has no cache — which is what "never bound" *is*, card-side — and the
        closure answers the sentence rather than raising.
        """
        card = WorkspaceTool(workspace_id=WORKSPACE_NAME, workspace_rag_search=True)

        assert card._document_cache is None
        assert card._rag_search_factory(WorkspaceRagSearch())("payment").note == _UNAVAILABLE

    def test_an_embed_that_raises_falls_back_to_the_keyword_leg(
        self, search: SearchHarness
    ) -> None:
        """One warning, no exception, and the lexical half still answers."""
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])
        search.embedder.embed_error = RuntimeError("the embedding provider is down")

        answer = search.run("payment")

        assert paths(answer) == ["invoice.md"]
        assert answer.hits[0].match is MatchKind.KEYWORD

    def test_an_embed_that_returns_nothing_falls_back_to_the_keyword_leg(
        self, search: SearchHarness
    ) -> None:
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])
        search.embedder.embed_returns = []

        answer = search.run("payment")

        assert [hit.match for hit in answer.hits] == [MatchKind.KEYWORD]
        assert search.store.searches == []

    def test_a_search_that_raises_falls_back_to_the_keyword_leg(
        self, search: SearchHarness
    ) -> None:
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])
        search.store.search_error = RuntimeError("cluster unreachable")

        answer = search.run("payment")

        assert [hit.match for hit in answer.hits] == [MatchKind.KEYWORD]

    def test_a_failing_vector_leg_never_raises_out_of_the_search(
        self, search: SearchHarness
    ) -> None:
        """This actor owns the write gate; a retrieval failure must not reach it."""
        search.store.search_error = RuntimeError("cluster unreachable")

        search.run("nothing is indexed at all")  # must not raise

    def test_an_embedder_that_cannot_even_be_built_falls_back_to_the_keyword_leg(
        self, search: SearchHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``build_embedding_service`` imports the ``[vector_search]`` extra.

        **The successor to ``test_rag_pipeline.py``'s row of the same name**, which
        guarded ``enable_rag`` against a failing build back when the actor was the
        one building an embedder. It no longer builds one, so that row would have
        gone vacuous; the property is real and belongs where the build now happens.
        """
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])

        def _explodes(model: str, provider: str) -> object:
            raise RuntimeError("the vector_search extra is not installed")

        monkeypatch.setattr(
            "akgentic.tool.vector_store.embedding_actor.build_embedding_service", _explodes
        )

        answer = search.run("payment")  # must not raise

        assert [hit.match for hit in answer.hits] == [MatchKind.KEYWORD]
        assert search.store.searches == []

    def test_a_card_that_resolved_no_store_spends_nothing_before_the_sentence(
        self, search: SearchHarness
    ) -> None:
        """**Rewritten in story 57-1, and it is not a weakened guard — read this.**

        It was ``test_a_card_that_resolved_no_store_falls_back_to_the_keyword_leg``
        and it asserted ``"keyword match" in answer``. That assertion described the
        *harness*: :class:`SearchHarness` announces the store to the actor by hand,
        so a card that resolved none sat beside an actor that has one. The common
        production shape has no such split — a card that resolves no store
        announces none (``_announce_vector_store`` returns early), the actor
        degrades too, and the answer is the unavailable sentence, which is exactly
        what the card's own gate now returns directly and with no ask.

        So the assertion follows the reality rather than the harness, and the gate
        was **not** weakened to preserve the old answer.

        **One production shape does reach the old state, and the answer changes
        there.** ``_bind_vector_store`` catches every resolution failure and leaves
        ``_vector_store`` at ``None`` with a WARNING saying *retrieval stays off
        for this card*; a sibling card of the same team may still have announced a
        store, and that card used to borrow the actor's keyword leg through the
        announcement. It answers the sentence now — the warning's own words made
        true. See :meth:`RagFactories._retrieval_bound`.

        The two spend assertions are kept rather than claimed as new: they were
        already green before this move, because ``_vector_hits`` short-circuits on
        the very condition the gate now tests. They pin that the hoisted gate did
        not *start* spending something.
        """
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])
        assert search.card is not None
        search.card._vector_store = None

        answer = search.run("payment")

        assert (answer.note, answer.hits) == (_UNAVAILABLE, [])
        assert search.store.searches == []
        assert search.embedder.embeds == []

    def test_a_degraded_tree_asked_with_a_wildcard_prefix_is_refused_for_the_prefix(
        self, search: SearchHarness
    ) -> None:
        """The gate ordering, pinned — reversing two of them flips this answer.

        The order is: cache absent → the wildcard check → availability → the legs.
        Swap the middle two and a degraded tree asked with a wildcard prefix
        answers *retrieval unavailable* instead of *the prefix is refused*, which
        is a silent change of meaning with nothing else in the suite to catch it:
        every other prefix spec runs against a tree that is **not** degraded, and
        every other degradation spec passes no prefix.

        **Both halves are asserted on ``note``, deliberately.** ``note is not
        None`` — or ``not hits`` — would pass under the very swap this row exists
        to catch, because both notes are set and both answers are hitless. The
        two sentences being *different strings* is what makes the spec work, and
        that is what the second assertion says.
        """
        assert search.card is not None
        search.card._vector_store = None
        assert search.card._retrieval_bound() is False

        answer = search.run("payment", path_prefix="report*")

        assert answer.note == _REJECTED_PREFIX
        assert answer.note != _UNAVAILABLE

    def test_no_hits_is_a_sentence_that_is_not_the_unavailable_one(
        self, search: SearchHarness
    ) -> None:
        """ "Nothing matched" and "nothing is indexed" have different next steps."""
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND], embedded=False)

        answer = search.run("bicycles")

        assert answer.hits == []
        assert answer.note == _NO_HITS
        assert answer.note != _UNAVAILABLE
        assert "workspace_rag_list" in (answer.note or "")


class TestTheVectorLeg:
    """It is scoped, over-fetched, and thresholded on the raw cosine.

    **Driven through the card**, because that is where the leg runs now. Every
    assertion below is byte-identical to the one it replaced: what moved is *where
    the call is made*, not *what the backend receives*.
    """

    def test_every_query_carries_the_workspace_as_its_scope(self, search: SearchHarness) -> None:
        """One ``workspace_chunks`` class holds every workspace of every team."""
        search.index("invoice.md", _INVOICE, [_FIRST])

        search.run("payment")

        [(collection, _, scope, _prefix)] = search.store.searches
        assert (collection, scope) == (RAG_COLLECTION, WORKSPACE_PATH)

    def test_an_empty_prefix_reaches_the_backend_as_none_rather_than_as_a_string(
        self, search: SearchHarness
    ) -> None:
        """``None`` filters nothing; ``""`` would be a predicate the backend applies."""
        search.index("invoice.md", _INVOICE, [_FIRST])

        search.run("payment")

        assert search.store.searches[0][3] is None

    def test_a_prefix_is_passed_to_the_backend(self, search: SearchHarness) -> None:
        """The predicate goes to the store, so the budget is not spent locally."""
        search.index("reports/invoice.md", _INVOICE, [_FIRST])

        search.run("payment", path_prefix="reports/")

        assert search.store.searches[0][3] == "reports/"

    def test_the_backend_is_over_fetched_because_fusion_reorders(
        self, search: SearchHarness
    ) -> None:
        search.index("invoice.md", _INVOICE, [_FIRST])

        search.run("payment", top_k=5)

        assert search.store.searches[0][1] == 5 * OVERFETCH

    def test_the_score_threshold_drops_weak_vector_hits(self, search: SearchHarness) -> None:
        """Applied to the raw cosine, before normalisation, so it stays absolute.

        The body is not cached, so the keyword leg contributes nothing and what
        the threshold drops is the whole of the answer.

        The threshold is **card configuration** rather than a call argument — it
        always was, and the closure spent it on the actor before this story and
        spends it on its own leg after — so the two cases are two cards.
        """
        search.index("holiday.md", "Holiday policy\n", [(0, 15, ["Holiday"])], cache=False)

        kept = search.run_with("holiday", score_threshold=0.0)
        dropped = search.run_with("holiday", score_threshold=1.5)

        assert paths(kept) == ["holiday.md"]
        assert (dropped.hits, dropped.note) == ([], _NO_HITS)


class TestTheKeywordLeg:
    """Over the bodies the actor holds — and never over one it does not."""

    def test_it_matches_case_insensitively(self, search: SearchHarness) -> None:
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])
        search.embedder.embed_error = RuntimeError("vector leg off")

        assert paths(search.run("PAYMENT")) == ["invoice.md"]

    def test_an_evicted_body_contributes_nothing_and_is_never_sliced(
        self, search: SearchHarness
    ) -> None:
        """ADR-045 §3/§4: the offsets of an evicted file are provenance.

        The file stays indexed and its vector hits still render — what it loses is
        the lexical leg, which is a degradation and never an error. Slicing a
        ``None`` body would be a ``TypeError`` on the gate's own thread.
        """
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND], cache=False)
        search.embedder.embed_error = RuntimeError("vector leg off")

        answer = search.run("payment")  # must not raise

        assert (answer.hits, answer.note) == ([], _NO_HITS)

    def test_an_evicted_body_still_answers_through_the_vector_leg(
        self, search: SearchHarness
    ) -> None:
        """This is what keeps ``max_documents`` a bound on state, not on the corpus."""
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND], cache=False)

        answer = search.run("payment")

        assert set(paths(answer)) == {"invoice.md"}
        assert {hit.match for hit in answer.hits} == {MatchKind.SEMANTIC}
        assert any("Payment terms are net thirty." in hit.text for hit in answer.hits)

    def test_a_body_whose_digest_no_longer_matches_the_row_is_skipped(
        self, search: SearchHarness
    ) -> None:
        """The two maps have different lifetimes; mismatched offsets belong to neither."""
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])
        stale = stored_docs(search.actor)["invoice.md"]
        seed_extract(
            search.actor, "invoice.md", stale.model_copy(update={"source_sha": "a-different-digest"})
        )
        search.embedder.embed_error = RuntimeError("vector leg off")

        assert search.run("payment").note == _NO_HITS

    def test_a_body_cut_by_another_extractor_is_skipped(self, search: SearchHarness) -> None:
        """The second half of an extraction's identity, and the reason for the clause.

        An ``EXTRACTOR_VERSION`` bump leaves the source bytes alone, so
        ``indexed_sha`` still matches — and every cached body becomes a miss and
        is re-extracted. Without this guard the leg slices a **new** extraction
        with **old** offsets, which quotes text that belongs to neither.

        The same shape as the digest spec above, for the other half of the pair.
        """
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])
        current = stored_docs(search.actor)["invoice.md"]
        seed_extract(
            search.actor,
            "invoice.md",
            current.model_copy(update={"extractor_version": EXTRACTOR_VERSION + 1}),
        )
        search.embedder.embed_error = RuntimeError("vector leg off")

        assert search.run("payment").note == _NO_HITS

    def test_the_vector_leg_is_unaffected_by_an_extractor_bump(
        self, search: SearchHarness
    ) -> None:
        """"Degrades to vector-only" — without this, the spec above only proves a loss.

        The store holds its own copy of each chunk's text, which no offset of the
        row's is used to produce. The file keeps rendering from it.
        """
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])
        current = stored_docs(search.actor)["invoice.md"]
        seed_extract(
            search.actor,
            "invoice.md",
            current.model_copy(update={"extractor_version": EXTRACTOR_VERSION + 1}),
        )

        answer = search.run("payment")

        assert set(paths(answer)) == {"invoice.md"}
        assert {hit.match for hit in answer.hits} == {MatchKind.SEMANTIC}
        assert any("Payment terms are net thirty." in hit.text for hit in answer.hits)

    def test_an_extractor_bump_does_not_mark_the_row_stale(self, search: SearchHarness) -> None:
        """A bump is not a mutation of the tree, so no path may write a status off it.

        ``mark_paths_stale`` fires on writes; the row here was never written to.
        It stays ``EMBEDDED`` with its chunks intact and simply loses its lexical
        leg, which is the degradation ADR-045 §4 already promises.
        """
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])
        current = stored_docs(search.actor)["invoice.md"]
        seed_extract(
            search.actor,
            "invoice.md",
            current.model_copy(update={"extractor_version": EXTRACTOR_VERSION + 1}),
        )
        search.embedder.embed_error = RuntimeError("vector leg off")

        assert search.run("payment").note == _NO_HITS

        row = stored_rows(search.actor)["invoice.md"]
        assert row.status is RagStatus.EMBEDDED
        assert row.chunk_count == 2

    def test_a_row_that_predates_the_field_still_matches(self, search: SearchHarness) -> None:
        """``None`` is "written before this field", not "unknown, refuse".

        A required field — or a ``None`` treated as a mismatch — would drop the
        keyword leg for every row already on disk, which is a silent de-index of
        every tree in the field on the day this ships.
        """
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])
        legacy = stored_rows(search.actor)["invoice.md"]
        seed_row(
            search.actor,
            "invoice.md",
            legacy.model_copy(update={"indexed_extractor_version": None}),
        )
        search.embedder.embed_error = RuntimeError("vector leg off")

        assert paths(search.run("payment")) == ["invoice.md"]

    def test_a_cached_path_absent_from_the_index_is_skipped_rather_than_raising(
        self, search: SearchHarness
    ) -> None:
        """The cache and the index have different caps as well as different lifetimes."""
        seed_extract(search.actor, "orphan.md", DocumentExtract(
            path="orphan.md",
            source_sha="sha",
            extractor_version=EXTRACTOR_VERSION,
            markdown="A payment note.\n",
            char_count=16,
            extracted_at=datetime.now(UTC),
        ))
        search.embedder.embed_error = RuntimeError("vector leg off")

        assert search.run("payment").note == _NO_HITS

    def test_only_the_chunk_whose_own_slice_carries_the_term_is_hit(
        self, search: SearchHarness
    ) -> None:
        """The offsets are what map a body match onto a chunk id."""
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])
        search.embedder.embed_error = RuntimeError("vector leg off")

        answer = search.run("refund")

        assert [hit.heading_path for hit in answer.hits] == [["Invoice", "Refunds"]]

    def test_the_prefix_filters_the_keyword_leg_too(self, search: SearchHarness) -> None:
        """The backend filters its own leg; nothing else would filter this one."""
        search.index("reports/invoice.md", _INVOICE, [_FIRST])
        search.index("notes/invoice.md", _INVOICE, [_FIRST])
        search.embedder.embed_error = RuntimeError("vector leg off")

        answer = search.run("payment", path_prefix="reports/")

        assert paths(answer) == ["reports/invoice.md"]

    def test_an_empty_query_hits_nothing_on_the_keyword_leg(self, search: SearchHarness) -> None:
        """A blank query must not match every chunk in the workspace."""
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])
        search.embedder.embed_error = RuntimeError("vector leg off")

        assert search.run("   ").note == _NO_HITS


class TestTheHit:
    """What one hit carries: where it is, which leg found it, and its text."""

    def test_a_hit_carries_its_path_and_heading_path(self, search: SearchHarness) -> None:
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])

        [hit] = search.run("payment", top_k=1).hits

        assert (hit.path, hit.heading_path) == ("invoice.md", ["Invoice", "Payment terms"])

    def test_a_keyword_only_hit_says_which_leg_found_it(self, search: SearchHarness) -> None:
        search.index("invoice.md", _INVOICE, [_FIRST], embedded=False)

        [hit] = search.run("payment").hits

        assert hit.match is MatchKind.KEYWORD

    def test_a_keyword_only_hit_has_no_score_at_all(self, search: SearchHarness) -> None:
        """``None``, never ``0.0`` — "was not scored" is not "matched, badly".

        ``fuse`` does not normalise the keyword leg, which is an indicator rather
        than a score, so there is no number to report. ``0.0`` would be a number,
        and an agent comparing it against another hit's cosine would read it as
        the worst possible match instead of as an absence.
        """
        search.index("invoice.md", _INVOICE, [_FIRST], embedded=False)

        [hit] = search.run("payment").hits

        assert hit.score is None

    def test_a_vector_only_hit_carries_its_raw_cosine(self, search: SearchHarness) -> None:
        """The raw cosine, because a fused score means nothing outside its own set.

        The number is arithmetic and not a recorded observation: the chunk's own
        text carries "invoice" and "payment", so its bag-of-words vector is
        ``[1, 1, 0, 0]`` against the query's ``[0, 1, 0, 0]`` — a cosine of
        ``1 / sqrt(2)``. A **fused** score at this alpha would be ``0.70``.

        **This is the claim the old spec was hiding.** It asserted the string
        ``"(semantic: 0.71)"`` — a *formatted* number at two decimals, which
        agrees with anything between ``0.705`` and ``0.715``. The field carries
        the cosine itself.
        """
        search.index("invoice.md", _INVOICE, [_FIRST], cache=False)

        [hit] = search.run("payment").hits

        assert hit.match is MatchKind.SEMANTIC
        assert hit.score == pytest.approx(1 / math.sqrt(2))

    def test_a_hit_confirmed_by_both_legs_says_hybrid(self, search: SearchHarness) -> None:
        search.index("invoice.md", _INVOICE, [_FIRST])

        [hit] = search.run("payment", top_k=1).hits

        assert hit.match is MatchKind.HYBRID

    def test_the_text_of_a_hit_comes_from_the_store_and_not_from_the_cache(
        self, search: SearchHarness
    ) -> None:
        """``SearchHit.text`` is what survives an eviction; a slice is not."""
        search.index("invoice.md", _INVOICE, [_FIRST])
        held = stored_docs(search.actor)["invoice.md"]
        seed_extract(
            search.actor,
            "invoice.md",
            held.model_copy(update={"markdown": _INVOICE.replace("net thirty", "REPLACED")}),
        )

        [hit] = search.run("payment", top_k=1).hits

        assert "net thirty" in hit.text
        assert "REPLACED" not in hit.text

    def test_the_text_is_carried_as_the_leg_supplied_it_and_is_not_stripped(
        self, search: SearchHarness
    ) -> None:
        """A behaviour change, stated: the render's ``.strip()`` is gone with it.

        The strip existed to make joined blocks read cleanly. There are no blocks,
        and a caller that wants the passage's own leading and trailing whitespace
        — to line it up against the file it came from — now gets it.
        """
        search.index("invoice.md", _INVOICE, [_FIRST], cache=False)

        [hit] = search.run("payment").hits

        assert hit.text == _INVOICE[_FIRST[0] : _FIRST[1]]
        assert hit.text.endswith("\n\n")

    def test_a_hit_whose_row_will_not_resolve_is_returned_with_null_coordinates(
        self, search: SearchHarness
    ) -> None:
        """The chunk text is still the answer, so a hit is never dropped for this.

        **Three claims where the old spec made one weak one.** It asserted
        ``answer.startswith("invoice.md (semantic:")`` and inferred "no heading
        path" from the *absence* of a ``>`` in a prefix — which a changed
        separator would also satisfy. Each coordinate is now named.

        ``ordinal`` is the exception and is deliberate: it falls back to what the
        vector store itself reported. It is the one coordinate still known, and
        discarding it would leave the caller with nothing at all.
        """
        search.index("invoice.md", _INVOICE, [_FIRST], cache=False)
        drop_row(search.actor, "invoice.md")

        [hit] = search.run("payment").hits

        assert hit.path == "invoice.md"
        assert hit.heading_path == []
        assert hit.chunk_count is None
        assert hit.start_line is None
        assert hit.end_line is None
        assert hit.ordinal == 0
        assert hit.match is MatchKind.SEMANTIC
        assert "Payment terms are net thirty." in hit.text

    def test_two_hits_come_back_in_fused_order_each_with_its_own_fields(
        self, search: SearchHarness
    ) -> None:
        """The whole list as one comparison, so a drifted ordering is a diff.

        **The successor to story 57-1's byte-for-byte shape guard.** Its subject —
        one composed string — no longer exists, so what it guarded is asserted at
        the new boundary: the *order* of the two hits, and the fields attached to
        each. It must not degrade into "two hits came back".

        The two scores are arithmetic rather than recorded observations. The
        query ``payment refund`` is ``[0, 1, 0, 1]`` over the four-word
        vocabulary. The first chunk carries "invoice" and "payment", so it is
        ``[1, 1, 0, 0]`` — a cosine of ``1 / (sqrt(2) * sqrt(2)) = 0.50``. The
        second carries "refund" alone, so it is ``[0, 0, 0, 1]`` — a cosine of
        ``1 / sqrt(2) = 0.71``. The second therefore ranks first, and both are
        ``HYBRID`` because the keyword leg hits both. The text of each is
        ``SearchHit.text`` — the store's copy — **unstripped**.
        """
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])

        answer = search.run("payment refund", top_k=2)

        assert [
            (hit.path, hit.ordinal, hit.chunk_count, hit.heading_path, hit.match, hit.text)
            for hit in answer.hits
        ] == [
            (
                "invoice.md",
                1,
                2,
                ["Invoice", "Refunds"],
                MatchKind.HYBRID,
                "A refund is issued on request.\n",
            ),
            (
                "invoice.md",
                0,
                2,
                ["Invoice", "Payment terms"],
                MatchKind.HYBRID,
                "# Invoice\n\nPayment terms are net thirty.\n\n",
            ),
        ]
        assert [hit.score for hit in answer.hits] == [
            pytest.approx(1 / math.sqrt(2)),
            pytest.approx(0.5),
        ]

    def test_two_hits_are_two_entries_rather_than_one_run_together(
        self, search: SearchHarness
    ) -> None:
        """What the blank-line separator used to buy, now structural.

        A chunk's own text routinely contains blank lines, so "hits are separated
        by a blank line" could never be read back reliably — which is why
        ``hit_count`` counted score labels instead. A list needs no separator.
        """
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])

        answer = search.run("payment refund", top_k=2)

        assert len(answer.hits) == 2
        assert answer.hits[0].text != answer.hits[1].text
        assert answer.note is None


class TestTheCoordinates:
    """Where a hit is: its ordinal out of how many, and its line range."""

    def test_a_hit_carries_the_line_range_its_record_holds(self, search: SearchHarness) -> None:
        """Transport, and only transport — the numbers are literals, not arithmetic.

        Re-deriving the pair here with ``line_starts`` would make the spec assert
        the harness's own arithmetic. The derivation is pinned by 58-2's
        ``TestTheLineRange``; what 58-3 owes is that what the record holds is what
        the hit carries, and a hand-chosen pair proves that and nothing else.
        """
        search.index(
            "invoice.md", _INVOICE, [_FIRST, _SECOND], cache=False, lines=[(1, 3), (5, 5)]
        )

        answer = search.run("payment refund", top_k=2)

        assert [(hit.ordinal, hit.start_line, hit.end_line) for hit in answer.hits] == [
            (1, 5, 5),
            (0, 1, 3),
        ]

    def test_a_keyword_only_hit_carries_the_range_too(self, search: SearchHarness) -> None:
        """The leg that never reaches the row still answers the coordinates.

        ``_KeywordMatch`` holds the chunk itself, so the range rides along with no
        lookup at all — and a keyword-only hit is the degraded mode this whole
        design turns on.
        """
        search.index("invoice.md", _INVOICE, [_FIRST], embedded=False, lines=[(1, 3)])

        [hit] = search.run("payment").hits

        assert (hit.match, hit.start_line, hit.end_line) == (MatchKind.KEYWORD, 1, 3)

    def test_a_hit_says_which_chunk_of_how_many_it_is(self, search: SearchHarness) -> None:
        """``chunk_count`` is the row's, so "chunk 2 of 3" is answerable."""
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND], cache=False)

        [hit] = search.run("refund", top_k=1).hits

        assert (hit.ordinal, hit.chunk_count) == (1, 2)

    def test_a_chunk_whose_row_predates_the_line_range_answers_nulls_and_its_text(
        self, search: SearchHarness
    ) -> None:
        """No migration and no de-indexing in the field, at the boundary.

        Asserted directly rather than inferred from the model's defaults: a row
        written before 58-2 carries ``start_line=None`` / ``end_line=None``, and
        every other field of the hit is populated as usual.
        """
        search.index("invoice.md", _INVOICE, [_FIRST], cache=False)

        [hit] = search.run("payment").hits

        assert (hit.start_line, hit.end_line) == (None, None)
        assert hit.ordinal == 0
        assert hit.chunk_count == 1
        assert hit.heading_path == ["Invoice", "Payment terms"]
        assert "Payment terms are net thirty." in hit.text

    def test_a_hit_the_store_reported_no_ordinal_for_is_returned_with_null_coordinates(
        self, search: SearchHarness
    ) -> None:
        """The other way a hit locates nowhere: the store itself carried no ordinal.

        Story 45-6 put ``ordinal`` on ``SearchHit``, and an entry written before
        it — or by a producer that does not partition — has none. There is then
        nothing to look the row up by, and the chunk text is still the answer.
        """
        search.store.backend.add(
            RAG_COLLECTION,
            [
                VectorEntry(
                    ref_type="workspace_chunk",
                    ref_id="an-entry-with-no-ordinal",
                    text="A payment note with no ordinal.",
                    vector=vector_for("payment"),
                    scope=WORKSPACE_PATH,
                    path="invoice.md",
                    ordinal=None,
                )
            ],
        )

        [hit] = search.run("payment").hits

        assert hit.path == "invoice.md"
        assert (hit.ordinal, hit.chunk_count, hit.start_line, hit.end_line) == (
            None,
            None,
            None,
            None,
        )
        assert hit.heading_path == []
        assert hit.text == "A payment note with no ordinal."

    def test_a_fused_key_that_resolves_to_neither_leg_spends_no_result_slot(
        self, search: SearchHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The skip the budget is spent **after**, pinned where it cannot arise alone.

        Every key ``fuse`` returns comes from one of the two mappings handed to
        it, so no production input reaches this branch — which is exactly why it
        needs a spec rather than an argument. A phantom key is fused in ahead of
        both real ones; it must contribute no hit **and** cost neither of them
        its slot, which is what "the budget is spent after filtering" means.
        """
        from akgentic.tool.vector_store import hybrid  # noqa: PLC0415 — patched per spec

        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND], cache=False)
        real = hybrid.fuse

        def phantom(
            keyword_keys: Any, vector_scores: Any, *, alpha: float = DEFAULT_ALPHA
        ) -> dict[str, float]:
            return {"a-key-neither-leg-has": 99.0, **real(keyword_keys, vector_scores, alpha=alpha)}

        monkeypatch.setattr(hybrid, "fuse", phantom)

        answer = search.run("payment refund", top_k=2)

        assert len(answer.hits) == 2
        assert set(paths(answer)) == {"invoice.md"}

    def test_the_coordinates_cost_one_record_read_per_vector_hit(
        self, search: SearchHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``cache.entry(path)`` is a YAML file opened and parsed, once per hit.

        ``DocumentCache.entry`` → ``DocumentStore.get_document`` → ``_read`` of a
        file. ``chunk_count`` is a field of the same row the chunk comes from, so
        resolving it through a **second** ``entry()`` call would double a search's
        disk reads for one integer. The bodies are evicted here so the keyword leg
        contributes nothing and every hit takes the lookup path.
        """
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND], cache=False)
        assert search.card is not None
        cache = search.card._document_cache
        assert cache is not None
        reads: list[str] = []
        original = cache.entry

        def counting(path: str) -> Any:
            reads.append(path)
            return original(path)

        monkeypatch.setattr(cache, "entry", counting)

        answer = search.run("payment refund", top_k=2)

        assert len(answer.hits) == 2
        assert reads == ["invoice.md", "invoice.md"]


class TestTheResultIsAModelAndNotAString:
    """ADR-054 Decision 1: a plain ``BaseModel``, and what its dump carries."""

    def test_the_result_dumps_exactly_two_keys(self) -> None:
        assert set(RagSearchResult(note="x").model_dump()) == {"hits", "note"}

    def test_a_hit_dumps_exactly_its_nine_fields(self) -> None:
        hit = RagSearchHit(path="a.md", match=MatchKind.KEYWORD, text="t")

        assert set(hit.model_dump()) == {
            "path",
            "ordinal",
            "chunk_count",
            "start_line",
            "end_line",
            "heading_path",
            "score",
            "match",
            "text",
        }

    def test_no_model_path_discriminator_reaches_the_payload(self) -> None:
        """Why these are plain ``BaseModel`` and not ``SerializableBaseModel``.

        That base declares a ``@model_serializer`` which appends
        ``__model__ = <module path>`` to every dump. A search result is
        serialised into a prompt and consumed, never reconstructed, so the
        discriminator would buy nothing and would put this module's own import
        path in front of the model once per hit — pinning where ``rag/search.py``
        happens to live into the agent-visible wire format.
        """
        dumped = RagSearchResult(
            hits=[RagSearchHit(path="a.md", match=MatchKind.SEMANTIC, text="t", score=0.5)]
        ).model_dump()

        assert "__model__" not in dumped
        assert "__model__" not in dumped["hits"][0]

    def test_the_match_kind_dumps_as_its_bare_value(self) -> None:
        """The dumped value is a plain ``str``, not the member that equals one.

        **This spec changed meaning in 58-4.** It asserted
        ``model_dump()["match"] == "hybrid"`` and its docstring claimed a
        ``StrEnum`` "serialises to the string". Neither held: a plain
        ``model_dump()`` left the *member* in the dict, and
        ``MatchKind.HYBRID == "hybrid"`` is ``True``, so the assertion agreed
        with the broken code as readily as with the fixed one. The type is the
        only claim that tells the two apart.
        """
        hit = RagSearchHit(path="a.md", match=MatchKind.HYBRID, text="t")

        assert type(hit.model_dump()["match"]) is str

    @pytest.mark.parametrize("kind", list(MatchKind))
    def test_a_fully_populated_result_survives_an_unsafe_dump_and_a_safe_load(
        self, kind: MatchKind
    ) -> None:
        """The writer/reader pair that empties an event log in the field.

        ``akgentic-team`` persists the LLM history with an **unsafe** YAML dumper
        and reads it back with ``yaml.safe_load_all``. An enum member that
        survives ``model_dump()`` is written as
        ``!!python/object/apply:...MatchKind`` — a tag the safe loader refuses,
        so the read yields **zero** events for the whole log and resume fails
        with a misleading "No Orchestrator StartMessage found".

        Three properties of this row are load bearing and each is one keystroke
        from being useless:

        - **plain** ``model_dump()``, never ``mode="json"`` — that mode converts
          the enum itself, so the row would be green before *and* after the fix.
          It is why this package's other ``StrEnum`` (``RagFile.status``) was
          never affected: both of the tool's own write sites dump in JSON mode.
        - ``yaml.dump``, never ``yaml.safe_dump`` — the safe dumper *raises* on
          an unrepresentable value instead of emitting a tag, which is a
          different failure from the silent one being reproduced here.
        - the assertion is on the **emitted text**. ``isinstance(match, str)`` is
          vacuous for a ``StrEnum``, and ``== "semantic"`` agrees with the broken
          code.

        Every field carries a non-default value, so this is also the executable
        form of the field audit: a field added later whose value is not
        YAML-plain reddens here with nobody remembering this story.
        """
        result = RagSearchResult(
            hits=[
                RagSearchHit(
                    path="invoice.md",
                    ordinal=2,
                    chunk_count=7,
                    start_line=11,
                    end_line=19,
                    heading_path=["Invoice", "Payment terms"],
                    score=0.71,
                    match=kind,
                    text="Payment terms are net thirty.",
                )
            ],
            note="carried so both fields of the parent hold a value too",
        )

        dumped = yaml.dump(result.model_dump(), default_flow_style=False)

        assert "!!python/" not in dumped
        assert yaml.safe_load(dumped) == result.model_dump()

    def test_a_result_from_the_production_builder_round_trips_the_same_way(
        self, search: SearchHarness
    ) -> None:
        """The same round trip over the path the defect actually travelled.

        A hand-built instance proves the model; only the builder proves the
        answer an agent's tool call returns.
        """
        search.index("invoice.md", _INVOICE, [_FIRST])

        result = search.run("payment", top_k=1)
        dumped = yaml.dump(result.model_dump(), default_flow_style=False)

        assert result.hits
        assert "!!python/" not in dumped
        assert yaml.safe_load(dumped) == result.model_dump()

    def test_the_safe_loaded_match_re_validates_as_its_kind(self) -> None:
        """What makes the stored string a *value* and not a lossy rendering.

        The attribute stays a real ``MatchKind`` — the serializer acts at dump
        time, not at validation time — so identity still holds on the way back
        in, and the member compares equal to the string that was stored.
        """
        result = RagSearchResult(
            hits=[RagSearchHit(path="a.md", match=MatchKind.SEMANTIC, text="t")]
        )

        reloaded = yaml.safe_load(yaml.dump(result.model_dump(), default_flow_style=False))

        assert reloaded["hits"][0]["match"] == "semantic"
        assert RagSearchHit.model_validate(reloaded["hits"][0]).match is MatchKind.SEMANTIC
        assert MatchKind.SEMANTIC == "semantic"


class TestTopK:
    """Honoured after filtering, and never under-filled by another scope."""

    def test_the_hit_list_is_cut_to_top_k(self, search: SearchHarness) -> None:
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])

        answer = search.run("payment refund", top_k=1)

        assert len(answer.hits) == 1

    def test_a_larger_budget_returns_both(self, search: SearchHarness) -> None:
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])

        answer = search.run("payment refund", top_k=5)

        assert len(answer.hits) == 2

    def test_a_zero_budget_is_clamped_rather_than_over_fetching_nothing(
        self, search: SearchHarness
    ) -> None:
        """``top_k * OVERFETCH`` at zero would ask the backend for no rows at all."""
        search.index("invoice.md", _INVOICE, [_FIRST])

        answer = search.run("payment", top_k=0)

        assert paths(answer) == ["invoice.md"]
        assert search.store.searches[0][1] == OVERFETCH


class TestScopeIsolation:
    """Epic row 1: two workspaces in one collection, and the read path honours it."""

    def test_a_search_returns_only_its_own_workspaces_chunks_at_full_top_k(
        self, search: SearchHarness
    ) -> None:
        """The other workspace's chunks are a better cosine match and still absent.

        ``InMemoryBackend._map_search_hits`` resolves ``{ref_id: entry}``
        last-one-wins, so two entries sharing a ``ref_id`` across scopes would be
        indistinguishable on the read path. Story 45-7 put the scope inside the
        ``chunk_id`` digest, which makes the ids distinct; this is the spec that
        proves the read path honours the predicate as well.
        """
        search.index("mine.md", _INVOICE, [_FIRST, _SECOND])
        for ordinal in range(6):
            search.store.store_chunk(
                chunk_id("other-workspace", "theirs.md", "sha", ordinal),
                "other-workspace",
                "theirs.md",
                ordinal,
                "payment payment payment",
            )

        answer = search.run("payment", top_k=5)

        assert set(paths(answer)) == {"mine.md"}

    def test_the_other_workspace_sees_only_its_own(
        self,
        workspace_tree: Path,
        store: SearchStore,
        orchestrator_proxy: FakeOrchestratorProxy,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Symmetry, so the spec is about the predicate and not about one ordering.

        The second workspace's card is bound with ``workspace_id="other-workspace"``
        so its resolved path — and therefore the ``scope`` its vector leg sends —
        is the one its actor was built for. A card on the first tree would have
        proved nothing about the predicate.
        """
        mine = SearchHarness(build_actor(), store)
        mine.install(monkeypatch)
        mine.enable()
        mine.bind_card(orchestrator_proxy, monkeypatch)
        mine.index("mine.md", _INVOICE, [_FIRST])

        theirs = SearchHarness(build_actor(workspace_path_for("other-workspace")), store)
        theirs.install(monkeypatch)
        theirs.enable()
        theirs.bind_card(orchestrator_proxy, monkeypatch, "other-workspace")
        theirs.index("theirs.md", _INVOICE, [_FIRST])

        answer = theirs.run("payment", top_k=5)

        assert set(paths(answer)) == {"theirs.md"}

    def test_the_two_workspaces_mint_distinct_ids_for_the_same_file(
        self, search: SearchHarness
    ) -> None:
        """Without this the predicate would be filtering entries that had collided."""
        assert chunk_id(WORKSPACE_PATH, "a.md", "sha", 0) != chunk_id(
            "other-workspace", "a.md", "sha", 0
        )


class TestThePathPrefixDecision:
    """Epic row 2, decided here: a metacharacter is refused at the boundary.

    ``WeaviateBackend`` builds ``Filter.by_property(path).like(f"{prefix}*")`` and
    Weaviate's ``Like`` treats both ``*`` and ``?`` as wildcards;
    ``InMemoryBackend`` uses ``str.startswith`` and treats both literally. Both
    characters are legal in a POSIX filename and the v4 filter API offers no
    escape, so the same query would mean two different things depending on where
    the collection happens to live. Refusing is the only answer under which the
    two backends agree.
    """

    @pytest.mark.parametrize("prefix", ["report*", "report?.md", "a/*/b", "?"])
    def test_a_prefix_carrying_a_metacharacter_is_refused(
        self, search: SearchHarness, prefix: str
    ) -> None:
        answer = search.run("payment", path_prefix=prefix)

        assert answer.note == _REJECTED_PREFIX
        assert answer.note != _UNAVAILABLE

    @pytest.mark.parametrize("backend", ["local", "inmemory", "weaviate"])
    def test_the_same_sentence_comes_back_whatever_the_backend(
        self, search: SearchHarness, backend: str
    ) -> None:
        """The refusal is at the caller, so the backends cannot disagree.

        What the parametrisation buys is the *param*: three different backends
        named in the resolved collection, one refusal, proving the sentence is not
        derived from the backend the param happens to name. The param is set on
        the bound card rather than declared on a fresh one because ``inmemory`` is
        **refused at bind** for a workspace (``WORKSPACE_IN_MEMORY_REFUSED``), so
        no card can be built naming it — and it is exactly the value a spec about
        "the sentence does not depend on the backend" wants to include.

        It reads a private attribute deliberately, in the shape this module
        already uses for ``_vector_store``: the resolved param is not something an
        author writes, and the closure re-captures it on every ``get_tools()``.
        """
        assert search.card is not None
        search.card._resolved_store = VectorStoreParam(backend=backend)

        answer = search.run("payment", path_prefix="report?.md")

        assert answer.note == _REJECTED_PREFIX
        assert answer.note != _UNAVAILABLE

    def test_a_refused_prefix_never_reaches_the_backend(self, search: SearchHarness) -> None:
        """No embed is spent either — the refusal is the first thing that happens."""
        search.index("invoice.md", _INVOICE, [_FIRST])

        search.run("payment", path_prefix="report*")

        assert search.store.searches == []
        assert search.embedder.embeds == []

    def test_the_in_memory_backend_treats_a_metacharacter_literally(self) -> None:
        """Half of the divergence the refusal exists for, pinned against real code."""
        from akgentic.tool.vector_store.backends.inmemory import _entry_matches

        entry = VectorEntry(
            ref_type="workspace_chunk", ref_id="c", text="t", vector=[1.0], path="report?.md"
        )
        other = entry.model_copy(update={"path": "reportX.md"})

        assert _entry_matches(entry, None, "report?") is True
        assert _entry_matches(other, None, "report?") is False

    def test_a_legal_prefix_is_not_refused(self, search: SearchHarness) -> None:
        """The refusal must not spread to the ordinary case it exists to protect."""
        search.index("reports/invoice.md", _INVOICE, [_FIRST])

        answer = search.run("payment", path_prefix="reports/")

        assert paths(answer) == ["reports/invoice.md"]


class TestTheFusionKnobs:
    """``alpha`` and ``score_threshold`` reach the rule the package shares."""

    def test_the_cards_default_alpha_is_the_fusion_modules_own(self) -> None:
        """The literal in ``rag/params.py`` is written out to avoid an import edge."""
        assert WorkspaceRagSearch().alpha == DEFAULT_ALPHA

    def test_alpha_none_takes_the_module_default(self, search: SearchHarness) -> None:
        """Driven on :func:`search_documents`, because that branch is only reachable there.

        ``alpha`` is card configuration and the closure always sends a float, so
        ``None`` reaches the search from no production caller at all — it is the
        default of the moved function's own signature, and this is the spec that
        says that default agrees with the fusion module's.
        """
        search.index("invoice.md", _INVOICE, [_FIRST])
        assert search.card is not None
        cache = search.card._document_cache
        assert cache is not None

        defaulted = search_documents(
            cache, None, None, "payment", top_k=5, scope=WORKSPACE_PATH, alpha=None
        )
        explicit = search_documents(
            cache, None, None, "payment", top_k=5, scope=WORKSPACE_PATH, alpha=DEFAULT_ALPHA
        )

        assert defaulted == explicit

    def test_pure_keyword_fusion_still_returns_the_keyword_hit(self, search: SearchHarness) -> None:
        """``alpha=0.0`` is pure keyword, and a vector-only hit then scores zero."""
        search.index("invoice.md", _INVOICE, [_FIRST])

        assert paths(search.run_with("payment", alpha=0.0)) == ["invoice.md"]


class TestTheStateItNeverTouches:
    """A search is a read: it must persist nothing and mutate nothing."""

    def test_a_search_writes_nothing(self, search: SearchHarness) -> None:
        """A write on a read path is a defect until a decision says otherwise."""
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])
        writes = watch_store(search.actor)

        assert set(paths(search.run("payment"))) == {"invoice.md"}

        assert writes.puts == []
        assert writes.evicted == []

    def test_a_search_leaves_the_index_untouched(self, search: SearchHarness) -> None:
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])
        before = stored_rows(search.actor)["invoice.md"].model_copy(deep=True)

        search.run("payment")

        assert stored_rows(search.actor)["invoice.md"] == before


class TestTheDocumentStoreContract:
    """``rag_search`` reads records the store actually persists.

    The successor to the state-field contract: what used to be two declared
    fields on the actor's own state class is one record on disk, and the claim
    worth pinning is the same one — a half that were not persisted would empty on
    every process start.
    """

    def test_neither_half_lives_on_the_actor_state_any_more(self) -> None:
        """A field here would be a second, divergent copy of what is on disk.

        The state class itself went with the host in 52-6; the actor is
        parameterised on core's ``BaseState``, which has no fields at all, so the
        assertion is read off the type the actor actually carries rather than off
        a class this package still owns.
        """
        assert "documents" not in BaseState.model_fields
        assert "rag_index" not in BaseState.model_fields
        assert BaseState.model_fields == {}

    def test_both_halves_are_fields_of_the_stored_record(self) -> None:
        assert {"extract", "row"} <= set(DocumentEntry.model_fields)

    def test_a_search_reads_both_halves_through_a_second_store_object(
        self, search: SearchHarness
    ) -> None:
        """The keyword leg joins the two halves, so both must have reached disk."""
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])
        entry = YamlDocumentStore().get_document(WORKSPACE_PATH, "invoice.md")
        assert entry is not None
        assert entry.extract is not None and entry.extract.markdown is not None
        assert entry.row is not None and entry.row.chunks
