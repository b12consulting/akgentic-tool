"""``workspace_rag_search`` on the actor: two legs, fusion, the render, degradation.

The vector store double here wraps a **real** :class:`InMemoryBackend` rather than
returning a canned hit list, and that is what makes the scope-isolation spec worth
anything: ``scope`` and ``path_prefix`` are honoured by the code that will honour
them in a deployment, including ``_map_search_hits``'s last-one-wins resolution of
``{ref_id: entry}`` — the mechanism epic row 1 records. A double that filtered in
the test would have proved that the test filters.

Embeddings are a fixed four-word bag rather than a network call, so a cosine
ordering is deterministic and a spec can say which hit comes first.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from akgentic.tool.vector_store.hybrid import DEFAULT_ALPHA, OVERFETCH
from akgentic.tool.vector_store.inmemory import InMemoryBackend
from akgentic.tool.vector_store.protocol import CollectionConfig, SearchResult
from akgentic.tool.vector_store.vector import VectorEntry
from akgentic.tool.workspace.actor import (
    WORKSPACE_ACTOR_ROLE,
    WorkspaceActor,
    workspace_actor_name,
)
from akgentic.tool.workspace.card.params import WorkspaceRagIndex, WorkspaceRagSearch
from akgentic.tool.workspace.documents.models import (
    EXTRACTOR_VERSION,
    RAG_COLLECTION,
    DocumentExtract,
    RagChunk,
    RagFile,
    RagStatus,
    chunk_id,
)
from akgentic.tool.workspace.models import WorkspaceConfig, WorkspaceState, content_sha
from akgentic.tool.workspace.readers import DocumentReader

from tests.conftest import MockActorAddress
from tests.workspace.conftest import WORKSPACE_NAME, DeadAddress

_UNAVAILABLE = "Retrieval indexing is not available for this workspace."
_NO_HITS = (
    "Nothing in the retrieval index matched that query. "
    "Use workspace_rag_list to see which files are indexed."
)

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
        self.backend.create_collection(RAG_COLLECTION, CollectionConfig(backend="inmemory"))
        self.searches: list[tuple[str, int, str | None, str | None]] = []
        self.embeds: list[list[str]] = []
        self.embed_error: Exception | None = None
        self.search_error: Exception | None = None
        self.embed_returns: list[list[float]] | None = None

    def create_collection(self, name: str, config: CollectionConfig) -> None:
        self.backend.create_collection(name, config)

    def add(
        self,
        collection: str,
        entries: list[VectorEntry],
        requester: Any = None,
        request_ref: str | None = None,
    ) -> None:
        self.backend.add(collection, entries)

    def remove(
        self,
        collection: str,
        ref_ids: list[str],
        scope: str | None = None,
        path_prefix: str | None = None,
    ) -> None:
        self.backend.remove(collection, ref_ids, scope=scope, path_prefix=path_prefix)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embeds.append(list(texts))
        if self.embed_error is not None:
            raise self.embed_error
        if self.embed_returns is not None:
            return self.embed_returns
        return [vector_for(text) for text in texts]

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


class SearchHarness:
    """An inert actor whose vector-store proxy is a :class:`SearchStore`."""

    def __init__(self, actor: WorkspaceActor, store: SearchStore) -> None:
        self.actor = actor
        self.store = store
        self.orchestrator = DeadAddress("orchestrator")
        self.vs_address: MockActorAddress | None = MockActorAddress("#VectorStore")

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.actor._orchestrator = self.orchestrator
        monkeypatch.setattr(self.actor, "proxy_ask", self._ask)
        monkeypatch.setattr(self.actor, "proxy_tell", self._tell)

    def enable(self) -> None:
        self.actor.enable_rag(
            "alice",
            WorkspaceRagIndex(),
            DocumentReader(llm_client=None),
            CollectionConfig(backend="inmemory"),
        )

    def index(
        self,
        path: str,
        body: str,
        spans: list[tuple[int, int, list[str]]],
        *,
        scope: str | None = None,
        cache: bool = True,
        embedded: bool = True,
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

        Returns:
            The digest both maps agree on.
        """
        owner = scope or self.actor.config.workspace_name
        sha = content_sha(body.encode("utf-8"))
        chunks: list[RagChunk] = []
        for ordinal, (start, end, heading) in enumerate(spans):
            identity = chunk_id(owner, path, sha, ordinal)
            chunks.append(
                RagChunk(
                    chunk_id=identity,
                    ordinal=ordinal,
                    start=start,
                    end=end,
                    heading_path=heading,
                )
            )
            if embedded:
                self.store.store_chunk(identity, owner, path, ordinal, body[start:end])
        if owner == self.actor.config.workspace_name:
            self.actor.state.rag_index[path] = RagFile(
                path=path,
                status=RagStatus.EMBEDDED,
                indexed_sha=sha,
                chunks=chunks,
                chunk_count=len(chunks),
                updated_at=datetime.now(UTC),
            )
            self.actor.state.documents[path] = DocumentExtract(
                path=path,
                source_sha=sha,
                extractor_version=EXTRACTOR_VERSION,
                markdown=body if cache else None,
                char_count=len(body),
                extracted_at=datetime.now(UTC),
            )
        return sha

    def _ask(self, address: Any, actor_type: Any = None, timeout: int | None = None) -> Any:
        if address is self.orchestrator:
            return SimpleNamespace(get_team_member=lambda name: self.vs_address)
        if address is self.vs_address:
            return self.store
        raise AssertionError(f"unexpected ask target {address}")

    def _tell(self, address: Any, actor_type: Any = None) -> Any:
        return self.store


def build_actor(workspace_name: str = WORKSPACE_NAME) -> WorkspaceActor:
    """A started actor over *workspace_name*, with no actor thread."""
    started = WorkspaceActor(
        config=WorkspaceConfig(
            name=workspace_actor_name(workspace_name),
            role=WORKSPACE_ACTOR_ROLE,
            workspace_name=workspace_name,
        )
    )
    started.on_start()
    return started


@pytest.fixture
def store() -> SearchStore:
    return SearchStore()


@pytest.fixture
def search(
    workspace_tree: Path, store: SearchStore, monkeypatch: pytest.MonkeyPatch
) -> SearchHarness:
    """A harness with retrieval already enabled — the case every search assumes."""
    built = SearchHarness(build_actor(), store)
    built.install(monkeypatch)
    built.enable()
    return built


##
## The document the specs search
##

_INVOICE = "# Invoice\n\nPayment terms are net thirty.\n\nA refund is issued on request.\n"
_SPLIT = _INVOICE.index("A refund")
_FIRST = (0, _SPLIT, ["Invoice", "Payment terms"])
_SECOND = (_SPLIT, len(_INVOICE), ["Invoice", "Refunds"])


def hit_count(answer: str) -> int:
    """How many hits a rendered answer carries.

    Counted by their score labels rather than by splitting on the blank line
    between blocks: a chunk's own text routinely contains blank lines, so the
    obvious split over-counts and the spec would be measuring the document.
    """
    return sum(answer.count(label) for label in ("(hybrid: ", "(semantic: ", "(keyword match)"))


class TestDegradation:
    """Every failure mode answers a sentence and none of them raises."""

    def test_a_workspace_with_no_vector_store_answers_the_sentence(
        self, workspace_tree: Path, monkeypatch: pytest.MonkeyPatch, store: SearchStore
    ) -> None:
        """Retrieval was never enabled — the same sentence the indexer answers."""
        harness = SearchHarness(build_actor(), store)
        harness.install(monkeypatch)

        assert harness.actor.rag_search("payment") == _UNAVAILABLE

    def test_an_actor_with_a_proxy_but_no_parameters_still_answers_the_sentence(
        self, search: SearchHarness
    ) -> None:
        """Both halves of enablement are required; a half-enabled tree is degraded."""
        search.actor._rag_params = None

        assert search.actor.rag_search("payment") == _UNAVAILABLE

    def test_an_embed_that_raises_falls_back_to_the_keyword_leg(
        self, search: SearchHarness
    ) -> None:
        """One warning, no exception, and the lexical half still answers."""
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])
        search.store.embed_error = RuntimeError("the embedding provider is down")

        answer = search.actor.rag_search("payment")

        assert "invoice.md" in answer
        assert "keyword match" in answer

    def test_an_embed_that_returns_nothing_falls_back_to_the_keyword_leg(
        self, search: SearchHarness
    ) -> None:
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])
        search.store.embed_returns = []

        answer = search.actor.rag_search("payment")

        assert "keyword match" in answer
        assert search.store.searches == []

    def test_a_search_that_raises_falls_back_to_the_keyword_leg(
        self, search: SearchHarness
    ) -> None:
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])
        search.store.search_error = RuntimeError("cluster unreachable")

        answer = search.actor.rag_search("payment")

        assert "keyword match" in answer

    def test_a_failing_vector_leg_never_raises_out_of_the_search(
        self, search: SearchHarness
    ) -> None:
        """This actor owns the write gate; a retrieval failure must not reach it."""
        search.store.search_error = RuntimeError("cluster unreachable")

        search.actor.rag_search("nothing is indexed at all")  # must not raise

    def test_no_hits_is_a_sentence_that_is_not_the_unavailable_one(
        self, search: SearchHarness
    ) -> None:
        """ "Nothing matched" and "nothing is indexed" have different next steps."""
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND], embedded=False)

        answer = search.actor.rag_search("bicycles")

        assert answer == _NO_HITS
        assert answer != _UNAVAILABLE
        assert "workspace_rag_list" in answer


class TestTheVectorLeg:
    """It is scoped, over-fetched, and thresholded on the raw cosine."""

    def test_every_query_carries_the_workspace_as_its_scope(self, search: SearchHarness) -> None:
        """One ``workspace_chunks`` class holds every workspace of every team."""
        search.index("invoice.md", _INVOICE, [_FIRST])

        search.actor.rag_search("payment")

        [(collection, _, scope, _prefix)] = search.store.searches
        assert (collection, scope) == (RAG_COLLECTION, WORKSPACE_NAME)

    def test_an_empty_prefix_reaches_the_backend_as_none_rather_than_as_a_string(
        self, search: SearchHarness
    ) -> None:
        """``None`` filters nothing; ``""`` would be a predicate the backend applies."""
        search.index("invoice.md", _INVOICE, [_FIRST])

        search.actor.rag_search("payment")

        assert search.store.searches[0][3] is None

    def test_a_prefix_is_passed_to_the_backend(self, search: SearchHarness) -> None:
        """The predicate goes to the store, so the budget is not spent locally."""
        search.index("reports/invoice.md", _INVOICE, [_FIRST])

        search.actor.rag_search("payment", path_prefix="reports/")

        assert search.store.searches[0][3] == "reports/"

    def test_the_backend_is_over_fetched_because_fusion_reorders(
        self, search: SearchHarness
    ) -> None:
        search.index("invoice.md", _INVOICE, [_FIRST])

        search.actor.rag_search("payment", top_k=5)

        assert search.store.searches[0][1] == 5 * OVERFETCH

    def test_the_score_threshold_drops_weak_vector_hits(self, search: SearchHarness) -> None:
        """Applied to the raw cosine, before normalisation, so it stays absolute.

        The body is not cached, so the keyword leg contributes nothing and what
        the threshold drops is the whole of the answer.
        """
        search.index("holiday.md", "Holiday policy\n", [(0, 15, ["Holiday"])], cache=False)

        kept = search.actor.rag_search("holiday", score_threshold=0.0)
        dropped = search.actor.rag_search("holiday", score_threshold=1.5)

        assert "holiday.md" in kept
        assert dropped == _NO_HITS


class TestTheKeywordLeg:
    """Over the bodies the actor holds — and never over one it does not."""

    def test_it_matches_case_insensitively(self, search: SearchHarness) -> None:
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])
        search.store.embed_error = RuntimeError("vector leg off")

        assert "invoice.md" in search.actor.rag_search("PAYMENT")

    def test_an_evicted_body_contributes_nothing_and_is_never_sliced(
        self, search: SearchHarness
    ) -> None:
        """ADR-045 §3/§4: the offsets of an evicted file are provenance.

        The file stays indexed and its vector hits still render — what it loses is
        the lexical leg, which is a degradation and never an error. Slicing a
        ``None`` body would be a ``TypeError`` on the gate's own thread.
        """
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND], cache=False)
        search.store.embed_error = RuntimeError("vector leg off")

        answer = search.actor.rag_search("payment")  # must not raise

        assert answer == _NO_HITS

    def test_an_evicted_body_still_renders_through_the_vector_leg(
        self, search: SearchHarness
    ) -> None:
        """This is what keeps ``max_documents`` a bound on state, not on the corpus."""
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND], cache=False)

        answer = search.actor.rag_search("payment")

        assert "invoice.md" in answer
        assert "Payment terms are net thirty." in answer
        assert "semantic:" in answer

    def test_a_body_whose_digest_no_longer_matches_the_row_is_skipped(
        self, search: SearchHarness
    ) -> None:
        """The two maps have different lifetimes; mismatched offsets belong to neither."""
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])
        stale = search.actor.state.documents["invoice.md"]
        search.actor.state.documents["invoice.md"] = stale.model_copy(
            update={"source_sha": "a-different-digest"}
        )
        search.store.embed_error = RuntimeError("vector leg off")

        assert search.actor.rag_search("payment") == _NO_HITS

    def test_a_cached_path_absent_from_the_index_is_skipped_rather_than_raising(
        self, search: SearchHarness
    ) -> None:
        """The cache and the index have different caps as well as different lifetimes."""
        search.actor.state.documents["orphan.md"] = DocumentExtract(
            path="orphan.md",
            source_sha="sha",
            extractor_version=EXTRACTOR_VERSION,
            markdown="A payment note.\n",
            char_count=16,
            extracted_at=datetime.now(UTC),
        )
        search.store.embed_error = RuntimeError("vector leg off")

        assert search.actor.rag_search("payment") == _NO_HITS

    def test_only_the_chunk_whose_own_slice_carries_the_term_is_hit(
        self, search: SearchHarness
    ) -> None:
        """The offsets are what map a body match onto a chunk id."""
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])
        search.store.embed_error = RuntimeError("vector leg off")

        answer = search.actor.rag_search("refund")

        assert "Refunds" in answer
        assert "Payment terms" not in answer

    def test_the_prefix_filters_the_keyword_leg_too(self, search: SearchHarness) -> None:
        """The backend filters its own leg; nothing else would filter this one."""
        search.index("reports/invoice.md", _INVOICE, [_FIRST])
        search.index("notes/invoice.md", _INVOICE, [_FIRST])
        search.store.embed_error = RuntimeError("vector leg off")

        answer = search.actor.rag_search("payment", path_prefix="reports/")

        assert "reports/invoice.md" in answer
        assert "notes/invoice.md" not in answer

    def test_an_empty_query_hits_nothing_on_the_keyword_leg(self, search: SearchHarness) -> None:
        """A blank query must not match every chunk in the workspace."""
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])
        search.store.embed_error = RuntimeError("vector leg off")

        assert search.actor.rag_search("   ") == _NO_HITS


class TestTheRender:
    """Path, heading path, score label, and the chunk's text."""

    def test_a_hit_renders_its_path_and_heading_path(self, search: SearchHarness) -> None:
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])

        answer = search.actor.rag_search("payment", top_k=1)

        assert answer.startswith("invoice.md > Invoice > Payment terms (")

    def test_a_keyword_only_hit_is_labelled_as_one(self, search: SearchHarness) -> None:
        search.index("invoice.md", _INVOICE, [_FIRST], embedded=False)

        assert "(keyword match)" in search.actor.rag_search("payment")

    def test_a_vector_only_hit_is_labelled_semantic_with_its_raw_cosine(
        self, search: SearchHarness
    ) -> None:
        """The raw cosine, because a fused score means nothing outside its own set.

        ``0.71`` is the arithmetic and not a recorded observation: the chunk's own
        text carries "invoice" and "payment", so its bag-of-words vector is
        ``[1, 1, 0, 0]`` against the query's ``[0, 1, 0, 0]`` — a cosine of
        ``1 / sqrt(2)``. A **fused** score at this alpha would be ``0.70``.
        """
        search.index("invoice.md", _INVOICE, [_FIRST], cache=False)

        assert "(semantic: 0.71)" in search.actor.rag_search("payment")

    def test_a_hit_confirmed_by_both_legs_is_labelled_hybrid(self, search: SearchHarness) -> None:
        search.index("invoice.md", _INVOICE, [_FIRST])

        assert "(hybrid: " in search.actor.rag_search("payment", top_k=1)

    def test_the_text_of_a_hit_comes_from_the_store_and_not_from_the_cache(
        self, search: SearchHarness
    ) -> None:
        """``SearchHit.text`` is what survives an eviction; a slice is not."""
        search.index("invoice.md", _INVOICE, [_FIRST])
        held = search.actor.state.documents["invoice.md"]
        search.actor.state.documents["invoice.md"] = held.model_copy(
            update={"markdown": _INVOICE.replace("net thirty", "REPLACED")}
        )

        answer = search.actor.rag_search("payment", top_k=1)

        assert "net thirty" in answer
        assert "REPLACED" not in answer

    def test_a_hit_with_no_resolvable_ordinal_renders_without_a_heading_path(
        self, search: SearchHarness
    ) -> None:
        """The chunk text is still the answer, so a hit is never dropped for this."""
        search.index("invoice.md", _INVOICE, [_FIRST], cache=False)
        search.actor.state.rag_index.pop("invoice.md")

        answer = search.actor.rag_search("payment")

        assert answer.startswith("invoice.md (semantic:")

    def test_hits_are_separated_by_a_blank_line(self, search: SearchHarness) -> None:
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])

        answer = search.actor.rag_search("payment refund", top_k=2)

        assert hit_count(answer) == 2
        assert "\n\n" in answer


class TestTopK:
    """Honoured after filtering, and never under-filled by another scope."""

    def test_the_render_is_cut_to_top_k(self, search: SearchHarness) -> None:
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])

        answer = search.actor.rag_search("payment refund", top_k=1)

        assert hit_count(answer) == 1

    def test_a_larger_budget_returns_both(self, search: SearchHarness) -> None:
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])

        answer = search.actor.rag_search("payment refund", top_k=5)

        assert hit_count(answer) == 2

    def test_a_zero_budget_is_clamped_rather_than_over_fetching_nothing(
        self, search: SearchHarness
    ) -> None:
        """``top_k * OVERFETCH`` at zero would ask the backend for no rows at all."""
        search.index("invoice.md", _INVOICE, [_FIRST])

        answer = search.actor.rag_search("payment", top_k=0)

        assert "invoice.md" in answer
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

        answer = search.actor.rag_search("payment", top_k=5)

        assert "theirs.md" not in answer
        assert "mine.md" in answer

    def test_the_other_workspace_sees_only_its_own(
        self, workspace_tree: Path, store: SearchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Symmetry, so the spec is about the predicate and not about one ordering."""
        mine = SearchHarness(build_actor(), store)
        mine.install(monkeypatch)
        mine.enable()
        mine.index("mine.md", _INVOICE, [_FIRST])

        theirs = SearchHarness(build_actor("other-workspace"), store)
        theirs.install(monkeypatch)
        theirs.enable()
        theirs.index("theirs.md", _INVOICE, [_FIRST])

        answer = theirs.actor.rag_search("payment", top_k=5)

        assert "theirs.md" in answer
        assert "mine.md" not in answer

    def test_the_two_workspaces_mint_distinct_ids_for_the_same_file(
        self, search: SearchHarness
    ) -> None:
        """Without this the predicate would be filtering entries that had collided."""
        assert chunk_id(WORKSPACE_NAME, "a.md", "sha", 0) != chunk_id(
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
        answer = search.actor.rag_search("payment", path_prefix=prefix)

        assert "cannot contain" in answer
        assert answer != _UNAVAILABLE

    @pytest.mark.parametrize("backend", ["inmemory", "weaviate"])
    def test_the_same_sentence_comes_back_whatever_the_backend(
        self,
        workspace_tree: Path,
        store: SearchStore,
        monkeypatch: pytest.MonkeyPatch,
        backend: str,
    ) -> None:
        """The refusal is at the caller, so the two backends cannot disagree."""
        harness = SearchHarness(build_actor(), store)
        harness.install(monkeypatch)
        harness.actor.enable_rag(
            "alice",
            WorkspaceRagIndex(),
            DocumentReader(llm_client=None),
            CollectionConfig(backend=backend),
        )

        answers = {harness.actor.rag_search("payment", path_prefix="report?.md")}

        assert len(answers) == 1
        assert "cannot contain" in answers.pop()

    def test_a_refused_prefix_never_reaches_the_backend(self, search: SearchHarness) -> None:
        """No embed is spent either — the refusal is the first thing that happens."""
        search.index("invoice.md", _INVOICE, [_FIRST])

        search.actor.rag_search("payment", path_prefix="report*")

        assert search.store.searches == []
        assert search.store.embeds == []

    def test_the_in_memory_backend_treats_a_metacharacter_literally(self) -> None:
        """Half of the divergence the refusal exists for, pinned against real code."""
        from akgentic.tool.vector_store.inmemory import _entry_matches

        entry = VectorEntry(
            ref_type="workspace_chunk", ref_id="c", text="t", vector=[1.0], path="report?.md"
        )
        other = entry.model_copy(update={"path": "reportX.md"})

        assert _entry_matches(entry, None, "report?") is True
        assert _entry_matches(other, None, "report?") is False

    def test_a_legal_prefix_is_not_refused(self, search: SearchHarness) -> None:
        """The refusal must not spread to the ordinary case it exists to protect."""
        search.index("reports/invoice.md", _INVOICE, [_FIRST])

        answer = search.actor.rag_search("payment", path_prefix="reports/")

        assert "reports/invoice.md" in answer


class TestTheFusionKnobs:
    """``alpha`` and ``score_threshold`` reach the rule the package shares."""

    def test_the_cards_default_alpha_is_the_fusion_modules_own(self) -> None:
        """The literal in ``card/params.py`` is written out to avoid an import edge."""
        assert WorkspaceRagSearch().alpha == DEFAULT_ALPHA

    def test_alpha_none_takes_the_module_default(self, search: SearchHarness) -> None:
        search.index("invoice.md", _INVOICE, [_FIRST])

        assert search.actor.rag_search("payment", alpha=None) == search.actor.rag_search(
            "payment", alpha=DEFAULT_ALPHA
        )

    def test_pure_keyword_fusion_still_returns_the_keyword_hit(self, search: SearchHarness) -> None:
        """``alpha=0.0`` is pure keyword, and a vector-only hit then scores zero."""
        search.index("invoice.md", _INVOICE, [_FIRST])

        assert "invoice.md" in search.actor.rag_search("payment", alpha=0.0)


class TestTheStateItNeverTouches:
    """A search is a read: it must notify nothing and mutate nothing."""

    def test_a_search_makes_no_state_notification(self, search: SearchHarness) -> None:
        """A notify on a read path is a defect until a decision says otherwise."""
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])
        seen: list[Any] = []
        search.actor.state.observer(SimpleNamespace(notify_state_change=seen.append))
        seen.clear()

        search.actor.rag_search("payment")

        assert seen == []

    def test_a_search_leaves_the_index_untouched(self, search: SearchHarness) -> None:
        search.index("invoice.md", _INVOICE, [_FIRST, _SECOND])
        before = search.actor.state.rag_index["invoice.md"].model_copy(deep=True)

        search.actor.rag_search("payment")

        assert search.actor.state.rag_index["invoice.md"] == before


class TestTheWorkspaceStateContract:
    """``rag_search`` reads two maps that the state actually declares."""

    def test_both_maps_are_state_fields(self) -> None:
        """A map that were not persisted would empty on every resume."""
        assert {"documents", "rag_index"} <= set(WorkspaceState.model_fields)
