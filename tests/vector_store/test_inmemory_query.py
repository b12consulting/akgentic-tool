"""In-memory backend query refinement — filters and score threshold.

The in-memory backend has no native query engine, so it applies ``VectorQuery``
constraints client-side; these tests pin that behaviour.
"""

from __future__ import annotations

from akgentic.tool.vector_store.inmemory import InMemoryBackend
from akgentic.tool.vector_store.protocol import CollectionConfig, VectorQuery
from akgentic.tool.vector_store.vector import VectorEntry


def _backend() -> InMemoryBackend:
    backend = InMemoryBackend()
    backend.create_collection("c", CollectionConfig(dimension=3))
    backend.add(
        "c",
        [
            VectorEntry(ref_type="entity", ref_id="a", text="a", vector=[1.0, 0.0, 0.0]),
            VectorEntry(ref_type="relation", ref_id="b", text="b", vector=[0.9, 0.1, 0.0]),
            VectorEntry(ref_type="entity", ref_id="c", text="c", vector=[0.0, 1.0, 0.0]),
        ],
    )
    return backend


class TestNoQuery:
    def test_plain_search_returns_all_ranked(self) -> None:
        result = _backend().search("c", [1.0, 0.0, 0.0], top_k=10)
        assert [h.ref_id for h in result.hits] == ["a", "b", "c"]


class TestFilters:
    def test_filter_by_ref_type(self) -> None:
        result = _backend().search(
            "c", [1.0, 0.0, 0.0], top_k=10, query=VectorQuery(filters={"ref_type": "entity"})
        )
        assert {h.ref_id for h in result.hits} == {"a", "c"}
        assert all(h.ref_type == "entity" for h in result.hits)

    def test_filter_match_any(self) -> None:
        result = _backend().search(
            "c",
            [1.0, 0.0, 0.0],
            top_k=10,
            query=VectorQuery(filters={"ref_id": ["a", "c"]}),
        )
        assert {h.ref_id for h in result.hits} == {"a", "c"}

    def test_unknown_filter_key_matches_nothing(self) -> None:
        result = _backend().search(
            "c", [1.0, 0.0, 0.0], top_k=10, query=VectorQuery(filters={"nope": "x"})
        )
        assert result.hits == []


class TestScoreThreshold:
    def test_threshold_drops_low_scorers(self) -> None:
        # Query aligned with 'a'; 'c' is orthogonal (cosine 0) and must be dropped.
        result = _backend().search(
            "c", [1.0, 0.0, 0.0], top_k=10, query=VectorQuery(score_threshold=0.5)
        )
        assert {h.ref_id for h in result.hits} == {"a", "b"}

    def test_threshold_and_filter_combined(self) -> None:
        result = _backend().search(
            "c",
            [1.0, 0.0, 0.0],
            top_k=10,
            query=VectorQuery(filters={"ref_type": "entity"}, score_threshold=0.5),
        )
        # entity + score>=0.5 leaves only 'a' ('c' is entity but orthogonal).
        assert {h.ref_id for h in result.hits} == {"a"}

    def test_top_k_still_caps_filtered_results(self) -> None:
        result = _backend().search(
            "c", [1.0, 0.0, 0.0], top_k=1, query=VectorQuery(filters={"ref_type": "entity"})
        )
        assert len(result.hits) == 1
        assert result.hits[0].ref_id == "a"
