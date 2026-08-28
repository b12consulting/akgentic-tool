"""Unit tests for akgentic.tool.vector_store.hybrid — the shared fusion rule.

``fuse`` is the single point of truth for how keyword and vector hits are
ranked, and ``semantic_scores`` is the single point where a failing vector
store degrades to keyword-only. Both the knowledge-graph and the planning
search depend on them, so they are tested here directly rather than only
through those actors.

The fusion reproduces Weaviate's relativeScoreFusion, so the tests are written
against that contract — min-max normalisation per leg, weighted by ``alpha`` —
rather than against particular constants.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from akgentic.tool.vector_store.hybrid import (
    DEFAULT_ALPHA,
    OVERFETCH,
    fuse,
    hybrid_search,
    semantic_scores,
)
from akgentic.tool.vector_store.protocol import CollectionStatus, SearchHit, SearchResult

# ---------------------------------------------------------------------------
# fuse
# ---------------------------------------------------------------------------


class TestFuse:
    """The keyword/vector scoring rule."""

    def test_alpha_default_matches_the_weaviate_client(self) -> None:
        """The client sends alpha=0.7 by default; drifting from it reorders results."""
        assert DEFAULT_ALPHA == pytest.approx(0.7)

    def test_sole_vector_hit_normalises_to_the_top_of_its_leg(self) -> None:
        assert fuse([], {"a": 0.31}) == {"a": pytest.approx(DEFAULT_ALPHA)}

    def test_keyword_only_hit_scores_the_keyword_weight(self) -> None:
        assert fuse(["a"], {}) == {"a": pytest.approx(1.0 - DEFAULT_ALPHA)}

    def test_hit_in_both_legs_sums_both_weights(self) -> None:
        assert fuse(["a"], {"a": 0.8}) == {"a": pytest.approx(1.0)}

    def test_dual_hit_outranks_either_leg_alone(self) -> None:
        fused = fuse(["a", "b"], {"a": 0.9, "c": 0.9})
        assert fused["a"] > fused["b"]
        assert fused["a"] > fused["c"]

    def test_vector_leg_is_min_max_normalised(self) -> None:
        """The weakest vector hit normalises to zero — relativeScoreFusion's rule."""
        fused = fuse([], {"a": 0.9, "b": 0.7, "c": 0.5})
        assert fused["a"] == pytest.approx(DEFAULT_ALPHA)
        assert fused["b"] == pytest.approx(DEFAULT_ALPHA * 0.5)
        assert fused["c"] == pytest.approx(0.0)

    def test_normalisation_is_relative_not_absolute(self) -> None:
        """Two result sets with the same spread fuse identically, whatever the raw scores."""
        assert fuse([], {"a": 0.9, "b": 0.5}) == fuse([], {"a": 0.5, "b": 0.1})

    def test_equal_vector_scores_all_normalise_to_the_top(self) -> None:
        """No spread means nothing to rank on, so nothing is penalised."""
        fused = fuse([], {"a": 0.4, "b": 0.4})
        assert fused == {"a": pytest.approx(DEFAULT_ALPHA), "b": pytest.approx(DEFAULT_ALPHA)}

    def test_alpha_one_ignores_the_keyword_leg(self) -> None:
        fused = fuse(["b"], {"a": 0.9, "b": 0.5}, alpha=1.0)
        assert fused["b"] == pytest.approx(0.0)

    def test_alpha_zero_ignores_the_vector_leg(self) -> None:
        fused = fuse(["b"], {"a": 0.9, "b": 0.5}, alpha=0.0)
        assert fused == {"a": pytest.approx(0.0), "b": pytest.approx(1.0)}

    def test_scores_stay_within_the_unit_interval(self) -> None:
        fused = fuse(["a", "d"], {"a": 0.95, "b": 0.6, "c": 0.1})
        assert all(0.0 <= score <= 1.0 for score in fused.values())

    def test_covers_the_union_of_both_legs(self) -> None:
        assert set(fuse(["a", "b"], {"b": 0.5, "c": 0.4})) == {"a", "b", "c"}

    def test_repeated_keyword_key_is_counted_only_once(self) -> None:
        assert fuse(["a", "a", "a"], {}) == {"a": pytest.approx(1.0 - DEFAULT_ALPHA)}

    def test_empty_legs_produce_no_hits(self) -> None:
        assert fuse([], {}) == {}

    def test_does_not_mutate_the_vector_scores_argument(self) -> None:
        vector_scores = {"a": 0.5}
        fuse(["a", "b"], vector_scores)
        assert vector_scores == {"a": 0.5}


# ---------------------------------------------------------------------------
# hybrid_search
# ---------------------------------------------------------------------------


class TestHybridSearch:
    """The whole algorithm: semantic leg, threshold, fusion, ranking."""

    def test_orders_best_first(self) -> None:
        proxy = _proxy(
            [
                SearchHit(ref_type="t", ref_id="a", text="", score=0.9),
                SearchHit(ref_type="t", ref_id="b", text="", score=0.5),
            ]
        )
        result = hybrid_search(["b"], proxy, "c", "query", top_k=5)
        assert [key for key, _ in result.ranked] == ["a", "b"]

    def test_returns_the_raw_vector_scores_for_labelling(self) -> None:
        """Planning renders the absolute cosine, which fusion would otherwise hide."""
        proxy = _proxy([SearchHit(ref_type="t", ref_id="a", text="", score=0.83)])
        assert hybrid_search([], proxy, "c", "query", top_k=5).vector_scores == {
            "a": pytest.approx(0.83)
        }

    def test_threshold_applies_to_the_raw_score_before_normalisation(self) -> None:
        """A weak hit is dropped on its absolute cosine, not on its rank in the set."""
        proxy = _proxy(
            [
                SearchHit(ref_type="t", ref_id="a", text="", score=0.9),
                SearchHit(ref_type="t", ref_id="b", text="", score=0.2),
            ]
        )
        result = hybrid_search([], proxy, "c", "query", top_k=5, score_threshold=0.5)
        assert [key for key, _ in result.ranked] == ["a"]
        assert "b" not in result.vector_scores

    def test_over_fetches_past_top_k(self) -> None:
        """Fusion and caller-side filtering both drop candidates, so ask for more."""
        proxy = _proxy([])
        hybrid_search([], proxy, "c", "query", top_k=5)
        assert proxy.search.call_args.args[2] == 5 * OVERFETCH

    def test_result_is_not_cut_to_top_k(self) -> None:
        """Callers slice after materialising, so a stale key costs no result slot."""
        proxy = _proxy(
            [SearchHit(ref_type="t", ref_id=str(i), text="", score=0.9) for i in range(6)]
        )
        assert len(hybrid_search([], proxy, "c", "query", top_k=2).ranked) == 6

    def test_no_vector_hits_degrades_to_keyword_order(self) -> None:
        """The empty semantic leg needs no special case — keyword keys tie and hold order."""
        result = hybrid_search(["a", "b", "c"], None, "c", "query", top_k=5)
        assert [key for key, _ in result.ranked] == ["a", "b", "c"]
        assert result.vector_scores == {}

    def test_keyword_and_vector_hits_are_unioned(self) -> None:
        proxy = _proxy([SearchHit(ref_type="t", ref_id="a", text="", score=0.9)])
        result = hybrid_search(["b"], proxy, "c", "query", top_k=5)
        assert {key for key, _ in result.ranked} == {"a", "b"}

    def test_alpha_is_forwarded_to_the_fusion(self) -> None:
        proxy = _proxy([SearchHit(ref_type="t", ref_id="a", text="", score=0.9)])
        result = hybrid_search(["b"], proxy, "c", "query", top_k=5, alpha=0.1)
        assert [key for key, _ in result.ranked] == ["b", "a"]


# ---------------------------------------------------------------------------
# semantic_scores
# ---------------------------------------------------------------------------


def _proxy(hits: list[SearchHit]) -> MagicMock:
    """Build a vector store proxy returning *hits* from a successful search."""
    proxy = MagicMock()
    proxy.embed.return_value = [[0.1, 0.2, 0.3]]
    proxy.search.return_value = SearchResult(
        hits=hits, status=CollectionStatus.READY, indexing_pending=0
    )
    return proxy


class TestSemanticScores:
    """Embedding + search, degrading to no hits instead of raising."""

    def test_maps_hits_to_ref_id_scores(self) -> None:
        proxy = _proxy(
            [
                SearchHit(ref_type="entity", ref_id="a", text="", score=0.9),
                SearchHit(ref_type="entity", ref_id="b", text="", score=0.4),
            ]
        )
        assert semantic_scores(proxy, "c", "query", 5) == {"a": 0.9, "b": 0.4}

    def test_returns_raw_scores_not_normalised_ones(self) -> None:
        """Thresholding happens on the raw cosine, so it must survive this call."""
        proxy = _proxy([SearchHit(ref_type="entity", ref_id="a", text="", score=0.42)])
        assert semantic_scores(proxy, "c", "query", 5) == {"a": pytest.approx(0.42)}

    def test_preserves_backend_ranking_order(self) -> None:
        proxy = _proxy(
            [
                SearchHit(ref_type="entity", ref_id="a", text="", score=0.9),
                SearchHit(ref_type="entity", ref_id="b", text="", score=0.4),
            ]
        )
        assert list(semantic_scores(proxy, "c", "query", 5)) == ["a", "b"]

    def test_forwards_collection_and_top_k_to_the_backend(self) -> None:
        proxy = _proxy([])
        semantic_scores(proxy, "planning", "query", 7)
        proxy.search.assert_called_once_with("planning", [0.1, 0.2, 0.3], 7)

    def test_no_proxy_yields_no_hits(self) -> None:
        assert semantic_scores(None, "c", "query", 5) == {}

    def test_empty_embedding_yields_no_hits(self) -> None:
        proxy = _proxy([])
        proxy.embed.return_value = []
        assert semantic_scores(proxy, "c", "query", 5) == {}
        proxy.search.assert_not_called()

    def test_failing_embed_yields_no_hits(self) -> None:
        proxy = _proxy([])
        proxy.embed.side_effect = RuntimeError("embedding service down")
        assert semantic_scores(proxy, "c", "query", 5) == {}

    def test_failing_search_yields_no_hits(self) -> None:
        proxy = _proxy([])
        proxy.search.side_effect = RuntimeError("backend unreachable")
        assert semantic_scores(proxy, "c", "query", 5) == {}
