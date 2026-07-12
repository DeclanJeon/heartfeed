"""Tests for retrieval components."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from dating_rag.domain.models import RetrievalResult
from dating_rag.retrieval.diversifier import (
    _is_duplicate,
    _text_similarity,
    diversify,
    diversify_results,
)
from dating_rag.retrieval.hybrid import HybridRetriever, hybrid_search, reciprocal_rank_fusion


# ---------------------------------------------------------------------------
# RRF Tests
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    """Tests for RRF fusion."""

    def test_single_list(self) -> None:
        results = [
            {"id": "a", "score": 0.9, "payload": {}},
            {"id": "b", "score": 0.8, "payload": {}},
        ]
        fused = reciprocal_rank_fusion([results])
        assert len(fused) == 2
        # Order preserved for single list
        assert fused[0]["id"] == "a"

    def test_two_lists_merging(self) -> None:
        list1 = [
            {"id": "a", "score": 0.9, "payload": {}},
            {"id": "b", "score": 0.8, "payload": {}},
        ]
        list2 = [
            {"id": "b", "score": 0.7, "payload": {}},
            {"id": "c", "score": 0.6, "payload": {}},
        ]
        fused = reciprocal_rank_fusion([list1, list2])
        # "b" appears in both lists, should rank highest
        assert fused[0]["id"] == "b"

    def test_empty_lists(self) -> None:
        fused = reciprocal_rank_fusion([[], []])
        assert fused == []

    def test_three_lists(self) -> None:
        """Item appearing in all 3 lists beats items in only 1."""
        list1 = [{"id": "x", "score": 1.0, "payload": {}}]
        list2 = [{"id": "x", "score": 1.0, "payload": {}}]
        list3 = [{"id": "y", "score": 1.0, "payload": {}}]
        fused = reciprocal_rank_fusion([list1, list2, list3])
        assert fused[0]["id"] == "x"

    def test_custom_k(self) -> None:
        results = [{"id": "a", "score": 0.9, "payload": {}}]
        fused_k10 = reciprocal_rank_fusion([results], k=10)
        fused_k100 = reciprocal_rank_fusion([results], k=100)
        # Smaller k -> higher score
        assert fused_k10[0]["score"] > fused_k100[0]["score"]


# ---------------------------------------------------------------------------
# HybridRetriever Tests
# ---------------------------------------------------------------------------


class TestHybridRetriever:
    """Tests for HybridRetriever class."""

    def _make_retriever(self) -> tuple[HybridRetriever, MagicMock, MagicMock]:
        """Create a HybridRetriever with mocked store and embedder."""
        store = MagicMock()
        embedder = MagicMock()
        embedder.encode_query.return_value = {
            "dense": np.array([0.1] * 1024),
            "sparse": {100: 0.5, 200: 0.3, 300: 0.2},
        }
        retriever = HybridRetriever(
            store=store,
            embedder=embedder,
            collection_name="test_collection",
            dense_top_k=5,
            sparse_top_k=10,
            dense_threshold=0.3,
            rrf_k=60,
        )
        return retriever, store, embedder

    def test_dense_search(self) -> None:
        retriever, store, _ = self._make_retriever()
        store.search.return_value = [
            {"id": "1", "score": 0.9, "payload": {"text": "hello"}},
        ]
        results = retriever.dense_search([0.1] * 1024)
        assert len(results) == 1
        store.search.assert_called_once_with(
            collection_name="test_collection",
            query_vector=[0.1] * 1024,
            vector_name="dense",
            top_k=5,
            score_threshold=0.3,
            query_filter=None,
        )

    def test_dense_search_with_filters(self) -> None:
        retriever, store, _ = self._make_retriever()
        store.search.return_value = []
        filters = {"category": "approach"}
        retriever.dense_search([0.1] * 1024, filters=filters)
        store.search.assert_called_once_with(
            collection_name="test_collection",
            query_vector=[0.1] * 1024,
            vector_name="dense",
            top_k=5,
            score_threshold=0.3,
            query_filter=filters,
        )

    def test_sparse_search(self) -> None:
        from qdrant_client.models import SparseVector

        retriever, store, _ = self._make_retriever()
        store.search.return_value = []
        sv = SparseVector(indices=[100, 200], values=[0.5, 0.3])
        retriever.sparse_search(sv)
        store.search.assert_called_once_with(
            collection_name="test_collection",
            query_vector=sv,
            vector_name="sparse",
            top_k=10,
            score_threshold=0.0,
            query_filter=None,
        )

    def test_rrf_fusion(self) -> None:
        retriever, _, _ = self._make_retriever()
        dense = [{"id": "a", "score": 0.9, "payload": {}}]
        sparse = [{"id": "b", "score": 0.8, "payload": {}}]
        fused = retriever.rrf_fusion(dense, sparse)
        assert len(fused) == 2

    def test_search_end_to_end(self) -> None:
        retriever, store, embedder = self._make_retriever()
        store.search.side_effect = [
            # Dense results
            [
                {"id": "c1", "score": 0.9, "payload": {"text": "dense hit", "video_id": "v1"}},
                {"id": "c2", "score": 0.8, "payload": {"text": "also dense", "video_id": "v2"}},
            ],
            # Sparse results
            [
                {"id": "c2", "score": 0.7, "payload": {"text": "also dense", "video_id": "v2"}},
                {"id": "c3", "score": 0.6, "payload": {"text": "sparse hit", "video_id": "v3"}},
            ],
        ]
        results = retriever.search("how to approach women", limit=2)
        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        # c2 appears in both lists -> should rank first
        assert results[0].chunk_id == "c2"
        embedder.encode_query.assert_called_once_with("how to approach women")

    def test_search_with_limit(self) -> None:
        retriever, store, _ = self._make_retriever()
        store.search.side_effect = [
            [{"id": f"c{i}", "score": 1.0 - i * 0.1, "payload": {"text": f"t{i}"}} for i in range(5)],
            [{"id": f"c{i}", "score": 1.0 - i * 0.1, "payload": {"text": f"t{i}"}} for i in range(5)],
        ]
        results = retriever.search("test", limit=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Reranker Tests
# ---------------------------------------------------------------------------


class TestPassageReranker:
    """Tests for PassageReranker class."""

    def test_rerank_reorders_by_score(self) -> None:
        import sys

        mock_ce_module = MagicMock()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.9, 0.5])
        mock_ce_module.CrossEncoder.return_value = mock_model

        sys.modules.setdefault("sentence_transformers", mock_ce_module)
        from dating_rag.retrieval.reranker import PassageReranker

        reranker = PassageReranker()
        candidates = [
            RetrievalResult(chunk_id="1", text="low relevance", score=0.9),
            RetrievalResult(chunk_id="2", text="high relevance", score=0.8),
            RetrievalResult(chunk_id="3", text="medium relevance", score=0.7),
        ]
        results = reranker.rerank("test query", candidates, top_k=2)
        assert len(results) == 2
        # Highest reranker score first
        assert results[0].chunk_id == "2"
        assert results[0].score == 0.9

    def test_rerank_empty(self) -> None:
        import sys

        mock_ce_module = MagicMock()
        mock_ce_module.CrossEncoder.return_value = MagicMock()

        sys.modules.setdefault("sentence_transformers", mock_ce_module)
        from dating_rag.retrieval.reranker import PassageReranker

        reranker = PassageReranker()
        results = reranker.rerank("test", [])
        assert results == []

    def test_backward_compat_alias(self) -> None:
        from dating_rag.retrieval.reranker import PassageReranker, Reranker

        assert Reranker is PassageReranker


# ---------------------------------------------------------------------------
# Diversifier Tests
# ---------------------------------------------------------------------------


class TestDiversifyResults:
    """Tests for result diversification."""

    def test_channel_limit(self) -> None:
        results = [
            RetrievalResult(
                chunk_id=f"chunk{i}",
                text=f"Text {i}",
                score=1.0 - i * 0.1,
                metadata={"channel_id": "ch1", "video_id": f"vid{i}"},
            )
            for i in range(5)
        ]
        diversified = diversify_results(results, max_per_channel=2)
        assert len(diversified) == 2

    def test_video_limit(self) -> None:
        results = [
            RetrievalResult(
                chunk_id=f"chunk{i}",
                text=f"Text {i}",
                score=1.0 - i * 0.1,
                metadata={"channel_id": f"ch{i}", "video_id": "vid1"},
            )
            for i in range(5)
        ]
        diversified = diversify_results(results, max_per_video=2)
        assert len(diversified) == 2

    def test_empty_results(self) -> None:
        diversified = diversify_results([])
        assert diversified == []

    def test_all_different_sources(self) -> None:
        results = [
            RetrievalResult(
                chunk_id=f"chunk{i}",
                text=f"Text {i}",
                score=1.0 - i * 0.1,
                metadata={"channel_id": f"ch{i}", "video_id": f"vid{i}"},
            )
            for i in range(5)
        ]
        diversified = diversify_results(results, max_per_channel=1)
        assert len(diversified) == 5

    def test_limit_parameter(self) -> None:
        results = [
            RetrievalResult(
                chunk_id=f"chunk{i}",
                text=f"Unique text {i}",
                score=1.0 - i * 0.1,
                metadata={"channel_id": f"ch{i}", "video_id": f"vid{i}"},
            )
            for i in range(10)
        ]
        diversified = diversify_results(results, limit=3)
        assert len(diversified) == 3

    def test_deduplication(self) -> None:
        """Near-identical chunks should be deduplicated."""
        results = [
            RetrievalResult(chunk_id="1", text="how to approach women at a bar", score=0.9),
            RetrievalResult(chunk_id="2", text="how to approach women at a bar", score=0.85),
            RetrievalResult(chunk_id="3", text="completely different topic here", score=0.8),
        ]
        diversified = diversify_results(results, dedup_threshold=0.85)
        # Second result is a duplicate of the first
        assert len(diversified) == 2
        assert diversified[0].chunk_id == "1"
        assert diversified[1].chunk_id == "3"


class TestTextSimilarity:
    """Tests for text similarity utility."""

    def test_identical_texts(self) -> None:
        assert _text_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self) -> None:
        assert _text_similarity("hello world", "foo bar") == 0.0

    def test_partial_overlap(self) -> None:
        sim = _text_similarity("hello world foo", "hello world bar")
        assert 0.0 < sim < 1.0

    def test_empty_texts(self) -> None:
        assert _text_similarity("", "") == 1.0


class TestIsDuplicate:
    """Tests for deduplication check."""

    def test_exact_duplicate_detected(self) -> None:
        existing = [RetrievalResult(chunk_id="1", text="same text here", score=0.9)]
        candidate = RetrievalResult(chunk_id="2", text="same text here", score=0.8)
        assert _is_duplicate(candidate, existing) is True

    def test_unique_text_not_flagged(self) -> None:
        existing = [RetrievalResult(chunk_id="1", text="completely different", score=0.9)]
        candidate = RetrievalResult(chunk_id="2", text="new unique content", score=0.8)
        assert _is_duplicate(candidate, existing) is False


class TestDiversifyWrapper:
    """Tests for the convenience diversify function."""

    def test_diversify_delegates(self) -> None:
        results = [
            RetrievalResult(
                chunk_id=f"c{i}",
                text=f"text {i}",
                score=1.0 - i * 0.1,
                metadata={"channel_id": "ch1", "video_id": f"v{i}"},
            )
            for i in range(5)
        ]
        diversified = diversify(results, limit=2, max_per_channel=3)
        assert len(diversified) == 2


class TestHybridSearchStandalone:
    """Tests for the standalone hybrid_search function."""

    def test_basic_hybrid_search(self) -> None:
        store = MagicMock()
        store.search.side_effect = [
            [{"id": "a", "score": 0.9, "payload": {"text": "dense"}}],
            [{"id": "b", "score": 0.8, "payload": {"text": "sparse"}}],
        ]
        results = hybrid_search(store, "col", [0.1] * 1024)
        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_hybrid_search_with_filter(self) -> None:
        store = MagicMock()
        store.search.side_effect = [[], []]
        filters = {"category": "mindset"}
        hybrid_search(store, "col", [0.1] * 1024, query_filter=filters)
        # Both calls should receive the filter
        for call in store.search.call_args_list:
            assert call.kwargs.get("query_filter") == filters
