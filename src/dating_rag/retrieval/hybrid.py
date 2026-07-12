"""Hybrid retrieval combining dense and sparse search."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qdrant_client.models import SparseVector

from dating_rag.domain.models import RetrievalResult
from dating_rag.store.qdrant import QdrantStore

if TYPE_CHECKING:
    from dating_rag.embeddings.bge_m3 import BgeM3Embedder


def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        result_lists: Lists of search results to fuse.
        k: RRF constant (typically 60).

    Returns:
        Fused and re-ranked results.
    """
    scores: dict[str, float] = {}
    id_to_result: dict[str, dict[str, Any]] = {}

    for results in result_lists:
        for rank, result in enumerate(results):
            result_id = str(result["id"])
            scores[result_id] = scores.get(result_id, 0.0) + 1.0 / (k + rank + 1)
            id_to_result[result_id] = result

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    return [
        {**id_to_result[id_], "score": scores[id_]}
        for id_ in sorted_ids
    ]


def _to_sparse_vector(lexical_weights: dict[int, float] | dict[str, float]) -> SparseVector:
    """Convert a token_id→weight dict to a Qdrant SparseVector.

    Args:
        lexical_weights: Dict mapping token IDs to their weights.

    Returns:
        Qdrant SparseVector with sorted indices and values.
    """
    indices = sorted(int(k) for k in lexical_weights)
    values = [float(lexical_weights[k]) for k in indices]
    return SparseVector(indices=indices, values=values)


class HybridRetriever:
    """Hybrid retriever combining dense and sparse search with RRF fusion.

    Uses QdrantStore for vector search and BgeM3Embedder for query encoding.
    Metadata filters narrow results before fusion.
    """

    def __init__(
        self,
        store: QdrantStore,
        embedder: BgeM3Embedder,
        collection_name: str = "transcripts",
        *,
        dense_top_k: int = 30,
        sparse_top_k: int = 30,
        dense_threshold: float = 0.35,
        rrf_k: int = 60,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            store: Qdrant store instance.
            embedder: Embedding model for query encoding.
            collection_name: Default collection to search.
            dense_top_k: Number of dense results to retrieve.
            sparse_top_k: Number of sparse results to retrieve.
            dense_threshold: Minimum dense similarity score.
            rrf_k: RRF fusion constant.
        """
        self.store = store
        self.embedder = embedder
        self.collection_name = collection_name
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.dense_threshold = dense_threshold
        self.rrf_k = rrf_k

    def dense_search(
        self,
        vector: list[float],
        *,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Perform dense vector search.

        Args:
            vector: Dense query embedding.
            filters: Optional metadata filters.
            limit: Override default dense_top_k.

        Returns:
            List of search result dicts with id, score, payload.
        """
        return self.store.search(
            collection_name=self.collection_name,
            query_vector=vector,
            vector_name="dense",
            top_k=limit or self.dense_top_k,
            score_threshold=self.dense_threshold,
            query_filter=filters,
        )

    def sparse_search(
        self,
        sparse_vector: SparseVector,
        *,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Perform sparse (lexical) search using BGE-M3 sparse vectors.

        Args:
            sparse_vector: Sparse vector from BGE-M3 lexical encoding.
            filters: Optional metadata filters.
            limit: Override default sparse_top_k.

        Returns:
            List of search result dicts with id, score, payload.
        """
        return self.store.search(
            collection_name=self.collection_name,
            query_vector=sparse_vector,
            vector_name="sparse",
            top_k=limit or self.sparse_top_k,
            score_threshold=0.0,
            query_filter=filters,
        )

    def rrf_fusion(
        self,
        dense_results: list[dict[str, Any]],
        sparse_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Fuse dense and sparse results using Reciprocal Rank Fusion.

        Args:
            dense_results: Results from dense search.
            sparse_results: Results from sparse search.

        Returns:
            Fused and re-ranked results.
        """
        return reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=self.rrf_k,
        )

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[RetrievalResult]:
        """Perform full hybrid search: encode query, search both vectors, fuse.

        Args:
            query: Natural language query text.
            filters: Optional metadata filters (category, channel_id, views, etc.).
            limit: Maximum results to return (defaults to dense_top_k).

        Returns:
            Fused retrieval results as RetrievalResult objects.
        """
        # Encode query into dense + sparse vectors
        encoded = self.embedder.encode_query(query)
        dense_vector: list[float] = encoded["dense"].tolist()
        sparse_vector = _to_sparse_vector(encoded["sparse"])

        # Dense + sparse retrieval
        dense_results = self.dense_search(dense_vector, filters=filters)
        sparse_results = self.sparse_search(sparse_vector, filters=filters)

        # Fuse with RRF
        fused = self.rrf_fusion(dense_results, sparse_results)

        # Convert to domain objects, apply limit
        results = [
            RetrievalResult(
                chunk_id=str(r["id"]),
                text=r["payload"].get("text", ""),
                score=r["score"],
                source_type="transcript",
                metadata=r["payload"],
            )
            for r in fused
        ]

        if limit is not None:
            results = results[:limit]

        return results


def hybrid_search(
    store: QdrantStore,
    collection_name: str,
    dense_vector: list[float],
    *,
    dense_top_k: int = 30,
    sparse_top_k: int = 30,
    dense_threshold: float = 0.35,
    rrf_k: int = 60,
    query_filter: dict[str, Any] | None = None,
) -> list[RetrievalResult]:
    """Perform hybrid dense-only retrieval with RRF fusion.

    Standalone function variant — uses two dense passes (low threshold + wide net)
    as a placeholder when no sparse vector is available.
    Prefer HybridRetriever for production use with real sparse vectors.

    Args:
        store: Qdrant store instance.
        collection_name: Collection to search.
        dense_vector: Dense embedding of the query.
        dense_top_k: Number of dense results.
        sparse_top_k: Number of sparse results (wide net).
        dense_threshold: Minimum dense score threshold.
        rrf_k: RRF constant.
        query_filter: Optional metadata filter dict.

    Returns:
        Fused retrieval results.
    """
    dense_results = store.search(
        collection_name=collection_name,
        query_vector=dense_vector,
        vector_name="dense",
        top_k=dense_top_k,
        score_threshold=dense_threshold,
        query_filter=query_filter,
    )

    # Wide-net second pass as sparse stand-in
    sparse_results = store.search(
        collection_name=collection_name,
        query_vector=dense_vector,
        vector_name="dense",
        top_k=sparse_top_k,
        score_threshold=0.0,
        query_filter=query_filter,
    )

    fused = reciprocal_rank_fusion([dense_results, sparse_results], k=rrf_k)

    return [
        RetrievalResult(
            chunk_id=str(r["id"]),
            text=r["payload"].get("text", ""),
            score=r["score"],
            source_type="transcript",
            metadata=r["payload"],
        )
        for r in fused
    ]
