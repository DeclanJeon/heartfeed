"""Qdrant vector store client with dense + sparse hybrid support."""

from __future__ import annotations

from typing import Any, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from dating_rag.domain.models import TranscriptChunk

# Payload fields to index for filtering
PAYLOAD_INDEXES: list[tuple[str, PayloadSchemaType]] = [
    ("video_id", PayloadSchemaType.KEYWORD),
    ("channel_id", PayloadSchemaType.KEYWORD),
    ("category", PayloadSchemaType.KEYWORD),
    ("language", PayloadSchemaType.KEYWORD),
    ("views", PayloadSchemaType.INTEGER),
]


def build_qdrant_filter(filters: dict[str, Any] | None) -> Filter | None:
    """Convert a dict of metadata filters to a Qdrant Filter object.

    Supports:
        - Exact match: {"category": "approach"} -> MatchValue
        - Range: {"views": {"gte": 1000}} -> Range

    Args:
        filters: Dict of field -> value or field -> {"gte"/"lte"} conditions.

    Returns:
        Qdrant Filter or None if no filters.
    """
    if not filters:
        return None

    conditions: list[FieldCondition] = []
    for key, value in filters.items():
        if isinstance(value, dict):
            conditions.append(
                FieldCondition(key=key, range=Range(
                    gte=value.get("gte"),
                    lte=value.get("lte"),
                ))
            )
        else:
            conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )

    return Filter(must=conditions)


class QdrantStore:
    """Client for storing and querying transcript chunks in Qdrant.

    Supports dense and sparse named vectors for hybrid retrieval.
    """

    def __init__(self, url: str = "http://localhost:6333", api_key: str = "") -> None:
        """Initialize Qdrant client.

        Args:
            url: Qdrant server URL.
            api_key: Optional API key for authentication.
        """
        self.client = QdrantClient(url=url, api_key=api_key or None)

    def create_collection(
        self,
        collection_name: str,
        dense_vector_size: int = 1024,
    ) -> None:
        """Create a collection with dense + sparse named vectors.

        Creates payload indexes for: video_id, channel_id, category,
        language, views.

        Args:
            collection_name: Name of the collection.
            dense_vector_size: Dimensionality of the dense vectors.
        """
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(size=dense_vector_size, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams()),
            },
        )
        for field_name, field_type in PAYLOAD_INDEXES:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_type,
            )

    def collection_exists(self, collection_name: str) -> bool:
        """Check whether a collection exists.

        Args:
            collection_name: Name of the collection.

        Returns:
            True if the collection exists.
        """
        return self.client.collection_exists(collection_name=collection_name)

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection.

        Args:
            collection_name: Name of the collection to delete.
        """
        self.client.delete_collection(collection_name=collection_name)

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[TranscriptChunk],
        dense_embeddings: Sequence[list[float] | Any],
        sparse_embeddings: Sequence[dict[int, float]] | None = None,
    ) -> None:
        """Insert or update transcript chunks with dense + sparse embeddings.

        Args:
            collection_name: Target collection name.
            chunks: List of transcript chunks.
            dense_embeddings: Corresponding dense embeddings.
            sparse_embeddings: Optional sparse embeddings from BGE-M3
                (list of token_id→weight dicts).
        """
        points = []
        for i, (chunk, dense_emb) in enumerate(zip(chunks, dense_embeddings)):
            payload = chunk.model_dump()
            if payload.get("published_at"):
                payload["published_at"] = payload["published_at"].isoformat()

            vectors: dict[str, Any] = {"dense": list(dense_emb)}

            if sparse_embeddings is not None:
                sparse = sparse_embeddings[i]
                indices = sorted(sparse.keys())
                vectors["sparse"] = SparseVector(
                    indices=indices,
                    values=[float(sparse[idx]) for idx in indices],
                )

            points.append(
                PointStruct(
                    id=chunk.chunk_id,
                    vector=vectors,
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=collection_name, points=points)

    def search(
        self,
        collection_name: str,
        query_vector: list[float] | SparseVector,
        vector_name: str = "dense",
        top_k: int = 10,
        score_threshold: float = 0.0,
        query_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar chunks using a named vector.

        Args:
            collection_name: Collection to search.
            query_vector: Dense embedding (list[float]) or SparseVector.
            vector_name: Which named vector to use ('dense' or 'sparse').
            top_k: Maximum results to return.
            score_threshold: Minimum similarity score.
            query_filter: Optional dict of field→value filters.

        Returns:
            List of search results with scores and payloads.
        """
        qdrant_filter = build_qdrant_filter(query_filter)

        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vector_name,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
        )

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload or {},
            }
            for hit in results.points
        ]

    def hybrid_search(
        self,
        collection_name: str,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        *,
        dense_top_k: int = 30,
        sparse_top_k: int = 30,
        dense_threshold: float = 0.0,
        sparse_threshold: float = 0.0,
        rrf_k: int = 60,
        query_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid dense + sparse search with Reciprocal Rank Fusion.

        Performs both dense and sparse searches in parallel, then fuses
        results using RRF to combine lexical and semantic relevance.

        Args:
            collection_name: Collection to search.
            dense_vector: Dense embedding of the query.
            sparse_vector: Sparse embedding of the query.
            dense_top_k: Number of dense results to retrieve.
            sparse_top_k: Number of sparse results to retrieve.
            dense_threshold: Minimum dense score.
            sparse_threshold: Minimum sparse score.
            rrf_k: RRF constant (typically 60).
            query_filter: Optional dict of field→value filters.

        Returns:
            Fused results sorted by RRF score (descending).
        """
        dense_results = self.search(
            collection_name=collection_name,
            query_vector=dense_vector,
            vector_name="dense",
            top_k=dense_top_k,
            score_threshold=dense_threshold,
            query_filter=query_filter,
        )
        sparse_results = self.search(
            collection_name=collection_name,
            query_vector=sparse_vector,
            vector_name="sparse",
            top_k=sparse_top_k,
            score_threshold=sparse_threshold,
            query_filter=query_filter,
        )

        # Reciprocal Rank Fusion
        scores: dict[str, float] = {}
        id_to_result: dict[str, dict[str, Any]] = {}

        for result_list in (dense_results, sparse_results):
            for rank, result in enumerate(result_list):
                rid = str(result["id"])
                scores[rid] = scores.get(rid, 0.0) + 1.0 / (rrf_k + rank + 1)
                id_to_result[rid] = result

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        return [
            {**id_to_result[rid], "score": scores[rid]}
            for rid in sorted_ids
        ]
