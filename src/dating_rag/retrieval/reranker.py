"""Cross-encoder reranker for improved relevance."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dating_rag.domain.models import RetrievalResult

if TYPE_CHECKING:
    pass


class PassageReranker:
    """Cross-encoder reranker using sentence-transformers.

    Scores query-passage pairs for relevance, reordering retrieval
    results to surface the most relevant passages first.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu") -> None:
        """Initialize the reranker.

        Args:
            model_name: HuggingFace model identifier for the reranker.
            device: Device to run inference on.
        """
        from sentence_transformers import CrossEncoder  # noqa: PLC0415
        self.model = CrossEncoder(model_name, max_length=512, device=device)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Rerank retrieval results by relevance to the query.

        Args:
            query: The user's query.
            candidates: Candidate retrieval results.
            top_k: Number of top results to return.

        Returns:
            Reranked results limited to top_k.
        """
        if not candidates:
            return []

        pairs = [(query, r.text) for r in candidates]
        scores = self.model.predict(pairs)

        scored = list(zip(candidates, scores, strict=True))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(
                chunk_id=r.chunk_id,
                text=r.text,
                score=float(score),
                source_type=r.source_type,
                metadata=r.metadata,
            )
            for r, score in scored[:top_k]
        ]


# Backward-compatible alias
Reranker = PassageReranker
