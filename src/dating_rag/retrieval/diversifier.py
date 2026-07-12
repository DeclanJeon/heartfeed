"""Result diversification to ensure source variety."""

from __future__ import annotations

from collections import Counter

from dating_rag.domain.models import RetrievalResult


def _text_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity between two texts at word level.

    Args:
        a: First text.
        b: Second text.

    Returns:
        Jaccard similarity coefficient in [0, 1].
    """
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _is_duplicate(result: RetrievalResult, seen: list[RetrievalResult], threshold: float = 0.85) -> bool:
    """Check if a result is near-identical to any already-selected result.

    Args:
        result: Candidate result to check.
        seen: Already-selected results.
        threshold: Jaccard similarity threshold for deduplication.

    Returns:
        True if the result is a near-duplicate.
    """
    for existing in seen:
        if _text_similarity(result.text, existing.text) >= threshold:
            return True
    return False


def diversify_results(
    results: list[RetrievalResult],
    *,
    limit: int | None = None,
    max_per_channel: int = 3,
    max_per_video: int = 2,
    penalty: float = 0.3,
    dedup_threshold: float = 0.85,
) -> list[RetrievalResult]:
    """Diversify results to avoid over-representing a single source.

    Applies per-channel and per-video caps, a diversity penalty, and
    deduplicates near-identical chunks.

    Args:
        results: Sorted retrieval results (highest score first).
        limit: Maximum total results to return.
        max_per_channel: Maximum results from any single channel.
        max_per_video: Maximum results from any single video.
        penalty: Score penalty applied per prior occurrence from same channel.
        dedup_threshold: Jaccard similarity above which chunks are considered duplicates.

    Returns:
        Diversified and deduplicated results.
    """
    channel_counts: Counter[str] = Counter()
    video_counts: Counter[str] = Counter()
    diversified: list[RetrievalResult] = []

    for result in results:
        channel_id = str(result.metadata.get("channel_id", ""))
        video_id = str(result.metadata.get("video_id", ""))

        # Cap per channel
        if channel_counts[channel_id] >= max_per_channel:
            continue

        # Cap per video
        if video_counts[video_id] >= max_per_video:
            continue

        # Deduplicate near-identical chunks
        if _is_duplicate(result, diversified, threshold=dedup_threshold):
            continue

        # Apply diversity penalty based on channel frequency
        adjusted_score = result.score - (penalty * channel_counts[channel_id])

        diversified.append(
            RetrievalResult(
                chunk_id=result.chunk_id,
                text=result.text,
                score=adjusted_score,
                source_type=result.source_type,
                metadata=result.metadata,
            )
        )

        channel_counts[channel_id] += 1
        video_counts[video_id] += 1

        # Stop early if limit reached
        if limit is not None and len(diversified) >= limit:
            break

    return diversified


def diversify(
    candidates: list[RetrievalResult],
    *,
    limit: int | None = None,
    max_per_channel: int = 3,
    max_per_video: int = 2,
) -> list[RetrievalResult]:
    """Convenience wrapper for diversify_results with sensible defaults.

    Args:
        candidates: Sorted retrieval results (highest score first).
        limit: Maximum total results to return.
        max_per_channel: Maximum results from any single channel.
        max_per_video: Maximum results from any single video.

    Returns:
        Diversified results.
    """
    return diversify_results(
        candidates,
        limit=limit,
        max_per_channel=max_per_channel,
        max_per_video=max_per_video,
    )
