"""Qdrant filter construction from QueryPlan."""

from __future__ import annotations


from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

from dating_rag.domain.models import QueryPlan


def build_qdrant_filter(
    plan: QueryPlan,
    *,
    include_category: bool = True,
    language: str | None = None,
    min_views: int | None = None,
    require_transcript_evidence: bool = False,
) -> Filter | None:
    """Build a Qdrant filter from a QueryPlan.

    Args:
        include_category: Whether to apply the analyzer's inferred category.
        plan: The analyzed query plan.
        language: Optional language code filter (e.g. "ko", "en").
        min_views: Optional minimum view count filter.

    Returns:
        A Qdrant Filter object, or None if no filters apply.
    """

    conditions: list[FieldCondition] = []

    if require_transcript_evidence:
        conditions.extend(
            [
                FieldCondition(
                    key="corpus_type",
                    match=MatchValue(value="transcript"),
                ),
                FieldCondition(
                    key="evidence_role",
                    match=MatchValue(value="source_evidence"),
                ),
                FieldCondition(
                    key="transcript_status",
                    match=MatchValue(value="available"),
                ),
            ]
        )

    if include_category and plan.category_filter:
        conditions.append(
            FieldCondition(
                key="category",
                match=MatchValue(value=plan.category_filter),
            )
        )

    # Channel filters — match any of the listed channels
    for channel in plan.channel_filters:
        conditions.append(
            FieldCondition(
                key="channel_name",
                match=MatchValue(value=channel),
            )
        )

    # Language filter
    if language:
        conditions.append(
            FieldCondition(
                key="language",
                match=MatchValue(value=language),
            )
        )

    # Minimum views filter
    if min_views is not None and min_views > 0:
        conditions.append(
            FieldCondition(
                key="views",
                range=Range(gte=min_views),
            )
        )

    if not conditions:
        return None

    return Filter(must=conditions)  # type: ignore[arg-type]
