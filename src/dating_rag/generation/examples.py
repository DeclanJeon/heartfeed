"""Illustrative story examples kept separate from retrieved evidence."""

from __future__ import annotations

from typing import Any

from dating_rag.domain.models import QueryPlan


def select_story_examples(
    examples: list[dict[str, Any]],
    plan: QueryPlan,
    *,
    limit: int = 1,
) -> list[dict[str, Any]]:
    """Select at most *limit* explicitly requested illustrative examples."""
    if limit <= 0 or not examples:
        return []
    if plan.intent == "high_risk" or not plan.allow_illustrative_examples:
        return []

    query_topics = set(plan.topics)
    if plan.category_filter:
        query_topics.add(plan.category_filter)

    selected: list[dict[str, Any]] = []
    def eligible(example: dict[str, Any]) -> bool:
        if not example.get("production_eligible", False):
            return False
        if example.get("evidence_role") != "non_evidence":
            return False
        if example.get("editorial_status") != "human_reviewed":
            return False
        rights = example.get("rights", {})
        if isinstance(rights, dict) and rights.get("status") in {"rejected", "takedown_pending"}:
            return False
        provenance = example.get("provenance", {})
        if isinstance(rights, dict) and rights.get("status") != "approved":
            return False
        return isinstance(provenance, dict) and provenance.get("derived_from_original_text") is False

    for example in examples:
        if not eligible(example):
            continue
        example_topics = {str(topic) for topic in example.get("topics", [])}
        if query_topics.intersection(example_topics):
            selected.append(example)
        if len(selected) >= limit:
            break
    return selected
