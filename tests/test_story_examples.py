"""Tests for illustrative story examples kept separate from evidence."""

from pathlib import Path

import yaml

from dating_rag.domain.models import QueryPlan
from dating_rag.generation.examples import select_story_examples
from dating_rag.retrieval.context_builder import ContextBuilder


def _examples() -> list[dict[str, object]]:
    config = yaml.safe_load(Path("config/story_examples.yaml").read_text(encoding="utf-8"))
    return config["examples"]


def test_story_registry_requires_non_evidence_provenance() -> None:
    for example in _examples():
        assert example["evidence_role"] == "non_evidence"
        assert example["production_eligible"] is True
        assert example["editorial_status"] == "human_reviewed"
        assert example["rights"]["status"] == "approved"
        assert example["content_hash"].startswith("sha256:")
        assert "source_urls" in example["rights"]


def test_story_examples_require_explicit_illustrative_intent() -> None:
    selected = select_story_examples(
        _examples(),
        QueryPlan(
            intent="specific_example",
            allow_illustrative_examples=True,
            category_filter="long-distance",
            topics=["long-distance"],
        ),
    )

    assert len(selected) == 1
    assert selected[0]["title"] == "토끼와 거북이"


def test_topic_match_alone_does_not_inject_story() -> None:
    selected = select_story_examples(
        _examples(),
        QueryPlan(category_filter="long-distance", topics=["long-distance"]),
    )
    assert selected == []


def test_selected_story_is_labeled_non_evidence() -> None:
    selected = select_story_examples(
        _examples(),
        QueryPlan(
            intent="specific_example",
            allow_illustrative_examples=True,
            category_filter="long-distance",
            topics=["long-distance"],
        ),
    )
    context = ContextBuilder().build_context(
        [],
        [],
        QueryPlan(use_transcripts=False, use_okf=False),
        illustrative_examples=selected,
    )
    assert "Illustrative Example (not source evidence)" in context
    assert "[E1] [예시]" in context
    assert "[S1]" not in context


def test_unapproved_story_cannot_be_selected() -> None:
    example = dict(_examples()[0])
    example["rights"] = {"status": "review_required"}
    assert (
        select_story_examples(
            [example],
            QueryPlan(
                intent="specific_example",
                allow_illustrative_examples=True,
                topics=["long-distance"],
            ),
        )
        == []
    )


def test_high_risk_queries_do_not_receive_story_analogies() -> None:
    selected = select_story_examples(
        _examples(),
        QueryPlan(intent="high_risk", category_filter="breakup", topics=["breakup"]),
    )
    assert selected == []
