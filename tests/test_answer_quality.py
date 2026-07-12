"""Regression tests for evidence-grounded answer assembly."""

import asyncio
import json
from pathlib import Path

import httpx

from dating_rag.api.app import CitationValidator
from dating_rag.domain.models import QueryPlan, RetrievalResult
from dating_rag.generation.generator import AnswerGenerator
from dating_rag.retrieval.context_builder import ContextBuilder
from scripts.index_chunks import _load_catalog, _timestamp_url


def _source() -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk-1",
        text="The source recommends asking an open-ended question.",
        score=0.8,
        metadata={
            "channel_name": "Coach Kim",
            "title": "Conversation Tips",
            "timestamp_url": "https://youtube.com/watch?v=abc&t=42",
        },
    )


def test_generator_uses_context_builder_labels() -> None:
    source = _source()
    context = ContextBuilder().build_context([source], [], QueryPlan())
    generator = AnswerGenerator(api_key="")

    messages = generator._build_messages(
        "How do I start a conversation?",
        [source],
        QueryPlan(),
        context_text=context,
    )

    user_message = messages[1]["content"]
    assert "[S1]" in user_message
    assert "Conversation Tips" in user_message
    assert "youtube.com/watch?v=abc&t=42" in user_message


def test_generation_request_contains_selected_context() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Use an open question [S1]."}}]},
        )

    async def run() -> str:
        generator = AnswerGenerator(api_url="https://llm.test/v1", api_key="test")
        await generator._client.aclose()
        generator._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            return await generator.generate(
                "How do I start a conversation?",
                [_source()],
                QueryPlan(),
                context_text="## Transcript Evidence\n[S1] Conversation Tips\nEvidence",
            )
        finally:
            await generator.close()

    assert asyncio.run(run()) == "Use an open question [S1]."
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert "## Transcript Evidence" in messages[1]["content"]


def test_citation_validator_accepts_labels_and_rejects_unknown_labels() -> None:
    validator = CitationValidator()
    source = _source()

    assert validator.validate("Try an open question [S1].", [source]) == []
    warnings = validator.validate("Try an open question [S9].", [source])
    assert any("S9" in warning for warning in warnings)


def test_citation_validator_flags_uncited_answer() -> None:
    warnings = CitationValidator().validate("Try an open question.", [_source()])
    assert "Answer contains no source citations" in warnings


def test_catalog_metadata_produces_category_and_timestamp(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        '{"videos": [{"id": "abc", "category": "conversation", '
        '"url": "https://youtube.com/watch?v=abc"}]}',
        encoding="utf-8",
    )

    loaded = _load_catalog(catalog)
    assert loaded["abc"]["category"] == "conversation"
    assert _timestamp_url(loaded["abc"]["url"], 42) == "https://youtube.com/watch?v=abc&t=42"

def test_inferred_category_can_be_disabled_without_losing_topics() -> None:
    from dating_rag.retrieval.filters import build_qdrant_filter
    from dating_rag.retrieval.query_analyzer import QueryAnalyzer

    plan = QueryAnalyzer().analyze("카톡으로 대화를 이어가는 방법")
    assert plan.category_filter == "texting"
    assert build_qdrant_filter(plan, include_category=False) is None
