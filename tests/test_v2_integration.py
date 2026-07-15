"""Integration test for v2 pipeline with real LLM generation.

Tests the generation path with mocked retrieval but real LLM call.
Requires LLM_API_KEY in environment.
"""
from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import MagicMock

import pytest

from dating_rag.domain.models import (
    ChatV2Answered,
    ChatV2SafetyEscalation,
    ChatV2InsufficientEvidence,
    QueryPlan,
    RetrievalResult,
)
from dating_rag.generation.generator import AnswerGenerator
from dating_rag.generation.prompts import build_v2_prompt
from dating_rag.retrieval.context_builder import CitationRegistry, ContextBuilder
from dating_rag.privacy.redaction import redact_concern
from dating_rag.safety.router import route_safety


def _make_result(chunk_id: str, text: str, channel: str, title: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        score=score,
        source_type="transcript",
        metadata={
            "channel_name": channel,
            "title": title,
            "timestamp_url": f"https://youtube.com/watch?v=test&t=60",
        },
    )


@pytest.mark.skipif(
    not os.environ.get("LLM_API_KEY"),
    reason="LLM_API_KEY not set",
)
class TestV2GenerationIntegration:
    """Test v2 generation with real LLM."""

    def test_basic_answer_generation(self) -> None:
        """Test that v2 generation produces structured answer."""
        api_url = os.environ.get("LLM_API_URL", "https://openrouter.ai/api/v1")
        api_key = os.environ["LLM_API_KEY"]
        model = os.environ.get("LLM_MODEL", "xiaomi/mimo-v2.5")

        generator = AnswerGenerator(api_url=api_url, api_key=api_key, model=model)

        sources = [
            _make_result("c1", "첫 데이트에서는 가벼운 질문으로 시작하는 것이 좋습니다. 상대방의 관심사나 취미를 물어보면 자연스럽게 대화가 이어져요.", "코치김", "첫 데이트 팁", 0.8),
            _make_result("c2", "대화가 막힐 때는 주변 환경을 활용하세요. 메뉴나 분위기에 대한 이야기는 부담 없는 주제입니다.", "데이트가이드", "대화 기술", 0.7),
        ]

        builder = ContextBuilder()
        plan = QueryPlan(intent="specific_example", topics=["first_dates"], category_filter="first_dates")
        context_text, registry = builder.build_context_with_registry(sources, [], plan)

        prompt = build_v2_prompt(
            question="첫 데이트에서 무슨 이야기를 해야 할까요?",
            context_text=context_text,
            plan=plan,
            registry_citation_ids=[c.citation_id for c in registry.get_all_citations()],
        )

        async def _run():
            result = await generator.generate(
                question="첫 데이트에서 무슨 이야기를 해야 할까요?",
                context=sources,
                plan=plan,
                context_text=context_text,
                max_tokens=1500,
            )
            return result

        answer = asyncio.run(_run())
        assert len(answer) > 50, f"Answer too short: {answer!r}"
        print(f"\n=== LLM Response ===\n{answer[:500]}\n====================")

    def test_safety_routing_no_llm(self) -> None:
        """Test safety routing works without LLM."""
        assessment = route_safety("자살하고 싶어요", redact_concern("자살하고 싶어요"))
        assert assessment is not None
        assert assessment.risk_kind == "self_harm"

        assessment2 = route_safety("첫 데이트 팁 알려주세요", redact_concern("첫 데이트 팁 알려주세요"))
        assert assessment2 is None

    def test_redaction_preserves_emotion(self) -> None:
        """Test that redaction preserves emotional content."""
        result = redact_concern("김민수와 헤어졌는데 너무 힘들어요")
        assert "김민수" not in result.redacted_text
        assert "이름" in result.redacted_text
        assert "힘들" in result.redacted_text or result.original_emotional_content
