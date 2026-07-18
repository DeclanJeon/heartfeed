"""Rescue BRT-14 core unit tests (no live LLM)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from dating_rag.domain.models import ChatV2Request, Consent, TrackContext
from dating_rag.orchestration.chat_service import ChatService
from dating_rag.privacy.redaction import RedactedConcern
from dating_rag.rescue.guards import (
    assert_production_llm_safe,
    is_breakup_related,
    is_free_llm_model,
    looks_general_dating_not_breakup,
)
from dating_rag.rescue.limits import GenerationAdmission
from dating_rag.rescue.track import load_brt14_track, track_day, track_hints_for_day
from dating_rag.safety.router import route_safety


def test_free_llm_detection():
    assert is_free_llm_model("google/gemma-4-31b-it:free")
    assert not is_free_llm_model("mimo-v2.5")


def test_prod_guard_blocks_free_model():
    with pytest.raises(RuntimeError):
        assert_production_llm_safe(
            product_mode="rescue_brt14",
            env="production",
            provider="openai",
            model="foo:free",
            fallback_model="bar",
            allow_free_llm=False,
        )


def test_prod_guard_allows_paid():
    assert_production_llm_safe(
        product_mode="rescue_brt14",
        env="production",
        provider="openai",
        model="mimo-v2.5",
        fallback_model="google/gemma-3-27b-it",
        allow_free_llm=False,
    )


def test_breakup_detection():
    assert is_breakup_related("이별 3일차인데 연락하고 싶어요")
    assert not looks_general_dating_not_breakup("이별 후 자책")
    assert looks_general_dating_not_breakup("첫 데이트 대화 주제 추천")


def test_brt14_yaml_days():
    load_brt14_track.cache_clear()
    data = load_brt14_track()
    assert data["track_id"] == "brt14"
    assert len(data["days"]) == 14
    d0 = track_day(0)
    assert d0 is not None
    assert "actions" in d0
    hints = track_hints_for_day(3)
    assert hints["day_index"] == 3
    assert hints["suggested_day_actions"]


def test_safety_timeout_failsafe_path_uses_other():
    red = RedactedConcern(redacted_text="평범한 질문", original_emotional_content="")
    assert route_safety("오늘 날씨 어때", red) is None


def test_stalking_ex_patterns():
    red = RedactedConcern(redacted_text="x", original_emotional_content="")
    a = route_safety("전 여친 위치 추적하는 방법 알려줘", red)
    assert a is not None
    assert a.risk_kind == "stalking"


def test_track_context_on_request():
    req = ChatV2Request(
        schema_version="2",
        request_id="r1",
        conversation_id="c1",
        question="이별 후 연락 참는 법",
        consent=Consent(),
        track=TrackContext(day_index=2, contact_status="no_contact", primary_goal="stabilize"),
    )
    assert req.track is not None
    assert req.track.id == "brt14"


def test_generation_admission_saturates():
    async def _run():
        adm = GenerationAdmission(limit=1)
        assert await adm.try_acquire() is True
        assert await adm.try_acquire() is False
        await adm.release()
        assert await adm.try_acquire() is True
        await adm.release()

    asyncio.run(_run())


def test_rescue_ood_without_breakup():
    async def _run():
        state = MagicMock()
        state.settings = MagicMock()
        state.settings.product_mode = "rescue_brt14"
        state.settings.allow_general = False
        state.settings.safety_timeout = 2.0
        state.settings.retrieval_timeout = 2.0
        state.settings.generation_timeout = 2.0
        state.settings.total_timeout = 10.0
        state.settings.max_output_tokens = 700
        state.settings.rescue_retrieval_top_k = 4
        svc = ChatService(state)
        req = ChatV2Request(
            schema_version="2",
            request_id="r-ood",
            conversation_id="c1",
            question="첫 데이트에서 무슨 대화 하지?",
            consent=Consent(),
        )
        out = await svc.answer(req)
        assert getattr(out, "status", None) == "insufficient_evidence"
        assert getattr(out, "reason_code", None) == "out_of_domain"

    asyncio.run(_run())


def test_safety_timeout_failsafe_escalates(monkeypatch):
    async def _run():
        state = MagicMock()
        state.settings = MagicMock()
        state.settings.product_mode = ""
        state.settings.allow_general = True
        state.settings.safety_timeout = 0.01
        state.settings.retrieval_timeout = 2.0
        state.settings.generation_timeout = 2.0
        state.settings.total_timeout = 10.0
        state.settings.max_output_tokens = 700
        state.settings.rescue_retrieval_top_k = 4
        svc = ChatService(state)

        def slow_route(*_a, **_k):
            import time

            time.sleep(0.05)
            return None

        monkeypatch.setattr(
            "dating_rag.orchestration.chat_service.route_safety",
            slow_route,
        )
        req = ChatV2Request(
            schema_version="2",
            request_id="r-safe",
            conversation_id="c1",
            question="그냥 질문",
            consent=Consent(),
        )
        out = await svc.answer(req)
        assert getattr(out, "status", None) == "safety_escalation"

    asyncio.run(_run())
