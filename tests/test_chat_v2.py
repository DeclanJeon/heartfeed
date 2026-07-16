"""Tests for the /v2/chat endpoint and ChatService pipeline."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from dating_rag.domain.models import (
    AnsweredContent,
    ChatV2Answered,
    ChatV2Request,
    ChatV2Response,
    EvidenceDecision,
    ProblemEnvelope,
    RetrievalResult,
)
from dating_rag.intake.planner import IntakeAction, IntakeDecision, plan_minimal_intake
from dating_rag.orchestration.chat_service import ChatService
from dating_rag.privacy.redaction import RedactedConcern
from dating_rag.safety.router import SafetyAssessment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def request_data() -> dict[str, Any]:
    return {
        "schema_version": "2",
        "request_id": "r1",
        "conversation_id": "c1",
        "question": "남자친구와 자주 싸워요",
        "consent": {
            "personalize_with_mbti": False,
            "cultural_saju_reflection": False,
        },
    }


@pytest.fixture()
def redacted() -> RedactedConcern:
    return RedactedConcern(
        redacted_text="[NAME]와 자주 싸워요",
        original_emotional_content="싸워요",
    )


@pytest.fixture()
def results() -> list[RetrievalResult]:
    return [RetrievalResult(chunk_id="c1", text="evidence", score=0.8)]


@pytest.fixture()
def accepted_decision(results: list[RetrievalResult]) -> EvidenceDecision:
    return EvidenceDecision(accepted=results, reason_code="accepted")


@pytest.fixture()
def rejected_decision() -> EvidenceDecision:
    return EvidenceDecision(
        accepted=[],
        reason_code="below_threshold",
        metrics={"count": 0},
    )


def _make_answered(request_id: str = "r1") -> ChatV2Answered:
    return ChatV2Answered(
        request_id=request_id,
        status="answered",
        answer=AnsweredContent(
            empathy="공감합니다",
            situation_framing="상황 정리",
            actions=[],
            boundaries="경계",
            summary="요약",
        ),
        evidence_claims=[],
        citations=[],
    )


@pytest.fixture()
def mock_state():
    """Minimal AppState-like object with mock pipeline components."""

    @dataclass
    class FakeState:
        retriever: Any = field(default_factory=MagicMock)
        context_builder: Any = field(default_factory=MagicMock)
        generator: Any = field(default_factory=MagicMock)
        _reranker: Any = None
        _reranker_attempted: bool = True
        settings: Any = field(default_factory=MagicMock)

        def get_reranker(self):
            return self._reranker

    state = FakeState()
    state.retriever.search = MagicMock(return_value=[])
    state.context_builder.build_context_with_registry = MagicMock(
        return_value=("context", MagicMock(citation_ids=MagicMock(return_value=["S1"]))),
    )
    state.generator.build_v2_response = AsyncMock(return_value=_make_answered())
    return state


def _run(coro):
    """Run an async coroutine to completion."""
    return asyncio.run(coro)


def _service(state: Any) -> ChatService:
    return ChatService(state)


# ---------------------------------------------------------------------------
# Safety escalation bypasses retrieval
# ---------------------------------------------------------------------------


@patch("dating_rag.orchestration.chat_service.plan_minimal_intake")
@patch("dating_rag.orchestration.chat_service.build_retrieval_query")
@patch("dating_rag.orchestration.chat_service.route_safety")
@patch("dating_rag.orchestration.chat_service.redact_concern")
def test_safety_escalation_bypasses_retrieval(
    mock_redact: MagicMock,
    mock_safety: MagicMock,
    mock_build_query: MagicMock,
    mock_intake: MagicMock,
    mock_state: Any,
    request_data: dict[str, Any],
    redacted: RedactedConcern,
) -> None:
    mock_redact.return_value = redacted
    mock_safety.return_value = SafetyAssessment(
        risk_kind="self_harm",
        urgency="precautionary",
        confidence=0.6,
        matched_keywords=["자살"],
    )

    service = _service(mock_state)
    result = _run(service.answer(ChatV2Request(**request_data)))

    assert result.status == "safety_escalation"
    mock_state.retriever.search.assert_not_called()
    mock_intake.assert_not_called()
    mock_build_query.assert_not_called()


# ---------------------------------------------------------------------------
# Insufficient evidence when gate rejects
# ---------------------------------------------------------------------------


@patch("dating_rag.orchestration.chat_service.EvidenceGate")
@patch("dating_rag.orchestration.chat_service.route_safety")
@patch("dating_rag.orchestration.chat_service.redact_concern")
def test_insufficient_evidence(
    mock_redact: MagicMock,
    mock_safety: MagicMock,
    mock_gate_cls: MagicMock,
    mock_state: Any,
    request_data: dict[str, Any],
    redacted: RedactedConcern,
    results: list[RetrievalResult],
    rejected_decision: EvidenceDecision,
) -> None:
    mock_redact.return_value = redacted
    mock_safety.return_value = None
    mock_state.retriever.search.return_value = results

    gate_instance = MagicMock()
    gate_instance.accept.return_value = rejected_decision
    mock_gate_cls.return_value = gate_instance

    service = _service(mock_state)
    result = _run(service.answer(ChatV2Request(**request_data)))

    assert result.status == "insufficient_evidence"
    assert result.reason_code == "below_threshold"


# ---------------------------------------------------------------------------
# Answered when pipeline succeeds
# ---------------------------------------------------------------------------


@patch("dating_rag.orchestration.chat_service.EvidenceGate")
@patch("dating_rag.orchestration.chat_service.route_safety")
@patch("dating_rag.orchestration.chat_service.redact_concern")
def test_answered_success(
    mock_redact: MagicMock,
    mock_safety: MagicMock,
    mock_gate_cls: MagicMock,
    mock_state: Any,
    request_data: dict[str, Any],
    redacted: RedactedConcern,
    results: list[RetrievalResult],
    accepted_decision: EvidenceDecision,
) -> None:
    mock_redact.return_value = redacted
    mock_safety.return_value = None
    mock_state.retriever.search.return_value = results

    gate_instance = MagicMock()
    gate_instance.accept.return_value = accepted_decision
    mock_gate_cls.return_value = gate_instance

    service = _service(mock_state)
    result = _run(service.answer(ChatV2Request(**request_data)))

    assert result.status == "answered"
    assert result.request_id == "r1"
    mock_safety.assert_called_once()
    mock_state.retriever.search.assert_called_once()
    gate_instance.accept.assert_called_once()


# ---------------------------------------------------------------------------
# Needs clarification when intake says ASK
# ---------------------------------------------------------------------------


@patch("dating_rag.orchestration.chat_service.plan_minimal_intake")
@patch("dating_rag.orchestration.chat_service.route_safety")
@patch("dating_rag.orchestration.chat_service.redact_concern")
def test_needs_clarification(
    mock_redact: MagicMock,
    mock_safety: MagicMock,
    mock_intake: MagicMock,
    mock_state: Any,
    request_data: dict[str, Any],
    redacted: RedactedConcern,
) -> None:
    mock_redact.return_value = redacted
    mock_safety.return_value = None
    mock_intake.return_value = IntakeDecision(
        action=IntakeAction.ASK_QUESTION,
        reason="vague",
        round_number=0,
    )

    service = _service(mock_state)
    result = _run(service.answer(ChatV2Request(**request_data)))

    assert result.status == "needs_clarification"
    mock_state.retriever.search.assert_not_called()


# ---------------------------------------------------------------------------
# Timeout returns 503 ProblemEnvelope
# ---------------------------------------------------------------------------


@patch("dating_rag.orchestration.chat_service.asyncio.wait_for")
def test_timeout_returns_503(
    mock_wait_for: MagicMock,
    mock_state: Any,
    request_data: dict[str, Any],
) -> None:
    mock_wait_for.side_effect = TimeoutError

    service = _service(mock_state)
    result = _run(service.answer(ChatV2Request(**request_data)))

    assert isinstance(result, ProblemEnvelope)
    assert result.status == 503


# ---------------------------------------------------------------------------
# Legacy /chat still works
# ---------------------------------------------------------------------------


def test_legacy_chat_registered() -> None:
    from dating_rag.api.app import app

    client = TestClient(app)
    routes = {getattr(r, "path", getattr(r, "url", None)) for r in app.routes}
    assert "/v2/chat" in routes

    resp = client.post("/chat", json={"question": "test"})
    # Endpoint exists — may return 503 if state not initialized, but not 404.
    assert resp.status_code != 404


def test_followup_without_relationship_keyword_inherits_scope_from_history() -> None:
    """A short follow-up lacking relationship keywords must NOT be out_of_scope
    when the prior turns clearly discuss a relationship topic."""
    followup = "그래서 저렇게 한다음에 어떤 주제로 이야기를 이어가면 좋을까?"
    history = [
        {"role": "user", "content": "한 공동체에서 한달정도 봤는데 이 친구가 너무 밝고 명량해"},
        {"role": "assistant", "content": "연애/관계 상황에서 다가가는 법을 정리해보면"},
    ]
    decision = plan_minimal_intake(followup, conversation_history=history)
    assert decision.reason != "out_of_scope"
    assert decision.action != IntakeAction.SKIP_OPTIONAL


def test_followup_truly_off_topic_still_out_of_scope() -> None:
    """Without any relationship context in history, an off-topic follow-up
    should still be flagged out_of_scope."""
    decision = plan_minimal_intake(
        "오늘 점심 뭐 먹을까?",
        conversation_history=[{"role": "user", "content": "날씨가 좋네요"}],
    )
    assert decision.reason == "out_of_scope"
