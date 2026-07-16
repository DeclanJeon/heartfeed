"""Orchestrates the v2 chat pipeline."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from dating_rag.domain.models import (
    ChatV2Answered,
    ChatV2InsufficientEvidence,
    ChatV2NeedsClarification,
    ChatV2Request,
    ChatV2Response,
    PersonalizationBlock,
    ProblemEnvelope,
)
from dating_rag.intake.planner import IntakeAction, plan_minimal_intake
from dating_rag.personalization.saju_adapter import LocalSajuAdapter
from dating_rag.privacy.redaction import build_retrieval_query, redact_concern
from dating_rag.retrieval.evidence_gate import EvidenceGate
from dating_rag.safety.router import generate_safety_response, route_safety

if TYPE_CHECKING:
    from dating_rag.api.app import AppState

logger = logging.getLogger(__name__)

# Timeout budgets (seconds)
_SAFETY_TIMEOUT = 5
_RETRIEVAL_TIMEOUT = 30
_GENERATION_TIMEOUT = 110
_SAJU_TIMEOUT = 5
_TOTAL_TIMEOUT = 150


class ChatService:
    """Orchestrates the v2 chat pipeline."""

    def __init__(self, state: AppState) -> None:
        self._state = state
        self._saju_adapter = LocalSajuAdapter()
        self._evidence_gate = EvidenceGate()

    async def answer(
        self, request: ChatV2Request,
    ) -> ChatV2Response | ProblemEnvelope:
        """Run the full v2 pipeline and return a typed response."""
        try:
            return await asyncio.wait_for(
                self._answer_inner(request), timeout=_TOTAL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            return ProblemEnvelope(
                type="about:blank",
                title="Service Unavailable",
                status=503,
                detail="Request timed out",
                instance="/v2/chat",
            )

    async def _answer_inner(
        self, request: ChatV2Request,
    ) -> ChatV2Response | ProblemEnvelope:
        state = self._state

        # ── 1. Redaction (needed by safety) ──────────────────────────────
        redacted = await asyncio.to_thread(redact_concern, request.question)

        # ── 2. Safety check ──────────────────────────────────────────────
        try:
            assessment = await asyncio.wait_for(
                asyncio.to_thread(
                    route_safety, request.question, redacted,
                ),
                timeout=_SAFETY_TIMEOUT,
            )
        except asyncio.TimeoutError:
            return ProblemEnvelope(
                type="about:blank",
                title="Service Unavailable",
                status=503,
                detail="Safety check timed out",
                instance="/v2/chat",
            )
        if assessment is not None:
            return generate_safety_response(
                assessment, request_id=request.request_id,
            )

        # ── 3. Intake check ──────────────────────────────────────────────
        decision = await asyncio.to_thread(
            plan_minimal_intake,
            request.question,
            request.clarification_answers,
            request.profile.model_dump() if request.profile else None,
            request.consent.model_dump(),
        )
        if decision.action == IntakeAction.ASK_QUESTION:
            return ChatV2NeedsClarification(
                request_id=request.request_id,
                status="needs_clarification",
                questions=[],
            )

        # Out-of-scope questions (not about relationships) → insufficient evidence
        if decision.reason == "out_of_scope":
            return ChatV2InsufficientEvidence(
                request_id=request.request_id,
                status="insufficient_evidence",
                reason_code="out_of_domain",
                message="질문이 연애/관계 주제와 관련이 없어 답변을 제공할 수 없습니다.",
                retrieval_summary={},
            )

        # ── 4. Evidence retrieval ────────────────────────────────────────
        query = build_retrieval_query(redacted, decision.reason)

        async def _retrieve() -> list:
            return await asyncio.to_thread(
                state.retriever.search, redacted.redacted_text,
            )

        try:
            results = await asyncio.wait_for(
                _retrieve(), timeout=_RETRIEVAL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            return ProblemEnvelope(
                type="about:blank",
                title="Service Unavailable",
                status=503,
                detail="Retrieval timed out",
                instance="/v2/chat",
            )

        # ── 5. Evidence gate ─────────────────────────────────────────────
        from dating_rag.domain.models import QueryPlan
        plan = QueryPlan(intent=decision.reason, topics=[])
        gate_decision = self._evidence_gate.accept(results, plan)

        if gate_decision.reason_code != "accepted":
            return ChatV2InsufficientEvidence(
                request_id=request.request_id,
                status="insufficient_evidence",
                reason_code=gate_decision.reason_code,  # type: ignore[arg-type]
                message=f"Insufficient evidence: {gate_decision.reason_code}",
                retrieval_summary=gate_decision.metrics,
            )

        accepted = gate_decision.accepted

        # ── 6. Rerank ────────────────────────────────────────────────────
        reranker = state.get_reranker()
        if reranker is not None:
            accepted = await asyncio.to_thread(
                reranker.rerank, redacted.redacted_text, accepted,
            )

        # ── 7. Context + registry ────────────────────────────────────────
        context_text, registry = state.context_builder.build_context_with_registry(
            accepted, [], plan,
        )

        # ── 8. Saju (if consented) ──────────────────────────────────────
        cultural_reflection = None
        if request.consent.cultural_saju_reflection and request.saju_input:
            saju = self._saju_adapter
            if saju.is_available():
                try:
                    chart = await asyncio.wait_for(
                        asyncio.to_thread(
                            saju.calculate_chart,
                            request.saju_input.birth_date,
                            request.saju_input.birth_time,
                            request.saju_input.birthplace or "",
                            request.saju_input.gender,
                            request.saju_input.calendar_type,
                            request.saju_input.timezone,
                        ),
                        timeout=_SAJU_TIMEOUT,
                    )
                    cultural_reflection = saju.generate_cultural_reflection(chart)
                except asyncio.TimeoutError:
                    logger.warning("Saju calculation timed out; omitting cultural block")
                except Exception:
                    logger.warning("Saju calculation failed; omitting cultural block")

        # ── 9. Generation ────────────────────────────────────────────────
        # Build conversation context from history
        conversation_context = ""
        if request.conversation_history:
            history_lines = []
            for turn in request.conversation_history[-6:]:  # Last 3 exchanges
                role_label = "사용자" if turn.role == "user" else "상담가"
                history_lines.append(f"{role_label}: {turn.content}")
            if history_lines:
                conversation_context = "\n\n## 이전 대화\n" + "\n".join(history_lines)

        lenses: list[str] = []
        if request.consent.personalize_with_mbti and request.profile and request.profile.mbti:
            lenses.append("mbti")
        if request.consent.personalize_with_observations and request.profile and request.profile.observed_tendencies:
            lenses.append("observations")
        personalization = PersonalizationBlock(
            lenses_used=lenses,
            disclaimer="개인화 정보는 조언 보조용이며 결정을 대체하지 않습니다.",
        )

        try:
            result = await asyncio.wait_for(
                state.generator.build_v2_response(
                    request.question,
                    accepted,
                    context_text,
                    plan,
                    registry,
                    personalization=personalization,
                    cultural_reflection=cultural_reflection,
                    conversation_context=conversation_context,
                ),
                timeout=_GENERATION_TIMEOUT,
            )
        except asyncio.TimeoutError:
            return ProblemEnvelope(
                type="about:blank",
                title="Service Unavailable",
                status=503,
                detail="Generation timed out",
                instance="/v2/chat",
            )
        except Exception as exc:
            logger.warning("Generation failed: %s", exc, exc_info=True)
            return ChatV2InsufficientEvidence(
                request_id=request.request_id,
                status="insufficient_evidence",
                reason_code="provider_schema_failure",
                message=f"Provider response could not be parsed: {type(exc).__name__}: {exc}",
                retrieval_summary={},
            )

        # ── 10. Return ───────────────────────────────────────────────────
        return ChatV2Answered(
            request_id=request.request_id,
            status="answered",
            answer=result.answer,
            evidence_claims=result.evidence_claims,
            citations=result.citations,
            personalization=personalization,
            cultural_reflection=cultural_reflection,
        )
