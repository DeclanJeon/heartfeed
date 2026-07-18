"""Orchestrates the v2 chat pipeline (Rescue-aware)."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from dating_rag.domain.models import (
    ChatV2Answered,
    ChatV2InsufficientEvidence,
    ChatV2NeedsClarification,
    ChatV2Request,
    ChatV2Response,
    PersonalizationBlock,
    ProblemEnvelope,
    TrackHints,
)
from dating_rag.intake.planner import IntakeAction, plan_minimal_intake
from dating_rag.personalization.saju_adapter import LocalSajuAdapter
from dating_rag.privacy.redaction import build_retrieval_query, redact_concern
from dating_rag.rescue.guards import looks_general_dating_not_breakup
from dating_rag.rescue.policy import should_refuse_ood, normalize_track
from dating_rag.rescue.limits import generation_admission
from dating_rag.rescue.track import track_hints_for_day
from dating_rag.retrieval.evidence_gate import EvidenceGate
from dating_rag.safety.router import generate_safety_response, route_safety

if TYPE_CHECKING:
    from dating_rag.api.app import AppState

logger = logging.getLogger(__name__)

# Legacy ceilings (overridden by settings when available)
_SAFETY_TIMEOUT = 5.0
_RETRIEVAL_TIMEOUT = 30.0
_GENERATION_TIMEOUT = 110.0
_SAJU_TIMEOUT = 5.0
_TOTAL_TIMEOUT = 150.0

_RESCUE_OOD_MESSAGE = (
    "현재 HeartFeed Rescue는 **이별 회복 14일 트랙** 중심입니다. "
    "이별·회복·경계·연락 충동과 관련된 질문으로 다시 물어보시면 더 잘 도와드릴 수 있어요."
)


class ChatService:
    """Orchestrates the v2 chat pipeline."""

    def __init__(self, state: AppState) -> None:
        self._state = state
        self._saju_adapter = LocalSajuAdapter()
        self._evidence_gate = EvidenceGate()

    def _budgets(self) -> tuple[float, float, float, float]:
        settings = getattr(self._state, "settings", None)
        if settings is None:
            return _SAFETY_TIMEOUT, _RETRIEVAL_TIMEOUT, _GENERATION_TIMEOUT, _TOTAL_TIMEOUT
        return (
            float(getattr(settings, "safety_timeout", _SAFETY_TIMEOUT)),
            float(getattr(settings, "retrieval_timeout", _RETRIEVAL_TIMEOUT)),
            float(getattr(settings, "generation_timeout", _GENERATION_TIMEOUT)),
            float(getattr(settings, "total_timeout", _TOTAL_TIMEOUT)),
        )

    def _product_mode(self) -> str:
        settings = getattr(self._state, "settings", None)
        return (getattr(settings, "product_mode", "") or "").strip()

    def _allow_general(self) -> bool:
        settings = getattr(self._state, "settings", None)
        return bool(getattr(settings, "allow_general", False))

    def _max_tokens(self) -> int:
        settings = getattr(self._state, "settings", None)
        return int(getattr(settings, "max_output_tokens", 700) or 700)

    async def answer(
        self, request: ChatV2Request,
    ) -> ChatV2Response | ProblemEnvelope:
        """Run the full v2 pipeline and return a typed response."""
        _, _, _, total_timeout = self._budgets()
        t0 = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self._answer_inner(request), timeout=total_timeout,
            )
            logger.info(
                "chat_done request_id=%s status=%s elapsed_ms=%.0f",
                request.request_id,
                getattr(result, "status", type(result).__name__),
                (time.perf_counter() - t0) * 1000,
            )
            return result
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
        safety_timeout, retrieval_timeout, generation_timeout, _ = self._budgets()
        stage_t0 = time.perf_counter()

        # ── 1. Redaction (needed by safety) ──────────────────────────────
        redacted = await asyncio.to_thread(redact_concern, request.question)
        logger.debug("stage=redact ms=%.0f", (time.perf_counter() - stage_t0) * 1000)

        # ── 2. Safety check (timeout → fail-safe escalate) ───────────────
        stage_t0 = time.perf_counter()
        try:
            assessment = await asyncio.wait_for(
                asyncio.to_thread(
                    route_safety, request.question, redacted,
                ),
                timeout=safety_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Safety check timed out; fail-safe escalation")
            from dating_rag.safety.router import SafetyAssessment

            assessment = SafetyAssessment(
                risk_kind="other",
                urgency="elevated",
                confidence=0.5,
                matched_keywords=["safety_timeout"],
            )
        logger.debug("stage=safety ms=%.0f", (time.perf_counter() - stage_t0) * 1000)
        if assessment is not None:
            return generate_safety_response(
                assessment, request_id=request.request_id,
            )

        # ── 2b. Rescue product mode OOD gate ─────────────────────────────
        mode = self._product_mode()
        track = normalize_track(request.track)
        if should_refuse_ood(
            product_mode=mode,
            allow_general=self._allow_general(),
            track=request.track,
            question=request.question,
        ):
            return ChatV2InsufficientEvidence(
                request_id=request.request_id,
                status="insufficient_evidence",
                reason_code="out_of_domain",
                message=_RESCUE_OOD_MESSAGE,
                retrieval_summary={
                    "product_mode": mode,
                    "general_dating_hint": looks_general_dating_not_breakup(
                        request.question
                    ),
                },
            )

        # ── 3. Intake check ──────────────────────────────────────────────
        decision = await asyncio.to_thread(
            plan_minimal_intake,
            request.question,
            request.clarification_answers,
            request.profile.model_dump() if request.profile else None,
            request.consent.model_dump(),
            request.conversation_history,
        )
        if decision.action == IntakeAction.ASK_QUESTION:
            from dating_rag.domain.models import ClarificationQuestion

            questions = []
            if decision.clarification_question:
                questions.append(
                    ClarificationQuestion(
                        question_id="intake-1",
                        prompt=decision.clarification_question,
                        reason=decision.reason or "clarification needed",
                        required=False,
                        answer_type="text",
                        skip_label="그냥 답변해 주세요",
                    )
                )
            return ChatV2NeedsClarification(
                request_id=request.request_id,
                status="needs_clarification",
                questions=questions,
            )

        if decision.reason == "out_of_scope":
            return ChatV2InsufficientEvidence(
                request_id=request.request_id,
                status="insufficient_evidence",
                reason_code="out_of_domain",
                message="질문이 연애/관계 주제와 관련이 없어 답변을 제공할 수 없습니다.",
                retrieval_summary={},
            )

        # ── 4. Evidence retrieval ────────────────────────────────────────
        stage_t0 = time.perf_counter()
        query = build_retrieval_query(redacted, decision.reason)

        async def _retrieve() -> list:
            return await asyncio.to_thread(
                lambda: state.retriever.search(
                    redacted.redacted_text,
                    fast=(mode == "rescue_brt14"),
                )
            )

        try:
            results = await asyncio.wait_for(
                _retrieve(), timeout=retrieval_timeout,
            )
        except asyncio.TimeoutError:
            return ProblemEnvelope(
                type="about:blank",
                title="Service Unavailable",
                status=503,
                detail="Retrieval timed out",
                instance="/v2/chat",
            )
        logger.info(
            "stage=retrieve ms=%.0f n=%s",
            (time.perf_counter() - stage_t0) * 1000,
            len(results) if results is not None else 0,
        )

        # ── 5. Evidence gate ─────────────────────────────────────────────
        from dating_rag.domain.models import QueryPlan
        intent = decision.reason or "general_advice"
        topics = ["relationship", "dating"]
        if track is not None:
            intent = "breakup"
            topics = ["breakup", "no_contact", f"day_{track.day_index}"]
        elif any(k in request.question for k in ("이별", "헤어진", "재회", "연락 끊")):
            topics = ["breakup", "no_contact"]
        elif any(k in request.question for k in ("첫 데이트", "썸", "소개팅")):
            topics = ["first_dates", "conversation"]
        elif any(k in request.question for k in ("장거리", "LD")):
            topics = ["long-distance", "texting"]
        elif any(k in request.question for k in ("MBTI", "mbti", "엠비티아이")):
            topics = ["mbti", "conversation"]
        elif any(k in request.question for k in ("싸움", "갈등", "화해", "다퉜")):
            topics = ["conversation", "counseling"]
        plan = QueryPlan(intent=intent, topics=topics)
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
        top_k = int(getattr(getattr(state, "settings", None), "rescue_retrieval_top_k", 4) or 4)
        if mode == "rescue_brt14" and accepted:
            accepted = accepted[:top_k]

        # ── 6. Rerank ────────────────────────────────────────────────────
        # Rescue: never lazy-load cross-encoder (cold load blows 15s budget).
        if mode != "rescue_brt14":
            reranker = state.get_reranker()
            if reranker is not None:
                accepted = await asyncio.to_thread(
                    reranker.rerank, redacted.redacted_text, accepted,
                )

        # ── 7. Context + registry ────────────────────────────────────────
        context_text, registry = state.context_builder.build_context_with_registry(
            accepted, [], plan,
        )

        # Inject track day hints into generation context
        track_hints_model: TrackHints | None = None
        if track is not None:
            hints = track_hints_for_day(track.day_index)
            track_hints_model = TrackHints(**hints)
            boundary = track.hard_boundary or ""
            context_text = (
                f"## Track BRT-14 day {track.day_index}\n"
                f"theme: {hints.get('theme','')}\n"
                f"contact_status: {track.contact_status}\n"
                f"primary_goal: {track.primary_goal}\n"
                f"hard_boundary: {boundary}\n"
                f"day_actions: {hints.get('suggested_day_actions')}\n"
                f"impulse_protocol: {hints.get('impulse_protocol')}\n\n"
                + context_text
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

        # ── 9. Generation (admission control) ────────────────────────────
        conversation_context = ""
        if request.conversation_history:
            history_lines = []
            for turn in request.conversation_history[-6:]:
                role_label = "사용자" if turn.role == "user" else "상담가"
                history_lines.append(f"{role_label}: {turn.content}")
            if history_lines:
                conversation_context = "\n\n## 이전 대화\n" + "\n".join(history_lines)

        lenses: list[str] = []
        if request.consent.personalize_with_mbti and request.profile and request.profile.mbti:
            lenses.append("mbti")
        if (
            request.consent.personalize_with_observations
            and request.profile
            and request.profile.observed_tendencies
        ):
            lenses.append("observations")
        personalization = PersonalizationBlock(
            lenses_used=lenses,
            disclaimer="개인화 정보는 조언 보조용이며 결정을 대체하지 않습니다.",
        )

        acquired = await generation_admission.try_acquire()
        if not acquired:
            return ProblemEnvelope(
                type="about:blank",
                title="Service Unavailable",
                status=503,
                detail="Generation capacity saturated; retry shortly",
                instance="/v2/chat",
            )

        stage_t0 = time.perf_counter()
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
                    temperature=0.1,
                    max_tokens=self._max_tokens(),
                ),
                timeout=generation_timeout,
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
        finally:
            await generation_admission.release()

        logger.info(
            "stage=generate ms=%.0f",
            (time.perf_counter() - stage_t0) * 1000,
        )

        return ChatV2Answered(
            request_id=request.request_id,
            status="answered",
            answer=result.answer,
            evidence_claims=result.evidence_claims,
            citations=result.citations,
            personalization=personalization,
            cultural_reflection=cultural_reflection,
            track_hints=track_hints_model,
        )
