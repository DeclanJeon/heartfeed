"""Stateless intake planner for relationship advice questions."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

# ── models ──────────────────────────────────────────────────────────────────

MAX_ROUNDS = 2


class IntakeAction(str, Enum):
    """Possible intake decisions."""

    PROCEED = "PROCEED"
    ASK_QUESTION = "ASK_QUESTION"
    SKIP_OPTIONAL = "SKIP_OPTIONAL"


class IntakeDecision(BaseModel):
    """The intake planner's decision about what to do next."""

    model_config = {"frozen": True}

    action: IntakeAction
    reason: str
    round_number: int = Field(ge=0, le=MAX_ROUNDS)
    clarification_question: str | None = None


# ── heuristics ──────────────────────────────────────────────────────────────

_RELATIONSHIP_CONTEXT_RE = __import__("re").compile(
    r"(?:남자친구|여자친구|남친|여친|연인|파트너|상대방|애인|"
    r"데이트|만남|헤어|이별|결혼|약혼|썸|짝사랑|바람|불륜|"
    r"여인|다가가|좋아하|관심|마음|고백|소개팅|미팅|"
    r"연애|사랑|이성|호감|설레|데이트|만나|약속|"
    r"회피형|애착|불안형|재회|연락|거절|거리두기|이별 후|"
    r"relationship|attachment|breakup|reconcil|contact|distance|"
    r"boyfriend|girlfriend|partner|dating|relationship|breakup|married|crush)",
    __import__("re").IGNORECASE,
)

_VAGUE_PATTERNS = __import__("re").compile(
    r"^(?:[!?.\s]*|.{1,10}[!?.\s]*)$",
    __import__("re").IGNORECASE,
)

_PROFILE_KEYS_REQUIRING_CONSENT = {"mbti", "observed_tendencies", "saju"}


# ── public API ──────────────────────────────────────────────────────────────


def plan_minimal_intake(
    question: str,
    clarification_answers: list | None = None,
    profile: dict | None = None,
    consent: dict | None = None,
    conversation_history: list | None = None,
) -> IntakeDecision:
    """Decide the next intake step for a relationship advice question.

    Stateless: derives round count from len(clarification_answers).
    Max 2 rounds of clarification.

    Returns:
        IntakeDecision with one of PROCEED, ASK_QUESTION, SKIP_OPTIONAL.
    """
    answers = clarification_answers or []
    round_number = len(answers)
    has_context = bool(_RELATIONSHIP_CONTEXT_RE.search(question))
    is_vague = _is_vague(question) and not has_context

    # ── round cap: stop asking after MAX_ROUNDS ──────────────────────────
    if round_number >= MAX_ROUNDS:
        return IntakeDecision(
            action=IntakeAction.PROCEED,
            reason="최대 clarification rounds 도달 — 일반 조언으로 진행합니다.",
            round_number=round_number,
        )

    # ── question is clear enough → check profile needs ──────────────────
    if not is_vague:
        return _decide_with_profile(question, profile, consent, round_number, conversation_history)

    # ── follow-up within an ongoing conversation → never interrupt with a
    # clarification question; the counselor already has context. ──────────
    if conversation_history:
        return _decide_with_profile(question, profile, consent, round_number, conversation_history)

    # ── first round: ask one discriminating question ─────────────────────
    if round_number == 0:
        return IntakeDecision(
            action=IntakeAction.ASK_QUESTION,
            reason="질문이 모호하여 관계 맥락 파악이 필요합니다.",
            round_number=round_number,
            clarification_question="현재 어떤 관계 상황인지 (예: 연애 중, 이별 후, 썸 단계 등) 알려주시면 더 정확한 조언을 드릴 수 있어요.",
        )

    # ── second round after a vague follow-up → just proceed ─────────────
    return _decide_with_profile(question, profile, consent, round_number, conversation_history)


# ── internals ───────────────────────────────────────────────────────────────


def _is_vague(question: str) -> bool:
    """Check if a question is too vague to give useful advice."""
    stripped = question.strip()
    if len(stripped) < 5:
        return True
    return bool(_VAGUE_PATTERNS.match(stripped))


def _decide_with_profile(
    question: str,
    profile: dict | None,
    consent: dict | None,
    round_number: int,
    conversation_history: list | None = None,
) -> IntakeDecision:
    """Decide whether to request optional profile info or proceed."""
    # Check if question is about relationships (skip for vague/short questions)
    has_relationship_context = _RELATIONSHIP_CONTEXT_RE.search(question)
    if not has_relationship_context and conversation_history:
        # A follow-up lacking relationship keywords may still continue an
        # ongoing relationship conversation — inherit scope from history.
        history_text = " ".join(
            getattr(t, "content", str(t)) for t in conversation_history[-6:]
        )
        has_relationship_context = bool(_RELATIONSHIP_CONTEXT_RE.search(history_text))
    if not _is_vague(question) and not has_relationship_context:
        return IntakeDecision(
            action=IntakeAction.SKIP_OPTIONAL,
            reason="out_of_scope",
            round_number=round_number,
        )

    # Never require MBTI/tendencies/saju for ordinary advice
    has_optional_profile = bool(profile) and any(
        profile.get(k) for k in _PROFILE_KEYS_REQUIRING_CONSENT
    )

    if has_optional_profile:
        return IntakeDecision(
            action=IntakeAction.SKIP_OPTIONAL,
            reason="프로필 정보는 일반 조언에 필수가 아닙니다.",
            round_number=round_number,
        )

    # Profile consent not given but question is clear → proceed
    if consent and not consent.get("process_partner_birth_data", False):
        return IntakeDecision(
            action=IntakeAction.PROCEED,
            reason="질문이 명확하고 프로필 동의가 없어도 일반 안내가 가능합니다.",
            round_number=round_number,
        )

    return IntakeDecision(
        action=IntakeAction.PROCEED,
        reason="질문이 명확하여 바로 조언을 제공합니다.",
        round_number=round_number,
    )
