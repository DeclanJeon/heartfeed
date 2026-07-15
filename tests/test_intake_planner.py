"""Tests for intake planner — clear/vague question handling, round cap, consent."""

from __future__ import annotations

from dating_rag.intake.planner import IntakeAction, IntakeDecision, plan_minimal_intake


# ── clear question → PROCEED ────────────────────────────────────────────────


class TestClearQuestionProceed:
    """Questions with clear relationship context should proceed immediately."""

    def test_clear_question_with_context(self) -> None:
        decision = plan_minimal_intake(
            question="남자친구가 자주 연락을 안 하는데 어떻게 대화하면 좋을까요?",
        )
        assert decision.action == IntakeAction.PROCEED
        assert decision.round_number == 0

    def test_clear_breakup_question(self) -> None:
        decision = plan_minimal_intake(
            question="3년 연애 후 이별했는데 마음을 어떻게 추스르면 좋을까요?",
        )
        assert decision.action == IntakeAction.PROCEED

    def test_clear_dating_question(self) -> None:
        decision = plan_minimal_intake(
            question="데이트 상대방이 바람을 피우는 것 같아요. 어떻게 확인하면 좋을까요?",
        )
        assert decision.action == IntakeAction.PROCEED


# ── vague question → ASK ────────────────────────────────────────────────────


class TestVagueQuestionAsk:
    """Vague questions should trigger one discriminating question."""

    def test_very_short_question(self) -> None:
        decision = plan_minimal_intake(question="고민이요")
        assert decision.action == IntakeAction.ASK_QUESTION
        assert decision.clarification_question is not None
        assert decision.round_number == 0

    def test_vague_no_context(self) -> None:
        decision = plan_minimal_intake(question="어떡하죠?")
        assert decision.action == IntakeAction.ASK_QUESTION
        assert "관계" in decision.clarification_question or "상황" in decision.clarification_question


# ── round cap ───────────────────────────────────────────────────────────────


class TestRoundCap:
    """Max 2 rounds of clarification."""

    def test_after_one_round_vague_still_asks(self) -> None:
        decision = plan_minimal_intake(
            question="그냥 힘들어요",
            clarification_answers=["이별 후입니다"],
        )
        # round_number = 1, still under cap → can ask one more or proceed
        assert decision.round_number == 1
        assert decision.action in (IntakeAction.ASK_QUESTION, IntakeAction.PROCEED)

    def test_after_two_rounds_always_proceeds(self) -> None:
        decision = plan_minimal_intake(
            question="그냥 힘들어요",
            clarification_answers=["첫 번째 답", "두 번째 답"],
        )
        assert decision.action == IntakeAction.PROCEED
        assert decision.round_number == 2

    def test_empty_answers_count_as_zero(self) -> None:
        decision = plan_minimal_intake(
            question="남자친구랑 싸웠어요. 어떻게 화해하면 좋을까요?",
            clarification_answers=[],
        )
        assert decision.action == IntakeAction.PROCEED
        assert decision.round_number == 0

    def test_none_answers_count_as_zero(self) -> None:
        decision = plan_minimal_intake(
            question="남자친구랑 싸웠어요. 어떻게 화해하면 좋을까요?",
            clarification_answers=None,
        )
        assert decision.action == IntakeAction.PROCEED
        assert decision.round_number == 0


# ── optional profile skip ───────────────────────────────────────────────────


class TestOptionalProfileSkip:
    """MBTI/tendencies/saju should never be required."""

    def test_profile_with_mbti_skip(self) -> None:
        decision = plan_minimal_intake(
            question="남자친구와 성격 차이로 고민이에요.",
            profile={"mbti": "INFP"},
        )
        assert decision.action == IntakeAction.SKIP_OPTIONAL
        assert "필수가 아닙니다" in decision.reason

    def test_profile_with_tendencies_skip(self) -> None:
        decision = plan_minimal_intake(
            question="연인이 자주 화를 내요.",
            profile={"observed_tendencies": "감정 기복이 심함"},
        )
        assert decision.action == IntakeAction.SKIP_OPTIONAL

    def test_profile_with_saju_skip(self) -> None:
        decision = plan_minimal_intake(
            question="결혼 상대와 궁합이 맞는지 궁금해요.",
            profile={"saju": {"birth_year": 1995}},
        )
        assert decision.action == IntakeAction.SKIP_OPTIONAL

    def test_no_profile_proceeds(self) -> None:
        decision = plan_minimal_intake(
            question="남자친구와 대화가 잘 안 돼요.",
            profile=None,
        )
        assert decision.action == IntakeAction.PROCEED


# ── consent handling ────────────────────────────────────────────────────────


class TestConsentHandling:
    """Profile consent is not needed for general advice."""

    def test_no_consent_clear_question(self) -> None:
        decision = plan_minimal_intake(
            question="연인과의 갈등을 어떻게 해결하면 좋을까요?",
            consent=None,
        )
        assert decision.action == IntakeAction.PROCEED

    def test_partner_birth_consent_false(self) -> None:
        decision = plan_minimal_intake(
            question="남자친구 생일 선물 추천해주세요.",
            consent={"process_partner_birth_data": False},
        )
        assert decision.action == IntakeAction.PROCEED
        assert "동의" in decision.reason or "일반" in decision.reason

    def test_partner_birth_consent_true(self) -> None:
        decision = plan_minimal_intake(
            question="남자친구 생일 선물 추천해주세요.",
            consent={"process_partner_birth_data": True},
        )
        assert decision.action == IntakeAction.PROCEED

    def test_empty_consent_dict(self) -> None:
        decision = plan_minimal_intake(
            question="데이트 코스 추천해주세요.",
            consent={},
        )
        assert decision.action == IntakeAction.PROCEED
