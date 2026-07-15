"""Tests for safety router — Korean self-harm, stalking, violence, coercion detection."""

from __future__ import annotations

from dating_rag.privacy.redaction import RedactedConcern, redact_concern
from dating_rag.safety.router import SafetyAssessment, generate_safety_response, route_safety


def _redacted(text: str) -> RedactedConcern:
    """Shortcut: create a RedactedConcern with text as-is (no PII to strip)."""
    return RedactedConcern(redacted_text=text, original_emotional_content=text)


# ── self-harm ───────────────────────────────────────────────────────────────


class TestSelfHarmDetection:
    """Self-harm keyword detection in Korean."""

    def test_direct_self_harm_korean(self) -> None:
        raw = "요즘 너무 힘들어서 죽고 싶다는 생각이 자꾸 들어요."
        redacted = redact_concern(raw)
        result = route_safety(raw, redacted)
        assert result is not None
        assert result.risk_kind == "self_harm"
        assert result.urgency == "imminent"
        assert any("죽고" in kw for kw in result.matched_keywords)

    def test_indirect_self_harm_korean(self) -> None:
        raw = "더 이상 이렇게 살기 힘들어서 못살겠어. 자살이라는 단어가 계속 떠올라."
        redacted = redact_concern(raw)
        result = route_safety(raw, redacted)
        assert result is not None
        assert result.risk_kind == "self_harm"
        assert result.urgency == "imminent"

    def test_self_harm_slang_korean(self) -> None:
        raw = "자해를 하고 나면 잠시 마음이 편해져요."
        redacted = redact_concern(raw)
        result = route_safety(raw, redacted)
        assert result is not None
        assert result.risk_kind == "self_harm"

    def test_extreme_expression(self) -> None:
        raw = "극단적 선택을 할까봐 무서워요."
        redacted = redact_concern(raw)
        result = route_safety(raw, redacted)
        assert result is not None
        assert result.risk_kind == "self_harm"

    def test_life_keyword(self) -> None:
        raw = "목숨을 끊고 싶다는 생각이 들어요."
        redacted = redact_concern(raw)
        result = route_safety(raw, redacted)
        assert result is not None
        assert result.risk_kind == "self_harm"
        assert result.urgency == "imminent"

    def test_english_self_harm(self) -> None:
        raw = "I want to kill myself, I can't go on."
        redacted = redact_concern(raw)
        result = route_safety(raw, redacted)
        assert result is not None
        assert result.risk_kind == "self_harm"
        assert result.urgency == "imminent"


# ── stalking ────────────────────────────────────────────────────────────────


class TestStalkingDetection:
    """Stalking keyword detection in Korean."""

    def test_stalking_korean(self) -> None:
        raw = "전 남자친구가 계속 저를 따라다니고 집 앞에서 기다리고 있어요."
        redacted = redact_concern(raw)
        result = route_safety(raw, redacted)
        assert result is not None
        assert result.risk_kind == "stalking"
        assert any("따라다니" in kw for kw in result.matched_keywords)

    def test_stalking_contact(self) -> None:
        raw = "헤어진 후에도 계속 연락이 와서 무서워요."
        redacted = redact_concern(raw)
        result = route_safety(raw, redacted)
        assert result is not None
        assert result.risk_kind == "stalking"

    def test_stalking_english(self) -> None:
        raw = "He keeps following me and watching my house."
        redacted = redact_concern(raw)
        result = route_safety(raw, redacted)
        assert result is not None
        assert result.risk_kind == "stalking"


# ── violence ────────────────────────────────────────────────────────────────


class TestViolenceDetection:
    """Violence keyword detection."""

    def test_violence_korean(self) -> None:
        raw = "남자친구가 화가 나면 때리고 폭력을 써요."
        redacted = redact_concern(raw)
        result = route_safety(raw, redacted)
        assert result is not None
        assert result.risk_kind == "violence"
        assert result.urgency == "elevated"

    def test_threat_korean(self) -> None:
        raw = "상대방이 저를 위협하고 협박해서 두려워요."
        redacted = redact_concern(raw)
        result = route_safety(raw, redacted)
        assert result is not None
        assert result.risk_kind == "violence"


# ── coercion ────────────────────────────────────────────────────────────────


class TestCoercionDetection:
    """Coercion keyword detection."""

    def test_coercion_korean(self) -> None:
        raw = "남자친구가 모든 걸 강요하고 강제로 통제하려고 해요."
        redacted = redact_concern(raw)
        result = route_safety(raw, redacted)
        assert result is not None
        assert result.risk_kind == "coercion"

    def test_coercion_english(self) -> None:
        raw = "He tries to coerce and control everything I do."
        redacted = redact_concern(raw)
        result = route_safety(raw, redacted)
        assert result is not None
        assert result.risk_kind == "coercion"


# ── ambiguous / normal ─────────────────────────────────────────────────────


class TestAmbiguousAndNormal:
    """Ambiguous or normal concerns should not trigger safety."""

    def test_normal_relationship_concern(self) -> None:
        raw = "남자친구랑 자주 싸워서 고민이에요. 어떻게 대화하면 좋을까요?"
        redacted = redact_concern(raw)
        result = route_safety(raw, redacted)
        assert result is None

    def test_ambiguous_tired_expression(self) -> None:
        raw = "요즘 연애가 많이 힘들어요."
        redacted = redact_concern(raw)
        result = route_safety(raw, redacted)
        # "힘들" alone without safety keywords should not trigger
        assert result is None

    def test_breakup_sadness_not_self_harm(self) -> None:
        raw = "이별 후에 너무 슬프고 눈물이 나요."
        redacted = redact_concern(raw)
        result = route_safety(raw, redacted)
        assert result is None


# ── urgency levels ──────────────────────────────────────────────────────────


class TestUrgencyLevels:
    """Urgency classification: imminent, elevated, precautionary."""

    def test_imminent_self_harm(self) -> None:
        raw = "오늘 정말 죽고 싶어요. 더 이상 못 살겠어요."
        redacted = redact_concern(raw)
        result = route_safety(raw, redacted)
        assert result is not None
        assert result.urgency == "imminent"

    def test_elevated_stalking_pattern(self) -> None:
        raw = "매일 집 앞에서 기다리고 있어요. 계속 연락도 와요."
        redacted = redact_concern(raw)
        result = route_safety(raw, redacted)
        assert result is not None
        assert result.risk_kind == "stalking"
        assert result.urgency == "elevated"

    def test_precautionary_ambiguous_self_harm(self) -> None:
        raw = "자해라는 생각이 가끔 들어요."  # keyword but no time/immediacy
        redacted = redact_concern(raw)
        result = route_safety(raw, redacted)
        assert result is not None
        assert result.risk_kind == "self_harm"
        # No imminent/elevated markers → precautionary
        assert result.urgency == "precautionary"


# ── safety response ─────────────────────────────────────────────────────────


class TestSafetyResponse:
    """Generate Korean safety escalation responses."""

    def test_self_harm_imminent_resources(self) -> None:
        assessment = SafetyAssessment(
            risk_kind="self_harm", urgency="imminent",
            confidence=0.95, matched_keywords=["죽고"],
        )
        resp = generate_safety_response(assessment, request_id="test-1")
        assert resp.status == "safety_escalation"
        assert resp.risk_kind == "self_harm"
        assert resp.urgency == "imminent"
        contacts = [r["contact"] for r in resp.resources]
        assert "109" in contacts
        assert "112" in contacts
        assert "119" in contacts
        assert "자살" in resp.message or "안전" in resp.message

    def test_stalking_resources(self) -> None:
        assessment = SafetyAssessment(
            risk_kind="stalking", urgency="elevated",
            confidence=0.8, matched_keywords=["따라다니"],
        )
        resp = generate_safety_response(assessment, request_id="test-2")
        contacts = [r["contact"] for r in resp.resources]
        assert "112" in contacts
        assert "1366" in contacts

    def test_violence_resources(self) -> None:
        assessment = SafetyAssessment(
            risk_kind="violence", urgency="imminent",
            confidence=0.9, matched_keywords=["때리"],
        )
        resp = generate_safety_response(assessment, request_id="test-3")
        contacts = [r["contact"] for r in resp.resources]
        assert "112" in contacts
        assert "119" in contacts

    def test_response_has_policy_version(self) -> None:
        assessment = SafetyAssessment(
            risk_kind="coercion", urgency="precautionary",
            confidence=0.6, matched_keywords=["강요"],
        )
        resp = generate_safety_response(assessment, request_id="test-4")
        assert resp.safety_policy_version == "2026-07-13"
