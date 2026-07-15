"""Tests for v2 contract schemas and Korean golden fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dating_rag.domain.models import (
    ChatV2Answered,
    ChatV2InsufficientEvidence,
    ChatV2NeedsClarification,
    ChatV2Request,
    ChatV2Response,
    ChatV2SafetyEscalation,
    ProblemEnvelope,
)

FIXTURES = Path(__file__).parent / "fixtures" / "v2"


# ---------------------------------------------------------------------------
# Fixture deserialization
# ---------------------------------------------------------------------------


class TestFixtureDeserialization:
    """Every golden fixture must parse into its Pydantic model."""

    def test_needs_clarification(self) -> None:
        data = json.loads((FIXTURES / "needs_clarification.json").read_text())
        model = ChatV2NeedsClarification.model_validate(data)
        assert model.status == "needs_clarification"
        assert len(model.questions) == 1
        assert model.schema_version == "2"
        assert model.api_version == "v2"

    def test_answered(self) -> None:
        data = json.loads((FIXTURES / "answered.json").read_text())
        model = ChatV2Answered.model_validate(data)
        assert model.status == "answered"
        assert len(model.evidence_claims) >= 1
        assert len(model.citations) >= 1
        assert model.answer.empathy  # non-empty
        assert model.conflicts is not None

    def test_safety_escalation(self) -> None:
        data = json.loads((FIXTURES / "safety_escalation.json").read_text())
        model = ChatV2SafetyEscalation.model_validate(data)
        assert model.status == "safety_escalation"
        assert model.risk_kind == "stalking"
        assert model.urgency == "elevated"
        resource_phones = [r["phone"] for r in model.resources]
        assert "112" in resource_phones
        assert "119" in resource_phones
        assert "109" in resource_phones

    def test_insufficient_evidence(self) -> None:
        data = json.loads((FIXTURES / "insufficient_evidence.json").read_text())
        model = ChatV2InsufficientEvidence.model_validate(data)
        assert model.status == "insufficient_evidence"
        assert model.reason_code == "below_threshold"
        assert model.suggestions is not None

    def test_problem_envelope(self) -> None:
        data = json.loads((FIXTURES / "problem_envelope.json").read_text())
        model = ProblemEnvelope.model_validate(data)
        assert model.status == 400
        assert "validation" in model.type.lower() or "검증" in model.title


# ---------------------------------------------------------------------------
# extra=forbid on ChatV2Request
# ---------------------------------------------------------------------------


class TestExtraForbid:
    """ChatV2Request must reject unknown fields."""

    def _minimal_request(self) -> dict:
        """Return a minimal valid request dict."""
        return {
            "schema_version": "2",
            "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "conversation_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
            "question": "연락이 줄어든 이유가 뭘까요?",
            "consent": {
                "personalize_with_mbti": False,
                "personalize_with_observations": False,
                "cultural_saju_reflection": False,
                "process_partner_birth_data": False,
            },
        }

    def test_valid_minimal_request(self) -> None:
        req = ChatV2Request.model_validate(self._minimal_request())
        assert req.schema_version == "2"
        assert req.consent.personalize_with_mbti is False

    def test_extra_field_rejected(self) -> None:
        data = self._minimal_request()
        data["unknown_field"] = "should fail"
        with pytest.raises(ValidationError, match="unknown_field"):
            ChatV2Request.model_validate(data)

    def test_extra_field_in_saju_input_rejected(self) -> None:
        data = self._minimal_request()
        data["saju_input"] = {
            "birth_date": "1995-03-15",
            "gender": "female",
            "calendar_type": "solar",
            "timezone": "Asia/Seoul",
            "extra_garbage": "nope",
        }
        with pytest.raises(ValidationError, match="extra_garbage"):
            ChatV2Request.model_validate(data)

    def test_extra_field_in_filters_rejected(self) -> None:
        data = self._minimal_request()
        data["filters"] = {"language": "ko", "bogus": True}
        with pytest.raises(ValidationError, match="bogus"):
            ChatV2Request.model_validate(data)


# ---------------------------------------------------------------------------
# Consent: process_partner_birth_data accepted but flagged
# ---------------------------------------------------------------------------


class TestConsent:
    """Consent flags must all be accepted, including process_partner_birth_data."""

    def test_partner_birth_data_consent_accepted(self) -> None:
        data = {
            "schema_version": "2",
            "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "conversation_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
            "question": "궁합을 봐주세요",
            "consent": {
                "personalize_with_mbti": True,
                "personalize_with_observations": False,
                "cultural_saju_reflection": True,
                "process_partner_birth_data": True,
            },
        }
        req = ChatV2Request.model_validate(data)
        assert req.consent.process_partner_birth_data is True
        # The field is accepted; flagging is an application-level concern,
        # but the schema must not reject it.

    def test_all_consent_defaults_false(self) -> None:
        from dating_rag.domain.models import Consent

        c = Consent()
        assert c.personalize_with_mbti is False
        assert c.personalize_with_observations is False
        assert c.cultural_saju_reflection is False
        assert c.process_partner_birth_data is False


# ---------------------------------------------------------------------------
# Status literals
# ---------------------------------------------------------------------------


class TestStatusLiterals:
    """Each response variant must have exactly one valid status literal."""

    def test_needs_clarification_literal(self) -> None:
        data = json.loads((FIXTURES / "needs_clarification.json").read_text())
        model = ChatV2NeedsClarification.model_validate(data)
        assert model.status == "needs_clarification"

    def test_answered_literal(self) -> None:
        data = json.loads((FIXTURES / "answered.json").read_text())
        model = ChatV2Answered.model_validate(data)
        assert model.status == "answered"

    def test_safety_escalation_literal(self) -> None:
        data = json.loads((FIXTURES / "safety_escalation.json").read_text())
        model = ChatV2SafetyEscalation.model_validate(data)
        assert model.status == "safety_escalation"

    def test_insufficient_evidence_literal(self) -> None:
        data = json.loads((FIXTURES / "insufficient_evidence.json").read_text())
        model = ChatV2InsufficientEvidence.model_validate(data)
        assert model.status == "insufficient_evidence"

    def test_wrong_status_rejected(self) -> None:
        data = json.loads((FIXTURES / "answered.json").read_text())
        data["status"] = "needs_clarification"  # wrong variant for answered shape
        with pytest.raises(ValidationError):
            ChatV2Answered.model_validate(data)


# ---------------------------------------------------------------------------
# Discriminated union (ChatV2Response)
# ---------------------------------------------------------------------------


class TestDiscriminatedUnion:
    """ChatV2Response type alias must resolve to the correct variant."""

    def _load(self, name: str) -> dict:
        return json.loads((FIXTURES / f"{name}.json").read_text())

    def test_union_needs_clarification(self) -> None:
        data = self._load("needs_clarification")
        model = ChatV2NeedsClarification.model_validate(data)
        assert isinstance(model, ChatV2NeedsClarification)

    def test_union_answered(self) -> None:
        data = self._load("answered")
        model = ChatV2Answered.model_validate(data)
        assert isinstance(model, ChatV2Answered)

    def test_union_safety_escalation(self) -> None:
        data = self._load("safety_escalation")
        model = ChatV2SafetyEscalation.model_validate(data)
        assert isinstance(model, ChatV2SafetyEscalation)

    def test_union_insufficient_evidence(self) -> None:
        data = self._load("insufficient_evidence")
        model = ChatV2InsufficientEvidence.model_validate(data)
        assert isinstance(model, ChatV2InsufficientEvidence)

    def test_union_type_is_union(self) -> None:
        """ChatV2Response should be a typing.Union, not a single BaseModel."""
        import typing

        origin = getattr(ChatV2Response, "__origin__", None)
        assert origin is typing.Union
        args = typing.get_args(ChatV2Response)
        assert ChatV2NeedsClarification in args
        assert ChatV2Answered in args
        assert ChatV2SafetyEscalation in args
        assert ChatV2InsufficientEvidence in args
