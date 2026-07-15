"""Tests for PII redaction — Korean names, phones, dates, addresses, emotional content."""

from __future__ import annotations

from dating_rag.privacy.redaction import RedactedConcern, build_retrieval_query, redact_concern


# ── name redaction ──────────────────────────────────────────────────────────


class TestNameRedaction:
    """Korean and English name patterns."""

    def test_korean_name_redacted(self) -> None:
        raw = "김민수랑 3개월째 만나고 있는데 고민이에요."
        result = redact_concern(raw)
        assert "김민수" not in result.redacted_text
        assert "[이름]" in result.redacted_text
        assert "3개월째 만나" in result.redacted_text

    def test_multiple_korean_names(self) -> None:
        raw = "이서연이 박지호를 좋아하는데 박지호는 최유진을 좋아해요."
        result = redact_concern(raw)
        assert "이서연" not in result.redacted_text
        assert "박지호" not in result.redacted_text
        assert "최유진" not in result.redacted_text
        assert result.redacted_text.count("[이름]") >= 3

    def test_english_name_redacted(self) -> None:
        raw = "My friend John said my boyfriend David is wrong."
        result = redact_concern(raw)
        assert "John" not in result.redacted_text
        assert "David" not in result.redacted_text
        assert "[이름]" in result.redacted_text


# ── phone number redaction ──────────────────────────────────────────────────


class TestPhoneRedaction:
    """Korean phone number patterns."""

    def test_mobile_number(self) -> None:
        raw = "남자친구 번호가 010-1234-5678인데 확인해봤어요."
        result = redact_concern(raw)
        assert "010-1234-5678" not in result.redacted_text
        assert "[전화번호]" in result.redacted_text

    def test_mobile_no_dash(self) -> None:
        raw = "01012345678으로 연락이 왔어요."
        result = redact_concern(raw)
        assert "01012345678" not in result.redacted_text
        assert "[전화번호]" in result.redacted_text

    def test_landline_number(self) -> None:
        raw = "02-123-4567로 전화했어요."
        result = redact_concern(raw)
        assert "02-123-4567" not in result.redacted_text
        assert "[전화번호]" in result.redacted_text


# ── birth date redaction ────────────────────────────────────────────────────


class TestBirthDateRedaction:
    """Korean birth date patterns."""

    def test_korean_date_format(self) -> None:
        raw = "1995년 3월 15일생이에요."
        result = redact_concern(raw)
        assert "1995" not in result.redacted_text
        assert "[생년월일]" in result.redacted_text

    def test_slash_date_format(self) -> None:
        raw = "95/03/15 태어났어요."
        result = redact_concern(raw)
        assert "95/03/15" not in result.redacted_text
        assert "[생년월일]" in result.redacted_text

    def test_dash_date_format(self) -> None:
        raw = "1995-03-15에 태어났어요."
        result = redact_concern(raw)
        assert "1995-03-15" not in result.redacted_text
        assert "[생년월일]" in result.redacted_text


# ── address / place redaction ───────────────────────────────────────────────


class TestPlaceRedaction:
    """Korean address and place patterns."""

    def test_address_with_apt(self) -> None:
        raw = "서울시 강남구 역삼동 현대아파트에서 만나기로 했어요."
        result = redact_concern(raw)
        assert "현대아파트" not in result.redacted_text
        assert "[장소]" in result.redacted_text

    def test_university(self) -> None:
        raw = "서울대학교에서 만나기로 했어요."
        result = redact_concern(raw)
        assert "서울대학교" not in result.redacted_text
        assert "[장소]" in result.redacted_text


# ── emotional content preservation ──────────────────────────────────────────


class TestEmotionalContentPreservation:
    """Redaction must keep emotional and relationship context intact."""

    def test_emotional_keywords_preserved(self) -> None:
        raw = "김민수랑 헤어지고 너무 슬프고 외로워요."
        result = redact_concern(raw)
        assert "슬프" in result.redacted_text
        assert "외로" in result.redacted_text
        assert "헤어" in result.redacted_text
        assert "김민수" not in result.redacted_text

    def test_emotional_content_field(self) -> None:
        raw = "박지호가 저를 배신해서 상처받았어요. 사랑했는데."
        result = redact_concern(raw)
        assert "배신" in result.original_emotional_content
        assert "상처" in result.original_emotional_content
        assert "사랑" in result.original_emotional_content

    def test_relationship_context_preserved(self) -> None:
        raw = "남자친구 010-1234-5678 번호로 다른 여자한테 연락한 걸 봤어요."
        result = redact_concern(raw)
        assert "남자친구" in result.redacted_text
        assert "010-1234-5678" not in result.redacted_text
        assert "[전화번호]" in result.redacted_text

    def test_no_emotional_content_uses_redacted_text(self) -> None:
        """When no emotional keywords found, original_emotional_content falls back."""
        raw = "2024년 1월 5일에 서울시 강남구 역삼동 현대아파트에 갔어요."
        result = redact_concern(raw)
        # Should still be a valid string
        assert isinstance(result.original_emotional_content, str)
        assert len(result.original_emotional_content) > 0


# ── build_retrieval_query ───────────────────────────────────────────────────


class TestBuildRetrievalQuery:
    """Retrieval query must never include personal identifiers."""

    def test_no_names_in_query(self) -> None:
        raw = "김민수랑 이별 후 대처 방법이 궁금해요."
        redacted = redact_concern(raw)
        query = build_retrieval_query(redacted, "이별 대처")
        assert "김민수" not in query
        assert "[이름]" not in query

    def test_no_phone_in_query(self) -> None:
        raw = "010-1234-5678로 연락하는 게 좋을까요?"
        redacted = redact_concern(raw)
        query = build_retrieval_query(redacted, "연락 방법")
        assert "010" not in query
        assert "[전화번호]" not in query

    def test_includes_intent(self) -> None:
        raw = "남자친구랑 싸웠어요."
        redacted = redact_concern(raw)
        query = build_retrieval_query(redacted, "갈등 해결")
        assert "갈등 해결" in query

    def test_query_length_capped(self) -> None:
        raw = "아주 긴 사연입니다. " * 50
        redacted = redact_concern(raw)
        query = build_retrieval_query(redacted, "긴 사연")
        assert len(query) <= 200
