"""Tests for knowledge claim extraction and OKF concept files."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
import pytest

from dating_rag.domain.models import KnowledgeClaim, TranscriptChunk
from dating_rag.knowledge.extractor import (
    ClaimExtractor,
    _detect_stance,
    _extract_conditions,
    _is_claim_sentence,
    _split_sentences,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "okf" / "concepts"


def _make_chunk(text: str, **kwargs: object) -> TranscriptChunk:
    """Create a minimal TranscriptChunk for testing."""
    defaults = {
        "chunk_id": "test-chunk-001",
        "video_id": "test-video",
        "channel_id": "test-channel",
        "channel_name": "TestChannel",
        "title": "Test Video",
        "text": text,
        "language": "ko",
        "start_seconds": 0.0,
        "end_seconds": 60.0,
    }
    defaults.update(kwargs)
    return TranscriptChunk(**defaults)


# ---------------------------------------------------------------------------
# Unit tests: stance detection
# ---------------------------------------------------------------------------


class TestDetectStance:
    """Tests for _detect_stance."""

    def test_korean_support(self) -> None:
        assert _detect_stance("연락을 자주 해야 합니다.") == "for"

    def test_korean_against(self) -> None:
        assert _detect_stance("절대로 하면 안 됩니다.") == "against"

    def test_korean_conditional(self) -> None:
        assert _detect_stance("만약 그 사람이 먼저 연락하면 좋습니다.") in ("for", "conditional")

    def test_korean_warning(self) -> None:
        assert _detect_stance("주의하세요, 이것은 위험합니다.") == "against"

    def test_english_support(self) -> None:
        assert _detect_stance("You should always be honest.") == "for"

    def test_english_against(self) -> None:
        assert _detect_stance("You should never lie to your partner.") == "against"

    def test_english_conditional(self) -> None:
        assert _detect_stance("If you feel uncomfortable, set a boundary.") in ("conditional", "against")

    def test_neutral(self) -> None:
        assert _detect_stance("The sky is blue today.") == "neutral"


# ---------------------------------------------------------------------------
# Unit tests: condition extraction
# ---------------------------------------------------------------------------


class TestExtractConditions:
    """Tests for _extract_conditions."""

    def test_english_if_condition(self) -> None:
        applies, not_applies = _extract_conditions("If you just started dating, keep it light.")
        assert "just started dating" in applies

    def test_no_conditions(self) -> None:
        applies, not_applies = _extract_conditions("Be yourself.")
        assert applies == ""
        assert not_applies == ""


# ---------------------------------------------------------------------------
# Unit tests: sentence splitting
# ---------------------------------------------------------------------------


class TestSplitSentences:
    """Tests for _split_sentences."""

    def test_english_sentences(self) -> None:
        result = _split_sentences("Hello world. How are you? Fine!")
        assert len(result) == 3

    def test_korean_sentences(self) -> None:
        result = _split_sentences("안녕하세요. 오늘 날씨가 좋습니다!")
        assert len(result) == 2

    def test_single_sentence(self) -> None:
        result = _split_sentences("Just one sentence")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Unit tests: claim sentence detection
# ---------------------------------------------------------------------------


class TestIsClaimSentence:
    """Tests for _is_claim_sentence."""

    def test_advice_sentence(self) -> None:
        assert _is_claim_sentence("연락을 자주 해야 합니다.") is True

    def test_neutral_statement(self) -> None:
        # A very simple statement without advice markers may not be a claim
        assert _is_claim_sentence("그냥 그렇습니다.") is False


# ---------------------------------------------------------------------------
# Integration tests: ClaimExtractor
# ---------------------------------------------------------------------------


class TestClaimExtractor:
    """Integration tests for ClaimExtractor.extract_claims."""

    def test_korean_advice_chunk(self) -> None:
        chunk = _make_chunk(
            "연락을 자주 해야 합니다. 그러나 너무 집착하면 안 됩니다. "
            "만약 상대가 답장을 늦게 하면 조심하세요."
        )
        extractor = ClaimExtractor()
        claims = extractor.extract_claims([chunk], concept_id="test-concept")

        assert len(claims) >= 2  # At least the "should" and "should not" claims
        for claim in claims:
            assert isinstance(claim, KnowledgeClaim)
            assert claim.concept_id == "test-concept"
            assert claim.stance in ("for", "against", "neutral", "conditional")

    def test_english_advice_chunk(self) -> None:
        chunk = _make_chunk(
            "You should never chase someone who doesn't respond. "
            "If you feel anxious about a relationship, talk to your partner. "
            "Always set clear boundaries.",
            language="en",
        )
        extractor = ClaimExtractor()
        claims = extractor.extract_claims([chunk], concept_id="boundaries")

        assert len(claims) >= 2
        stances = {c.stance for c in claims}
        assert "against" in stances or "for" in stances

    def test_empty_chunks(self) -> None:
        extractor = ClaimExtractor()
        claims = extractor.extract_claims([], concept_id="empty")
        assert claims == []

    def test_no_advice_chunk(self) -> None:
        chunk = _make_chunk("The weather was nice. I went to the store.")
        extractor = ClaimExtractor()
        claims = extractor.extract_claims([chunk], concept_id="irrelevant")
        # May extract 0 or 1 claim depending on heuristic; shouldn't error
        assert isinstance(claims, list)

    def test_deduplication(self) -> None:
        chunk = _make_chunk("연락을 자주 해야 합니다. 연락을 자주 해야 합니다.")
        extractor = ClaimExtractor()
        claims = extractor.extract_claims([chunk], concept_id="dedup")
        # Same sentence should only appear once
        statements = [c.statement for c in claims]
        assert len(statements) == len(set(statements))

    def test_multiple_chunks(self) -> None:
        chunk1 = _make_chunk("연락을 자주 해야 합니다.", chunk_id="c1")
        chunk2 = _make_chunk("절대로 하면 안 됩니다.", chunk_id="c2")
        extractor = ClaimExtractor()
        claims = extractor.extract_claims([chunk1, chunk2], concept_id="multi")

        assert len(claims) >= 2
        chunk_ids = set()
        for c in claims:
            chunk_ids.update(c.evidence_chunk_ids)
        assert "c1" in chunk_ids or "c2" in chunk_ids


# ---------------------------------------------------------------------------
# OKF concept file validation
# ---------------------------------------------------------------------------


class TestOKFConceptFiles:
    """Validate all OKF concept files have valid YAML frontmatter and required sections."""

    @pytest.fixture()
    def concept_files(self) -> list[Path]:
        files = sorted(DATA_DIR.glob("*.md"))
        assert len(files) >= 10, f"Expected at least 10 concept files, found {len(files)}"
        return files

    def test_all_files_have_yaml_frontmatter(self, concept_files: list[Path]) -> None:
        for path in concept_files:
            content = path.read_text(encoding="utf-8")
            assert content.startswith("---"), f"{path.name}: missing YAML frontmatter"
            # Extract frontmatter between --- delimiters
            parts = content.split("---", 2)
            assert len(parts) >= 3, f"{path.name}: malformed frontmatter (need opening and closing ---)"
            frontmatter = parts[1]
            data = yaml.safe_load(frontmatter)
            assert isinstance(data, dict), f"{path.name}: frontmatter is not a dict"

    def test_required_frontmatter_fields(self, concept_files: list[Path]) -> None:
        required = {"type", "title", "resource", "tags", "timestamp"}
        for path in concept_files:
            content = path.read_text(encoding="utf-8")
            parts = content.split("---", 2)
            data = yaml.safe_load(parts[1])
            missing = required - set(data.keys())
            assert not missing, f"{path.name}: missing fields {missing}"

    def test_type_is_concept(self, concept_files: list[Path]) -> None:
        for path in concept_files:
            content = path.read_text(encoding="utf-8")
            parts = content.split("---", 2)
            data = yaml.safe_load(parts[1])
            assert data["type"] == "concept", f"{path.name}: type should be 'concept'"

    def test_resource_prefix(self, concept_files: list[Path]) -> None:
        for path in concept_files:
            content = path.read_text(encoding="utf-8")
            parts = content.split("---", 2)
            data = yaml.safe_load(parts[1])
            assert data["resource"].startswith("dating://concept/"), (
                f"{path.name}: resource should start with 'dating://concept/'"
            )

    def test_has_definition_section(self, concept_files: list[Path]) -> None:
        for path in concept_files:
            content = path.read_text(encoding="utf-8")
            assert "# Definition" in content, f"{path.name}: missing # Definition section"

    def test_has_common_goals_section(self, concept_files: list[Path]) -> None:
        for path in concept_files:
            content = path.read_text(encoding="utf-8")
            assert "# Common goals" in content, f"{path.name}: missing # Common goals section"

    def test_has_claims_section(self, concept_files: list[Path]) -> None:
        for path in concept_files:
            content = path.read_text(encoding="utf-8")
            assert "# Claims" in content, f"{path.name}: missing # Claims section"

    def test_claims_are_links(self, concept_files: list[Path]) -> None:
        link_re = re.compile(r"^\s*-\s*\[.*\]\(.*\)\s*$", re.MULTILINE)
        for path in concept_files:
            content = path.read_text(encoding="utf-8")
            claims_start = content.find("# Claims")
            if claims_start == -1:
                continue
            claims_section = content[claims_start:]
            assert link_re.search(claims_section), (
                f"{path.name}: # Claims section should contain markdown links"
            )
