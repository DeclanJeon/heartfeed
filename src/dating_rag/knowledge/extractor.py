"""Knowledge claim extraction from transcripts using keyword-based NLP."""

from __future__ import annotations

import re
from typing import Literal

from dating_rag.domain.models import KnowledgeClaim, TranscriptChunk

# ---------------------------------------------------------------------------
# Korean stance patterns
# ---------------------------------------------------------------------------

# "해야 합니다" — should do (supports / for)
_SUPPORT_KO_RE = re.compile(
    r"(?:~?해야\s*합니다|~?하는\s*것이\s*좋습니다|~?하는\s*게\s*좋습니다"
    r"|~?하세요|~?해야\s*돼요|~?하는\s*것이\s*좋아요"
    r"|~?하면\s*좋습니다|~?하면\s*좋아요|~?하세요\b"
    r"|추천합니다|추천해요|권장합니다|권장해요)",
    re.IGNORECASE,
)

# "하면 안 됩니다" — should not do (against)
_AGAINST_KO_RE = re.compile(
    r"(?:~?하면\s*안\s*됩니다|~?하지\s*말아야\s*합니다"
    r"|~?는\s*것이\s*나쁩니다|~?는\s*게\s*나빠요"
    r"|~?하지\s*마세요|~?하면\s*안\s*돼요"
    r"|~?하지\s*말아야\s*해요|~?는\s*것을\s*피하세요"
    r"|피해야\s*합니다|피해야\s*해요|조심하세요|조심해야\s*합니다)",
    re.IGNORECASE,
)

# Conditional patterns: "~면", "~하면", "~경우에는"
_CONDITIONAL_KO_RE = re.compile(
    r"(?:~?면\s|~?하면\s|~?경우에는?\s|~?때는?\s"
    r"|만약\s|만약에\s|~?다면\s|~?일\s때는?\s"
    r"|~?상황에서는?\s|~?조건에서는?\s)",
    re.IGNORECASE,
)

# Warning patterns
_WARNING_KO_RE = re.compile(
    r"(?:주의|경고|위험|조심|주의하세요|위험합니다"
    r"|~?하면\s*큰일|~?하면\s*후회|후회할\s*수\s*있습니다"
    r"|~?하지\s*않으면\s*안\s*됩니다)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# English stance patterns
# ---------------------------------------------------------------------------

_SUPPORT_EN_RE = re.compile(
    r"\b(?:should|you\s+should|it's\s+(?:best|good)\s+to"
    r"|I\s+(?:recommend|suggest)|best\s+practice"
    r"|it\s+works\s+best|the\s+(?:key|secret)\s+is)\b",
    re.IGNORECASE,
)

_AGAINST_EN_RE = re.compile(
    r"\b(?:should\s+not|shouldn't|don't|never|avoid"
    r"|it's\s+(?:bad|worst)\s+to|worst\s+(?:thing|mistake)"
    r"|biggest\s+mistake|common\s+mistake|red\s+flag)\b",
    re.IGNORECASE,
)

_CONDITIONAL_EN_RE = re.compile(
    r"\b(?:if\s+you|when\s+you|in\s+(?:case|situations?\s+where)"
    r"|only\s+(?:if|when)|unless|depends\s+on"
    r"|assuming\s+that|given\s+that)\b",
    re.IGNORECASE,
)

_WARNING_EN_RE = re.compile(
    r"\b(?:warning|caution|careful|danger|watch\s+out"
    r"|be\s+(?:aware|careful|warned)|heads\s+up"
    r"|beware\s+of|sign\s+of)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Condition extraction patterns
# ---------------------------------------------------------------------------

_APPLIES_WHEN_KO_RE = re.compile(
    r"(?:~?면|~?하면|~?경우에는?|~?때는?"
    r"|만약\s|만약에\s|~?다면|~?일\s때는?"
    r"|~?상황에서는?|~?조건에서는?)([^。.!?\n]{5,80})",
    re.IGNORECASE,
)

_DOES_NOT_APPLY_WHEN_KO_RE = re.compile(
    r"(?:~?하지만|그러나|다만|단|except|unless"
    r"|~?때는\s*안|~?경우는\s*제외)([^。.!?\n]{5,80})",
    re.IGNORECASE,
)

_APPLIES_WHEN_EN_RE = re.compile(
    r"(?:if\s+you|when\s+you|in\s+(?:case|situations?\s+where)"
    r"|only\s+(?:if|when)|depends\s+on)([^.!?\n]{5,80})",
    re.IGNORECASE,
)

_DOES_NOT_APPLY_WHEN_EN_RE = re.compile(
    r"(?:except\s+(?:when|if)|unless|however|but\s+not"
    r"|doesn't\s+apply\s+(?:when|if)|not\s+for)([^.!?\n]{5,80})",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_stance(text: str) -> Literal["for", "against", "neutral", "conditional"]:
    """Detect claim stance from text using keyword patterns."""
    # Check warnings first (they imply "against" with higher severity)
    if _WARNING_KO_RE.search(text) or _WARNING_EN_RE.search(text):
        return "against"

    # Check conditional (must be before for/against to catch "if you should...")
    if _CONDITIONAL_KO_RE.search(text) or _CONDITIONAL_EN_RE.search(text):
        # If conditional AND has support/against markers, use the stronger signal
        has_support = bool(_SUPPORT_KO_RE.search(text) or _SUPPORT_EN_RE.search(text))
        has_against = bool(_AGAINST_KO_RE.search(text) or _AGAINST_EN_RE.search(text))
        if has_against and not has_support:
            return "against"
        if has_support and not has_against:
            return "for"
        return "conditional"

    # Direct for/against
    if _AGAINST_KO_RE.search(text) or _AGAINST_EN_RE.search(text):
        return "against"
    if _SUPPORT_KO_RE.search(text) or _SUPPORT_EN_RE.search(text):
        return "for"

    return "neutral"


def _extract_conditions(text: str) -> tuple[str, str]:
    """Extract applies_when and does_not_apply_when from text."""
    applies_parts: list[str] = []
    not_applies_parts: list[str] = []

    for m in _APPLIES_WHEN_KO_RE.finditer(text):
        applies_parts.append(m.group(1).strip())
    for m in _APPLIES_WHEN_EN_RE.finditer(text):
        applies_parts.append(m.group(1).strip())

    for m in _DOES_NOT_APPLY_WHEN_KO_RE.finditer(text):
        not_applies_parts.append(m.group(1).strip())
    for m in _DOES_NOT_APPLY_WHEN_EN_RE.finditer(text):
        not_applies_parts.append(m.group(1).strip())

    return "; ".join(applies_parts), "; ".join(not_applies_parts)


def _is_claim_sentence(text: str) -> bool:
    """Heuristic: does this sentence contain a knowledge claim?"""
    # A sentence is claim-worthy if it contains stance markers
    stance = _detect_stance(text)
    if stance != "neutral":
        return True
    # Or if it has advice-like sentence endings
    if re.search(r"(?:합니다|해요|입니다|이에요|세요|돼요)\s*[.!?]?$", text):
        return True
    # English: imperative or declarative advice
    if re.search(r"\b(?:always|never|don't|make\s+sure|remember)\b", text, re.IGNORECASE):
        return True
    return False


def _confidence_for_stance(
    stance: Literal["for", "against", "neutral", "conditional"],
    evidence_count: int,
    creator_count: int,
) -> float:
    """Calculate confidence score based on stance and evidence."""
    base = {
        "for": 0.6,
        "against": 0.6,
        "conditional": 0.5,
        "neutral": 0.3,
    }[stance]
    # Boost with evidence
    evidence_boost = min(evidence_count * 0.05, 0.2)
    creator_boost = min(creator_count * 0.05, 0.2)
    return min(base + evidence_boost + creator_boost, 1.0)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, handling Korean and English punctuation."""
    # Split on sentence-ending punctuation, keeping the delimiter attached
    parts = re.split(r"([.!?。]+(?:\s|$))", text)
    sentences: list[str] = []
    current = ""
    for i, part in enumerate(parts):
        current += part
        if i % 2 == 1:  # delimiter was at odd index
            s = current.strip()
            if s:
                sentences.append(s)
            current = ""
    # Don't lose trailing text without punctuation
    if current.strip():
        sentences.append(current.strip())
    return sentences


# ---------------------------------------------------------------------------
# ClaimExtractor
# ---------------------------------------------------------------------------


class ClaimExtractor:
    """Keyword-based knowledge claim extractor for dating advice transcripts.

    Uses regex patterns to identify stance, conditions, and claims
    in Korean and English transcript text.
    """

    def __init__(self, *, min_sentence_len: int = 10) -> None:
        self.min_sentence_len = min_sentence_len

    def extract_claims(
        self,
        chunks: list[TranscriptChunk],
        *,
        concept_id: str = "",
    ) -> list[KnowledgeClaim]:
        """Extract knowledge claims from transcript chunks.

        Args:
            chunks: Transcript chunks to extract claims from.
            concept_id: The concept these claims relate to.

        Returns:
            List of extracted KnowledgeClaim objects.
        """
        claims: list[KnowledgeClaim] = []
        seen_statements: set[str] = set()

        for chunk in chunks:
            sentences = _split_sentences(chunk.text)
            for sentence in sentences:
                if len(sentence) < self.min_sentence_len:
                    continue
                if not _is_claim_sentence(sentence):
                    continue

                # Deduplicate
                normalized = sentence.strip().lower()
                if normalized in seen_statements:
                    continue
                seen_statements.add(normalized)

                stance = _detect_stance(sentence)
                applies_when, does_not_apply_when = _extract_conditions(sentence)

                confidence = _confidence_for_stance(
                    stance, evidence_count=1, creator_count=1,
                )

                claim = KnowledgeClaim(
                    concept_id=concept_id,
                    statement=sentence.strip(),
                    stance=stance,
                    applies_when=applies_when,
                    does_not_apply_when=does_not_apply_when,
                    evidence_chunk_ids=[chunk.chunk_id],
                    creator_ids=[chunk.channel_id],
                    evidence_count=1,
                    creator_count=1,
                    confidence=confidence,
                )
                claims.append(claim)

        return claims
