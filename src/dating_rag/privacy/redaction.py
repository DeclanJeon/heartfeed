"""Deterministic PII redaction for relationship concerns."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


class RedactedConcern(BaseModel):
    """Result of redacting PII from a raw concern."""

    model_config = {"frozen": True}

    redacted_text: str
    original_emotional_content: str


# ── patterns ────────────────────────────────────────────────────────────────

# Korean names: 2-4 syllable family+given patterns (common surnames)
# Korean names: 2-4 syllable family+given patterns (common surnames)
# Negative lookbehind prevents matching inside [bracket] replacement tags.
_KR_NAME = re.compile(
    r"(?<!\[)(?:[김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허노하곽성차주우구신]"
    r"[가-힣]{1,3})(?![\]은는이가을를와과에서만도한])"
)


# English names: capitalized word sequences (1-2 words, avoid common nouns)
_EN_NAME = re.compile(r"\b([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20})?)\b")

# Phone numbers: Korean mobile/landline patterns
_PHONE = re.compile(
    r"(?:01[016789]|02|0[3-6][1-5])[\s-]?\d{3,4}[\s-]?\d{4}"
)

# Birth dates: various formats
# Birth dates: various formats (no \b — Korean chars aren't \w)
_BIRTH_DATE = re.compile(
    r"(?:\d{4}[년./-]\s*\d{1,2}[월./-]\s*\d{1,2}[일]?|"
    r"\d{2}[./-]\d{2}[./-]\d{2})"
)

# Addresses/places: Korean address patterns + place keywords
_PLACE = re.compile(
    r"(?:[가-힣]+(?:시|도|구|군|동|읍|면|리|로|길)\s*[가-힣0-9\s]*"
    r"(?:아파트|오피스텔|빌라|호텔|카페|식당|공원|역|대학교?|병원|교회|성당|절))"
    r"|[가-힣]+(?:대학교?|고등학교|중학교)"
)

# ── redaction words (must avoid false positives on emotional content) ────────

_EMOTIONAL_PATTERNS = re.compile(
    r"(?:사랑|미운|보고\s*싶|외롭|슬프|행복|불안|걱정|화나|답답|"
    r"속상|서운|그리워|미안|고마|원망|분노|눈물|가슴\s*아프|"
    r"힘들|지쳐|포기|후회|외로|심란|우울|불행|상처|배신|"
    r"trust|love|hurt|angry|sad|lonely|happy|anxious|worried|"
    r"relationship|dating|partner|boyfriend|girlfriend|husband|wife)"
)


def _replace_names(text: str) -> str:
    """Replace Korean and English names while preserving sentence structure."""
    # Korean names
    text = _KR_NAME.sub("[이름]", text)
    # English names — skip common words that look like names
    _skip = {
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
        "Korea", "Seoul", "Busan",
    }

    def _en_replace(m: re.Match[str]) -> str:
        return "[이름]" if m.group(0) not in _skip else m.group(0)

    text = _EN_NAME.sub(_en_replace, text)
    return text


def redact_concern(raw_concern: str) -> RedactedConcern:
    """Deterministically redact PII while preserving emotional context.

    - Names → [이름]
    - Phone numbers → [전화번호]
    - Addresses/places → [장소]
    - Birth dates → [생년월일]
    """
    emotional_content = _EMOTIONAL_PATTERNS.findall(raw_concern)
    original_emotional_content = " ".join(dict.fromkeys(emotional_content))

    text = raw_concern
    text = _BIRTH_DATE.sub("[생년월일]", text)
    text = _PHONE.sub("[전화번호]", text)
    text = _PLACE.sub("[장소]", text)
    text = _replace_names(text)

    return RedactedConcern(
        redacted_text=text,
        original_emotional_content=original_emotional_content or text,
    )


def build_retrieval_query(redacted: RedactedConcern, intent: str) -> str:
    """Build a topic-level search query from a redacted concern.

    Never includes personal identifiers. Extracts the core relationship topic
    combined with the user's stated intent.
    """
    # Strip remaining bracket tags from redacted text
    cleaned = re.sub(r"\[(?:이름|전화번호|장소|생년월일)\]", "", redacted.redacted_text)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    parts = [intent.strip()]
    if cleaned:
        parts.append(cleaned)

    # Cap query length to keep it a useful search query
    query = " ".join(parts)
    if len(query) > 200:
        query = query[:200].rsplit(" ", 1)[0]

    return query
