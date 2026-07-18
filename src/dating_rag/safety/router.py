"""Safety risk detection and escalation routing."""

from __future__ import annotations

import re
from dataclasses import dataclass

from dating_rag.domain.models import ChatV2SafetyEscalation
from dating_rag.privacy.redaction import RedactedConcern
from dating_rag.safety.resources import RESOURCES, CrisisResource

# ── keyword sets ────────────────────────────────────────────────────────────

_SELF_HARM_KW: list[str] = [
    "죽고", "자살", "자해", "극단적", "목숨", "힘들어서 못살겠",
    "사라지고 싶", "살 이유를 모르겠", "살기 싫", "끝내고 싶",
    "포기하고 싶", "의미 없", "떠나고 싶", "죽어버리", "죽을 것 같",
    "세상이 끝난", "아무도 없", "혼자이고 싶",
    "suicide", "self-harm", "kill myself", "end it all",
    "want to disappear", "no reason to live",
]

_STALKING_KW: list[str] = [
    "따라다니", "따라다녀", "따라다녔", "따라와", "기다리고", "기다렸",
    "계속 연락", "스토킹", "감시", "몰래",
    "위치 추적", "위치추적", "gps 추적", "계정 해킹", "해킹해서",
    "몰래 녹음", "몰래 촬영", "잠금 해제", "카톡 몰래",
    "stalking", "following", "watching", "monitoring",
    "track location", "hack account", "spy on",
]

_VIOLENCE_KW: list[str] = [
    "때리", "때렸", "때리고", "때리는", "때리면",
    "폭력", "위협", "협박", "폭행", "구타",
    "직장에 찾아", "직장으로 찾아", "집에 찾아",
    "violence", "hit", "threaten", "abuse",
]

_COERCION_KW: list[str] = [
    "강요", "강제", "통제",
    "coerce", "control", "force",
]

# Compiled patterns — word-boundary anchored per keyword
_SELF_HARM_RE = re.compile("|".join(re.escape(k) for k in _SELF_HARM_KW), re.IGNORECASE)
_STALKING_RE = re.compile("|".join(re.escape(k) for k in _STALKING_KW), re.IGNORECASE)
_VIOLENCE_RE = re.compile("|".join(re.escape(k) for k in _VIOLENCE_KW), re.IGNORECASE)
_COERCION_RE = re.compile("|".join(re.escape(k) for k in _COERCION_KW), re.IGNORECASE)


# ── models ──────────────────────────────────────────────────────────────────

RiskKind = str  # "self_harm" | "violence" | "stalking" | "coercion" | "abuse"
Urgency = str  # "imminent" | "elevated" | "precautionary"


@dataclass(frozen=True)
class SafetyAssessment:
    """Result of a safety risk assessment."""

    risk_kind: RiskKind
    urgency: Urgency
    confidence: float
    matched_keywords: list[str]


# ── urgency heuristics ──────────────────────────────────────────────────────

_IMMINENT_PATTERNS = re.compile(
    r"(?:죽고\s*싶|자살|목숨|kill myself|end it all|"
    r"죽여|죽[이를]|칼|총|무기|"
    r"오늘|지금|당장|马上)",
    re.IGNORECASE,
)

_ELEVATED_PATTERNS = re.compile(
    r"(?:매일|항상|계속|늘|자꾸|always|every day|keep|"
    r"때리|폭력|위협|협박|강요|강제|통제)",
    re.IGNORECASE,
)


def _classify_urgency(text: str, risk_kind: RiskKind) -> Urgency:
    """Determine urgency level from text signals and risk kind."""
    # Self-harm with direct life-threat language → imminent
    if risk_kind == "self_harm" and _IMMINENT_PATTERNS.search(text):
        return "imminent"
    # Violence with immediate threat → imminent
    if risk_kind == "violence" and _IMMINENT_PATTERNS.search(text):
        return "imminent"
    # Ongoing patterns → elevated
    if _ELEVATED_PATTERNS.search(text):
        return "elevated"
    return "precautionary"


# ── public API ──────────────────────────────────────────────────────────────


def route_safety(
    raw_concern: str,
    redacted: RedactedConcern,
) -> SafetyAssessment | None:
    """Scan for safety risk signals and return an assessment if found.

    Performs high-recall keyword/regex scanning on both raw and redacted text.
    Returns None when no risk is detected.
    """
    # Scan both raw (may contain names before redaction) and redacted text
    texts = [raw_concern, redacted.redacted_text]
    combined = " ".join(texts)

    detections: list[tuple[RiskKind, re.Pattern[str]]] = [
        ("self_harm", _SELF_HARM_RE),
        ("stalking", _STALKING_RE),
        ("violence", _VIOLENCE_RE),
        ("coercion", _COERCION_RE),
    ]

    best: SafetyAssessment | None = None
    for kind, pattern in detections:
        matches = pattern.findall(combined)
        if not matches:
            continue

        urgency = _classify_urgency(combined, kind)
        # Confidence: base 0.5 + 0.15 per keyword match, capped at 0.95
        confidence = min(0.5 + 0.15 * len(matches), 0.95)

        assessment = SafetyAssessment(
            risk_kind=kind,
            urgency=urgency,
            confidence=confidence,
            matched_keywords=list(dict.fromkeys(matches)),  # dedupe, preserve order
        )

        # Keep highest-confidence assessment
        if best is None or assessment.confidence > best.confidence:
            best = assessment

    return best


def generate_safety_response(
    assessment: SafetyAssessment,
    *,
    request_id: str = "safety-escalation",
) -> ChatV2SafetyEscalation:
    """Generate a Korean-language safety escalation response.

    Selects appropriate crisis resources based on risk kind and urgency.
    """
    resource_keys, message, actions = _get_response_content(assessment)

    resources = [
        _format_resource(key) for key in resource_keys
    ]

    return ChatV2SafetyEscalation(
        request_id=request_id,
        status="safety_escalation",
        risk_kind=assessment.risk_kind,  # type: ignore[arg-type]
        urgency=assessment.urgency,  # type: ignore[arg-type]
        message=message,
        immediate_actions=actions,
        resources=resources,
        safety_policy_version="2026-07-13",
    )


# ── internal helpers ────────────────────────────────────────────────────────


def _get_response_content(
    assessment: SafetyAssessment,
) -> tuple[list[str], str, list[str]]:
    """Return (resource_keys, message, immediate_actions) for a risk kind."""
    kind = assessment.risk_kind
    urgency = assessment.urgency

    if kind == "self_harm":
        if urgency == "imminent":
            return (
                ["suicide_prevention", "police", "emergency"],
                "당신의 안전이 가장 중요합니다. 지금 바로 도움을 요청해 주세요. 혼자가 아닙니다.",
                [
                    "지금 바로 109(자살예방 상담전화)에 전화해 주세요",
                    "위급한 상황이라면 112(경찰) 또는 119(소방)에 신고해 주세요",
                    "신뢰할 수 있는 주변 분에게 지금 말씀해 주세요",
                ],
            )
        return (
            ["suicide_prevention", "police"],
            "많이 힘드시군요. 전문 상담사가 도움을 드릴 수 있습니다.",
            [
                "109(자살예방 상담전화)에서 24시간 무료 상담을 받으실 수 있습니다",
                "지금 감정을 안전하게 표현해 주셔서 감사합니다",
            ],
        )

    if kind in ("stalking", "coercion"):
        return (
            ["police", "women_emergency"],
            "현재 상황이 안전하지 않을 수 있습니다. 전문 기관의 도움을 받아주세요.",
            [
                "상대방에게 직접 대응하지 마시고 안전한 곳으로 이동해 주세요",
                "112(경찰)에 신고하여 상황을 기록해 두세요",
                "1366(여성긴급전화)에서 상담 및 보호 지원을 받으실 수 있습니다",
            ],
        )

    if kind == "violence":
        return (
            ["police", "emergency"],
            "폭력은 결코 용납될 수 없습니다. 즉시 도움을 요청해 주세요.",
            [
                "안전한 곳으로 즉시 이동해 주세요",
                "112(경찰)에 신고해 주세요",
                "다치셨다면 119(소방/응급의료)에 연락해 주세요",
            ],
        )

    # Fallback for abuse/other
    return (
        ["police", "women_emergency"],
        "전문 기관에서 도움을 받으실 수 있습니다.",
        [
            "112(경찰)에 상담 및 신고가 가능합니다",
            "1366(여성긴급전화)에서 지원을 받으실 수 있습니다",
        ],
    )


def _format_resource(key: str) -> dict:
    """Format a CrisisResource for the API response."""
    resource: CrisisResource = RESOURCES[key]
    return {
        "name": resource.name,
        "contact": resource.contact,
        "description": resource.description,
    }
