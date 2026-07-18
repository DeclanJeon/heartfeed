"""Domain models for the dating RAG system."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class TranscriptSegment(BaseModel):
    """A raw transcript segment from a video."""

    start_seconds: float
    end_seconds: float
    text: str
    speaker: str | None = None


class TranscriptChunk(BaseModel):
    """A chunk of transcript ready for embedding and retrieval."""

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    video_id: str
    channel_id: str
    channel_name: str
    title: str
    text: str
    language: str = "en"
    start_seconds: float
    end_seconds: float
    timestamp_url: str = ""
    category: str = ""
    tags: list[str] = Field(default_factory=list)
    views: int = 0
    published_at: datetime | None = None
    chunk_index: int = 0
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    corpus_type: Literal["transcript", "unknown"] = "unknown"
    evidence_role: Literal["source_evidence", "non_evidence", "unknown"] = "unknown"
    source_origin: str = ""
    transcript_status: Literal["available", "unavailable", "unknown"] = "unknown"
    fallback_used: bool = False
    raw_document_id: str = ""
    raw_sha256: str = ""
    ingestion_run_id: str = ""
    chunk_policy_version: str = ""
    content_sha256: str = ""
    schema_version: int = 1
    chunk_id_version: int = 1

class KnowledgeClaim(BaseModel):
    """A structured knowledge claim extracted from transcript evidence."""

    claim_id: str = Field(default_factory=lambda: str(uuid4()))
    concept_id: str
    statement: str
    stance: Literal["for", "against", "neutral", "conditional"] = "neutral"
    applies_when: str = ""
    does_not_apply_when: str = ""
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    creator_ids: list[str] = Field(default_factory=list)
    evidence_count: int = 0
    creator_count: int = 0
    confidence: float = 0.0


class QueryPlan(BaseModel):
    """Parsed query intent and retrieval strategy."""

    intent: str = ""
    topics: list[str] = Field(default_factory=list)
    use_transcripts: bool = True
    use_okf: bool = True
    require_conflict_search: bool = False
    require_source_diversity: bool = True
    category_filter: str | None = None
    channel_filters: list[str] = Field(default_factory=list)
    allow_illustrative_examples: bool = False

class RetrievalResult(BaseModel):
    """A single retrieval result with provenance."""

    chunk_id: str
    text: str
    score: float
    source_type: Literal["transcript", "claim"] = "transcript"
    metadata: dict[str, object] = Field(default_factory=dict)
    metadata: dict[str, object] = Field(default_factory=dict)


class EvidenceDecision(BaseModel):
    """Result of the evidence gate evaluation."""

    accepted: list[RetrievalResult] = Field(default_factory=list)
    rejected: list[RetrievalResult] = Field(default_factory=list)
    reason_code: Literal[
        "accepted",
        "no_candidates",
        "below_threshold",
        "insufficient_diversity",
    ] = "accepted"
    metrics: dict[str, float | int] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """Final response returned to the user."""

    answer: str
    sources: list[RetrievalResult] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    citation_warnings: list[str] = Field(default_factory=list)
    plan: QueryPlan | None = None


# ---------------------------------------------------------------------------
# v2 contract models (append-only, frozen 2026-07-13)
# ---------------------------------------------------------------------------


class ClarificationAnswer(BaseModel):
    """User answer to a clarification question."""

    question_id: str
    value: str | list[str]


class ProfileInput(BaseModel):
    """Optional user profile information."""


    mbti: str | None = Field(default=None, pattern=r"^(?:[EI][SN][TF][JP]|unknown)$")
    observed_tendencies: str | None = Field(default=None, max_length=500)


class Consent(BaseModel):
    """Consent flags for data processing. All default to False."""

    personalize_with_mbti: bool = False
    personalize_with_observations: bool = False
    cultural_saju_reflection: bool = False
    process_partner_birth_data: bool = False


class SajuInput(BaseModel):
    """Solar-only saju input for v1. extra=forbid."""

    model_config = ConfigDict(extra="forbid")

    birth_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    birth_time: str | None = Field(default=None, pattern=r"^\d{2}:\d{2}$")
    birthplace: str | None = Field(default=None, max_length=200)
    gender: Literal["male", "female"]
    calendar_type: Literal["solar"]
    timezone: str


class RequestFilters(BaseModel):
    """Optional query filters. extra=forbid."""

    model_config = ConfigDict(extra="forbid")

    language: Literal["ko", "en"] | None = None


class TrackContext(BaseModel):
    """BRT-14 track context attached to a chat request."""

    model_config = ConfigDict(extra="forbid")

    id: Literal["brt14"] = "brt14"
    day_index: int = Field(default=0, ge=0, le=13)
    contact_status: Literal[
        "no_contact", "rare", "frequent", "cohabit_pending"
    ] = "no_contact"
    primary_goal: Literal[
        "stabilize", "no_contact_hold", "self_worth", "decide_therapy"
    ] = "stabilize"
    hard_boundary: str | None = Field(default=None, max_length=200)


class TrackHints(BaseModel):
    """Server-attached day plan hints for track-aware answers."""

    suggested_day_actions: list[str] = Field(default_factory=list)
    impulse_protocol: str = ""
    theme: str = ""
    day_index: int = 0


class ConversationTurn(BaseModel):
    """A single turn in a multi-turn conversation."""

    role: Literal["user", "assistant"]
    content: str


class ChatV2Request(BaseModel):
    """Top-level v2 chat request. extra=forbid."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2"]
    request_id: str
    conversation_id: str
    clarification_answers: list[ClarificationAnswer] | None = None
    conversation_history: list[ConversationTurn] | None = None
    question: str = Field(min_length=1, max_length=2000)
    profile: ProfileInput | None = None
    consent: Consent
    saju_input: SajuInput | None = None
    filters: RequestFilters | None = None
    track: TrackContext | None = None


# ---- response components --------------------------------------------------


class Citation(BaseModel):
    """A single citation linking an evidence claim to a source."""

    citation_id: str
    source_type: Literal["transcript", "claim"]
    source_id: str
    title: str
    creator: str
    timestamp_url: str | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    accepted_score: float
    media_kind: Literal["youtube", "book", "other"] = "youtube"
    excerpt: str | None = None
    source_origin: str | None = None
    rights: Literal["public_domain", "copyrighted_summary", "unknown"] = "unknown"


class EvidenceClaim(BaseModel):
    """A structured evidence claim used in an answer."""

    claim_id: str
    text: str
    citation_ids: list[str] = Field(min_length=1)
    support_state: Literal["supported", "disputed", "conditional"]


class ActionItem(BaseModel):
    """A recommended action item in the answer."""

    text: str
    basis: Literal["accepted_evidence", "user_statement", "policy_template"]
    citation_ids: list[str] | None = None
    example: str | None = Field(
        default=None,
        description="구체적 대화 예시 또는 시연. 적용 예시인 경우 반드시 포함.",
    )
    evidence_quote: str | None = Field(
        default=None,
        description="이 행동 제안의 근거가 되는 증거 원문 발췌. 사용자가 왜 이 조언을 따르면 좋은지 이해하도록 돕는다.",
    )


class AnsweredContent(BaseModel):
    """Structured answer body."""

    empathy: str
    situation_framing: str
    actions: list[ActionItem]
    boundaries: str
    summary: str
    narrative: str = Field(
        default="",
        description="이야깃거리 — 도서·영상 증거를 엮은 서사 2~5문단.",
    )


class ClarificationQuestion(BaseModel):
    """A question asked to the user for clarification."""

    question_id: str
    prompt: str
    reason: str
    required: bool
    answer_type: Literal["text", "single_choice", "multi_choice"]
    options: list[dict] | None = None
    skip_label: str | None = None


class ConflictItem(BaseModel):
    """A conflict between evidence claims."""

    claim_ids: list[str]
    description: str
    creator_diversity: int


class PersonalizationBlock(BaseModel):
    """Metadata about personalization applied."""

    lenses_used: list[str]
    disclaimer: str


class CulturalReflectionBlock(BaseModel):
    """Cultural / saju reflection metadata."""

    sections: list[dict]
    quality_flags: list[str]
    disclaimer: str
    adapter_version: str


# ---- v2 response variants -------------------------------------------------


class ChatV2Base(BaseModel):
    """Common fields for all v2 response variants."""

    request_id: str
    schema_version: Literal["2"] = "2"
    api_version: Literal["v2"] = "v2"


class ChatV2NeedsClarification(ChatV2Base):
    """Response when the system needs more information."""

    status: Literal["needs_clarification"]
    questions: list[ClarificationQuestion]
    can_answer_without_optional_profile: bool = True


class ChatV2Answered(ChatV2Base):
    """Response with a fully formed answer."""

    status: Literal["answered"]
    answer: AnsweredContent
    evidence_claims: list[EvidenceClaim]
    citations: list[Citation]
    conflicts: list[ConflictItem] | None = None
    personalization: PersonalizationBlock | None = None
    cultural_reflection: CulturalReflectionBlock | None = None
    track_hints: TrackHints | None = None


class ChatV2SafetyEscalation(ChatV2Base):
    """Response when a safety risk is detected."""

    status: Literal["safety_escalation"]
    risk_kind: Literal[
        "self_harm", "violence", "stalking", "coercion", "abuse", "other"
    ]
    urgency: Literal["imminent", "elevated", "precautionary"]
    message: str
    immediate_actions: list[str]
    resources: list[dict]
    safety_policy_version: str


class ChatV2InsufficientEvidence(ChatV2Base):
    """Response when evidence is insufficient to answer."""

    status: Literal["insufficient_evidence"]
    reason_code: Literal[
        "no_candidates",
        "below_threshold",
        "insufficient_diversity",
        "citation_validation_failed",
        "out_of_domain",
        "provider_schema_failure",
    ]
    message: str
    retrieval_summary: dict
    suggestions: list[str] | None = None


class ProblemEnvelope(BaseModel):
    """RFC 7807-style problem detail for error responses."""

    type: str
    title: str
    status: int
    detail: str
    instance: str


# Discriminated union for v2 responses.
ChatV2Response = (
    ChatV2NeedsClarification
    | ChatV2Answered
    | ChatV2SafetyEscalation
    | ChatV2InsufficientEvidence
)


class EvidenceDecision(BaseModel):
    """Result of evidence gate evaluation."""

    accepted: list[RetrievalResult] = Field(default_factory=list)
    rejected: list[RetrievalResult] = Field(default_factory=list)
    reason_code: str = "accepted"
    metrics: dict[str, object] = Field(default_factory=dict)
