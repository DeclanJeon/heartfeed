"""Domain models for the dating RAG system."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


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


class RetrievalResult(BaseModel):
    """A single retrieval result with provenance."""

    chunk_id: str
    text: str
    score: float
    source_type: Literal["transcript", "claim"] = "transcript"
    metadata: dict[str, object] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """Final response returned to the user."""

    answer: str
    sources: list[RetrievalResult] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    citation_warnings: list[str] = Field(default_factory=list)
    plan: QueryPlan | None = None
