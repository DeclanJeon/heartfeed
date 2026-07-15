"""Tests for CitationRegistry, build_v2_prompt, and build_context_with_registry."""

from __future__ import annotations

import pytest

from dating_rag.domain.models import (
    Citation,
    KnowledgeClaim,
    QueryPlan,
    RetrievalResult,
)
from dating_rag.generation.prompts import build_v2_prompt
from dating_rag.retrieval.context_builder import CitationRegistry, ContextBuilder


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_result(
    chunk_id: str = "chunk-1",
    text: str = "test evidence text",
    score: float = 0.85,
    source_type: str = "transcript",
    **metadata: object,
) -> RetrievalResult:
    defaults = {
        "channel_name": "TestChannel",
        "title": "Test Video",
        "timestamp_url": "https://youtube.com/watch?v=abc&t=120",
    }
    defaults.update(metadata)
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        score=score,
        source_type=source_type,  # type: ignore[arg-type]
        metadata=defaults,
    )


def _make_claim_result(
    chunk_id: str = "claim-1",
    text: str = "claim text",
    score: float = 0.9,
    **metadata: object,
) -> RetrievalResult:
    defaults = {
        "channel_name": "OKF",
        "title": "Knowledge Claim",
        "timestamp_url": "",
    }
    defaults.update(metadata)
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        score=score,
        source_type="claim",
        metadata=defaults,
    )


def _make_plan(**kwargs: object) -> QueryPlan:
    defaults: dict[str, object] = {
        "intent": "advice",
        "topics": ["dating"],
        "use_transcripts": True,
        "use_okf": True,
        "require_conflict_search": False,
        "require_source_diversity": True,
    }
    defaults.update(kwargs)
    return QueryPlan(**defaults)  # type: ignore[arg-type]


# ===========================================================================
# CitationRegistry tests
# ===========================================================================


class TestCitationRegistry:
    """Tests for CitationRegistry ID assignment and lookups."""

    def test_assigns_sequential_s_ids_for_transcripts(self) -> None:
        registry = CitationRegistry()
        results = [
            _make_result(chunk_id="c1", text="first"),
            _make_result(chunk_id="c2", text="second"),
            _make_result(chunk_id="c3", text="third"),
        ]
        registry.register(results)

        ids = registry.citation_ids()
        assert ids == ["S1", "S2", "S3"]

    def test_assigns_sequential_c_ids_for_claims(self) -> None:
        registry = CitationRegistry()
        results = [
            _make_claim_result(chunk_id="k1", text="claim one"),
            _make_claim_result(chunk_id="k2", text="claim two"),
        ]
        registry.register(results)

        ids = registry.citation_ids()
        assert ids == ["C1", "C2"]

    def test_mixed_types_get_independent_numbering(self) -> None:
        registry = CitationRegistry()
        results = [
            _make_result(chunk_id="t1", text="transcript 1"),
            _make_claim_result(chunk_id="k1", text="claim 1"),
            _make_result(chunk_id="t2", text="transcript 2"),
            _make_claim_result(chunk_id="k2", text="claim 2"),
        ]
        registry.register(results)

        ids = registry.citation_ids()
        assert ids == ["S1", "C1", "S2", "C2"]

    def test_get_citation_returns_citation_model(self) -> None:
        registry = CitationRegistry()
        results = [_make_result(chunk_id="c1", text="evidence")]
        registry.register(results)

        citation = registry.get_citation("S1")
        assert citation is not None
        assert isinstance(citation, Citation)
        assert citation.citation_id == "S1"
        assert citation.source_id == "c1"
        assert citation.source_type == "transcript"
        assert citation.title == "Test Video"
        assert citation.creator == "TestChannel"

    def test_get_citation_returns_none_for_unknown(self) -> None:
        registry = CitationRegistry()
        assert registry.get_citation("S99") is None

    def test_get_all_citations_returns_all(self) -> None:
        registry = CitationRegistry()
        results = [
            _make_result(chunk_id="c1"),
            _make_claim_result(chunk_id="k1"),
        ]
        registry.register(results)

        all_cites = registry.get_all_citations()
        assert len(all_cites) == 2
        assert all(isinstance(c, Citation) for c in all_cites)

    def test_len_returns_count(self) -> None:
        registry = CitationRegistry()
        results = [_make_result(chunk_id="c1"), _make_result(chunk_id="c2")]
        registry.register(results)
        assert len(registry) == 2

    def test_contains_membership(self) -> None:
        registry = CitationRegistry()
        results = [_make_result(chunk_id="c1")]
        registry.register(results)
        assert "S1" in registry
        assert "S2" not in registry

    def test_preserves_score_from_retrieval_result(self) -> None:
        registry = CitationRegistry()
        results = [_make_result(score=0.73)]
        registry.register(results)

        citation = registry.get_citation("S1")
        assert citation is not None
        assert citation.accepted_score == pytest.approx(0.73)

    def test_handles_empty_timestamp_url(self) -> None:
        registry = CitationRegistry()
        results = [_make_result(timestamp_url="")]
        registry.register(results)

        citation = registry.get_citation("S1")
        assert citation is not None
        assert citation.timestamp_url is None or citation.timestamp_url == ""

    def test_handles_none_start_end_seconds(self) -> None:
        registry = CitationRegistry()
        results = [_make_result()]
        registry.register(results)

        citation = registry.get_citation("S1")
        assert citation is not None
        # start_seconds/end_seconds may be None when not in metadata
        # (the default _make_result doesn't set them)


# ===========================================================================
# validate_citation_ids tests
# ===========================================================================


class TestValidateCitationIds:
    """Tests for CitationRegistry.validate_citation_ids."""

    def test_returns_all_valid_when_all_match(self) -> None:
        registry = CitationRegistry()
        registry.register([
            _make_result(chunk_id="c1"),
            _make_result(chunk_id="c2"),
        ])

        valid, invalid = registry.validate_citation_ids(["S1", "S2"])
        assert valid == ["S1", "S2"]
        assert invalid == []

    def test_returns_invalid_for_unknown_ids(self) -> None:
        registry = CitationRegistry()
        registry.register([_make_result(chunk_id="c1")])

        valid, invalid = registry.validate_citation_ids(["S1", "S99", "X1"])
        assert valid == ["S1"]
        assert sorted(invalid) == ["S99", "X1"]

    def test_empty_ids_returns_empty_lists(self) -> None:
        registry = CitationRegistry()
        registry.register([_make_result(chunk_id="c1")])

        valid, invalid = registry.validate_citation_ids([])
        assert valid == []
        assert invalid == []

    def test_mixed_valid_and_invalid(self) -> None:
        registry = CitationRegistry()
        registry.register([
            _make_result(chunk_id="t1"),
            _make_claim_result(chunk_id="k1"),
        ])

        valid, invalid = registry.validate_citation_ids(["S1", "C1", "S3", "C2"])
        assert sorted(valid) == ["C1", "S1"]
        assert sorted(invalid) == ["C2", "S3"]


# ===========================================================================
# build_v2_prompt tests
# ===========================================================================


class TestBuildV2Prompt:
    """Tests for build_v2_prompt."""

    def test_includes_context_text(self) -> None:
        plan = _make_plan()
        result = build_v2_prompt(
            question="어떻게 해야 하나요?",
            context_text="## Transcript Evidence\n\n[S1] test evidence",
            plan=plan,
            registry_citation_ids=["S1"],
        )
        assert "[S1] test evidence" in result

    def test_includes_citation_id_list(self) -> None:
        plan = _make_plan()
        result = build_v2_prompt(
            question="question",
            context_text="ctx",
            plan=plan,
            registry_citation_ids=["S1", "S2", "C1"],
        )
        assert "S1, S2, C1" in result

    def test_includes_available_citation_ids_section_header(self) -> None:
        plan = _make_plan()
        result = build_v2_prompt(
            question="q",
            context_text="ctx",
            plan=plan,
            registry_citation_ids=["S1"],
        )
        assert "## Available Citation IDs" in result

    def test_includes_response_format_instructions(self) -> None:
        plan = _make_plan()
        result = build_v2_prompt(
            question="q",
            context_text="ctx",
            plan=plan,
            registry_citation_ids=["S1"],
        )
        assert "## Response Format Instructions" in result
        assert "ChatV2Answered" in result

    def test_includes_question_section(self) -> None:
        plan = _make_plan()
        result = build_v2_prompt(
            question="데이트 조언 부탁해요",
            context_text="ctx",
            plan=plan,
            registry_citation_ids=[],
        )
        assert "데이트 조언 부탁해요" in result

    def test_includes_strategy_hints_for_conflict(self) -> None:
        plan = _make_plan(require_conflict_search=True)
        result = build_v2_prompt(
            question="q",
            context_text="ctx",
            plan=plan,
            registry_citation_ids=["S1"],
        )
        assert "comparing viewpoints" in result

    def test_handles_empty_citation_ids(self) -> None:
        plan = _make_plan()
        result = build_v2_prompt(
            question="q",
            context_text="ctx",
            plan=plan,
            registry_citation_ids=[],
        )
        assert "(none)" in result

    def test_includes_korean_language_instruction(self) -> None:
        plan = _make_plan()
        result = build_v2_prompt(
            question="q",
            context_text="ctx",
            plan=plan,
            registry_citation_ids=["S1"],
        )
        assert "Korean" in result


# ===========================================================================
# Existing build_context still works (regression)
# ===========================================================================


class TestBuildContextRegression:
    """Verify the original build_context API is unchanged."""

    def test_build_context_returns_string(self) -> None:
        builder = ContextBuilder(max_chunks=5, max_tokens=2000)
        results = [
            _make_result(chunk_id="c1", text="evidence one"),
            _make_result(chunk_id="c2", text="evidence two"),
        ]
        plan = _make_plan()
        ctx = builder.build_context(results, [], plan)
        assert isinstance(ctx, str)
        assert "[S1]" in ctx
        assert "[S2]" in ctx
        assert "evidence one" in ctx

    def test_build_context_includes_okf_claims(self) -> None:
        builder = ContextBuilder()
        claim = KnowledgeClaim(
            concept_id="test",
            statement="test claim",
            stance="for",
            confidence=0.8,
            creator_ids=["creator1"],
            evidence_count=3,
        )
        plan = _make_plan()
        ctx = builder.build_context([], [claim], plan)
        assert "[C1]" in ctx
        assert "test claim" in ctx

    def test_build_context_with_registry_returns_tuple(self) -> None:
        builder = ContextBuilder(max_chunks=5, max_tokens=2000)
        results = [
            _make_result(chunk_id="c1", text="evidence one"),
            _make_result(chunk_id="c2", text="evidence two"),
        ]
        plan = _make_plan()
        ctx, registry = builder.build_context_with_registry(results, [], plan)

        assert isinstance(ctx, str)
        assert isinstance(registry, CitationRegistry)
        assert "[S1]" in ctx
        assert registry.citation_ids() == ["S1", "S2"]

    def test_build_context_with_registry_populates_registry(self) -> None:
        builder = ContextBuilder()
        results = [
            _make_result(chunk_id="t1", text="first transcript"),
            _make_claim_result(chunk_id="k1", text="first claim"),
        ]
        plan = _make_plan()
        ctx, registry = builder.build_context_with_registry(results, [], plan)

        assert "S1" in registry
        # Claims from okf_claims get C1, but here we only have transcript results.
        # The claim result would be in transcript_results passed as-is.
        # In the actual pipeline, claims go via okf_claims, not transcript_results.
        # But the registry still assigns C1 to source_type=="claim".
        assert "C1" in registry
        s1 = registry.get_citation("S1")
        assert s1 is not None
        assert s1.source_id == "t1"

    def test_build_context_with_registry_mixed_sources(self) -> None:
        builder = ContextBuilder()
        transcript = [_make_result(chunk_id="t1", text="transcript evidence")]
        claim = KnowledgeClaim(
            concept_id="test",
            statement="knowledge claim",
            stance="neutral",
            confidence=0.7,
            creator_ids=["creator1"],
            evidence_count=2,
        )
        plan = _make_plan()
        ctx, registry = builder.build_context_with_registry(transcript, [claim], plan)

        assert "S1" in registry
        assert "[S1]" in ctx
        assert "[C1]" in ctx
