"""Tests for generation: prompts, citation validation, and generator logic."""

from __future__ import annotations


from dating_rag.domain.models import QueryPlan, RetrievalResult
from dating_rag.generation.citations import validate_citations
from dating_rag.generation.generator import (
    AnswerGenerator,
    _detect_conflicts,
    extract_cited_sources,
    format_citation,
)
from dating_rag.generation.prompts import SYSTEM_PROMPT, build_prompt


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_result(
    text: str,
    *,
    source_type: str = "transcript",
    channel_name: str = "DatingCoach",
    title: str = "Test Video",
    timestamp_url: str = "https://youtube.com/watch?v=abc&t=120",
    chunk_id: str = "chunk-1",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        score=0.9,
        source_type=source_type,  # type: ignore[arg-type]
        metadata={
            "channel_name": channel_name,
            "title": title,
            "timestamp_url": timestamp_url,
        },
    )


def _make_plan(**kwargs: object) -> QueryPlan:
    defaults: dict[str, object] = {
        "intent": "general_advice",
        "topics": ["confidence"],
        "use_transcripts": True,
        "use_okf": True,
        "require_conflict_search": False,
        "require_source_diversity": True,
    }
    defaults.update(kwargs)
    return QueryPlan(**defaults)  # type: ignore[arg-type]


# ===========================================================================
# SYSTEM_PROMPT tests
# ===========================================================================


class TestSystemPrompt:
    """Tests for the SYSTEM_PROMPT constant."""

    def test_contains_all_8_rules(self) -> None:
        for keyword in [
            "Base all claims",          # rule 1
            "Distinguish source",       # rule 2
            "Never diagnose",           # rule 3
            "When sources conflict",    # rule 4
            "practical next steps",     # rule 5
            "Cite with",               # rule 6
            "Never invent",            # rule 7
            "Prioritize safety",       # rule 8
        ]:
            assert keyword in SYSTEM_PROMPT, f"Missing rule keyword: {keyword!r}"

    def test_contains_response_format_section(self) -> None:
        assert "## Response Format" in SYSTEM_PROMPT

    def test_contains_boundaries_section(self) -> None:
        assert "## Boundaries" in SYSTEM_PROMPT

    def test_mentions_citation_format(self) -> None:
        assert "[Channel, Video title, MM:SS]" in SYSTEM_PROMPT


# ===========================================================================
# build_prompt tests
# ===========================================================================


class TestBuildPrompt:
    """Tests for the build_prompt function."""

    def test_basic_structure(self) -> None:
        plan = _make_plan()
        result = build_prompt(
            question="How do I start a conversation?",
            context="[S1] Be confident and smile.",
            plan=plan,
        )
        assert "## Retrieved Evidence" in result
        assert "[S1] Be confident and smile." in result
        assert "## Question" in result
        assert "How do I start a conversation?" in result

    def test_conflict_hint_included_when_required(self) -> None:
        plan = _make_plan(require_conflict_search=True)
        result = build_prompt(question="Q", context="ctx", plan=plan)
        assert "comparing viewpoints" in result
        assert "## Strategy Hints" in result

    def test_no_hints_section_when_no_hints(self) -> None:
        plan = _make_plan(
            require_conflict_search=False,
            require_source_diversity=False,
            intent="",
            topics=[],
        )
        result = build_prompt(question="Q", context="ctx", plan=plan)
        assert "## Strategy Hints" not in result

    def test_intent_and_topics_in_hints(self) -> None:
        plan = _make_plan(intent="compare_viewpoints", topics=["texting", "timing"])
        result = build_prompt(question="Q", context="ctx", plan=plan)
        assert "compare_viewpoints" in result
        assert "texting, timing" in result

    def test_source_diversity_hint(self) -> None:
        plan = _make_plan(require_source_diversity=True)
        result = build_prompt(question="Q", context="ctx", plan=plan)
        assert "multiple creators" in result

    def test_empty_context(self) -> None:
        plan = _make_plan()
        result = build_prompt(question="Q", context="", plan=plan)
        assert "## Retrieved Evidence" in result
        assert "## Question" in result


# ===========================================================================
# extract_cited_sources tests
# ===========================================================================


class TestExtractCitedSources:
    """Tests for extract_cited_sources."""

    def test_extracts_s_labels(self) -> None:
        assert extract_cited_sources("Based on [S1] and [S3], ...") == {"S1", "S3"}

    def test_extracts_c_labels(self) -> None:
        assert extract_cited_sources("See [C1] for details.") == {"C1"}

    def test_mixed_labels(self) -> None:
        assert extract_cited_sources("[S1] says X, but [C2] says Y.") == {"S1", "C2"}

    def test_no_labels(self) -> None:
        assert extract_cited_sources("No citations here.") == set()

    def test_deduplicates(self) -> None:
        assert extract_cited_sources("[S1] and [S1] again") == {"S1"}


# ===========================================================================
# format_citation tests
# ===========================================================================


class TestFormatCitation:
    """Tests for format_citation."""

    def test_full_citation(self) -> None:
        result = _make_result(
            "text",
            channel_name="CoachK",
            title="First Date Tips",
            timestamp_url="https://youtube.com/watch?v=xyz&t=3725",
        )
        citation = format_citation(result, "S1")
        assert "[S1]" in citation
        assert "CoachK" in citation
        assert "First Date Tips" in citation
        assert "62:05" in citation  # 3725s = 62min 5sec

    def test_no_timestamp(self) -> None:
        result = _make_result("text", timestamp_url="")
        citation = format_citation(result, "S2")
        assert "[S2]" in citation
        assert "@" not in citation

    def test_no_title(self) -> None:
        result = _make_result("text", title="")
        citation = format_citation(result, "S3")
        assert "—" not in citation


# ===========================================================================
# validate_citations tests
# ===========================================================================


class TestValidateCitations:
    """Tests for validate_citations."""

    def test_valid_single_source(self) -> None:
        sources = [_make_result("Be confident.")]
        answer = "Be confident [S1]. Smile often [S1]."
        v = validate_citations(answer, sources)
        assert v.is_valid
        assert v.errors == []
        assert "S1" in v.cited_labels

    def test_invalid_source_label(self) -> None:
        sources = [_make_result("text")]
        answer = "As shown in [S5], confidence matters."
        v = validate_citations(answer, sources)
        assert not v.is_valid
        assert len(v.errors) == 1
        assert "S5" in v.errors[0]

    def test_multiple_sources_valid(self) -> None:
        sources = [
            _make_result("text1", chunk_id="c1"),
            _make_result("text2", chunk_id="c2"),
            _make_result("text3", chunk_id="c3"),
        ]
        answer = "Point A [S1]. Point B [S2]. Point C [S3]."
        v = validate_citations(answer, sources)
        assert v.is_valid
        assert v.cited_labels == {"S1", "S2", "S3"}

    def test_claim_label_validation(self) -> None:
        sources = [_make_result("claim text", source_type="claim", chunk_id="c1")]
        answer = "See [C1] for evidence."
        v = validate_citations(answer, sources)
        assert v.is_valid
        assert "C1" in v.cited_labels

    def test_timestamp_valid(self) -> None:
        sources = [_make_result("text")]
        answer = 'Advice from [DatingCoach, "Test Video", 2:00] suggests confidence.'
        v = validate_citations(answer, sources)
        # 2:00 is valid
        assert not any("Invalid timestamp" in e for e in v.errors)

    def test_timestamp_invalid_seconds(self) -> None:
        sources = [_make_result("text")]
        answer = 'Advice from [DatingCoach, "Test Video", 2:99] suggests confidence.'
        v = validate_citations(answer, sources)
        assert not v.is_valid
        assert any("Invalid timestamp" in e for e in v.errors)

    def test_no_citations_passes(self) -> None:
        sources = [_make_result("text")]
        answer = "Just general advice with no citations."
        v = validate_citations(answer, sources)
        assert v.is_valid
        assert v.cited_labels == set()

    def test_diagnosis_warning(self) -> None:
        sources = [_make_result("text")]
        answer = "Based on [S1], you have an anxious attachment style."
        v = validate_citations(answer, sources)
        assert any("attachment style diagnosis" in w for w in v.warnings)

    def test_empty_sources_with_citation_errors(self) -> None:
        answer = "As [S1] says, confidence is key."
        v = validate_citations(answer, [])
        assert not v.is_valid
        assert "S1" in v.errors[0]


# ===========================================================================
# _detect_conflicts tests
# ===========================================================================


class TestDetectConflicts:
    """Tests for the conflict detection heuristic."""

    def test_detects_disagree(self) -> None:
        answer = (
            "Coach A says to text first. However, Coach B disagrees "
            "and recommends waiting at least a day before responding."
        )
        conflicts = _detect_conflicts(answer, [])
        assert len(conflicts) >= 1

    def test_no_conflict_in_aligned_advice(self) -> None:
        answer = "Both coaches agree that confidence is attractive and recommend practicing regularly."
        conflicts = _detect_conflicts(answer, [])
        assert len(conflicts) == 0

    def test_detects_on_the_other_hand(self) -> None:
        answer = (
            "Some say be direct. On the other hand, others prefer a more gradual approach to building rapport."
        )
        conflicts = _detect_conflicts(answer, [])
        assert len(conflicts) >= 1


# ===========================================================================
# AnswerGenerator unit tests (no network)
# ===========================================================================


class TestAnswerGenerator:
    """Unit tests for AnswerGenerator that don't require network calls."""

    def test_init_defaults(self) -> None:
        gen = AnswerGenerator(api_key="test-key")
        assert gen.model == "mimo-v2.5"
        assert gen.api_url == "https://api.xiaomimimo.com/v1"

    def test_init_custom(self) -> None:
        gen = AnswerGenerator(
            api_url="http://localhost:8080/v1/",
            api_key="k",
            model="llama-3",
        )
        assert gen.api_url == "http://localhost:8080/v1"  # trailing slash stripped
        assert gen.model == "llama-3"

    def test_headers_with_key(self) -> None:
        gen = AnswerGenerator(api_key="sk-abc")
        h = gen._headers()
        assert h["Authorization"] == "Bearer sk-abc"
        assert h["Content-Type"] == "application/json"

    def test_headers_without_key(self) -> None:
        gen = AnswerGenerator(api_key="")
        h = gen._headers()
        assert "Authorization" not in h

    def test_format_context(self) -> None:
        gen = AnswerGenerator(api_key="k")
        results = [
            _make_result("First point.", chunk_id="c1"),
            _make_result("Second point.", chunk_id="c2"),
        ]
        ctx = gen._format_context(results)
        assert "[S1]" in ctx
        assert "[S2]" in ctx
        assert "First point." in ctx
        assert "Second point." in ctx

    def test_format_context_with_claims(self) -> None:
        gen = AnswerGenerator(api_key="k")
        results = [
            _make_result("claim", source_type="claim", chunk_id="c1"),
        ]
        ctx = gen._format_context(results)
        assert "[C1]" in ctx

    def test_build_messages_structure(self) -> None:
        gen = AnswerGenerator(api_key="k")
        plan = _make_plan()
        results = [_make_result("evidence", chunk_id="c1")]
        msgs = gen._build_messages("question?", results, plan)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "question?" in msgs[1]["content"]
        assert "evidence" in msgs[1]["content"]

    def test_build_messages_custom_system_prompt(self) -> None:
        gen = AnswerGenerator(api_key="k")
        plan = _make_plan()
        msgs = gen._build_messages("Q", [], plan, system_prompt="Custom prompt.")
        assert msgs[0]["content"] == "Custom prompt."

    def test_generator_alias(self) -> None:
        from dating_rag.generation.generator import Generator
        assert Generator is AnswerGenerator

def test_default_system_prompt_requires_korean_grounding() -> None:
    from dating_rag.generation.prompts import SYSTEM_PROMPT

    assert "한국어" in SYSTEM_PROMPT
    assert "[S1]" in SYSTEM_PROMPT
    assert "근거" in SYSTEM_PROMPT
    assert "자료만으로" in SYSTEM_PROMPT

def test_active_yaml_prompt_matches_grounding_contract() -> None:
    from dating_rag.generation.prompts import load_prompts

    prompt = load_prompts()["system_prompt"]
    assert "한국어" in prompt
    assert "[S1]" in prompt
    assert "제공된 자료만으로는 확인하기 어렵습니다." in prompt
    assert "short narrative" in prompt
    assert "Illustrative Example" in prompt

def test_target_questions_receive_evidence_contract() -> None:
    from dating_rag.retrieval.query_analyzer import QueryAnalyzer

    analyzer = QueryAnalyzer()
    generator = AnswerGenerator(api_key="test")
    source = _make_result("Evidence about the target topic.")
    for question in (
        "MBTI별 연애 스타일",
        "카톡 대화 이어가기",
        "이별 후 no contact",
        "장거리 연애",
    ):
        messages = generator._build_messages(
            question,
            [source],
            analyzer.analyze(question),
            context_text="## Transcript Evidence\n[S1] Evidence",
        )
        assert question in messages[1]["content"]
        assert "[S1]" in messages[1]["content"]
