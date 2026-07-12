"""Tests for query analyzer, context builder, and filter builder."""

from dating_rag.domain.models import KnowledgeClaim, QueryPlan, RetrievalResult
from dating_rag.retrieval.context_builder import ContextBuilder
from dating_rag.retrieval.filters import build_qdrant_filter
from dating_rag.retrieval.query_analyzer import QueryAnalyzer


# ── QueryAnalyzer ──────────────────────────────────────────────────────────


class TestQueryAnalyzer:
    """Tests for intent classification and category detection."""

    def setup_method(self) -> None:
        self.analyzer = QueryAnalyzer()

    # -- Intent classification --

    def test_general_advice_english(self) -> None:
        plan = self.analyzer.analyze("How do I talk to someone I like?")
        assert plan.intent == "general_advice"
        assert plan.use_transcripts is True
        assert plan.use_okf is True

    def test_general_advice_korean(self) -> None:
        plan = self.analyzer.analyze("좋아하는 사람에게 어떻게 말을 걸어요?")
        assert plan.intent == "general_advice"

    def test_specific_example_intent(self) -> None:
        plan = self.analyzer.analyze("Can you give me an example of a good opening line?")
        assert plan.intent == "specific_example"

    def test_specific_example_korean(self) -> None:
        plan = self.analyzer.analyze("실제 예시를 들어주세요")
        assert plan.intent == "specific_example"

    def test_compare_viewpoints_intent(self) -> None:
        plan = self.analyzer.analyze("Should I text first or wait? Different opinions on this")
        assert plan.intent == "compare_viewpoints"
        assert plan.require_conflict_search is True

    def test_compare_viewpoints_korean(self) -> None:
        plan = self.analyzer.analyze("먼저 연락하는 거 vs 기다리는 거 비교해주세요")
        assert plan.intent == "compare_viewpoints"
        assert plan.require_conflict_search is True

    def test_creator_lookup_intent(self) -> None:
        plan = self.analyzer.analyze("What does Coach Kim say about confidence?")
        assert plan.intent == "creator_lookup"
        assert plan.require_source_diversity is False

    def test_creator_lookup_korean(self) -> None:
        plan = self.analyzer.analyze("김쌤이 자신감에 대해 뭐라고 했어요?")
        assert plan.intent == "creator_lookup"

    def test_definition_intent(self) -> None:
        plan = self.analyzer.analyze("What does 'negging' mean?")
        assert plan.intent == "definition"

    def test_definition_korean(self) -> None:
        plan = self.analyzer.analyze("내깅이 무슨 뜻이에요?")
        assert plan.intent == "definition"

    def test_high_risk_intent_english(self) -> None:
        plan = self.analyzer.analyze("Someone is stalking me, what should I do?")
        assert plan.intent == "high_risk"
        assert plan.use_okf is True
        assert plan.require_source_diversity is True

    def test_high_risk_intent_korean(self) -> None:
        plan = self.analyzer.analyze("스토킹을 당하고 있어요")
        assert plan.intent == "high_risk"

    def test_high_risk_takes_priority(self) -> None:
        """High-risk keywords override other intent signals."""
        plan = self.analyzer.analyze("예시를 들어주세요, 스토킹 관련")
        assert plan.intent == "high_risk"

    # -- Category detection --

    def test_category_texting(self) -> None:
        plan = self.analyzer.analyze("How long should I wait to reply to a text?")
        assert plan.category_filter == "texting"

    def test_category_first_dates(self) -> None:
        plan = self.analyzer.analyze("What should I do on a first date?")
        assert plan.category_filter == "first_dates"

    def test_category_online_dating(self) -> None:
        plan = self.analyzer.analyze("How to improve my Tinder profile?")
        assert plan.category_filter == "online_dating"

    def test_category_breakup_korean(self) -> None:
        plan = self.analyzer.analyze("이별 후에 어떻게 극복해요?")
        assert plan.category_filter == "breakup"

    def test_category_none_for_unrelated(self) -> None:
        plan = self.analyzer.analyze("What is the meaning of life?")
        assert plan.category_filter is None

    # -- Topics extraction --

    def test_topics_from_hashtags(self) -> None:
        plan = self.analyzer.analyze("Tips for #firstdate nervousness")
        assert "firstdate" in plan.topics

    def test_topics_from_quoted_terms(self) -> None:
        plan = self.analyzer.analyze('What is "peacocking" in dating?')
        assert "peacocking" in plan.topics

    # -- Filter overrides --

    def test_explicit_category_override(self) -> None:
        plan = self.analyzer.analyze(
            "How to text better?",
            filters={"category": "conversation"},
        )
        assert plan.category_filter == "conversation"

    def test_explicit_channel_filter(self) -> None:
        plan = self.analyzer.analyze(
            "Dating advice",
            filters={"channel": "CoachKim"},
        )
        assert plan.channel_filters == ["CoachKim"]

    # -- Default plan values --

    def test_default_flags(self) -> None:
        plan = self.analyzer.analyze("Help me with dating")
        assert plan.use_transcripts is True
        assert plan.use_okf is True
        assert plan.require_conflict_search is False
        assert plan.require_source_diversity is True


# ── ContextBuilder ─────────────────────────────────────────────────────────


class TestContextBuilder:
    """Tests for context formatting."""

    def setup_method(self) -> None:
        self.builder = ContextBuilder(max_chunks=8, max_tokens=3000)

    def _make_result(self, chunk_id: str, text: str, **meta: object) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=chunk_id,
            text=text,
            score=0.9,
            metadata={
                "channel_name": "TestChannel",
                "title": "Test Video",
                "timestamp_url": "https://youtube.com/watch?v=abc&t=120",
                **meta,
            },
        )

    def _make_claim(
        self,
        claim_id: str,
        statement: str,
        stance: str = "for",
    ) -> KnowledgeClaim:
        return KnowledgeClaim(
            claim_id=claim_id,
            concept_id="c1",
            statement=statement,
            stance=stance,
            confidence=0.8,
            evidence_count=3,
            creator_ids=["creator_a", "creator_b"],
        )

    def test_transcript_labels(self) -> None:
        results = [
            self._make_result("1", "First chunk text"),
            self._make_result("2", "Second chunk text"),
        ]
        plan = QueryPlan(use_transcripts=True, use_okf=False)
        ctx = self.builder.build_context(results, [], plan)

        assert "[S1]" in ctx
        assert "[S2]" in ctx
        assert "First chunk text" in ctx
        assert "Second chunk text" in ctx

    def test_transcript_includes_metadata(self) -> None:
        results = [self._make_result("1", "Some advice")]
        plan = QueryPlan(use_transcripts=True, use_okf=False)
        ctx = self.builder.build_context(results, [], plan)

        assert "TestChannel" in ctx
        assert "Test Video" in ctx
        assert "youtube.com" in ctx

    def test_okf_claims_section(self) -> None:
        claims = [
            self._make_claim("c1", "Eye contact builds trust"),
            self._make_claim("c2", "Too much eye contact is creepy", stance="against"),
        ]
        plan = QueryPlan(use_transcripts=False, use_okf=True)
        ctx = self.builder.build_context([], claims, plan)

        assert "[C1]" in ctx
        assert "[C2]" in ctx
        assert "Eye contact builds trust" in ctx
        assert "against" in ctx
        assert "80%" in ctx  # confidence formatted
        assert "creator_a" in ctx

    def test_applies_when_in_context(self) -> None:
        claim = self._make_claim("c1", "Be direct")
        claim.applies_when = "When she shows interest"
        claim.does_not_apply_when = "In professional settings"

        plan = QueryPlan(use_transcripts=False, use_okf=True)
        ctx = self.builder.build_context([], [claim], plan)

        assert "Applies when: When she shows interest" in ctx
        assert "Does not apply when: In professional settings" in ctx

    def test_conflict_hint_when_comparing(self) -> None:
        plan = QueryPlan(
            use_transcripts=True,
            use_okf=False,
            require_conflict_search=True,
        )
        ctx = self.builder.build_context([], [], plan)
        assert "comparing viewpoints" in ctx.lower()

    def test_no_conflict_hint_when_not_comparing(self) -> None:
        plan = QueryPlan(
            use_transcripts=True,
            use_okf=False,
            require_conflict_search=False,
        )
        ctx = self.builder.build_context([], [], plan)
        assert "comparing viewpoints" not in ctx.lower()

    def test_respects_max_chunks(self) -> None:
        builder = ContextBuilder(max_chunks=2, max_tokens=10000)
        results = [self._make_result(str(i), f"Chunk {i}") for i in range(5)]
        plan = QueryPlan(use_transcripts=True, use_okf=False)
        ctx = builder.build_context(results, [], plan)

        assert "[S1]" in ctx
        assert "[S2]" in ctx
        assert "[S3]" not in ctx

    def test_respects_token_budget(self) -> None:
        builder = ContextBuilder(max_chunks=100, max_tokens=2000)
        # Each entry with 500-word text ≈ 1554 tokens; header ≈ 17 tokens
        # After header: 1983 remaining. First entry: 1983 - 1554 = 429 left,
        # but next entry needs 1554 + 200 safety = 1754 > 429, so only 1 fits.
        results = [self._make_result(str(i), "word " * 500) for i in range(5)]
        plan = QueryPlan(use_transcripts=True, use_okf=False)
        ctx = builder.build_context(results, [], plan)

        assert "[S1]" in ctx
        assert "[S2]" not in ctx

    def test_empty_results(self) -> None:
        plan = QueryPlan()
        ctx = self.builder.build_context([], [], plan)
        assert ctx == ""

    def test_transcripts_skipped_when_disabled(self) -> None:
        results = [self._make_result("1", "Text")]
        plan = QueryPlan(use_transcripts=False, use_okf=False)
        ctx = self.builder.build_context(results, [], plan)
        assert "[S1]" not in ctx

    def test_okf_skipped_when_disabled(self) -> None:
        claims = [self._make_claim("c1", "Claim")]
        plan = QueryPlan(use_transcripts=False, use_okf=False)
        ctx = self.builder.build_context([], claims, plan)
        assert "[C1]" not in ctx


# ── Filter builder ─────────────────────────────────────────────────────────


class TestBuildQdrantFilter:
    """Tests for Qdrant filter generation."""

    def test_no_filters_returns_none(self) -> None:
        plan = QueryPlan()
        f = build_qdrant_filter(plan)
        assert f is None

    def test_category_filter(self) -> None:
        plan = QueryPlan(category_filter="texting")
        f = build_qdrant_filter(plan)
        assert f is not None
        assert len(f.must) == 1
        assert f.must[0].key == "category"
        assert f.must[0].match.value == "texting"

    def test_channel_filter(self) -> None:
        plan = QueryPlan(channel_filters=["CoachKim"])
        f = build_qdrant_filter(plan)
        assert f is not None
        assert f.must[0].key == "channel_name"
        assert f.must[0].match.value == "CoachKim"

    def test_language_filter(self) -> None:
        plan = QueryPlan()
        f = build_qdrant_filter(plan, language="ko")
        assert f is not None
        assert f.must[0].key == "language"
        assert f.must[0].match.value == "ko"

    def test_min_views_filter(self) -> None:
        plan = QueryPlan()
        f = build_qdrant_filter(plan, min_views=1000)
        assert f is not None
        assert f.must[0].key == "views"
        assert f.must[0].range.gte == 1000

    def test_combined_filters(self) -> None:
        plan = QueryPlan(
            category_filter="attraction",
            channel_filters=["CoachLee"],
        )
        f = build_qdrant_filter(plan, language="ko", min_views=500)
        assert f is not None
        assert len(f.must) == 4

        keys = {c.key for c in f.must}
        assert keys == {"category", "channel_name", "language", "views"}

    def test_zero_min_views_ignored(self) -> None:
        plan = QueryPlan()
        f = build_qdrant_filter(plan, min_views=0)
        assert f is None
