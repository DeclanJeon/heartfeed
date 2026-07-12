"""Intent classification and query analysis for retrieval routing."""

from __future__ import annotations

import re
from typing import Literal

from dating_rag.domain.models import QueryPlan

Intent = Literal[
    "specific_example",
    "general_advice",
    "compare_viewpoints",
    "creator_lookup",
    "definition",
    "high_risk",
]

# Korean keywords mapped to intents
_KR_INTENT_KEYWORDS: dict[str, list[str]] = {
    "specific_example": ["예시", "예를 들어", "경험", "사례", "어떻게 했", "실제로"],
    "compare_viewpoints": ["vs", "비교", "다른 의견", "어느 쪽", "찬반", "반대"],
    "creator_lookup": ["크리에이터", "유튜버", "쌤", "선생님", "님"],
    "definition": ["什么意思", "뜻이 뭐", "의미", "무슨 뜻", "뭔 뜻"],
    "high_risk": ["자살", "자해", "스토킹", "협박", "폭력", "성범죄", "미성년"],
}

# English keywords mapped to intents
_EN_INTENT_KEYWORDS: dict[str, list[str]] = {
    "specific_example": ["example", "for instance", "case study", "what happened", "story"],
    "compare_viewpoints": ["vs", "compare", "different opinions", "debate", "disagree"],
    "creator_lookup": ["coach", "creator", "youtuber", "channel"],
    "definition": ["what does", "what is", "meaning", "define", "explain"],
    "high_risk": ["suicide", "self-harm", "stalking", "threat", "violence", "abuse", "underage"],
}

# Category keyword mapping
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "texting": ["text", "message", "reply", "respond", " nhắn", "카톡", "메시지", "답장", "읽씹"],
    "first_dates": ["first date", "데이트", "初次约会", "만남", "약속"],
    "attraction": ["attract", "interest", "flirt", "매력", "끌리", "관심"],
    "confidence": ["confidence", "self-esteem", "mindset", "자신감", "멘탈", "마음가짐"],
    "online_dating": ["tinder", "bumble", "dating app", "profile", "swipe", "앱", "프로필", "온라인"],
    "conversation": ["conversation", "talk", "topic", "대화", "주제", "말하기"],
    "relationships": ["relationship", "girlfriend", "boyfriend", "partner", "연애", "커플", "관계"],
    "breakup": ["breakup", "break up", "ex", "move on", "이별", "헤어지", "전남친", "전여친"],
    "social_skills": ["social", "body language", "social skills", "사회성", "눈치", "센스"],
    "lifestyle": ["fitness", "fashion", "hobby", "grooming", "운동", "패션", "자기계발"],
}


def _detect_intent(question: str) -> Intent:
    """Classify query intent using keyword matching.

    Args:
        question: The user's question text.

    Returns:
        Detected intent label.
    """
    q_lower = question.lower()

    # High risk always takes priority
    for kw in _KR_INTENT_KEYWORDS.get("high_risk", []):
        if kw in question:
            return "high_risk"
    for kw in _EN_INTENT_KEYWORDS.get("high_risk", []):
        if kw in q_lower:
            return "high_risk"

    # Score each intent
    scores: dict[str, int] = {}
    for intent, keywords in _KR_INTENT_KEYWORDS.items():
        if intent == "high_risk":
            continue
        for kw in keywords:
            if kw in question:
                scores[intent] = scores.get(intent, 0) + 1
    for intent, keywords in _EN_INTENT_KEYWORDS.items():
        if intent == "high_risk":
            continue
        for kw in keywords:
            if kw in q_lower:
                scores[intent] = scores.get(intent, 0) + 1

    if scores:
        return max(scores, key=lambda k: scores[k])  # type: ignore[return-value]

    return "general_advice"


def _detect_category(question: str) -> str | None:
    """Detect the most relevant category from the question.

    Args:
        question: The user's question text.

    Returns:
        Category ID string, or None if no strong match.
    """
    q_lower = question.lower()
    scores: dict[str, int] = {}

    for cat_id, keywords in _CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in q_lower or kw in question:
                scores[cat_id] = scores.get(cat_id, 0) + 1

    if not scores:
        return None

    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] >= 1 else None


def _extract_topics(question: str) -> list[str]:
    """Extract topic keywords from the question.

    Args:
        question: The user's question text.

    Returns:
        List of topic strings.
    """
    topics: list[str] = []

    # Add matched category as a topic
    cat = _detect_category(question)
    if cat:
        topics.append(cat)

    # Extract quoted terms
    quoted = re.findall(r'["\u201c]([^"\u201d]+)["\u201d]', question)
    topics.extend(quoted)

    # Extract hashtags
    hashtags = re.findall(r"#(\w+)", question)
    topics.extend(hashtags)

    return topics


class QueryAnalyzer:
    """Analyzes user questions to produce a retrieval plan.

    Uses keyword-based intent classification and category detection
    to route queries through the retrieval pipeline appropriately.
    """

    def analyze(
        self,
        question: str,
        filters: dict[str, str] | None = None,
    ) -> QueryPlan:
        """Analyze a user question and produce a QueryPlan.

        Args:
            question: The user's question text.
            filters: Optional explicit filters (category, channel, language).

        Returns:
            A QueryPlan describing how to retrieve relevant context.
        """
        intent = _detect_intent(question)
        category = _detect_category(question)
        topics = _extract_topics(question)

        # Apply explicit filter overrides
        if filters:
            if "category" in filters:
                category = filters["category"]

        # Determine strategy flags
        use_transcripts = True
        use_okf = True
        require_conflict_search = intent == "compare_viewpoints"
        require_source_diversity = intent != "creator_lookup"

        # For high-risk queries, still retrieve but the generation layer
        # should handle safety framing
        if intent == "high_risk":
            use_okf = True
            require_source_diversity = True

        channel_filters: list[str] = []
        if filters and "channel" in filters:
            channel_filters = [filters["channel"]]

        return QueryPlan(
            intent=intent,
            topics=topics,
            use_transcripts=use_transcripts,
            use_okf=use_okf,
            require_conflict_search=require_conflict_search,
            require_source_diversity=require_source_diversity,
            category_filter=category,
            channel_filters=channel_filters,
        )
