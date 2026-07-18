"""Score and rank book/classic chunks for narrative (이야깃거리) + 참고 도서."""

from __future__ import annotations

import re
from dataclasses import dataclass

from dating_rag.domain.models import RetrievalResult

# Lightweight Korean/English topic lexicons for soft scoring without a second model.
_TOPIC_LEXICONS: dict[str, tuple[str, ...]] = {
    "breakup": (
        "이별", "헤어", "재회", "노컨택", "연락 끊", "그리움", "상실", "이별 후",
        "breakup", "no contact", "ex ", "lost love", "grief",
    ),
    "first_dates": (
        "첫 데이트", "소개팅", "썸", "호감", "첫인상", "설렘",
        "first date", "courtship", "crush", "banter",
    ),
    "long-distance": (
        "장거리", "멀리", "떨어져", "영상통화", "기다림",
        "long distance", "separation", "waiting", "reunion",
    ),
    "conflict": (
        "싸움", "갈등", "화해", "다퉜", "오해", "사과", "자존심",
        "conflict", "argument", "apology", "pride", "misunderstanding",
    ),
    "attachment": (
        "불안", "회피", "집착", "애착", "의존", "버림",
        "anxious", "avoidant", "attachment", "abandon",
    ),
    "self_worth": (
        "자존감", "자아", "독립", "경계", "자기",
        "self-worth", "boundary", "independence", "dignity",
    ),
    "infidelity": (
        "바람", "외도", "배신", "질투", "불신",
        "affair", "betrayal", "jealous", "trust",
    ),
    "marriage": (
        "결혼", "부부", "동거", "배우자",
        "marriage", "spouse", "partner", "domestic",
    ),
}


def _is_book(result: RetrievalResult) -> bool:
    meta = result.metadata or {}
    origin = str(meta.get("source_origin", "") or "")
    platform = str(meta.get("platform", "") or "")
    return (
        origin.startswith("book")
        or origin == "classic-literature"
        or platform == "book"
    )


def _is_classic(result: RetrievalResult) -> bool:
    origin = str((result.metadata or {}).get("source_origin", "") or "")
    return origin == "classic-literature"


def _blob(result: RetrievalResult) -> str:
    meta = result.metadata or {}
    parts = [
        str(meta.get("title", "")),
        str(meta.get("channel_name", "") or meta.get("channel", "")),
        str(meta.get("text", "") or result.text or ""),
    ]
    return " ".join(parts).lower()


@dataclass
class ScoredBook:
    result: RetrievalResult
    score: float
    reasons: list[str]


def score_book_for_query(
    result: RetrievalResult,
    query: str,
    topics: list[str] | None = None,
) -> ScoredBook:
    """Heuristic relevance score: vector score + lexical topic overlap + type prior."""
    reasons: list[str] = []
    q = (query or "").lower()
    blob = _blob(result)
    base = float(result.score or 0.0)
    score = base
    reasons.append(f"vector={base:.3f}")

    # Type prior: classics for narrative empathy; theory books for practical insight.
    if _is_classic(result):
        score += 0.08
        reasons.append("classic_prior+0.08")
    elif _is_book(result):
        score += 0.05
        reasons.append("book_prior+0.05")

    # Query token overlap (whitespace + hangul n-grams)
    tokens = [tok for tok in re.split(r"\s+", q) if len(tok) >= 2]
    if len(q) >= 2:
        for n in (2, 3):
            tokens.extend(q[i : i + n] for i in range(0, max(0, len(q) - n + 1)))
    seen_t: set[str] = set()
    uniq: list[str] = []
    for tok in tokens:
        if tok and tok not in seen_t:
            seen_t.add(tok)
            uniq.append(tok)
    hit = sum(1 for tok in uniq[:100] if tok in blob)
    if hit:
        bonus = min(0.45, 0.04 * hit)
        score += bonus
        reasons.append(f"lex_hits={hit}+{bonus:.2f}")

    # Topic lexicon soft match
    topic_keys = list(topics or [])
    # also infer from query
    for key, words in _TOPIC_LEXICONS.items():
        if any(w.lower() in q for w in words):
            if key not in topic_keys:
                topic_keys.append(key)
    topic_hits = 0
    for key in topic_keys:
        words = _TOPIC_LEXICONS.get(key, ())
        if any(w.lower() in blob for w in words):
            topic_hits += 1
    if topic_hits:
        bonus = min(0.2, 0.06 * topic_hits)
        score += bonus
        reasons.append(f"topics={topic_hits}+{bonus:.2f}")

    return ScoredBook(result=result, score=score, reasons=reasons)


def rank_books_for_narrative(
    results: list[RetrievalResult],
    query: str,
    topics: list[str] | None = None,
    *,
    top_k: int = 5,
) -> list[ScoredBook]:
    """Return book/classic chunks sorted by relevance for 이야깃거리 / 참고 도서."""
    books = [r for r in results if _is_book(r)]
    scored = [score_book_for_query(r, query, topics) for r in books]
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:top_k]


def merge_ranked_books_into_accepted(
    accepted: list[RetrievalResult],
    all_results: list[RetrievalResult],
    query: str,
    topics: list[str] | None = None,
    *,
    classic_k: int = 2,
    theory_k: int = 1,
) -> tuple[list[RetrievalResult], list[ScoredBook], bool]:
    """Ensure top-scoring books are present in accepted list (ordered).

    Returns (merged_accepted, ranked_books, low_coverage).
    low_coverage=True when fewer than classic_k+theory_k books available overall.
    """
    ranked = rank_books_for_narrative(
        all_results + accepted,
        query,
        topics,
        top_k=12,
    )
    classics = [s for s in ranked if _is_classic(s.result)]
    theories = [s for s in ranked if not _is_classic(s.result)]
    pick: list[RetrievalResult] = []
    for s in classics[:classic_k]:
        pick.append(s.result)
    for s in theories[:theory_k]:
        pick.append(s.result)

    # Rebuild accepted: books first (ranked), then other evidence
    seen: set[str] = set()
    merged: list[RetrievalResult] = []
    for r in pick:
        key = str(r.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        merged.append(r)
    for r in accepted:
        key = str(r.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        merged.append(r)

    need = classic_k + theory_k
    low = len(ranked) < need
    return merged, ranked[: max(need, 5)], low


def format_narrative_brief(ranked: list[ScoredBook]) -> str:
    """Inject into generation context so the model must use scored sources."""
    if not ranked:
        return (
            "## Narrative Source Brief\n\n"
            "NO ranked classic/book sources available for this query. "
            "Do NOT invent book or classic titles. Keep narrative as gentle situation storytelling "
            "from YouTube evidence only (no fake literature). Flag low 참고 도서 coverage.\n"
        )
    lines = [
        "## Narrative Source Brief — 이야깃거리 = 고전 스토리텔링 (필수)",
        "Ranked by relevance. You MUST tell the user a short STORY from the top classics, not abstract tips.",
        "Order: use sources top-down. Prefer classic-literature for the main tale; theory books as a short bridge.",
        "",
        "How to write `narrative` (Korean, 3~5 short paragraphs):",
        "1) Open with the user's feeling in one warm sentence.",
        "2) Main tale: 「작품명」(작가) 속 구체 장면 — who wanted what, what went wrong, what it cost. "
        "Use story beats from the evidence excerpt (시작·충돌·대가·여운).",
        "3) Optional second classic/book if listed, in one shorter beat.",
        "4) Bridge: '지금 당신 상황과 닿는 지점' 한 단락 + soft next-step hint (not a full action list).",
        "5) Cite with [S#] from Available Citation IDs. Never invent titles not listed.",
        "",
        "Tone: storytelling, not lecture. Empathy first. No bullet tips inside narrative.",
        "",
        "Ranked sources:",
    ]
    for i, s in enumerate(ranked, 1):
        meta = s.result.metadata or {}
        title = str(meta.get("title") or s.result.chunk_id)
        author = str(meta.get("channel_name") or meta.get("channel") or "")
        origin = str(meta.get("source_origin") or "")
        kind = "classic" if origin == "classic-literature" else "book"
        excerpt = (s.result.text or str(meta.get("text") or ""))[:450].replace("\n", " ")
        lines.append(
            f"{i}. [{kind}] score={s.score:.3f} | {title}"
            + (f" — {author}" if author else "")
            + f" | reasons={','.join(s.reasons)}"
        )
        lines.append(f"   excerpt: {excerpt}…")
    return "\n".join(lines) + "\n"
