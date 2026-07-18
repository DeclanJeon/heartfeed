"""Context builder for formatting retrieval results into LLM context."""

from __future__ import annotations

from dating_rag.domain.models import (
    Citation,
    KnowledgeClaim,
    QueryPlan,
    RetrievalResult,
)

# Approximate tokens per character (conservative for mixed CJK/English)
_TOKENS_PER_CHAR = 0.6


def _estimate_tokens(text: str) -> int:
    """Estimate token count for a text string."""
    return int(len(text) * _TOKENS_PER_CHAR)


def _format_timestamp_url(url: str) -> str:
    """Format a timestamp URL for citation.

    Args:
        url: The timestamp URL (e.g. https://youtube.com/watch?v=...&t=120).

    Returns:
        Formatted URL string or empty.
    """
    if not url:
        return ""
    return f"  ↳ Source: {url}"


# ---------------------------------------------------------------------------
# CitationRegistry
# ---------------------------------------------------------------------------


class CitationRegistry:
    """Registry that assigns stable citation IDs (S1, S2, …, C1, C2, …)
    to accepted retrieval results and exposes lookup helpers.

    Transcripts receive ``S1``, ``S2``, … labels; claims receive ``C1``, ``C2``, …
    in encounter order.
    """

    def __init__(self) -> None:
        self._citations: dict[str, Citation] = {}

    def register(self, results: list[RetrievalResult]) -> None:
        """Register retrieval results and assign citation IDs."""
        transcript_idx = 0
        claim_idx = 0
        for result in results:
            if result.source_type == "transcript":
                transcript_idx += 1
                cid = f"S{transcript_idx}"
            else:
                claim_idx += 1
                cid = f"C{claim_idx}"

            meta = result.metadata
            origin = str(meta.get("source_origin", "") or "")
            platform = str(meta.get("platform", "") or "")
            if origin.startswith("book") or origin == "classic-literature" or platform == "book":
                media_kind = "book"
            elif meta.get("timestamp_url") or "youtube" in str(meta.get("url", "")).lower():
                media_kind = "youtube"
            else:
                media_kind = "other"
            full_text = str(meta.get("text", "") or result.text or "")
            if origin == "classic-literature":
                rights = "public_domain"
                excerpt = full_text[:2500] or None
            elif origin.startswith("book") or platform == "book":
                rights = "copyrighted_summary"
                excerpt = full_text[:800] or None
            else:
                rights = "unknown"
                excerpt = full_text[:400] or None
            self._citations[cid] = Citation(
                citation_id=cid,
                source_type=str(result.source_type),
                source_id=str(result.chunk_id),
                title=str(meta.get("title", "")),
                creator=str(meta.get("channel_name", "Unknown")),
                timestamp_url=str(meta.get("timestamp_url", "") or meta.get("url", "") or "") or None,
                start_seconds=(
                    float(meta["start_seconds"])
                    if meta.get("start_seconds") is not None
                    else None
                ),
                end_seconds=(
                    float(meta["end_seconds"])
                    if meta.get("end_seconds") is not None
                    else None
                ),
                accepted_score=result.score,
                media_kind=media_kind,  # type: ignore[arg-type]
                excerpt=excerpt,
                source_origin=origin or None,
                rights=rights,  # type: ignore[arg-type]
            )

    # -- public lookups ---------------------------------------------------

    def get_citation(self, citation_id: str) -> Citation | None:
        """Return the Citation for *citation_id*, or ``None``."""
        return self._citations.get(citation_id)

    def get_all_citations(self) -> list[Citation]:
        """Return every registered citation."""
        return list(self._citations.values())

    def validate_citation_ids(self, ids: list[str]) -> tuple[list[str], list[str]]:
        """Partition *ids* into (valid, invalid) against the registry."""
        known = set(self._citations)
        return sorted(known & set(ids)), sorted(set(ids) - known)

    def citation_ids(self) -> list[str]:
        """Return all registered citation IDs in order."""
        return list(self._citations.keys())

    def __len__(self) -> int:
        return len(self._citations)

    def __contains__(self, citation_id: str) -> bool:
        return citation_id in self._citations


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------


class ContextBuilder:
    """Formats retrieval results into a structured context string for the LLM.

    Builds a context block with labeled transcript evidence [S1], [S2], ...
    and a separate OKF claims section, respecting token limits.
    """

    def __init__(
        self,
        max_chunks: int = 8,
        max_tokens: int = 3000,
    ) -> None:
        """Initialize the context builder.

        Args:
            max_chunks: Maximum number of transcript chunks to include.
            max_tokens: Approximate maximum token budget for the context.
        """
        self.max_chunks = max_chunks
        self.max_tokens = max_tokens

    def _build_context_core(
        self,
        transcript_results: list[RetrievalResult],
        okf_claims: list[KnowledgeClaim],
        plan: QueryPlan,
        illustrative_examples: list[dict[str, object]] | None = None,
    ) -> tuple[str, list[RetrievalResult]]:
        """Core rendering logic shared by ``build_context`` and
        ``build_context_with_registry``.

        Returns:
            ``(formatted_context, rendered_results)``
        """
        sections: list[str] = []
        rendered: list[RetrievalResult] = []
        token_budget = self.max_tokens

        if transcript_results and plan.use_transcripts:
            evidence_lines: list[str] = []
            rendered_index = 0

            def _is_book(result: RetrievalResult) -> bool:
                origin = str((result.metadata or {}).get("source_origin", "") or "")
                platform = str((result.metadata or {}).get("platform", "") or "")
                return (
                    origin.startswith("book")
                    or origin == "classic-literature"
                    or platform == "book"
                )

            # Prefer a mix: keep up to 2 book chunks even if YouTube ranks higher.
            # Otherwise first-date style queries only show video citations.
            books = [r for r in transcript_results if _is_book(r)]
            non_books = [r for r in transcript_results if not _is_book(r)]
            book_slots = min(3, len(books), max(0, self.max_chunks - 1))
            classic_books = [
                r
                for r in books
                if str((r.metadata or {}).get("source_origin", "")) == "classic-literature"
            ]
            theory_books = [r for r in books if r not in classic_books]
            # Reserve: up to 2 classic narratives + 1 theory book when available
            classic_slots = min(2, len(classic_books), book_slots)
            theory_slots = min(1, len(theory_books), max(0, book_slots - classic_slots))
            reserved_slots = classic_slots + theory_slots
            if reserved_slots == 0:
                reserved_slots = book_slots

            ordered: list[RetrievalResult] = []
            ordered.extend(non_books[: max(0, self.max_chunks - reserved_slots)])
            for b in classic_books[:classic_slots] + theory_books[: max(theory_slots, book_slots - classic_slots)]:
                if b not in ordered:
                    ordered.append(b)
            for r in transcript_results:
                if len(ordered) >= self.max_chunks:
                    break
                if r not in ordered:
                    ordered.append(r)

            def _entry_text(result: RetrievalResult) -> str:
                text = (result.text or "").strip()
                # Book HQ/templates are long; keep a tight excerpt so they fit
                # the token budget alongside YouTube evidence.
                if _is_book(result) and len(text) > 700:
                    return text[:700].rstrip() + "…"
                if len(text) > 1200:
                    return text[:1200].rstrip() + "…"
                return text

            # Render classics/books first so token exhaustion cannot drop them.
            book_pick = [r for r in ordered if _is_book(r)][: max(reserved_slots, book_slots)]
            other_pick = [r for r in ordered if not _is_book(r)]
            render_queue: list[RetrievalResult] = []
            seen_chunk: set[str] = set()
            for r in book_pick + other_pick:
                key = str(r.chunk_id)
                if key in seen_chunk:
                    continue
                seen_chunk.add(key)
                render_queue.append(r)
                if len(render_queue) >= self.max_chunks:
                    break

            for result in render_queue:
                channel = result.metadata.get("channel_name", "Unknown")
                title = result.metadata.get("title", "")
                timestamp_url = str(result.metadata.get("timestamp_url", ""))

                header = f"[S{rendered_index + 1}] ({channel}"
                if title:
                    header += f' — "{title}"'
                header += ")"

                text = _entry_text(result)
                source_line = _format_timestamp_url(timestamp_url)

                entry = header + "\n" + text
                if source_line:
                    entry += "\n" + source_line
                entry_tokens = _estimate_tokens(entry)

                if token_budget - entry_tokens < 200:
                    continue
                token_budget -= entry_tokens
                rendered_index += 1
                rendered.append(result)
                evidence_lines.append(entry)

            if evidence_lines:
                sections.append(
                    "## Transcript Evidence\n\n" + "\n\n".join(evidence_lines)
                )

        # --- OKF claims section ---
        if okf_claims and plan.use_okf:
            claim_lines: list[str] = []
            for i, claim in enumerate(okf_claims):
                label = f"C{i + 1}"
                stance = claim.stance
                confidence = f"{claim.confidence:.0%}"
                creators = (
                    ", ".join(claim.creator_ids) if claim.creator_ids else "unknown"
                )
                evidence_count = claim.evidence_count

                line = (
                    f"[{label}] {claim.statement}\n"
                    f"  Stance: {stance} | Confidence: {confidence} "
                    f"| Creators: {creators} | Evidence count: {evidence_count}"
                )
                if claim.applies_when:
                    line += f"\n  Applies when: {claim.applies_when}"
                if claim.does_not_apply_when:
                    line += f"\n  Does not apply when: {claim.does_not_apply_when}"

                entry_tokens = _estimate_tokens(line)
                if token_budget - entry_tokens < 100:
                    break
                token_budget -= entry_tokens
                claim_lines.append(line)

            if claim_lines:
                sections.append(
                    "## Knowledge Claims\n\n" + "\n\n".join(claim_lines)
                )

        # --- Illustrative example section ---
        if illustrative_examples:
            example = illustrative_examples[0]
            title = str(example.get("title", "이야기 예시"))
            analogy = str(example.get("analogy", ""))
            caution = str(example.get("caution", ""))
            example_lines = [
                "## Illustrative Example (not source evidence)",
                f"[E1] [예시] {title}",
                analogy,
            ]
            if caution:
                example_lines.append(f"주의: {caution}")
            example_text = "\n".join(example_lines)
            entry_tokens = _estimate_tokens(example_text)
            if token_budget - entry_tokens >= 100:
                sections.append(example_text)

        # --- Conflict hint ---
        if plan.require_conflict_search:
            sections.append(
                "## Note\n"
                "The user is comparing viewpoints. Ensure you surface "
                "disagreements between creators with evidence from both sides."
            )

        return "\n\n".join(sections), rendered

    # -- public API -------------------------------------------------------

    def build_context(
        self,
        transcript_results: list[RetrievalResult],
        okf_claims: list[KnowledgeClaim],
        plan: QueryPlan,
        illustrative_examples: list[dict[str, object]] | None = None,
    ) -> str:
        """Build a formatted context string from retrieval results.

        Args:
            transcript_results: Ranked transcript retrieval results.
            okf_claims: Relevant OKF knowledge claims.
            plan: The query plan (used for strategy hints).

        Returns:
            Formatted context string for the LLM prompt.
        """
        ctx, _ = self._build_context_core(
            transcript_results, okf_claims, plan, illustrative_examples,
        )
        return ctx

    def build_context_with_registry(
        self,
        transcript_results: list[RetrievalResult],
        okf_claims: list[KnowledgeClaim],
        plan: QueryPlan,
        illustrative_examples: list[dict[str, object]] | None = None,
    ) -> tuple[str, CitationRegistry]:
        """Build context string and a :class:`CitationRegistry` for the rendered evidence.

        Returns:
            ``(formatted_context, citation_registry)``
        """
        ctx, rendered = self._build_context_core(
            transcript_results, okf_claims, plan, illustrative_examples,
        )
        registry = CitationRegistry()
        registry.register(rendered)
        return ctx, registry
