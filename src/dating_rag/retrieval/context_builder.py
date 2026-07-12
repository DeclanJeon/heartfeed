"""Context builder for formatting retrieval results into LLM context."""

from __future__ import annotations

from dating_rag.domain.models import (
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

    def build_context(
        self,
        transcript_results: list[RetrievalResult],
        okf_claims: list[KnowledgeClaim],
        plan: QueryPlan,
    ) -> str:
        """Build a formatted context string from retrieval results.

        Args:
            transcript_results: Ranked transcript retrieval results.
            okf_claims: Relevant OKF knowledge claims.
            plan: The query plan (used for strategy hints).

        Returns:
            Formatted context string for the LLM prompt.
        """
        sections: list[str] = []
        token_budget = self.max_tokens

        # --- Transcript evidence section ---
        if transcript_results and plan.use_transcripts:
            evidence_lines: list[str] = []
            for i, result in enumerate(transcript_results[: self.max_chunks]):
                label = f"S{i + 1}"
                channel = result.metadata.get("channel_name", "Unknown")
                title = result.metadata.get("title", "")
                timestamp_url = str(result.metadata.get("timestamp_url", ""))

                header = f"[{label}] ({channel}"
                if title:
                    header += f" — \"{title}\""
                header += ")"

                text = result.text.strip()
                source_line = _format_timestamp_url(timestamp_url)

                entry = header + "\n" + text
                if source_line:
                    entry += "\n" + source_line

                entry_tokens = _estimate_tokens(entry)
                if token_budget - entry_tokens < 200:
                    break
                token_budget -= entry_tokens
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
                creators = ", ".join(claim.creator_ids) if claim.creator_ids else "unknown"
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

        # --- Conflict hint ---
        if plan.require_conflict_search:
            sections.append(
                "## Note\n"
                "The user is comparing viewpoints. Ensure you surface "
                "disagreements between creators with evidence from both sides."
            )

        return "\n\n".join(sections)
