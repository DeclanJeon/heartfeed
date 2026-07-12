"""Citation validation for generated answers.

Checks that source references ([S1], [S2], …) in the answer text correspond
to actual retrieval results, and that any timestamps mentioned are plausible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from dating_rag.domain.models import RetrievalResult


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# [S1], [C3], etc.
_LABEL_RE = re.compile(r"\[([SC])(\d+)\]")

# Timestamps like 12:34 or 1:02:03
_TIMESTAMP_RE = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b")

# Citation-style references: [Channel, "Title", MM:SS]
_FULL_CITATION_RE = re.compile(
    r"\[([^,\]]+),\s*\"?([^\",\]]+)\"?,\s*(\d{1,2}:\d{2}(?::\d{2})?)\]"
)


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------


@dataclass
class CitationValidation:
    """Result of validating citations in a generated answer."""

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cited_labels: set[str] = field(default_factory=set)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------


def validate_citations(
    answer: str,
    allowed_sources: list[RetrievalResult],
) -> CitationValidation:
    """Validate that citations in *answer* reference real sources.

    Checks performed:
    1. Every [S<N>] or [C<N>] label refers to an index that exists in
       *allowed_sources*.
    2. Timestamps embedded in ``[Channel, "Title", MM:SS]`` citations are
       syntactically valid (MM < 60, SS < 60).
    3. Warns when a source label is referenced but no matching full citation
       (channel + title + timestamp) is present.

    Args:
        answer: The generated answer text.
        allowed_sources: The retrieval results that were provided as context.
            Source labels are assigned as S1, S2, … (for transcripts) and
            C1, C2, … (for claims), matching ContextBuilder ordering.

    Returns:
        A :class:`CitationValidation` with collected errors and warnings.
    """
    result = CitationValidation()
    n_sources = len(allowed_sources)

    # Build the set of allowed labels.
    allowed_labels: set[str] = set()
    for i, src in enumerate(allowed_sources):
        prefix = "S" if src.source_type == "transcript" else "C"
        allowed_labels.add(f"{prefix}{i + 1}")

    # 1. Check [SN]/[CN] label references.
    found_labels: set[str] = set()
    for match in _LABEL_RE.finditer(answer):
        label = match.group(0)[1:-1]  # strip brackets → "S1"
        found_labels.add(label)
        if label not in allowed_labels:
            result.add_error(
                f"Answer references [{label}] but only {n_sources} sources were provided "
                f"(allowed: {', '.join(sorted(allowed_labels)) or 'none'})."
            )
    result.cited_labels = found_labels

    # 2. Validate timestamps in full citations [Channel, "Title", MM:SS].
    for match in _FULL_CITATION_RE.finditer(answer):
        minutes_str, seconds_str = match.group(3).split(":")[0], match.group(3).split(":")[1]
        try:
            minutes = int(minutes_str)
            seconds = int(seconds_str)
        except ValueError:
            result.add_error(f"Unparseable timestamp in citation: {match.group(0)!r}")
            continue
        if minutes >= 60 or seconds >= 60:
            result.add_error(
                f"Invalid timestamp {match.group(3)} in citation {match.group(0)!r} "
                f"(minutes and seconds must be < 60)."
            )

    # 3. Warn about labels cited without a full citation nearby.
    if found_labels and not _FULL_CITATION_RE.search(answer):
        for label in sorted(found_labels):
            result.add_warning(
                f"[{label}] is referenced but no full citation "
                f'[Channel, "Title", MM:SS] was found in the answer.'
            )

    # 4. Safety: flag if answer appears to make a clinical diagnosis.
    _diagnostic_patterns = re.compile(
        r"\b(you (?:have|are|seem to (?:have|be)) (?:an? )?)"
        r"(?:anxious|avoidant|disorganized|secure)\s+attachment\b",
        re.IGNORECASE,
    )
    if _diagnostic_patterns.search(answer):
        result.add_warning(
            "Answer appears to assign an attachment style diagnosis, which violates "
            "system prompt rule #3."
        )

    return result
