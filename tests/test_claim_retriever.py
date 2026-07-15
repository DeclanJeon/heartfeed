"""Tests for the claim retriever."""

from __future__ import annotations

import tempfile
from pathlib import Path

from dating_rag.domain.models import KnowledgeClaim
from dating_rag.retrieval.claim_retriever import ClaimRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_okf(
    tmp: Path,
    concepts: dict[str, str] | None = None,
    claims: dict[str, str] | None = None,
) -> Path:
    """Create an OKF directory structure in tmp."""
    okf = tmp / "okf"
    concepts_dir = okf / "concepts"
    claims_dir = okf / "claims"
    concepts_dir.mkdir(parents=True)
    claims_dir.mkdir(parents=True)

    # Always keep .gitkeep in claims
    (claims_dir / ".gitkeep").write_text("")

    for name, content in (concepts or {}).items():
        (concepts_dir / name).write_text(content)

    for name, content in (claims or {}).items():
        (claims_dir / name).write_text(content)

    return okf


_CONCEPT_WITH_CLAIMS = """\
---
type: concept
title: Dating app success strategies
resource: dating://concept/dating-app-success
tags:
  - dating-apps
  - online-dating
timestamp: 2026-07-12
---

# Definition

Dating app success strategies.

# Claims

- [Claim: Profile photos showing genuine smiles get significantly more matches](../claims/claim-028.md)
- [Claim: Writing specific interests attracts more compatible matches](../claims/claim-029.md)
"""

_CONCEPT_WITHOUT_CLAIMS = """\
---
type: concept
title: Simple topic
resource: dating://concept/simple-topic
tags:
  - basic
timestamp: 2026-07-12
---

# Definition

A simple topic with no claims.
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClaimRetriever:
    """Tests for ClaimRetriever.retrieve_claims()."""

    def test_empty_concepts_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            okf = _make_okf(Path(td))
            retriever = ClaimRetriever(okf_dir=okf)
            results = retriever.retrieve_claims("anything", [])
            assert results == []

    def test_non_matching_topic_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            okf = _make_okf(
                Path(td),
                concepts={"dating-app-success.md": _CONCEPT_WITH_CLAIMS},
            )
            retriever = ClaimRetriever(okf_dir=okf)
            # Topic doesn't match any concept filename
            results = retriever.retrieve_claims("quantum physics", [])
            assert results == []

    def test_matching_concept_with_existing_claims(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            okf = _make_okf(
                Path(td),
                concepts={"dating-app-success.md": _CONCEPT_WITH_CLAIMS},
                claims={"claim-028.md": "claim content"},
            )
            retriever = ClaimRetriever(okf_dir=okf)
            results = retriever.retrieve_claims("dating-app-success", ["chunk1"])
            # Only claim-028.md exists on disk
            assert len(results) == 1
            assert results[0].claim_id == "claim-028"
            assert results[0].concept_id == "dating-app-success"
            assert results[0].evidence_chunk_ids == ["chunk1"]

    def test_matching_concept_no_existing_claims(self) -> None:
        """Concept file references claims but none exist on disk (v1 stub)."""
        with tempfile.TemporaryDirectory() as td:
            okf = _make_okf(
                Path(td),
                concepts={"dating-app-success.md": _CONCEPT_WITH_CLAIMS},
            )
            retriever = ClaimRetriever(okf_dir=okf)
            results = retriever.retrieve_claims("dating-app-success", [])
            assert results == []

    def test_concept_without_claims_section(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            okf = _make_okf(
                Path(td),
                concepts={"simple-topic.md": _CONCEPT_WITHOUT_CLAIMS},
            )
            retriever = ClaimRetriever(okf_dir=okf)
            results = retriever.retrieve_claims("simple-topic", ["c1"])
            assert results == []

    def test_returns_knowledge_claim_type(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            okf = _make_okf(
                Path(td),
                concepts={"dating-app-success.md": _CONCEPT_WITH_CLAIMS},
                claims={"claim-028.md": "content", "claim-029.md": "content"},
            )
            retriever = ClaimRetriever(okf_dir=okf)
            results = retriever.retrieve_claims("dating-app-success", ["c1", "c2"])
            assert len(results) == 2
            for claim in results:
                assert isinstance(claim, KnowledgeClaim)
            ids = {c.claim_id for c in results}
            assert ids == {"claim-028", "claim-029"}

    def test_topic_matching_is_slug_based(self) -> None:
        """Topic with spaces matches filename with hyphens."""
        with tempfile.TemporaryDirectory() as td:
            okf = _make_okf(
                Path(td),
                concepts={"dating-app-success.md": _CONCEPT_WITH_CLAIMS},
                claims={"claim-028.md": "content"},
            )
            retriever = ClaimRetriever(okf_dir=okf)
            results = retriever.retrieve_claims("dating app success", [])
            assert len(results) == 1

    def test_concepts_dir_not_found(self) -> None:
        """Missing concepts directory returns empty list."""
        with tempfile.TemporaryDirectory() as td:
            okf = Path(td) / "nonexistent"
            retriever = ClaimRetriever(okf_dir=okf)
            results = retriever.retrieve_claims("anything", [])
            assert results == []
