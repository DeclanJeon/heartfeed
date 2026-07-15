"""Claim retriever that loads knowledge claims from OKF concept files.

v1 stub — the ``data/okf/claims/`` directory is empty, so this module
parses concept markdown for claim references but returns an empty list
when referenced claim files do not yet exist.
"""

from __future__ import annotations

import re
from pathlib import Path

from dating_rag.domain.models import KnowledgeClaim

_DEFAULT_OKF_DIR = Path(__file__).resolve().parents[3] / "data" / "okf"
_CLAIM_REF_PATTERN = re.compile(r"\.\./claims/(claim-\d+\.md)")


class ClaimRetriever:
    """Retrieves structured knowledge claims linked from OKF concepts.

    Args:
        okf_dir: Root OKF data directory containing ``concepts/`` and
            ``claims/`` subdirectories.
    """

    def __init__(self, okf_dir: Path | None = None) -> None:
        self._okf_dir = okf_dir or _DEFAULT_OKF_DIR
        self._concepts_dir = self._okf_dir / "concepts"
        self._claims_dir = self._okf_dir / "claims"

    def retrieve_claims(
        self,
        topic: str,
        transcript_chunk_ids: list[str],
    ) -> list[KnowledgeClaim]:
        """Retrieve knowledge claims relevant to *topic*.

        In v1 this scans OKF concept markdown files for the given topic,
        parses claim references, and returns claims whose files exist on
        disk.  Returns an empty list when no matching concept files are
        found or when referenced claim files don't exist yet.

        Args:
            topic: The topic string to match against concept file names.
            transcript_chunk_ids: Chunk IDs for provenance (unused in v1 stub).

        Returns:
            List of :class:`KnowledgeClaim` objects.  Empty in v1 when no
            claim files exist.
        """
        if not self._concepts_dir.is_dir():
            return []

        claims: list[KnowledgeClaim] = []

        # Match topic against concept filenames (slug-based matching)
        topic_slug = topic.lower().replace(" ", "-")
        for concept_file in sorted(self._concepts_dir.glob("*.md")):
            stem = concept_file.stem.lower()
            if topic_slug not in stem and stem not in topic_slug:
                continue

            text = concept_file.read_text(encoding="utf-8")
            concept_id = concept_file.stem

            for match in _CLAIM_REF_PATTERN.finditer(text):
                claim_filename = match.group(1)
                claim_path = self._claims_dir / claim_filename
                if not claim_path.exists():
                    continue
                # In v1 we just validate existence; claim content parsing
                # will be implemented when claims are authored.
                claim_id = claim_filename.replace(".md", "")
                claims.append(
                    KnowledgeClaim(
                        claim_id=claim_id,
                        concept_id=concept_id,
                        statement="",  # Will be populated from claim file
                        evidence_chunk_ids=transcript_chunk_ids,
                    )
                )

        return claims
