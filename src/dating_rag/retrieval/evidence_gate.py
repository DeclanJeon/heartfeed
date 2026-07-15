"""Evidence gate that filters retrieval results by quality thresholds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dating_rag.domain.models import EvidenceDecision, QueryPlan, RetrievalResult

_DEFAULT_THRESHOLTS_PATH = Path(__file__).resolve().parents[3] / "config" / "retrieval_thresholds.yaml"


def _load_thresholds(path: Path | None = None) -> dict[str, dict[str, float | int]]:
    """Load per-intent threshold config from YAML."""
    path = path or _DEFAULT_THRESHOLTS_PATH
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("retrieval_thresholds", {})


def _count_unique_channels(results: list[RetrievalResult]) -> int:
    """Count distinct channel_name values across results."""
    channels: set[str] = set()
    for r in results:
        ch = r.metadata.get("channel_name")
        if ch:
            channels.add(str(ch))
    return len(channels)


class EvidenceGate:
    """Filters retrieval results through per-intent quality thresholds.

    Args:
        thresholds_path: Optional path to a custom thresholds YAML.
            Defaults to ``config/retrieval_thresholds.yaml``.
    """

    def __init__(self, thresholds_path: Path | None = None) -> None:
        self._thresholds = _load_thresholds(thresholds_path)

    def accept(
        self,
        candidates: list[RetrievalResult],
        plan: QueryPlan,
        thresholds: dict[str, dict[str, float | int]] | None = None,
    ) -> EvidenceDecision:
        """Evaluate candidates against per-intent thresholds.

        Args:
            candidates: Retrieval results to evaluate.
            plan: The query plan (carries intent).
            thresholds: Optional override for threshold config.
                When *None*, uses the YAML-loaded defaults.

        Returns:
            An :class:`EvidenceDecision` with accepted/rejected lists,
            a reason code, and quality metrics.
        """
        cfg = thresholds or self._thresholds

        if not candidates:
            return EvidenceDecision(
                accepted=[],
                rejected=[],
                reason_code="no_candidates",
                metrics={"count": 0, "max_score": 0.0, "unique_channels": 0},
            )

        intent = plan.intent or "general_advice"
        intent_cfg = cfg.get(intent, cfg.get("general_advice", {}))
        min_score = float(intent_cfg.get("min_score", 0.015))
        min_count = int(intent_cfg.get("min_count", 1))
        min_diversity = int(intent_cfg.get("min_diversity", 1))

        # Sort candidates descending by score for consistent processing
        sorted_candidates = sorted(candidates, key=lambda r: r.score, reverse=True)

        accepted: list[RetrievalResult] = []
        rejected: list[RetrievalResult] = []

        for result in sorted_candidates:
            if result.score >= min_score:
                accepted.append(result)
            else:
                rejected.append(result)

        unique_channels = _count_unique_channels(accepted)

        metrics: dict[str, float | int] = {
            "count": len(accepted),
            "max_score": max((r.score for r in accepted), default=0.0),
            "unique_channels": unique_channels,
            "min_score_threshold": min_score,
            "min_count_threshold": min_count,
            "min_diversity_threshold": min_diversity,
        }

        if not accepted:
            return EvidenceDecision(
                accepted=[],
                rejected=rejected,
                reason_code="below_threshold",
                metrics=metrics,
            )

        if len(accepted) < min_count:
            return EvidenceDecision(
                accepted=accepted,
                rejected=rejected,
                reason_code="below_threshold",
                metrics=metrics,
            )

        if unique_channels < min_diversity:
            return EvidenceDecision(
                accepted=accepted,
                rejected=rejected,
                reason_code="insufficient_diversity",
                metrics=metrics,
            )

        return EvidenceDecision(
            accepted=accepted,
            rejected=rejected,
            reason_code="accepted",
            metrics=metrics,
        )
