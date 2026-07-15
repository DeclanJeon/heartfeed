"""Tests for the evidence gate."""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from dating_rag.domain.models import EvidenceDecision, QueryPlan, RetrievalResult
from dating_rag.retrieval.evidence_gate import EvidenceGate, _count_unique_channels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(
    chunk_id: str = "c1",
    score: float = 0.05,
    channel_name: str = "ch1",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        score=score,
        metadata={"channel_name": channel_name},
    )


def _plan(intent: str = "general_advice") -> QueryPlan:
    return QueryPlan(intent=intent, topics=["test"])


def _write_thresholds(tmp: Path, overrides: dict | None = None) -> Path:
    """Write a thresholds YAML with optional overrides."""
    base = {
        "retrieval_thresholds": {
            "general_advice": {"min_score": 0.015, "min_count": 2, "min_diversity": 2},
            "specific_example": {"min_score": 0.012, "min_count": 1, "min_diversity": 1},
            "compare_viewpoints": {"min_score": 0.012, "min_count": 3, "min_diversity": 2},
            "high_risk": {"min_score": 0.010, "min_count": 1, "min_diversity": 1},
            "creator_lookup": {"min_score": 0.010, "min_count": 1, "min_diversity": 1},
        }
    }
    if overrides:
        base["retrieval_thresholds"].update(overrides)
    path = tmp / "thresholds.yaml"
    path.write_text(yaml.dump(base))
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvidenceGate:
    """Tests for EvidenceGate.accept()."""

    def test_empty_input_returns_no_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = _write_thresholds(Path(td))
            gate = EvidenceGate(thresholds_path=path)
            decision = gate.accept([], _plan())
            assert decision.reason_code == "no_candidates"
            assert decision.accepted == []
            assert decision.rejected == []
            assert decision.metrics["count"] == 0

    def test_all_below_threshold_returns_below_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = _write_thresholds(Path(td))
            gate = EvidenceGate(thresholds_path=path)
            candidates = [
                _result("c1", score=0.001, channel_name="ch1"),
                _result("c2", score=0.002, channel_name="ch2"),
            ]
            decision = gate.accept(candidates, _plan("general_advice"))
            assert decision.reason_code == "below_threshold"
            assert len(decision.accepted) == 0
            assert len(decision.rejected) == 2

    def test_enough_quality_same_channel_insufficient_diversity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = _write_thresholds(Path(td))
            gate = EvidenceGate(thresholds_path=path)
            # general_advice requires min_diversity=2, but all same channel
            candidates = [
                _result("c1", score=0.020, channel_name="same_channel"),
                _result("c2", score=0.018, channel_name="same_channel"),
                _result("c3", score=0.016, channel_name="same_channel"),
            ]
            decision = gate.accept(candidates, _plan("general_advice"))
            assert decision.reason_code == "insufficient_diversity"
            assert len(decision.accepted) == 3
            assert decision.metrics["unique_channels"] == 1

    def test_good_diverse_results_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = _write_thresholds(Path(td))
            gate = EvidenceGate(thresholds_path=path)
            candidates = [
                _result("c1", score=0.050, channel_name="ch1"),
                _result("c2", score=0.040, channel_name="ch2"),
                _result("c3", score=0.030, channel_name="ch3"),
            ]
            decision = gate.accept(candidates, _plan("general_advice"))
            assert decision.reason_code == "accepted"
            assert len(decision.accepted) == 3
            assert len(decision.rejected) == 0
            assert decision.metrics["unique_channels"] == 3

    def test_per_intent_threshold_differences(self) -> None:
        """compare_viewpoints needs min_count=3 and min_diversity=2."""
        with tempfile.TemporaryDirectory() as td:
            path = _write_thresholds(Path(td))
            gate = EvidenceGate(thresholds_path=path)
            # Only 2 results — compare_viewpoints needs 3
            candidates = [
                _result("c1", score=0.020, channel_name="ch1"),
                _result("c2", score=0.018, channel_name="ch2"),
            ]
            decision = gate.accept(candidates, _plan("compare_viewpoints"))
            assert decision.reason_code == "below_threshold"
            assert decision.metrics["min_count_threshold"] == 3

    def test_high_risk_lower_threshold(self) -> None:
        """high_risk has min_score=0.010, so lower scores pass."""
        with tempfile.TemporaryDirectory() as td:
            path = _write_thresholds(Path(td))
            gate = EvidenceGate(thresholds_path=path)
            candidates = [
                _result("c1", score=0.011, channel_name="ch1"),
            ]
            decision = gate.accept(candidates, _plan("high_risk"))
            assert decision.reason_code == "accepted"
            assert len(decision.accepted) == 1

    def test_specific_example_single_result(self) -> None:
        """specific_example needs only min_count=1."""
        with tempfile.TemporaryDirectory() as td:
            path = _write_thresholds(Path(td))
            gate = EvidenceGate(thresholds_path=path)
            candidates = [
                _result("c1", score=0.015, channel_name="ch1"),
            ]
            decision = gate.accept(candidates, _plan("specific_example"))
            assert decision.reason_code == "accepted"
            assert len(decision.accepted) == 1

    def test_unknown_intent_falls_back_to_general_advice(self) -> None:
        """An intent not in config falls back to general_advice thresholds."""
        with tempfile.TemporaryDirectory() as td:
            path = _write_thresholds(Path(td))
            gate = EvidenceGate(thresholds_path=path)
            candidates = [
                _result("c1", score=0.020, channel_name="ch1"),
            ]
            # "definition" is not in the thresholds config
            decision = gate.accept(candidates, _plan("definition"))
            # general_advice fallback: min_count=2, only 1 result
            assert decision.reason_code == "below_threshold"


class TestCountUniqueChannels:
    """Tests for the _count_unique_channels helper."""

    def test_empty(self) -> None:
        assert _count_unique_channels([]) == 0

    def test_all_same(self) -> None:
        results = [_result(f"c{i}", channel_name="ch1") for i in range(3)]
        assert _count_unique_channels(results) == 1

    def test_all_different(self) -> None:
        results = [_result(f"c{i}", channel_name=f"ch{i}") for i in range(4)]
        assert _count_unique_channels(results) == 4

    def test_missing_channel_name(self) -> None:
        r = RetrievalResult(chunk_id="c1", text="t", score=0.5, metadata={})
        assert _count_unique_channels([r]) == 0


class TestEvidenceDecisionModel:
    """Tests for the EvidenceDecision model itself."""

    def test_default_values(self) -> None:
        d = EvidenceDecision()
        assert d.accepted == []
        assert d.rejected == []
        assert d.reason_code == "accepted"
        assert d.metrics == {}

    def test_roundtrip(self) -> None:
        d = EvidenceDecision(
            accepted=[_result("c1", 0.5)],
            reason_code="accepted",
            metrics={"count": 1},
        )
        data = d.model_dump()
        assert data["reason_code"] == "accepted"
        assert len(data["accepted"]) == 1
