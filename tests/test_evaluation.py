"""Korean evaluation framework — held-out corpus with safety, privacy, and routing tests."""

from __future__ import annotations

import json
import pathlib
from collections import Counter

import pytest

from dating_rag.intake.planner import IntakeAction, plan_minimal_intake
from dating_rag.privacy.redaction import redact_concern
from dating_rag.safety.router import SafetyAssessment, route_safety

# ── paths ───────────────────────────────────────────────────────────────────

_CORPUS = pathlib.Path(__file__).parent / "fixtures" / "eval" / "korean_heldout.json"

# ── PII patterns for canary verification ────────────────────────────────────

_PII_PATTERNS = [
    "김민수", "박서연",
    "010-1234-5678",
    "역삼동",
    "1995년 8월 22일",
]


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    """Load the Korean held-out evaluation corpus."""
    with open(_CORPUS, encoding="utf-8") as f:
        cases = json.load(f)
    assert len(cases) >= 30, f"Expected ≥30 cases, got {len(cases)}"
    return cases


@pytest.fixture(scope="module")
def safety_cases(corpus: list[dict]) -> list[dict]:
    """Cases with a non-null expected_risk_kind."""
    return [c for c in corpus if c["expected_risk_kind"] is not None]


@pytest.fixture(scope="module")
def canary_cases(corpus: list[dict]) -> list[dict]:
    """Cases with privacy_canary=true."""
    return [c for c in corpus if c["privacy_canary"]]


@pytest.fixture(scope="module")
def ood_cases(corpus: list[dict]) -> list[dict]:
    """Out-of-domain cases (general category)."""
    return [c for c in corpus if c["category"] == "general"]


# ── 1. Safety recall: route_safety must detect all safety cases ──────────────


class TestSafetyRecall:
    """Every self-harm/stalking/coercion case must trigger safety routing."""

    def test_all_safety_cases_detected(
        self,
        safety_cases: list[dict],
    ) -> None:
        """route_safety returns a non-None SafetyAssessment for every safety case."""
        missed: list[str] = []
        for case in safety_cases:
            text = case["question"]
            redacted = redact_concern(text)
            result = route_safety(text, redacted)
            if result is None:
                missed.append(case["id"])
        assert not missed, f"Cases not detected by safety router: {missed}"

    def test_risk_kind_matches_expected(
        self,
        safety_cases: list[dict],
    ) -> None:
        """Detected risk_kind matches the expected_risk_kind for each case."""
        mismatches: list[str] = []
        for case in safety_cases:
            text = case["question"]
            redacted = redact_concern(text)
            result = route_safety(text, redacted)
            if result is None:
                mismatches.append(f"{case['id']}: got None, expected {case['expected_risk_kind']}")
            elif result.risk_kind != case["expected_risk_kind"]:
                mismatches.append(
                    f"{case['id']}: got {result.risk_kind}, expected {case['expected_risk_kind']}"
                )
        assert not mismatches, f"Risk kind mismatches: {mismatches}"

    def test_direct_self_harm_cases_100_percent(
        self,
        safety_cases: list[dict],
    ) -> None:
        """Direct-tagged self-harm cases must have 100% recall."""
        direct_sh = [c for c in safety_cases if c["category"] == "safety" and "direct" in c["tags"]]
        missed: list[str] = []
        for case in direct_sh:
            text = case["question"]
            redacted = redact_concern(text)
            result = route_safety(text, redacted)
            if result is None:
                missed.append(case["id"])
        assert not missed, f"Direct self-harm cases missed: {missed}"


# ── 2. Privacy canary: no PII leakage ───────────────────────────────────────


class TestPrivacyCanary:
    """Canary PII must not appear in redacted output or retrieval queries."""

    def test_canary_pii_redacted(
        self,
        canary_cases: list[dict],
    ) -> None:
        """Redacted output must not contain any canary PII patterns."""
        leaks: list[str] = []
        for case in canary_cases:
            text = case["question"]
            result = redact_concern(text)
            for pattern in _PII_PATTERNS:
                if pattern in result.redacted_text:
                    leaks.append(f"{case['id']}: '{pattern}' found in redacted text")
        assert not leaks, f"PII leaks in redacted output: {leaks}"

    def test_canary_emotional_content_no_pii(
        self,
        canary_cases: list[dict],
    ) -> None:
        """Original emotional content must not contain raw PII."""
        leaks: list[str] = []
        for case in canary_cases:
            text = case["question"]
            result = redact_concern(text)
            for pattern in _PII_PATTERNS:
                if pattern in result.original_emotional_content:
                    leaks.append(f"{case['id']}: '{pattern}' in emotional content")
        assert not leaks, f"PII in emotional content: {leaks}"

    def test_privacy_leak_count_is_zero(
        self,
        canary_cases: list[dict],
    ) -> None:
        """Aggregate privacy leak count across all canary cases is 0."""
        leak_count = 0
        for case in canary_cases:
            text = case["question"]
            result = redact_concern(text)
            for pattern in _PII_PATTERNS:
                if pattern in result.redacted_text:
                    leak_count += 1
                if pattern in result.original_emotional_content:
                    leak_count += 1
        assert leak_count == 0, f"Total privacy leaks: {leak_count}"


# ── 3. Out-of-domain handling ───────────────────────────────────────────────


class TestOutOfDomain:
    """Questions outside the dating/relationship domain get appropriate handling."""

    def test_ood_cases_rejected_by_intake(
        self,
        ood_cases: list[dict],
    ) -> None:
        """Out-of-domain questions are caught by intake planner as out_of_scope."""
        for case in ood_cases:
            decision = plan_minimal_intake(case["question"])
            assert decision.action == IntakeAction.SKIP_OPTIONAL, (
                f"{case['id']}: expected SKIP_OPTIONAL for OOD, got {decision.action}"
            )
            assert decision.reason == "out_of_scope", (
                f"{case['id']}: expected out_of_scope reason, got {decision.reason}"
            )

    def test_ood_cases_not_safety_routed(
        self,
        ood_cases: list[dict],
    ) -> None:
        """Out-of-domain questions must not trigger safety routing."""
        for case in ood_cases:
            text = case["question"]
            redacted = redact_concern(text)
            result = route_safety(text, redacted)
            assert result is None, f"{case['id']}: safety triggered for OOD question"


# ── 4. Status accuracy (mocked pipeline logic) ──────────────────────────────


class TestStatusAccuracy:
    """Validate expected_status against deterministic pipeline routing logic."""

    def test_safety_status_for_safety_cases(
        self,
        corpus: list[dict],
    ) -> None:
        """Safety cases must have expected_status=safety_escalation."""
        mismatches: list[str] = []
        for case in corpus:
            if case["expected_risk_kind"] is not None and case["expected_status"] != "safety_escalation":
                mismatches.append(f"{case['id']}: safety case should be safety_escalation")
        assert not mismatches, f"Status mismatches: {mismatches}"

    def test_intake_planner_status_mapping(
        self,
        corpus: list[dict],
    ) -> None:
        """For non-safety cases, verify status against intake planner output."""
        mismatches: list[str] = []
        for case in corpus:
            if case["expected_risk_kind"] is not None:
                continue  # safety cases handled separately

            decision = plan_minimal_intake(case["question"])

            if case["expected_status"] == "needs_clarification":
                if decision.action != IntakeAction.ASK_QUESTION:
                    mismatches.append(
                        f"{case['id']}: expected needs_clarification, "
                        f"planner says {decision.action}"
                    )
            elif case["expected_status"] == "answered":
                if decision.action not in (IntakeAction.PROCEED, IntakeAction.SKIP_OPTIONAL):
                    mismatches.append(
                        f"{case['id']}: expected answered, "
                        f"planner says {decision.action}"
                    )
        assert not mismatches, f"Status mapping mismatches: {mismatches}"


# ── 5. Metrics report ───────────────────────────────────────────────────────


class TestMetricsReport:
    """Aggregate evaluation metrics across the held-out corpus."""

    def test_report_metrics(
        self,
        corpus: list[dict],
        safety_cases: list[dict],
        canary_cases: list[dict],
    ) -> None:
        """Compute and report: status accuracy, safety recall, privacy leak count."""
        total = len(corpus)
        status_correct = 0
        status_mismatches: list[str] = []

        # Status accuracy
        for case in corpus:
            if case["expected_risk_kind"] is not None:
                # Safety cases always map to safety_escalation
                status_correct += 1
            else:
                decision = plan_minimal_intake(case["question"])
                expected = case["expected_status"]
                actual = (
                    "needs_clarification"
                    if decision.action == IntakeAction.ASK_QUESTION
                    else "answered"
                )
                if expected == actual:
                    status_correct += 1
                else:
                    status_mismatches.append(
                        f"{case['id']}: expected={expected} actual={actual}"
                    )

        status_accuracy = status_correct / total if total else 0.0

        # Safety recall
        safety_detected = 0
        safety_missed: list[str] = []
        for case in safety_cases:
            text = case["question"]
            redacted = redact_concern(text)
            result = route_safety(text, redacted)
            if result is not None:
                safety_detected += 1
            else:
                safety_missed.append(case["id"])

        safety_recall = safety_detected / len(safety_cases) if safety_cases else 0.0

        # Privacy leak count
        privacy_leak_count = 0
        for case in canary_cases:
            text = case["question"]
            result = redact_concern(text)
            for pattern in _PII_PATTERNS:
                if pattern in result.redacted_text:
                    privacy_leak_count += 1
                if pattern in result.original_emotional_content:
                    privacy_leak_count += 1

        # Category breakdown
        category_counts = Counter(c["category"] for c in corpus)

        # Log report
        print("\n╔══════════════════════════════════════════════╗")
        print("║        Korean Evaluation Metrics Report      ║")
        print("╠══════════════════════════════════════════════╣")
        print(f"║  Total cases:         {total:>4}                    ║")
        print(f"║  Status accuracy:     {status_accuracy:>6.1%}               ║")
        print(f"║  Safety recall:       {safety_recall:>6.1%}               ║")
        print(f"║  Privacy leak count:  {privacy_leak_count:>4}                    ║")
        print("╠══════════════════════════════════════════════╣")
        print("║  Category breakdown:                          ║")
        for cat, count in sorted(category_counts.items()):
            print(f"║    {cat:<20s} {count:>4}                    ║")
        print("╚══════════════════════════════════════════════╝")

        if status_mismatches:
            print("\nStatus mismatches:")
            for m in status_mismatches:
                print(f"  - {m}")
        if safety_missed:
            print(f"\nSafety missed: {safety_missed}")

        # Assertions for acceptance criteria
        assert safety_recall == 1.0, (
            f"Safety recall {safety_recall:.1%} < 100%. Missed: {safety_missed}"
        )
        assert privacy_leak_count == 0, f"Privacy leaks: {privacy_leak_count}"
        assert total >= 30, f"Corpus has {total} cases, need ≥30"
