#!/usr/bin/env python3
"""Run QA tests on generated questions through the v2 pipeline.

Runs a stratified sample and records metrics for each query.
"""
import asyncio
import json
import time
import sys
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=False)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from httpx import ASGITransport, AsyncClient


async def run_single(client: AsyncClient, q: dict) -> dict:
    """Run a single question through the v2 pipeline."""
    payload = {
        "schema_version": "2",
        "request_id": f"qa-{q['id']}",
        "conversation_id": f"qa-conv-{q['id']}",
        "question": q["question"],
        "consent": {
            "process_my_birth_data": False,
            "store_my_data": False,
            "process_partner_birth_data": False,
        },
    }

    t0 = time.perf_counter()
    try:
        resp = await client.post("/v2/chat", json=payload, timeout=120)
        elapsed = time.perf_counter() - t0
        body = resp.json()
        status = body.get("status", "error")
        answer = body.get("answer", {})
        evidence = body.get("evidence_claims", [])
        citations = body.get("citations", [])

        # Quality checks
        has_example = False
        has_evidence_quote = False
        has_timestamp_link = False

        if status == "answered":
            actions = answer.get("actions", [])
            for a in actions:
                if a.get("example"):
                    has_example = True
                if a.get("evidence_quote"):
                    has_evidence_quote = True
                eq = a.get("evidence_quote", "")
                if "youtube.com" in str(eq) and "t=" in str(eq) and "t=0" not in str(eq):
                    has_timestamp_link = True

        return {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "expected_status": q.get("expected_status", "answered"),
            "actual_status": status,
            "status_match": status == q.get("expected_status", "answered"),
            "elapsed_sec": round(elapsed, 1),
            "evidence_count": len(evidence),
            "citation_count": len(citations),
            "has_example": has_example,
            "has_evidence_quote": has_evidence_quote,
            "has_timestamp_link": has_timestamp_link,
            "answer_summary": answer.get("summary", "")[:200] if answer else "",
            "error": None,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "expected_status": q.get("expected_status", "answered"),
            "actual_status": "error",
            "status_match": False,
            "elapsed_sec": round(elapsed, 1),
            "evidence_count": 0,
            "citation_count": 0,
            "has_example": False,
            "has_evidence_quote": False,
            "has_timestamp_link": False,
            "answer_summary": "",
            "error": str(e)[:200],
        }


async def run_qa_batch(questions: list[dict], concurrency: int = 1) -> list[dict]:
    """Run questions through the pipeline with controlled concurrency."""
    from dating_rag.api.app import app, lifespan

    results = []
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with lifespan(app):
            total = len(questions)
            for i, q in enumerate(questions):
                result = await run_single(client, q)
                results.append(result)

                if (i + 1) % 10 == 0:
                    answered = sum(1 for r in results if r["actual_status"] == "answered")
                    print(f"  [{i+1}/{total}] answered={answered} errors={sum(1 for r in results if r['actual_status']=='error')}")

    return results


def select_sample(questions: list[dict], n: int = 200) -> list[dict]:
    """Select a stratified sample across categories."""
    import random
    rng = random.Random(42)

    # Group by category
    by_cat: dict[str, list[dict]] = {}
    for q in questions:
        by_cat.setdefault(q["category"], []).append(q)

    # Sample proportionally
    sample = []
    per_cat = max(1, n // len(by_cat))
    for cat, qs in by_cat.items():
        k = min(per_cat, len(qs))
        sample.extend(rng.sample(qs, k))

    # Fill remaining with random
    if len(sample) < n:
        remaining = [q for q in questions if q not in sample]
        sample.extend(rng.sample(remaining, min(n - len(sample), len(remaining))))

    return sample[:n]


def main() -> None:
    questions_path = Path(__file__).parent.parent / "data" / "eval" / "korean_10k_questions.json"
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(questions)} questions")

    # Select sample
    sample = select_sample(questions, n=200)
    print(f"Selected {len(sample)} for QA run")

    # Run
    print("\nRunning QA...")
    results = asyncio.run(run_qa_batch(sample))

    # Save results
    output_dir = Path(__file__).parent.parent / "data" / "eval"
    results_path = output_dir / "qa_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to {results_path}")

    # Summary
    total = len(results)
    statuses = Counter(r["actual_status"] for r in results)
    cats = Counter(r["category"] for r in results)
    answered = [r for r in results if r["actual_status"] == "answered"]
    avg_time = sum(r["elapsed_sec"] for r in results) / total if total else 0
    avg_evidence = sum(r["evidence_count"] for r in answered) / len(answered) if answered else 0
    avg_citations = sum(r["citation_count"] for r in answered) / len(answered) if answered else 0
    with_example = sum(1 for r in answered if r["has_example"])
    with_evidence_quote = sum(1 for r in answered if r["has_evidence_quote"])
    with_ts_link = sum(1 for r in answered if r["has_timestamp_link"])
    status_match = sum(1 for r in results if r["status_match"])

    print(f"\n{'='*60}")
    print(f"QA SUMMARY ({total} questions)")
    print(f"{'='*60}")
    print(f"\nStatus distribution:")
    for s, c in sorted(statuses.items(), key=lambda x: -x[1]):
        print(f"  {s}: {c} ({c/total*100:.1f}%)")

    print(f"\nCategory distribution:")
    for cat, c in sorted(cats.items(), key=lambda x: -x[1]):
        cat_answered = sum(1 for r in results if r["category"] == cat and r["actual_status"] == "answered")
        print(f"  {cat}: {c} total, {cat_answered} answered ({cat_answered/c*100:.0f}%)")

    print(f"\nQuality metrics (answered only, n={len(answered)}):")
    print(f"  Avg response time: {avg_time:.1f}s")
    print(f"  Avg evidence count: {avg_evidence:.1f}")
    print(f"  Avg citation count: {avg_citations:.1f}")
    print(f"  Has concrete example: {with_example}/{len(answered)} ({with_example/len(answered)*100:.0f}%)")
    print(f"  Has evidence quote: {with_evidence_quote}/{len(answered)} ({with_evidence_quote/len(answered)*100:.0f}%)")
    print(f"  Has timestamp link (t>0): {with_ts_link}/{len(answered)} ({with_ts_link/len(answered)*100:.0f}%)")
    print(f"  Status match (expected vs actual): {status_match}/{total} ({status_match/total*100:.0f}%)")

    # Errors
    errors = [r for r in results if r["actual_status"] == "error"]
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:5]:
            print(f"  {e['id']}: {e['error']}")


if __name__ == "__main__":
    main()
