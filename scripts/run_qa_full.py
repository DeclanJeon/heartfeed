#!/usr/bin/env python3
"""Run all 10000 questions through the v2 pipeline with parallel execution.

Uses asyncio.Semaphore for controlled concurrency.
Saves results incrementally to avoid losing progress on timeout.
"""
import asyncio
import json
import time
import sys
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from httpx import ASGITransport, AsyncClient

CONCURRENCY = 2  # parallel requests (CPU embedding bottleneck)
RESULTS_PATH = Path(__file__).parent.parent / "data" / "eval" / "qa_full_results.jsonl"


async def run_single(client: AsyncClient, q: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        payload = {
            "schema_version": "2",
            "request_id": f"qa-{q['id']}",
            "conversation_id": f"qa-{q['id']}",
            "question": q["question"],
            "consent": {"process_my_birth_data": False, "store_my_data": False, "process_partner_birth_data": False},
        }
        t0 = time.perf_counter()
        try:
            resp = await client.post("/v2/chat", json=payload, timeout=180)
            elapsed = time.perf_counter() - t0
            body = resp.json()
            status = body.get("status", "error")
            answer = body.get("answer", {})
            actions = answer.get("actions", []) if answer else []
            return {
                "id": q["id"], "cat": q["category"],
                "expected": q.get("expected_status", "answered"),
                "status": status, "elapsed": round(elapsed, 1),
                "citations": len(body.get("citations", [])),
                "has_example": any(a.get("example") for a in actions),
                "has_evidence_quote": any(a.get("evidence_quote") for a in actions),
                "summary": answer.get("summary", "")[:150] if answer else "",
            }
        except Exception as e:
            return {
                "id": q["id"], "cat": q["category"],
                "expected": q.get("expected_status", "answered"),
                "status": "error", "elapsed": round(time.perf_counter() - t0, 1),
                "error": str(e)[:100],
            }


async def main():
    from dating_rag.api.app import app, lifespan

    qpath = Path(__file__).parent.parent / "data" / "eval" / "korean_10k_questions.json"
    questions = json.loads(qpath.read_text(encoding="utf-8"))
    total = len(questions)
    print(f"Total questions: {total}, concurrency: {CONCURRENCY}")

    # Check for existing progress
    done_ids = set()
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except:
                    pass
    remaining = [q for q in questions if q["id"] not in done_ids]
    print(f"Already done: {len(done_ids)}, remaining: {len(remaining)}")

    if not remaining:
        print("All questions already processed!")
        return

    sem = asyncio.Semaphore(CONCURRENCY)
    transport = ASGITransport(app=app)
    count = 0
    t_start = time.perf_counter()

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with lifespan(app):
            # Process in batches of CONCURRENCY * 2 for incremental saving
            batch_size = CONCURRENCY * 4
            for batch_start in range(0, len(remaining), batch_size):
                batch = remaining[batch_start:batch_start + batch_size]
                tasks = [run_single(client, q, sem) for q in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Save results incrementally
                with open(RESULTS_PATH, "a") as f:
                    for r in results:
                        if isinstance(r, Exception):
                            r = {"id": "unknown", "status": "error", "error": str(r)[:100]}
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                        count += 1

                elapsed = time.perf_counter() - t_start
                rate = count / elapsed if elapsed > 0 else 0
                eta = (len(remaining) - count) / rate if rate > 0 else 0
                answered = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "answered")
                print(f"  [{count}/{len(remaining)}] batch answered={answered}/{len(batch)} rate={rate:.1f}/s ETA={eta/60:.0f}min")

    total_elapsed = time.perf_counter() - t_start
    print(f"\nDone! {count} questions in {total_elapsed/60:.1f} minutes")


if __name__ == "__main__":
    asyncio.run(main())
