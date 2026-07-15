#!/usr/bin/env python3
"""Background QA runner - sequential, incremental save."""
import json, time, sys
from pathlib import Path

# Setup paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

import asyncio
from httpx import ASGITransport, AsyncClient

RESULTS_PATH = ROOT / "data" / "eval" / "qa_full_results.jsonl"


def load_done_ids():
    done = set()
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except:
                    pass
    return done


async def main():
    from dating_rag.api.app import app, lifespan

    questions = json.loads((ROOT / "data" / "eval" / "korean_10k_questions.json").read_text())
    done_ids = load_done_ids()
    remaining = [q for q in questions if q["id"] not in done_ids]
    total_remaining = len(remaining)
    print(f"Total: {len(questions)}, Done: {len(done_ids)}, Remaining: {total_remaining}", flush=True)

    if not remaining:
        print("All done!")
        return

    transport = ASGITransport(app=app)
    count = 0
    errors = 0
    t_start = time.perf_counter()

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with lifespan(app):
            for q in remaining:
                payload = {
                    "schema_version": "2",
                    "request_id": f"qa-{q['id']}",
                    "conversation_id": f"qa-{q['id']}",
                    "question": q["question"],
                    "consent": {
                        "process_my_birth_data": False,
                        "store_my_data": False,
                        "process_partner_birth_data": False,
                    },
                }
                t0 = time.perf_counter()
                try:
                    resp = await client.post("/v2/chat", json=payload, timeout=300)
                    elapsed = time.perf_counter() - t0
                    body = resp.json()
                    status = body.get("status", "error")
                    answer = body.get("answer", {})
                    actions = answer.get("actions", []) if answer else []
                    result = {
                        "id": q["id"],
                        "cat": q["category"],
                        "expected": q.get("expected_status", "answered"),
                        "status": status,
                        "elapsed": round(elapsed, 1),
                        "citations": len(body.get("citations", [])),
                        "has_example": any(a.get("example") for a in actions),
                        "has_evidence_quote": any(a.get("evidence_quote") for a in actions),
                        "summary": answer.get("summary", "")[:150] if answer else "",
                    }
                    if status == "error":
                        errors += 1
                except Exception as e:
                    elapsed = time.perf_counter() - t0
                    result = {
                        "id": q["id"],
                        "cat": q["category"],
                        "status": "error",
                        "elapsed": round(elapsed, 1),
                        "error": str(e)[:100],
                    }
                    errors += 1

                with open(RESULTS_PATH, "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                count += 1

                if count % 10 == 0:
                    elapsed_total = time.perf_counter() - t_start
                    rate = count / elapsed_total
                    eta_min = (total_remaining - count) / rate / 60
                    print(
                        f"[{count}/{total_remaining}] "
                        f"rate={rate:.2f}/s ETA={eta_min:.0f}min errors={errors}",
                        flush=True,
                    )

    total_time = time.perf_counter() - t_start
    print(f"Done! {count} questions in {total_time/60:.1f}min, errors={errors}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
