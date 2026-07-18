#!/usr/bin/env python3
"""Latency bench harness for Rescue (mock-friendly).

Usage:
  LLM_MOCK=1 python scripts/bench_latency.py
  BACKEND_URL=http://localhost:8000/v2/chat python scripts/bench_latency.py
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from pathlib import Path

import httpx

SEED = [
    "이별 3일차인데 밤마다 연락하고 싶어요. 지금 어떻게 버틸까요?",
    "전 연인이 스토리 올리면 확인하게 됩니다. 끊는 방법이 있을까요?",
    "친구가 재회 앱을 추천해요. 들어야 할까요?",
    "상대가 먼저 연락이 왔습니다. 답장 기준을 어떻게 잡을까요?",
    "이별 후 자책이 심합니다. 오늘 할 수 있는 일 3가지만 알려주세요.",
]


async def one(client: httpx.AsyncClient, url: str, q: str, i: int) -> dict:
    payload = {
        "schema_version": "2",
        "request_id": f"bench-{i}",
        "conversation_id": "bench",
        "question": q,
        "track": {
            "id": "brt14",
            "day_index": min(i, 13),
            "contact_status": "no_contact",
            "primary_goal": "stabilize",
        },
        "consent": {
            "personalize_with_mbti": False,
            "personalize_with_observations": False,
            "cultural_saju_reflection": False,
            "process_partner_birth_data": False,
        },
    }
    t0 = time.perf_counter()
    try:
        r = await client.post(url, json=payload)
        ms = (time.perf_counter() - t0) * 1000
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        return {
            "ok": r.status_code < 500,
            "status_code": r.status_code,
            "status": body.get("status"),
            "ms": ms,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "ms": (time.perf_counter() - t0) * 1000}


async def main() -> None:
    url = os.getenv("BACKEND_URL", "http://localhost:8000/v2/chat")
    runs = int(os.getenv("BENCH_RUNS", "3"))
    results = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        idx = 0
        for _ in range(runs):
            for q in SEED:
                results.append(await one(client, url, q, idx))
                idx += 1
    ms = [r["ms"] for r in results if "ms" in r]
    ms_sorted = sorted(ms)
    p95 = ms_sorted[int(len(ms_sorted) * 0.95) - 1] if ms_sorted else None
    report = {
        "n": len(results),
        "ok_rate": sum(1 for r in results if r.get("ok")) / max(1, len(results)),
        "p50_ms": statistics.median(ms) if ms else None,
        "p95_ms": p95,
        "mean_ms": statistics.mean(ms) if ms else None,
        "target_p95_ms": 15000,
        "pass_p95": bool(p95 is not None and p95 <= 15000),
        "samples": results[:10],
    }
    out = Path(__file__).resolve().parents[2] / "artifacts" / "rescue" / "bench-latency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
