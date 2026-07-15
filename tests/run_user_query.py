"""Run the user's query through the v2 chat pipeline and print the result."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time

# Ensure .env is loaded
from dotenv import load_dotenv
load_dotenv(override=False)

from httpx import ASGITransport, AsyncClient

sys.path.insert(0, "src")


async def main() -> None:
    from dating_rag.api.app import app, lifespan

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with lifespan(app):
            payload = {
                "schema_version": "2",
                "request_id": "test-req-001",
                "conversation_id": "test-conv-001",
                "question": "마음에 담아두고 있는 여인이 있는데 어떻게 다가가야될지 모르겠다.",
                "consent": {
                    "process_my_birth_data": False,
                    "store_my_data": False,
                    "process_partner_birth_data": False,
                },
            }
            print("=" * 70)
            print(f"QUERY: {payload['question']}")
            print("=" * 70)
            t0 = time.perf_counter()
            resp = await client.post("/v2/chat", json=payload, timeout=120)
            elapsed = time.perf_counter() - t0
            print(f"\nStatus: {resp.status_code}  ({elapsed:.1f}s)")
            print("-" * 70)
            body = resp.json()
            print(json.dumps(body, ensure_ascii=False, indent=2))
            print("-" * 70)

            # Also test legacy /chat
            print("\n\n[LEGACY /chat]")
            legacy_payload = {"question": payload["question"]}
            t0 = time.perf_counter()
            resp2 = await client.post("/chat", json=legacy_payload, timeout=120)
            elapsed2 = time.perf_counter() - t0
            print(f"Status: {resp2.status_code}  ({elapsed2:.1f}s)")
            print("-" * 70)
            body2 = resp2.json()
            print(json.dumps(body2, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
