#!/usr/bin/env python3
"""Fix timestamp_url in Qdrant for chunks that have start_seconds=0.

Uses catalog video duration and chunk count to estimate proper timestamps.
Handles corrupted array payloads from previous failed bulk update.
"""
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

QDRANT_URL = "http://localhost:6333"
COLLECTION = "datewise_transcripts"
CATALOG_PATH = Path(__file__).parent.parent / "data" / "catalog-index.json"


def load_catalog() -> dict[str, dict]:
    if not CATALOG_PATH.exists():
        return {}
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return {str(v["id"]): v for v in data.get("videos", []) if v.get("id")}


def scroll_all_points() -> list[dict]:
    points = []
    offset = None
    while True:
        body: dict = {"limit": 100, "with_payload": True, "with_vector": False}
        if offset:
            body["offset"] = offset
        resp = httpx.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll", json=body)
        result = resp.json()["result"]
        batch = result.get("points", [])
        if not batch:
            break
        points.extend(batch)
        offset = result.get("next_page_offset")
        if not offset:
            break
    return points


def compute_timestamps(points: list[dict], catalog: dict[str, dict]) -> list[dict]:
    video_points: dict[str, list[dict]] = {}
    for p in points:
        vid = p["payload"].get("video_id", "")
        if not vid:
            continue
        video_points.setdefault(vid, []).append(p)

    updates = []
    for vid, vpoints in video_points.items():
        cat = catalog.get(vid, {})
        duration = cat.get("duration", 0)
        base_url = cat.get("url", "")
        vpoints.sort(key=lambda p: p["payload"].get("chunk_index", 0))
        n_chunks = len(vpoints)

        for p in vpoints:
            payload = p["payload"]
            current_start = payload.get("start_seconds", 0)
            current_url = payload.get("timestamp_url", "")

            # Handle corrupted array data
            if isinstance(current_start, list):
                current_start = 0
            if isinstance(current_url, list):
                current_url = current_url[0] if current_url else ""

            # Skip if already has proper timestamp
            if isinstance(current_start, (int, float)) and current_start > 0:
                continue

            chunk_idx = payload.get("chunk_index", 0)
            if duration and n_chunks:
                window = duration / n_chunks
                start_sec = chunk_idx * window
                end_sec = min((chunk_idx + 1) * window, duration)
            else:
                start_sec = chunk_idx * 30
                end_sec = (chunk_idx + 1) * 30

            if base_url:
                sep = "&" if "?" in base_url else "?"
                ts_url = f"{base_url}{sep}t={int(start_sec)}"
            elif current_url:
                ts_url = current_url.replace("&t=0", f"&t={int(start_sec)}").replace("?t=0", f"?t={int(start_sec)}")
            else:
                ts_url = ""

            updates.append({
                "point_id": p["id"],
                "start_seconds": round(start_sec, 1),
                "end_seconds": round(end_sec, 1),
                "timestamp_url": ts_url,
            })

    return updates


def apply_updates(updates: list[dict]) -> int:
    """Update each point individually using PUT to replace corrupted arrays."""
    applied = 0
    for i, u in enumerate(updates):
        resp = httpx.request(
            "PUT",
            f"{QDRANT_URL}/collections/{COLLECTION}/points/payload",
            json={
                "points": [u["point_id"]],
                "payload": {
                    "start_seconds": u["start_seconds"],
                    "end_seconds": u["end_seconds"],
                    "timestamp_url": u["timestamp_url"],
                },
            },
        )
        if resp.status_code == 200:
            applied += 1
        else:
            print(f"  ERROR on {u['point_id'][:12]}: {resp.status_code}")

        if (i + 1) % 50 == 0:
            print(f"  Updated {i + 1}/{len(updates)}")

    return applied


def verify_sample(n: int = 5) -> None:
    resp = httpx.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
        json={"limit": n, "with_payload": ["video_id", "chunk_index", "start_seconds", "end_seconds", "timestamp_url"], "with_vector": False},
    )
    points = resp.json()["result"]["points"]
    print("\nVerification sample:")
    for p in points:
        pl = p["payload"]
        ss = pl.get("start_seconds")
        es = pl.get("end_seconds")
        url = pl.get("timestamp_url", "")
        print(f"  {pl.get('video_id', '?')}[{pl.get('chunk_index', '?')}] start={ss} end={es} url={str(url)[:80]}")


def main() -> None:
    print("=== Timestamp Fix Script (v2 - individual PUT) ===")
    catalog = load_catalog()
    print(f"Catalog entries: {len(catalog)}")

    print("\nScrolling Qdrant points...")
    points = scroll_all_points()
    print(f"Total points: {len(points)}")

    print("\nComputing timestamps...")
    updates = compute_timestamps(points, catalog)
    print(f"Points needing update: {len(updates)}")

    if not updates:
        print("No updates needed.")
        return

    for u in updates[:3]:
        print(f"  {u['point_id'][:12]}... start={u['start_seconds']} end={u['end_seconds']} url={u['timestamp_url'][:60]}")

    print(f"\nApplying {len(updates)} updates (individual PUT)...")
    applied = apply_updates(updates)
    print(f"\nApplied: {applied}/{len(updates)}")

    verify_sample()


if __name__ == "__main__":
    main()
