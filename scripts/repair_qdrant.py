#!/usr/bin/env python3
"""Fix corrupted Qdrant payload: restore video_id and fix timestamps."""
import json
import sys
from pathlib import Path

import httpx

QDRANT_URL = "http://localhost:6333"
COLLECTION = "datewise_transcripts"
CATALOG_PATH = Path(__file__).parent.parent / "data" / "catalog-index.json"


def load_catalog() -> dict[str, dict]:
    if not CATALOG_PATH.exists():
        return {}
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return {str(v["id"]): v for v in data.get("videos", []) if v.get("id")}


def load_chunk_mapping() -> dict[str, dict]:
    """Load chunk_id -> {video_id, chunk_index, ...} mapping."""
    mapping = {}
    for subdir in ["chunks", "chunks-target", "chunks-target-remaining"]:
        d = Path(__file__).parent.parent / "data" / subdir
        if not d.exists():
            continue
        for f in d.glob("*.json"):
            data = json.load(open(f))
            items = data if isinstance(data, list) else [data]
            for item in items:
                cid = item.get("chunk_id", "")
                if cid:
                    mapping[cid] = item
    return mapping


def main() -> None:
    print("=== Qdrant Payload Repair ===")

    catalog = load_catalog()
    chunk_map = load_chunk_mapping()
    print(f"Catalog: {len(catalog)} videos, Chunk map: {len(chunk_map)} chunks")

    # Scroll all points
    points = []
    offset = None
    while True:
        body = {"limit": 100, "with_payload": True, "with_vector": False}
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

    print(f"Total points: {len(points)}")

    # Group by video_id
    video_groups: dict[str, list[dict]] = {}
    fixed = 0
    for p in points:
        pl = p["payload"]
        vid = pl.get("video_id", "")
        cid = pl.get("chunk_id", p["id"])

        # Restore missing video_id
        if not vid:
            chunk_data = chunk_map.get(cid, {})
            vid = chunk_data.get("video_id", "")
            if vid:
                httpx.request(
                    "PUT",
                    f"{QDRANT_URL}/collections/{COLLECTION}/points/payload",
                    json={"points": [p["id"]], "payload": {"video_id": vid}},
                )
                fixed += 1

        if vid:
            video_groups.setdefault(vid, []).append(p)

    print(f"Restored video_id: {fixed}")

    # Fix timestamps for ALL points
    print("\nFixing timestamps...")
    updated = 0
    for vid, vpoints in video_groups.items():
        cat = catalog.get(vid, {})
        duration = cat.get("duration", 0)
        base_url = cat.get("url", "")

        vpoints.sort(key=lambda p: p["payload"].get("chunk_index", 0))
        n_chunks = len(vpoints)

        for p in vpoints:
            pl = p["payload"]
            chunk_idx = pl.get("chunk_index", 0)

            if duration and n_chunks:
                window = duration / n_chunks
                start_sec = round(chunk_idx * window, 1)
                end_sec = round(min((chunk_idx + 1) * window, duration), 1)
            else:
                start_sec = chunk_idx * 30.0
                end_sec = (chunk_idx + 1) * 30.0

            if base_url:
                sep = "&" if "?" in base_url else "?"
                ts_url = f"{base_url}{sep}t={int(start_sec)}"
            else:
                ts_url = ""

            resp = httpx.request(
                "PUT",
                f"{QDRANT_URL}/collections/{COLLECTION}/points/payload",
                json={
                    "points": [p["id"]],
                    "payload": {
                        "start_seconds": start_sec,
                        "end_seconds": end_sec,
                        "timestamp_url": ts_url,
                    },
                },
            )
            if resp.status_code == 200:
                updated += 1

            if updated % 100 == 0 and updated > 0:
                print(f"  {updated}...")

    print(f"Updated timestamps: {updated}")

    # Verify
    resp = httpx.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
        json={"limit": 5, "with_payload": ["video_id", "chunk_index", "start_seconds", "timestamp_url"], "with_vector": False},
    )
    print("\nVerification:")
    for p in resp.json()["result"]["points"]:
        pl = p["payload"]
        print(f"  {pl.get('video_id','?')}[{pl.get('chunk_index','?')}] start={pl.get('start_seconds')} url={str(pl.get('timestamp_url',''))[:70]}")


if __name__ == "__main__":
    main()
