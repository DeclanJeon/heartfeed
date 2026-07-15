#!/usr/bin/env python3
"""Restore full Qdrant payload from original chunk files.

Point IDs in Qdrant match chunk_id in the chunk JSON files.
This script loads all chunks, matches by ID, and restores the full payload
using PUT (which replaces all payload fields).
"""
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


def load_all_chunks() -> dict[str, dict]:
    """Load chunk_id -> full chunk data mapping."""
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
    print("=== Full Payload Restoration ===")

    catalog = load_catalog()
    chunk_map = load_all_chunks()
    print(f"Catalog: {len(catalog)} videos, Chunks: {len(chunk_map)}")

    # Get all point IDs
    point_ids = []
    offset = None
    while True:
        body = {"limit": 100, "with_payload": False, "with_vector": False}
        if offset:
            body["offset"] = offset
        resp = httpx.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll", json=body)
        result = resp.json()["result"]
        batch = result.get("points", [])
        if not batch:
            break
        point_ids.extend(p["id"] for p in batch)
        offset = result.get("next_page_offset")
        if not offset:
            break

    print(f"Total points: {len(point_ids)}")

    matched = 0
    updated = 0
    for i, pid in enumerate(point_ids):
        chunk = chunk_map.get(pid)
        if not chunk:
            continue
        matched += 1

        vid = chunk.get("video_id", "")
        cat = catalog.get(vid, {})
        base_url = cat.get("url", "")
        duration = cat.get("duration", 0)
        chunk_idx = chunk.get("chunk_index", 0)

        # Calculate proper timestamps
        # Count chunks for this video
        video_chunks = [c for c in chunk_map.values() if c.get("video_id") == vid]
        n_chunks = len(video_chunks)

        start_sec = chunk.get("start_seconds", 0)
        end_sec = chunk.get("end_seconds", 0)

        if (not start_sec or start_sec == 0) and duration and n_chunks:
            window = duration / n_chunks
            start_sec = round(chunk_idx * window, 1)
            end_sec = round(min((chunk_idx + 1) * window, duration), 1)

        # Build timestamp URL
        ts_url = chunk.get("timestamp_url", "")
        if not ts_url and base_url:
            sep = "&" if "?" in base_url else "?"
            ts_url = f"{base_url}{sep}t={int(start_sec)}"

        # Build full payload
        payload = {
            "chunk_id": chunk.get("chunk_id", ""),
            "video_id": vid,
            "channel_id": chunk.get("channel_id", ""),
            "channel_name": chunk.get("channel_name", ""),
            "title": chunk.get("title", ""),
            "text": chunk.get("text", ""),
            "language": chunk.get("language", "ko"),
            "start_seconds": start_sec,
            "end_seconds": end_sec,
            "timestamp_url": ts_url,
            "category": chunk.get("category", ""),
            "tags": chunk.get("tags", []),
            "views": chunk.get("views", 0),
            "published_at": chunk.get("published_at"),
            "chunk_index": chunk_idx,
            "previous_chunk_id": chunk.get("previous_chunk_id"),
            "next_chunk_id": chunk.get("next_chunk_id"),
            "corpus_type": chunk.get("corpus_type", "unknown"),
            "evidence_role": chunk.get("evidence_role", "unknown"),
            "source_origin": chunk.get("source_origin", ""),
            "transcript_status": chunk.get("transcript_status", "unknown"),
            "fallback_used": chunk.get("fallback_used", False),
            "raw_document_id": chunk.get("raw_document_id", ""),
            "raw_sha256": chunk.get("raw_sha256", ""),
            "ingestion_run_id": chunk.get("ingestion_run_id", ""),
            "chunk_policy_version": chunk.get("chunk_policy_version", ""),
            "content_sha256": chunk.get("content_sha256", ""),
            "schema_version": chunk.get("schema_version", 1),
            "chunk_id_version": chunk.get("chunk_id_version", 1),
        }

        resp = httpx.request(
            "PUT",
            f"{QDRANT_URL}/collections/{COLLECTION}/points/payload",
            json={"points": [pid], "payload": payload},
        )
        if resp.status_code == 200:
            updated += 1

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(point_ids)} (matched: {matched}, updated: {updated})")

    print(f"\nMatched: {matched}/{len(point_ids)}")
    print(f"Updated: {updated}")

    # Verify
    resp = httpx.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
        json={"limit": 5, "with_payload": ["video_id", "chunk_index", "start_seconds", "timestamp_url", "title"], "with_vector": False},
    )
    print("\nVerification:")
    for p in resp.json()["result"]["points"]:
        pl = p["payload"]
        print(f"  {pl.get('video_id','?')}[{pl.get('chunk_index','?')}] start={pl.get('start_seconds')} \"{str(pl.get('title',''))[:40]}\" url={str(pl.get('timestamp_url',''))[:60]}")


if __name__ == "__main__":
    main()
