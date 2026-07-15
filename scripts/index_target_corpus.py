#!/usr/bin/env python3
"""Ingest and index the bounded Flucto target corpus."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5
sys.path.insert(0, str(Path(__file__).parent))

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dating_rag.ingestion.pipeline import run_ingestion
from index_chunks import index_chunks
from dating_rag.store.qdrant import QdrantStore

def _stabilize_chunk_ids(chunks_dir: Path) -> None:
    """Assign deterministic IDs derived from content and chunk policy."""
    rows_by_path: dict[Path, list[dict[str, object]]] = {}
    old_to_new: dict[str, str] = {}
    for path in sorted(chunks_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else [payload]
        rows_by_path[path] = rows
        for row in rows:
            video_id = str(row.get("video_id", ""))
            chunk_index = int(row.get("chunk_index", 0))
            start_ms = int(float(row.get("start_seconds", 0) or 0) * 1000)
            content_hash = str(row.get("content_sha256", ""))[:24]
            policy = str(row.get("chunk_policy_version", "v1"))
            stable_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"datewise:transcript:{policy}:{video_id}:{start_ms}:{chunk_index}:{content_hash}",
                )
            )
            old_to_new[str(row["chunk_id"])] = stable_id
            row["chunk_id"] = stable_id
            row["chunk_id_version"] = 2
    for rows in rows_by_path.values():
        for row in rows:
            for key in ("previous_chunk_id", "next_chunk_id"):
                value = row.get(key)
                if value:
                    row[key] = old_to_new.get(str(value), str(value))
    for path, rows in rows_by_path.items():
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def _chunk_video_ids(chunks_dir: Path) -> set[str]:
    video_ids: set[str] = set()
    for path in chunks_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else [payload]
        video_ids.update(str(row["video_id"]) for row in rows if row.get("video_id"))
    return video_ids


def _clear_target_points(collection: str, video_ids: set[str]) -> None:
    if not video_ids:
        return
    from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchAny

    store = QdrantStore()
    if not store.collection_exists(collection):
        return
    store.client.delete(
        collection_name=collection,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="video_id", match=MatchAny(any=sorted(video_ids)))]
            )
        )
    )
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/source/flucto-target/corpus")
    parser.add_argument("--chunks", default="data/chunks-target")
    parser.add_argument("--collection", default="datewise_transcripts")
    parser.add_argument("--catalog", default="data/catalog-index.json")
    parser.add_argument(
        "--keep-chunks",
        action="store_true",
        help="Keep existing generated chunks instead of replacing them",
    )
    parser.add_argument(
        "--state",
        default="data/source/flucto-target/indexed-video-ids.json",
        help="State file used to remove points from the previous target corpus",
    )
    parser.add_argument("--manifest", default=None, help="JSONL source manifest output")
    parser.add_argument("--audit", default=None, help="JSON source/chunk audit output")
    args = parser.parse_args()

    source = Path(args.source)
    chunks = Path(args.chunks)
    state = Path(args.state)
    previous_ids: set[str] = set()
    if state.exists():
        payload = json.loads(state.read_text(encoding="utf-8"))
        previous_ids = {str(video_id) for video_id in payload.get("video_ids", [])}
    if chunks.exists() and not args.keep_chunks:
        shutil.rmtree(chunks)

    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    manifest_path = Path(args.manifest) if args.manifest else source.parent / "manifest.jsonl"
    audit_path = Path(args.audit) if args.audit else source.parent / "ingestion-audit.json"
    count = run_ingestion(
        source,
        chunks,
        source_origin="flucto-target",
        ingestion_run_id=run_id,
        manifest_path=manifest_path,
        audit_path=audit_path,
    )
    _stabilize_chunk_ids(chunks)
    current_ids = _chunk_video_ids(chunks)
    _clear_target_points(args.collection, previous_ids | current_ids)
    print(f"Generated {count} transcript chunks from {source}")
    index_chunks(str(chunks), args.collection, args.catalog)
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "ingestion_run_id": run_id,
                "video_ids": sorted(current_ids),
                "manifest": str(manifest_path),
                "audit": str(audit_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
