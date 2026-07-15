#!/usr/bin/env python3
"""Index new YouTube corpus into Qdrant alongside existing datewise_transcripts.

Usage:
  1. Run collect_youtube_corpus.py first
  2. python3 scripts/collection/index_new_corpus.py --source data/source/new-corpus --collection datewise_transcripts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project src and scripts to path
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "scripts"))

from dating_rag.ingestion.pipeline import run_ingestion
from dating_rag.store.qdrant import QdrantStore
from index_chunks import index_chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/source/new-corpus", help="Directory with .md files")
    parser.add_argument("--chunks", default="data/chunks-new-corpus", help="Output chunks directory")
    parser.add_argument("--collection", default="datewise_transcripts", help="Qdrant collection name")
    parser.add_argument("--source-origin", default="youtube-search", help="Source origin label")
    parser.add_argument("--catalog", default="data/catalog-index.json", help="Catalog index path")
    args = parser.parse_args()

    source = Path(args.source)
    chunks = Path(args.chunks)

    if not source.exists():
        print(f"Source directory not found: {source}")
        return

    md_files = list(source.glob("*.md"))
    print(f"Found {len(md_files)} markdown files in {source}")

    if not md_files:
        print("No files to index")
        return

    # Run ingestion
    from datetime import datetime, timezone
    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    manifest_path = source.parent / "new-corpus-manifest.jsonl"
    audit_path = source.parent / "new-corpus-audit.json"

    count = run_ingestion(
        source,
        chunks,
        source_origin=args.source_origin,
        ingestion_run_id=run_id,
        manifest_path=manifest_path,
        audit_path=audit_path,
    )
    print(f"Generated {count} chunks")

    # Index into Qdrant
    index_chunks(str(chunks), args.collection, args.catalog)
    print(f"Indexed into {args.collection}")

    # Verify
    store = QdrantStore()
    info = store.client.get_collection(args.collection)
    print(f"Total points in collection: {info.points_count}")


if __name__ == "__main__":
    main()
