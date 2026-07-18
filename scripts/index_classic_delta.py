#!/usr/bin/env python3
"""Index only missing classic-literature markdown files into Qdrant (delta).

Much faster than full re-scan with large batches on CPU BGE-M3.
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dating_rag.embeddings.bge_m3 import BgeM3Embedder
from dating_rag.store.qdrant import QdrantStore


def parse_frontmatter(content: str) -> tuple[dict, str]:
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            fm_text = content[3:end]
            body = content[end + 3 :].strip()
            fm: dict[str, str] = {}
            for line in fm_text.strip().split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    fm[key.strip()] = val.strip().strip('"').strip("'")
            return fm, body
    return {}, content


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-dir",
        default="data/source/classic-literature/corpus",
    )
    parser.add_argument("--collection", default="datewise_transcripts")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="0 = all missing")
    args = parser.parse_args()

    corpus = Path(args.corpus_dir)
    md_files = sorted(corpus.glob("*.md"))
    print(f"Found {len(md_files)} markdown files", flush=True)

    qdrant_url = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
    store = QdrantStore(url=qdrant_url, api_key=os.environ.get("QDRANT_API_KEY", ""))
    client = store.client

    # Scroll all classic ids currently in collection
    existing: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=args.collection,
            scroll_filter={
                "must": [
                    {
                        "key": "source_origin",
                        "match": {"value": "classic-literature"},
                    }
                ]
            },
            limit=256,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        for p in points:
            existing.add(str(p.id))
        if offset is None:
            break
    print(f"Existing classic points in Qdrant: {len(existing)}", flush=True)

    todo: list[tuple[str, dict, str, Path]] = []
    for md in md_files:
        content = md.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(content)
        chunk_id = fm.get("id", md.stem)
        source_origin = fm.get("source_origin", "classic-literature")
        chunk_uuid = str(uuid5(NAMESPACE_URL, f"{source_origin}:{chunk_id}"))
        if chunk_uuid in existing:
            continue
        title = fm.get("title", md.stem)
        channel = fm.get("channel", "")
        text = f"{title}\n\n{body}"
        todo.append(
            (
                chunk_uuid,
                {
                    "title": title,
                    "channel": channel,
                    "source_origin": source_origin,
                    "file_path": str(md),
                },
                text,
                md,
            )
        )

    if args.limit and args.limit > 0:
        todo = todo[: args.limit]
    print(f"Missing to index: {len(todo)}", flush=True)
    if not todo:
        print("Nothing to do.", flush=True)
        return

    print("Loading BGE-M3…", flush=True)
    embedder = BgeM3Embedder(
        model_name=os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"),
        device=os.environ.get("EMBEDDING_DEVICE", "cpu"),
    )

    from qdrant_client.models import SparseVector

    total = 0
    bs = max(1, args.batch_size)
    for i in range(0, len(todo), bs):
        chunk = todo[i : i + bs]
        texts = [t for _, _, t, _ in chunk]
        print(f"Embedding batch {i // bs + 1} size={len(texts)}…", flush=True)
        encoded = embedder.encode_texts(texts)
        dense_embs = encoded["dense"]
        sparse_embs = encoded["sparse"]
        points = []
        for (cid, meta, text, _md), dense_emb, sparse_emb in zip(
            chunk, dense_embs, sparse_embs
        ):
            sparse_indices = sorted(sparse_emb.keys())
            sparse_vector = SparseVector(
                indices=sparse_indices,
                values=[float(sparse_emb[idx]) for idx in sparse_indices],
            )
            points.append(
                {
                    "id": cid,
                    "vector": {
                        "dense": list(dense_emb),
                        "sparse": sparse_vector,
                    },
                    "payload": {
                        "text": text,
                        "title": meta["title"],
                        "channel_name": meta["channel"],
                        "source_origin": meta["source_origin"],
                        "platform": "book",
                        "file_path": meta["file_path"],
                        "uploaded": datetime.now().isoformat(),
                    },
                }
            )
        client.upsert(collection_name=args.collection, points=points)
        total += len(points)
        print(f"  Upserted {len(points)} (total new {total}/{len(todo)})", flush=True)

    print(f"Done. Newly indexed: {total}", flush=True)


if __name__ == "__main__":
    main()
