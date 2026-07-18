#!/usr/bin/env python3
"""Force re-upsert all classic-literature markdown into Qdrant (content refresh)."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dating_rag.embeddings.bge_m3 import BgeM3Embedder
from dating_rag.store.qdrant import QdrantStore
from qdrant_client.models import SparseVector


def parse_fm(content: str) -> tuple[dict, str]:
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            fm: dict[str, str] = {}
            for line in content[3:end].strip().split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    fm[k.strip()] = v.strip().strip('"').strip("'")
            return fm, content[end + 3 :].strip()
    return {}, content


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--corpus-dir",
        default="data/source/classic-literature/corpus",
    )
    ap.add_argument("--collection", default="datewise_transcripts")
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()

    corpus = Path(args.corpus_dir)
    files = sorted(corpus.glob("*.md"))
    print(f"reindex files {len(files)}", flush=True)

    store = QdrantStore(
        url=os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"),
        api_key=os.environ.get("QDRANT_API_KEY", ""),
    )
    embedder = BgeM3Embedder(
        model_name=os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"),
        device=os.environ.get("EMBEDDING_DEVICE", "cpu"),
    )

    batch: list[tuple] = []
    total = 0
    bs = max(1, args.batch_size)
    for i, md in enumerate(files):
        content = md.read_text(encoding="utf-8")
        fm, body = parse_fm(content)
        chunk_id = fm.get("id", md.stem)
        origin = fm.get("source_origin", "classic-literature")
        cid = str(uuid5(NAMESPACE_URL, f"{origin}:{chunk_id}"))
        title = fm.get("title", md.stem)
        channel = fm.get("channel", "")
        text = f"{title}\n\n{body}"
        batch.append((cid, title, channel, origin, text, str(md)))
        if len(batch) >= bs or i == len(files) - 1:
            texts = [b[4] for b in batch]
            enc = embedder.encode_texts(texts)
            points = []
            for (cid, title, channel, origin, text, fp), dense, sparse in zip(
                batch, enc["dense"], enc["sparse"]
            ):
                idxs = sorted(sparse.keys())
                points.append(
                    {
                        "id": cid,
                        "vector": {
                            "dense": list(dense),
                            "sparse": SparseVector(
                                indices=idxs,
                                values=[float(sparse[j]) for j in idxs],
                            ),
                        },
                        "payload": {
                            "text": text,
                            "title": title,
                            "channel_name": channel,
                            "source_origin": origin,
                            "platform": "book",
                            "file_path": fp,
                            "uploaded": datetime.now().isoformat(),
                        },
                    }
                )
            store.client.upsert(collection_name=args.collection, points=points)
            total += len(points)
            print(f"upserted {total}", flush=True)
            batch = []
    print(f"done {total}", flush=True)


if __name__ == "__main__":
    main()
