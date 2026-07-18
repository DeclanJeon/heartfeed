#!/usr/bin/env python3
"""Index book insight markdown files into Qdrant."""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from uuid import uuid5, NAMESPACE_URL

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dating_rag.embeddings.bge_m3 import BgeM3Embedder
from dating_rag.store.qdrant import QdrantStore


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown."""
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            fm_text = content[3:end]
            body = content[end+3:].strip()
            fm = {}
            for line in fm_text.strip().split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    fm[key.strip()] = val.strip().strip('"').strip("'")
            return fm, body
    return {}, content


def index_book_insights(
    corpus_dir: str,
    collection: str = "datewise_transcripts",
    batch_size: int = 8,
):
    """Index all book insight files into Qdrant."""
    
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        print(f"Corpus directory not found: {corpus_dir}")
        return
    
    md_files = list(corpus_path.glob("*.md"))
    print(f"Found {len(md_files)} markdown files")
    
    # Initialize
    print("Initializing BGE-M3 embedder...")
    embedder = BgeM3Embedder()
    
    print("Connecting to Qdrant...")
    store = QdrantStore()
    
    # Process in batches
    total_indexed = 0
    batch = []
    
    for i, md_file in enumerate(md_files):
        content = md_file.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(content)
        
        # Extract key info
        chunk_id = fm.get("id", md_file.stem)
        # Convert to UUID for Qdrant compatibility
        source_origin = fm.get("source_origin", "book-insight")
        chunk_uuid = str(uuid5(NAMESPACE_URL, f"{source_origin}:{chunk_id}"))
        title = fm.get("title", "")
        channel = fm.get("channel", "")
        source_origin = fm.get("source_origin", "book-insight")
        
        # Skip if already exists
        try:
            existing = store.client.get_points(
                collection_name=collection,
                ids=[chunk_uuid],
            )
            if existing.points:
                continue
        except:
            pass
        
        # Prepare text for embedding
        text = f"{title}\n\n{body}"
        
        batch.append({
            "id": chunk_uuid,
            "text": text,
            "title": title,
            "channel": channel,
            "source_origin": source_origin,
            "file_path": str(md_file),
        })
        
        if len(batch) >= batch_size or i == len(md_files) - 1:
            if batch:
                # Generate embeddings
                texts = [b["text"] for b in batch]
                encoded = embedder.encode_texts(texts)
                dense_embs = encoded["dense"]
                sparse_embs = encoded["sparse"]
                
                # Prepare points
                from qdrant_client.models import SparseVector
                points = []
                for b, dense_emb, sparse_emb in zip(batch, dense_embs, sparse_embs):
                    # Convert sparse to Qdrant SparseVector
                    sparse_indices = sorted(sparse_emb.keys())
                    sparse_vector = SparseVector(
                        indices=sparse_indices,
                        values=[float(sparse_emb[idx]) for idx in sparse_indices],
                    )
                    
                    points.append({
                        "id": b["id"],
                        "vector": {
                            "dense": list(dense_emb),
                            "sparse": sparse_vector,
                        },
                        "payload": {
                            "text": b["text"],
                            "title": b["title"],
                            "channel_name": b["channel"],
                            "source_origin": b["source_origin"],
                            "platform": "book",
                            "file_path": b["file_path"],
                            "uploaded": datetime.now().isoformat(),
                        },
                    })
                
                # Upsert
                try:
                    store.client.upsert(
                        collection_name=collection,
                        points=points,
                    )
                    total_indexed += len(batch)
                    print(f"  Indexed batch {i//batch_size + 1}: {len(batch)} points (total: {total_indexed})")
                except Exception as e:
                    print(f"  Error indexing batch: {e}")
                
                batch = []
    
    print(f"\nDone! Total indexed: {total_indexed} points")
    return total_indexed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", default="data/source/book-insights/corpus")
    parser.add_argument("--collection", default="datewise_transcripts")
    args = parser.parse_args()
    
    index_book_insights(args.corpus_dir, args.collection)
