#!/usr/bin/env python3
"""Index transcript chunks into the Datewise Qdrant collection."""

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dating_rag.domain.models import TranscriptChunk
from dating_rag.embeddings.bge_m3 import BgeM3Embedder
from dating_rag.store.qdrant import QdrantStore


def _load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    """Load the video catalog keyed by YouTube video ID."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(item["id"]): item
        for item in payload.get("videos", [])
        if isinstance(item, dict) and item.get("id")
    }


def _timestamp_url(url: str, start_seconds: float) -> str:
    """Add a stable YouTube timestamp to a source URL."""
    if not url:
        return ""
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}t={max(0, int(start_seconds))}"


def index_chunks(
    chunks_dir: str = "data/chunks",
    collection: str = "datewise_transcripts",
    catalog_path: str = "data/catalog-index.json",
):

    """Index all chunks into Qdrant, enriching payloads from the video catalog."""
    
    chunks_path = Path(chunks_dir)
    catalog = _load_catalog(Path(catalog_path))
    if not chunks_path.exists():
        print(f"Chunks directory not found: {chunks_dir}")
        return
    
    # Load all chunks
    all_raw = []
    for chunk_file in sorted(chunks_path.glob("*.json")):
        with open(chunk_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                all_raw.extend(data)
            else:
                all_raw.append(data)
    
    print(f"Loaded {len(all_raw)} raw chunks from {chunks_path}")
    
    if not all_raw:
        print("No chunks to index")
        return
    
    # Convert to TranscriptChunk objects
    chunks: list[TranscriptChunk] = []
    for raw in all_raw:
        try:
            # Ensure required fields
            if "chunk_id" not in raw:
                continue
            video_id = str(raw.get("video_id", ""))
            catalog_item = catalog.get(video_id, {})
            start_seconds = float(raw.get("start_seconds", 0) or 0)
            source_url = str(catalog_item.get("url", ""))
            chunk = TranscriptChunk(
                chunk_id=raw["chunk_id"],
                video_id=video_id,
                channel_id=raw.get("channel_id", ""),
                channel_name=raw.get("channel_name") or catalog_item.get("channel", ""),
                title=raw.get("title") or catalog_item.get("title", ""),
                text=raw.get("text", ""),
                language=raw.get("language", "ko"),
                start_seconds=start_seconds,
                end_seconds=raw.get("end_seconds", 0),
                timestamp_url=raw.get("timestamp_url") or _timestamp_url(source_url, start_seconds),
                category=raw.get("category") or catalog_item.get("category", ""),
                tags=raw.get("tags", []),
                views=int(raw["views"]) if raw.get("views") is not None else int(catalog_item.get("views", 0) or 0),
                chunk_index=raw.get("chunk_index", 0),
                previous_chunk_id=raw.get("previous_chunk_id"),
                next_chunk_id=raw.get("next_chunk_id"),
            )
            chunks.append(chunk)
        except Exception as e:
            print(f"  Skip chunk: {e}")
            continue
    
    print(f"Valid chunks: {len(chunks)}")
    
    if not chunks:
        print("No valid chunks to index")
        return
    
    # Initialize embedder and store
    print("Initializing BGE-M3 embedder...")
    embedder = BgeM3Embedder()
    
    print("Connecting to Qdrant...")
    store = QdrantStore()
    
    if not store.collection_exists(collection):
        store.create_collection(collection, dense_vector_size=1024)
        print(f"Created collection '{collection}'")
    else:
        print(f"Collection '{collection}' already exists")
    
    # Index in batches
    batch_size = 32
    total = len(chunks)
    indexed = 0
    
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c.text for c in batch]
        
        print(f"Encoding batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size}...")
        embeddings = embedder.encode_texts(texts)
        
        store.upsert_chunks(
            collection_name=collection,
            chunks=batch,
            dense_embeddings=embeddings["dense"],
            sparse_embeddings=embeddings["sparse"],
        )
        indexed += len(batch)
        print(f"  Indexed {indexed}/{total}")
    
    print(f"\nDone! Indexed {indexed} chunks into '{collection}'")

if __name__ == "__main__":
    chunks_dir = sys.argv[1] if len(sys.argv) > 1 else "data/chunks"
    collection = sys.argv[2] if len(sys.argv) > 2 else "datewise_transcripts"
    catalog_path = sys.argv[3] if len(sys.argv) > 3 else "data/catalog-index.json"
    index_chunks(chunks_dir, collection, catalog_path)
