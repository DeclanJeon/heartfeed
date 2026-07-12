"""Integration tests for embeddings + Qdrant store."""

from __future__ import annotations

import uuid

import pytest

from dating_rag.domain.models import TranscriptChunk


@pytest.fixture(scope="module")
def store():
    """Qdrant store connected to local instance."""
    from dating_rag.store.qdrant import QdrantStore

    s = QdrantStore(url="http://localhost:6333")
    yield s
    # Cleanup
    for name in ("test_embeddings", "test_hybrid"):
        if s.collection_exists(name):
            s.delete_collection(name)


@pytest.fixture(scope="module")
def embedder():
    """BGE-M3 embedder (downloads ~1.5GB model on first use)."""
    from dating_rag.embeddings.bge_m3 import BgeM3Embedder

    return BgeM3Embedder(device="cpu")


def _make_chunk(text: str, **kwargs) -> TranscriptChunk:
    """Create a test chunk."""
    defaults = {
        "video_id": "vid_001",
        "channel_id": "ch_001",
        "channel_name": "Test Channel",
        "title": "Test Video",
        "text": text,
        "language": "en",
        "start_seconds": 0.0,
        "end_seconds": 10.0,
        "category": "approach",
    }
    defaults.update(kwargs)
    return TranscriptChunk(**defaults)


class TestQdrantStore:
    """Tests for Qdrant collection and upsert operations."""

    def test_create_and_check_collection(self, store):
        store.create_collection("test_embeddings", dense_vector_size=1024)
        assert store.collection_exists("test_embeddings")

    def test_delete_collection(self, store):
        name = f"test_delete_{uuid.uuid4().hex[:8]}"
        store.create_collection(name, dense_vector_size=1024)
        assert store.collection_exists(name)
        store.delete_collection(name)
        assert not store.collection_exists(name)


class TestBgeM3Embedder:
    """Tests for BGE-M3 embedding."""

    def test_encode_texts_dense_shape(self, embedder):
        texts = ["How to approach someone at a bar", "데이팅 팁"]
        result = embedder.encode_texts(texts)
        assert "dense" in result
        assert "sparse" in result
        assert result["dense"].shape == (2, 1024)

    def test_encode_texts_sparse(self, embedder):
        texts = ["confidence is key in dating"]
        result = embedder.encode_texts(texts)
        assert len(result["sparse"]) == 1
        sparse = result["sparse"][0]
        assert len(sparse) > 0  # should have some token weights
        # All values should be positive floats
        assert all(v > 0 for v in sparse.values())

    def test_encode_query(self, embedder):
        result = embedder.encode_query("how to be more confident")
        assert "dense" in result
        assert "sparse" in result
        assert result["dense"].shape == (1024,)
        assert len(result["sparse"]) > 0

    def test_korean_text(self, embedder):
        result = embedder.encode_texts(["데이팅에서 자신감을 높이는 방법"])
        assert result["dense"].shape == (1, 1024)
        assert len(result["sparse"][0]) > 0

    def test_embed_dense_backward_compat(self, embedder):
        """Ensure backward-compatible helpers work."""
        dense = embedder.embed_dense(["test"])
        assert dense.shape == (1, 1024)

        query_vec = embedder.embed_query_dense("test")
        assert query_vec.shape == (1024,)


class TestHybridIntegration:
    """End-to-end: embed → store → search."""

    def test_upsert_and_search(self, store, embedder):
        store.create_collection("test_hybrid", dense_vector_size=1024)

        chunks = [
            _make_chunk("Be confident when approaching women"),
            _make_chunk("Humor is attractive on first dates"),
            _make_chunk("Active listening builds connection"),
        ]

        encoded = embedder.encode_texts([c.text for c in chunks])

        store.upsert_chunks(
            collection_name="test_hybrid",
            chunks=chunks,
            dense_embeddings=encoded["dense"],
            sparse_embeddings=encoded["sparse"],
        )

        # Dense search
        query = embedder.encode_query("how to approach women")
        dense_results = store.search(
            "test_hybrid",
            query_vector=list(query["dense"]),
            vector_name="dense",
            top_k=3,
        )
        assert len(dense_results) > 0
        assert "payload" in dense_results[0]

    def test_hybrid_search(self, store, embedder):
        """Test full hybrid dense+sparse RRF search."""
        from qdrant_client.models import SparseVector

        query = embedder.encode_query("how to be confident dating")
        sparse_raw = query["sparse"]
        indices = sorted(sparse_raw.keys())
        sparse_vec = SparseVector(
            indices=indices,
            values=[float(sparse_raw[idx]) for idx in indices],
        )

        results = store.hybrid_search(
            collection_name="test_hybrid",
            dense_vector=list(query["dense"]),
            sparse_vector=sparse_vec,
            dense_top_k=5,
            sparse_top_k=5,
        )
        assert len(results) > 0
        # Each result should have score and payload
        assert all("score" in r and "payload" in r for r in results)
