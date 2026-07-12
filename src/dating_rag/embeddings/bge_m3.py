"""BGE-M3 embedding model wrapper using FlagEmbedding."""

from __future__ import annotations

import numpy as np
from FlagEmbedding import BGEM3FlagModel


class BgeM3Embedder:
    """Wrapper for BAAI/bge-m3 embedding model.

    Produces dense and sparse vectors in a single forward pass.
    BGE-M3 natively handles Korean, English, and 100+ other languages.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu") -> None:
        """Initialize the embedder.

        Args:
            model_name: HuggingFace model identifier.
            device: Device to run inference on ('cpu', 'cuda', etc.).
        """
        self.model = BGEM3FlagModel(model_name, use_fp16=device != "cpu")
        self.model_name = model_name
        self.device = device

    def encode_texts(self, texts: list[str], batch_size: int = 32) -> dict[str, np.ndarray]:
        """Encode a batch of texts into dense + sparse vectors.

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for encoding.

        Returns:
            Dict with 'dense' (2D array, shape [N, dim]) and
            'sparse' (list of dicts with 'indices' and 'values').
        """
        output = self.model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return {
            "dense": np.array(output["dense_vecs"], dtype=np.float32),
            "sparse": output["lexical_weights"],
        }

    def encode_query(self, query: str) -> dict[str, np.ndarray | dict]:
        """Encode a single query into dense + sparse vectors.

        Args:
            query: Query text to embed.

        Returns:
            Dict with 'dense' (1D array) and
            'sparse' (dict with 'indices' and 'values').
        """
        output = self.model.encode(
            [query],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return {
            "dense": np.array(output["dense_vecs"][0], dtype=np.float32),
            "sparse": output["lexical_weights"][0],
        }

    def embed_dense(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Compute dense embeddings only (backward-compatible helper).

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for encoding.

        Returns:
            2D numpy array of shape (len(texts), embedding_dim).
        """
        result = self.encode_texts(texts, batch_size=batch_size)
        return result["dense"]

    def embed_query_dense(self, query: str) -> np.ndarray:
        """Embed a single query to a dense vector (backward-compatible helper).

        Args:
            query: Query text to embed.

        Returns:
            1D numpy array representing the query embedding.
        """
        result = self.encode_query(query)
        return result["dense"]  # type: ignore[return-value]

