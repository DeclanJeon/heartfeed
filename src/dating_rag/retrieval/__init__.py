"""Retrieval strategies and ranking."""

from dating_rag.retrieval.claim_retriever import ClaimRetriever
from dating_rag.retrieval.context_builder import CitationRegistry, ContextBuilder
from dating_rag.retrieval.diversifier import diversify, diversify_results
from dating_rag.retrieval.evidence_gate import EvidenceGate
from dating_rag.retrieval.hybrid import HybridRetriever, hybrid_search, reciprocal_rank_fusion
from dating_rag.retrieval.reranker import PassageReranker, Reranker

__all__ = [
    "ClaimRetriever",
    "CitationRegistry",
    "ContextBuilder",
    "EvidenceGate",
    "HybridRetriever",
    "PassageReranker",
    "Reranker",
    "diversify",
    "diversify_results",
    "hybrid_search",
    "reciprocal_rank_fusion",
]
