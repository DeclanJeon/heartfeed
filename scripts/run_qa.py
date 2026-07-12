#!/usr/bin/env python3
"""
QA 테스트 스크립트 - 100개 연애 고민으로 RAG 시스템 검증
"""

import json
import sys
import time
from dataclasses import dataclass, asdict

from dating_rag.retrieval.query_analyzer import QueryAnalyzer
from dating_rag.retrieval.hybrid import HybridRetriever
from dating_rag.retrieval.diversifier import diversify_results
from dating_rag.retrieval.context_builder import ContextBuilder
from dating_rag.embeddings.bge_m3 import BgeM3Embedder
from dating_rag.store.qdrant import QdrantStore


@dataclass
class QAResult:
    question_id: str
    question: str
    category: str
    intent: str
    chunks_retrieved: int
    chunks_after_diversify: int
    context_length: int
    has_evidence: bool
    timestamp_links: int
    retrieval_time_ms: float
    status: str
    error: str | None = None


def run_qa_test(questions_file: str, output_file: str, limit: int = 100):
    """Run QA tests on dating questions."""
    
    # Load questions
    questions = []
    with open(questions_file) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    
    questions = questions[:limit]
    # Initialize components
    analyzer = QueryAnalyzer()
    store = QdrantStore()
    embedder = BgeM3Embedder()
    retriever = HybridRetriever(store=store, embedder=embedder)
    context_builder = ContextBuilder()
    results: list[QAResult] = []
    stats = {
        "total": len(questions),
        "success": 0,
        "failed": 0,
        "no_evidence": 0,
        "by_intent": {},
        "by_category": {},
        "avg_retrieval_time_ms": 0,
        "avg_chunks": 0,
    }
    
    for i, q in enumerate(questions):
        qid = q["id"]
        question = q["question"]
        category = q.get("category", "unknown")
        
        print(f"\n[{i+1}/{len(questions)}] {qid}: {question[:50]}...")
        
        try:
            # 1. Query analysis
            start = time.time()
            plan = analyzer.analyze(question)
            intent = plan.intent
            
            # 2. Retrieval (no category filter - categories not in chunks)
            candidates = retriever.search(
                query=question,
                limit=20,
            )
            retrieval_time = (time.time() - start) * 1000
            
            # 3. Diversification
            diversified = diversify_results(candidates, limit=8)
            
            # 4. Context building
            context = context_builder.build_context(
                transcript_results=diversified,
                okf_claims=[],
                plan=plan,
            )
            
            # 5. Count timestamp links
            timestamp_links = context.count("youtube.com/watch")
            
            result = QAResult(
                question_id=qid,
                question=question,
                category=category,
                intent=intent,
                chunks_retrieved=len(candidates),
                chunks_after_diversify=len(diversified),
                context_length=len(context),
                has_evidence=len(diversified) > 0,
                timestamp_links=timestamp_links,
                retrieval_time_ms=round(retrieval_time, 1),
                status="success" if len(diversified) > 0 else "no_evidence",
            )
            
            stats["success"] += 1
            if len(diversified) == 0:
                stats["no_evidence"] += 1
            
            stats["by_intent"][intent] = stats["by_intent"].get(intent, 0) + 1
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
            stats["avg_retrieval_time_ms"] += retrieval_time
            stats["avg_chunks"] += len(diversified)
            
            print(f"  ✓ intent={intent}, chunks={len(diversified)}, time={retrieval_time:.0f}ms")
            
        except Exception as e:
            result = QAResult(
                question_id=qid,
                question=question,
                category=category,
                intent="error",
                chunks_retrieved=0,
                chunks_after_diversify=0,
                context_length=0,
                has_evidence=False,
                timestamp_links=0,
                retrieval_time_ms=0,
                status="error",
                error=str(e),
            )
            stats["failed"] += 1
            print(f"  ✗ Error: {e}")
        
        results.append(result)
    
    # Calculate averages
    if stats["success"] > 0:
        stats["avg_retrieval_time_ms"] = round(stats["avg_retrieval_time_ms"] / stats["success"], 1)
        stats["avg_chunks"] = round(stats["avg_chunks"] / stats["success"], 1)
    
    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stats": stats,
        "results": [asdict(r) for r in results],
    }
    
    with open(output_file, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("QA TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total questions: {stats['total']}")
    print(f"Success: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"No evidence: {stats['no_evidence']}")
    print(f"Avg retrieval time: {stats['avg_retrieval_time_ms']}ms")
    print(f"Avg chunks per query: {stats['avg_chunks']}")
    print()
    print("By Intent:")
    for intent, count in sorted(stats["by_intent"].items(), key=lambda x: -x[1]):
        print(f"  {intent}: {count}")
    print()
    print("By Category:")
    for cat, count in sorted(stats["by_category"].items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    print()
    print(f"Results saved to: {output_file}")
    
    return stats


if __name__ == "__main__":
    questions_file = sys.argv[1] if len(sys.argv) > 1 else "data/eval/dating_questions.jsonl"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "data/eval/qa_results.json"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    run_qa_test(questions_file, output_file, limit)
