# HeartFeed Path Ownership Manifest

Generated: 2026-07-13
Session: 019f5b77-7d3c-7000-b6f7-5a3943243d38

## Conventions Discovered

- **Backend:** Python 3.12+, Pydantic v2, FastAPI, hatchling build
- **Frontend:** Next.js 16, React 19, TypeScript strict, shadcn/ui, @/* path alias
- **Tests:** pytest (flat `tests/` directory, `test_*.py` naming)
- **Config:** YAML under `config/`
- **Source:** `src/dating_rag/` with subpackages

## Goal Path Ownership

### G001 — Discovery + path freeze + contract freeze
- `datewise-rag/src/dating_rag/domain/models.py` (append v2 models)
- `datewise-rag/tests/fixtures/v2/*.json` (new, Korean golden fixtures)
- `datewise-rag/tests/test_v2_schemas.py` (new)

### G002 — Safety + redaction + minimal intake
- `datewise-rag/src/dating_rag/privacy/redaction.py` (new)
- `datewise-rag/src/dating_rag/safety/router.py` (new)
- `datewise-rag/src/dating_rag/safety/resources.py` (new)
- `datewise-rag/src/dating_rag/intake/planner.py` (new)
- `datewise-rag/src/dating_rag/safety/__init__.py` (new)
- `datewise-rag/src/dating_rag/privacy/__init__.py` (new)
- `datewise-rag/src/dating_rag/intake/__init__.py` (new)
- `datewise-rag/tests/test_safety_router.py` (new)
- `datewise-rag/tests/test_redaction.py` (new)
- `datewise-rag/tests/test_intake_planner.py` (new)

### G003 — Shadow corpus + OKF + evidence gate
- `datewise-rag/src/dating_rag/retrieval/evidence_gate.py` (new)
- `datewise-rag/src/dating_rag/retrieval/claim_retriever.py` (new)
- `datewise-rag/src/dating_rag/retrieval/query_analyzer.py` (modify: add intents)
- `datewise-rag/src/dating_rag/retrieval/filters.py` (modify: force evidence role)
- `datewise-rag/src/dating_rag/store/qdrant.py` (modify: alias support)
- `datewise-rag/config/retrieval_thresholds.yaml` (new)
- `datewise-rag/tests/test_evidence_gate.py` (new)
- `datewise-rag/tests/test_claim_retriever.py` (new)

### G004 — Schema-constrained generation + exact citations
- `datewise-rag/src/dating_rag/generation/generator.py` (modify: add build_v2_response)
- `datewise-rag/src/dating_rag/generation/prompts.py` (modify: separate sections)
- `datewise-rag/src/dating_rag/retrieval/context_builder.py` (modify: citation registry)
- `datewise-rag/tests/test_citation_schema.py` (new)

### G005 — Saju adapter
- `datewise-rag/src/dating_rag/personalization/saju_adapter.py` (new)
- `datewise-rag/src/dating_rag/personalization/saju/engine/` (vendored subset)
- `datewise-rag/src/dating_rag/personalization/__init__.py` (new)
- `datewise-rag/tests/test_saju_adapter.py` (new)

### G006 — FastAPI v2 orchestration
- `datewise-rag/src/dating_rag/api/app.py` (modify: add /v2/chat, keep /chat)
- `datewise-rag/src/dating_rag/orchestration/chat_service.py` (new)
- `datewise-rag/src/dating_rag/orchestration/__init__.py` (new)
- `datewise-rag/tests/test_chat_v2.py` (new)

### G007 — Next.js BFF + UI
- `ui_reference/v1/src/app/api/ask/route.ts` (rewrite: FastAPI proxy)
- `ui_reference/v1/src/app/page.tsx` (rewrite: union state machine)
- `ui_reference/v1/src/components/answer-panel.tsx` (adapt to union)
- `ui_reference/v1/src/lib/heartfeed-api.ts` (new: TS types + fetch helper)
- `ui_reference/v1/src/lib/__tests__/heartfeed-api.test.ts` (new)

### G008 — Evaluation + rollout
- `datewise-rag/tests/fixtures/eval/` (new: Korean held-out corpus)
- `datewise-rag/tests/test_evaluation.py` (new)

## Unchanged Paths (preserve as-is)

- `datewise-rag/src/dating_rag/domain/models.py` existing classes (TranscriptChunk, KnowledgeClaim, QueryPlan, RetrievalResult, ChatResponse)
- `datewise-rag/src/dating_rag/api/app.py` existing chat() endpoint
- `datewise-rag/src/dating_rag/retrieval/hybrid.py` (HybridRetriever)
- `datewise-rag/src/dating_rag/retrieval/reranker.py` (PassageReranker)
- `datewise-rag/src/dating_rag/embeddings/bge_m3.py` (BgeM3Embedder)
- `datewise-rag/src/dating_rag/store/qdrant.py` (QdrantStore, PAYLOAD_INDEXES)
- `datewise-rag/config/categories.yaml`
- `datewise-rag/config/prompts.yaml`
- `datewise-rag/config/story_examples.yaml`
- All existing test files
- `ui_reference/v1/src/components/ui/` (shadcn components)
- `ui_reference/v1/src/app/layout.tsx`
- `ui_reference/v1/src/components/rotating-questions.tsx`
