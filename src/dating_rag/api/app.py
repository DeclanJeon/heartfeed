"""FastAPI application for the dating RAG chatbot."""

from __future__ import annotations

import json as _json
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dating_rag import __version__
from dating_rag.config import Settings, get_settings
from dating_rag.domain.models import (
    ChatResponse,
    ChatV2Request,
    ChatV2Response,
    KnowledgeClaim,
    ProblemEnvelope,
    QueryPlan,
    RetrievalResult,
)
from dating_rag.orchestration.chat_service import ChatService

from dating_rag.embeddings.bge_m3 import BgeM3Embedder
from dating_rag.generation.generator import Generator
from dating_rag.generation.prompts import load_prompts
from dating_rag.generation.examples import select_story_examples
from dating_rag.retrieval.context_builder import ContextBuilder
from dating_rag.retrieval.diversifier import diversify
from dating_rag.retrieval.filters import build_qdrant_filter
from dating_rag.retrieval.hybrid import HybridRetriever
from dating_rag.retrieval.query_analyzer import QueryAnalyzer
from dating_rag.retrieval.reranker import PassageReranker
from dating_rag.store.qdrant import PAYLOAD_INDEXES, QdrantStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """POST /chat request body."""

    question: str = Field(..., min_length=1, description="User's dating advice question")
    filters: dict[str, str] | None = Field(
        default=None,
        description="Optional metadata filters (category, channel, language)",
    )


class SearchRequest(BaseModel):
    """POST /search request body."""

    query: str = Field(..., min_length=1, description="Search query text")
    filters: dict[str, str] | None = Field(default=None, description="Optional metadata filters")
    limit: int = Field(default=10, ge=1, le=100, description="Max results to return")


class HealthResponse(BaseModel):
    """GET /health response."""

    status: str
    version: str
    qdrant_connected: bool
    qdrant_collection_count: int | None = None


class StatsResponse(BaseModel):
    """GET /stats response."""

    collection_name: str
    vector_count: int
    indexed_fields: list[str]
    qdrant_url: str


# ---------------------------------------------------------------------------
class CitationValidator:
    """Validate source-label citations against the retrieved evidence."""

    def validate(
        self,
        answer: str,
        sources: list[RetrievalResult],
        context_text: str = "",
    ) -> list[str]:
        """Return warnings for labels absent from the rendered context."""
        if not sources:
            return []

        import re

        if context_text:
            available = {
                match.group(1)
                for match in re.finditer(r"(?m)^\[(S\d+)\]\s*\(", context_text)
            }
            available.update(
                match.group(1)
                for match in re.finditer(r"(?m)^\[(C\d+)\]\s+", context_text)
            )
        else:
            available = {
                f"S{i + 1}" if source.source_type == "transcript" else f"C{i + 1}"
                for i, source in enumerate(sources)
            }
        cited = {match.group(1) for match in re.finditer(r"\[([SC]\d+)\]", answer)}
        warnings: list[str] = []
        invalid = sorted(cited - available)
        if invalid:
            warnings.append(f"Answer cites unavailable sources: {', '.join(invalid)}")
        if not cited:
            warnings.append("Answer contains no source citations")
        return warnings


# ---------------------------------------------------------------------------
# Application state (initialized once at startup)
# ---------------------------------------------------------------------------


@dataclass
class AppState:
    """Holds all initialized pipeline components."""

    settings: Settings
    store: QdrantStore
    embedder: BgeM3Embedder
    retriever: HybridRetriever
    context_builder: ContextBuilder
    analyzer: QueryAnalyzer
    generator: Generator
    citation_validator: CitationValidator
    prompts: dict[str, str]
    collection_name: str = "datewise_transcripts"
    story_examples: list[dict[str, Any]] = field(default_factory=list)
    # Lazy-loaded reranker fields
    _reranker: PassageReranker | None = field(default=None, repr=False)
    _reranker_model: str = field(default="", repr=False)
    _reranker_device: str = field(default="cpu", repr=False)
    _reranker_attempted: bool = field(default=False, repr=False)

    def get_reranker(self) -> PassageReranker | None:
        """Lazy-load the reranker on first use to avoid blocking startup."""
        if self._reranker_attempted:
            return self._reranker
        self._reranker_attempted = True
        try:
            self._reranker = PassageReranker(
                model_name=self._reranker_model,
                device=self._reranker_device,
            )
            logger.info("Reranker loaded: %s", self._reranker_model)
        except Exception:
            logger.warning("Reranker not available; skipping cross-encoder reranking")
            self._reranker = None
        return self._reranker


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    """Initialize pipeline components on startup."""
    settings = get_settings()

    # Load retrieval config for tuning parameters
    retrieval_cfg = settings.retrieval_config.get("retrieval", {})

    # Initialize lightweight components
    store = QdrantStore(url=settings.qdrant.url, api_key=settings.qdrant.api_key)
    if not store.collection_exists(settings.collection_name):
        store.create_collection(settings.collection_name, dense_vector_size=1024)
        logger.info("Created empty Qdrant collection: %s", settings.collection_name)
    store.ensure_payload_indexes(settings.collection_name)
    embedder = BgeM3Embedder(
        model_name=settings.embeddings.model_name,
        device=settings.embeddings.device,
    )
    retriever = HybridRetriever(
        store=store,
        embedder=embedder,
        collection_name=settings.collection_name,
        dense_top_k=retrieval_cfg.get("dense_top_k", 30),
        sparse_top_k=retrieval_cfg.get("sparse_top_k", 30),
        dense_threshold=retrieval_cfg.get("dense_score_threshold", 0.35),
        rrf_k=retrieval_cfg.get("rrf_k", 60),
    )

    context_builder = ContextBuilder(
        max_chunks=retrieval_cfg.get("max_context_chunks", 8),
        max_tokens=retrieval_cfg.get("max_context_tokens", 3000),
    )

    generator = Generator(
        api_url=settings.llm.api_url,
        api_key=settings.llm.api_key,
        model=settings.llm.model,
        fallback_api_url=settings.llm.fallback_api_url,
        fallback_api_key=settings.llm.fallback_api_key,
        fallback_model=settings.llm.fallback_model,
        provider=settings.llm.provider,
        nous_model=settings.llm.nous_model,
        nous_auth_path=settings.llm.nous_auth_path,
        timeout=35.0,
    )

    prompts = load_prompts(config_dir=settings.config_dir)
    story_config = settings.load_yaml_config("story_examples")
    story_examples = story_config.get("examples", [])

    state = AppState(
        settings=settings,
        store=store,
        embedder=embedder,
        retriever=retriever,
        context_builder=context_builder,
        analyzer=QueryAnalyzer(),
        generator=generator,
        citation_validator=CitationValidator(),
        prompts=prompts,
        collection_name=settings.collection_name,
        story_examples=story_examples,
        # Reranker model config — loaded lazily on first request
        _reranker_model=retrieval_cfg.get("rerank_model", "BAAI/bge-reranker-v2-m3"),
        _reranker_device=settings.embeddings.device,
    )

    app.state.rag = state  # type: ignore[attr-defined]
    logger.info("Dating RAG API initialized (version=%s)", __version__)
    yield
    logger.info("Shutting down Dating RAG API")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


app = FastAPI(
    title="Datewise RAG",
    description="Evidence-first hybrid RAG assistant for dating and relationship questions",
    version=__version__,
    lifespan=lifespan,
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
try:
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
except Exception:  # pragma: no cover - static dir optional at import time
    pass


def _get_state(request: Request) -> AppState:
    """Retrieve initialized app state from request."""
    state: AppState | None = getattr(request.app.state, "rag", None)
    if state is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return state


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Health check endpoint with Qdrant connectivity status."""
    state = _get_state(request)
    qdrant_connected = False
    collection_count: int | None = None
    try:
        collections = state.store.client.get_collections()
        collection_count = len(collections.collections)
        qdrant_connected = True
    except Exception as exc:
        logger.warning("Qdrant health check failed: %s", exc)

    return HealthResponse(
        status="ok",
        version=__version__,
        qdrant_connected=qdrant_connected,
        qdrant_collection_count=collection_count,
    )


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the web chat client."""
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/stats", response_model=StatsResponse)
async def stats(request: Request) -> StatsResponse:
    """Collection statistics endpoint."""
    state = _get_state(request)
    try:
        info = state.store.client.get_collection(state.collection_name)
        return StatsResponse(
            collection_name=state.collection_name,
            vector_count=info.points_count or 0,
            indexed_fields=[name for name, _ in PAYLOAD_INDEXES],
            qdrant_url=state.settings.qdrant.url,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to retrieve collection stats: {exc}",
        ) from exc


@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """Main chat endpoint — full RAG+OKF pipeline.

    Pipeline: QueryAnalysis → HybridRetrieval → Rerank → Diversify → ContextBuild → Generate.
    """
    state = _get_state(request)
    start = time.monotonic()

    try:
        # 1. Analyze query → plan
        plan: QueryPlan = state.analyzer.analyze(body.question, filters=body.filters)

        # 2. Build metadata filters without mutating the query plan.
        include_category = state.settings.auto_category_filters or bool(
            body.filters and "category" in body.filters
        )
        qdrant_filter = build_qdrant_filter(
            plan,
            include_category=include_category,
            require_transcript_evidence=plan.use_transcripts,
        )
        filter_dict: dict[str, Any] | None = None
        if qdrant_filter is not None:
            filter_dict = _qdrant_filter_to_dict(qdrant_filter)
        # 3. Hybrid retrieval (offload sync embedding + Qdrant calls to thread)
        retrieval_cfg = state.settings.retrieval_config.get("retrieval", {})
        search_limit = retrieval_cfg.get("dense_top_k", 30)
        results: list[RetrievalResult] = await asyncio.to_thread(
            lambda: state.retriever.search(
                body.question, filters=filter_dict, limit=search_limit
            )
        )

        # 4. Rerank (lazy-loaded, also CPU-bound)
        if not results:
            return ChatResponse(
                answer="현재 질문을 직접 뒷받침하는 자료를 찾지 못했습니다. 질문을 조금 더 구체적으로 입력해 주세요.",
                sources=[],
                plan=plan,
            )
        reranker = state.get_reranker()
        if reranker and results:
            rerank_top_k = retrieval_cfg.get("rerank_top_k", 10)
            results = await asyncio.to_thread(
                reranker.rerank, body.question, results, rerank_top_k
            )

        # 5. Diversify
        if retrieval_cfg.get("diversify", True) and results:
            results = diversify(
                results,
                max_per_channel=retrieval_cfg.get("max_per_channel", 3),
                max_per_video=retrieval_cfg.get("max_per_video", 2),
            )
        okf_claims: list[KnowledgeClaim] = []
        # 6. Build the exact labeled context passed to the generator.
        context_text = state.context_builder.build_context(
            results,
            okf_claims,
            plan,
            illustrative_examples=select_story_examples(
                state.story_examples,
                plan,
            ),
        )

        if not state.generator.api_key:
            raise HTTPException(
                status_code=503,
                detail="LLM is not configured; set LLM_API_KEY before using /chat",
            )

        # 7. Generate response from the labeled evidence context.
        system_prompt = state.prompts.get("system_prompt", "")
        chat_response = await state.generator.build_response(
            question=body.question,
            context=results,
            context_text=context_text,
            plan=plan,
            system_prompt=system_prompt,
        )

        # 8. Validate citations
        citation_warnings = state.citation_validator.validate(
            chat_response.answer,
            results,
            context_text,
        )

        chat_response.citation_warnings = citation_warnings
        elapsed = time.monotonic() - start
        logger.info(
            "Chat completed: intent=%s, sources=%d, %.2fs",
            plan.intent,
            len(results),
            elapsed,
        )

        return chat_response

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Chat endpoint error")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


@app.post("/v2/chat")
async def chat_v2(request: Request, body: ChatV2Request) -> ChatV2Response | ProblemEnvelope:
    """v2 chat endpoint — full safety → intake → retrieval → generation pipeline."""
    state = _get_state(request)
    service = ChatService(state)
    return await service.answer(body)

@app.post("/v2/chat/stream")
async def chat_v2_stream(request: Request, body: ChatV2Request):
    """v2 streaming chat endpoint — returns answer text as SSE chunks."""
    state = _get_state(request)
    service = ChatService(state)

    async def event_generator():
        try:
            result = await service.answer(body)
            # For non-answered responses, send the full JSON as a single event
            if not hasattr(result, "answer"):
                yield f"data: {_json.dumps(result.model_dump(), ensure_ascii=False)}\n\n"
                return

            # For answered responses, stream the answer text in chunks
            answer_text = result.answer.summary or result.answer.empathy
            for i in range(0, len(answer_text), 50):
                chunk = answer_text[i:i+50]
                yield f"data: {_json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}\n\n"

            # Send the full response as the final event
            yield f"data: {_json.dumps({'type': 'complete', 'data': result.model_dump()}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {_json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/search", response_model=list[RetrievalResult])
async def search(request: Request, body: SearchRequest) -> list[RetrievalResult]:
    """Search endpoint for testing retrieval without generation."""
    state = _get_state(request)

    try:
        plan = state.analyzer.analyze(body.query, filters=body.filters)
        include_category = state.settings.auto_category_filters or bool(
            body.filters and "category" in body.filters
        )
        qdrant_filter = build_qdrant_filter(
            plan,
            include_category=include_category,
            require_transcript_evidence=plan.use_transcripts,
        )
        filter_dict: dict[str, Any] | None = None
        if qdrant_filter is not None:
            filter_dict = _qdrant_filter_to_dict(qdrant_filter)

        results = await asyncio.to_thread(
            lambda: state.retriever.search(
                body.query, filters=filter_dict, limit=body.limit
            )
        )

        reranker = state.get_reranker()
        if reranker and results:
            results = await asyncio.to_thread(
                reranker.rerank, body.query, results, body.limit
            )

        return results

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Search endpoint error")
        raise HTTPException(status_code=500, detail=f"Search error: {exc}") from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _qdrant_filter_to_dict(qdrant_filter: Any) -> dict[str, Any]:
    """Convert a Qdrant Filter to the dict format expected by QdrantStore.search.

    QdrantStore.search internally calls build_qdrant_filter (from store module)
    which accepts a dict[str, Any].  This helper extracts the conditions from
    a qdrant_client.models.Filter object so we can pass them through.
    """
    if qdrant_filter is None:
        return {}

    result: dict[str, Any] = {}
    conditions = getattr(qdrant_filter, "must", None) or []
    for cond in conditions:
        key = getattr(cond, "key", None)
        if key is None:
            continue

        match = getattr(cond, "match", None)
        if match is not None:
            value = getattr(match, "value", None)
            if value is not None:
                result[key] = value
                continue

        range_val = getattr(cond, "range", None)
        if range_val is not None:
            gte = getattr(range_val, "gte", None)
            lte = getattr(range_val, "lte", None)
            range_dict: dict[str, Any] = {}
            if gte is not None:
                range_dict["gte"] = gte
            if lte is not None:
                range_dict["lte"] = lte
            if range_dict:
                result[key] = range_dict

    return result
