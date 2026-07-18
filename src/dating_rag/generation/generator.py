"""LLM-based response generation with streaming and citation support."""

from __future__ import annotations
import logging
import asyncio

import json
import re
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import httpx

from dating_rag.domain.models import (
    ChatResponse,
    ChatV2Answered,
    ActionItem,
    AnsweredContent,
    Citation,
    EvidenceClaim,
    PersonalizationBlock,
    CulturalReflectionBlock,
    ConflictItem,
    QueryPlan,
    RetrievalResult,
)
from dating_rag.generation.prompts import SYSTEM_PROMPT, V2_SYSTEM_SUFFIX, build_prompt, build_v2_prompt
if TYPE_CHECKING:
    from dating_rag.retrieval.context_builder import CitationRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Citation extraction helpers
# ---------------------------------------------------------------------------

# Matches [S1], [S2], [C1], etc.  — source labels injected by ContextBuilder.
_SOURCE_LABEL_RE = re.compile(r"\[([SC]\d+)\]")


def extract_cited_sources(answer: str) -> set[str]:
    """Return the set of source labels (e.g. ``{'S1', 'C2'}``) cited in *answer*."""
    return {m.group(1) for m in _SOURCE_LABEL_RE.finditer(answer)}


def format_citation(
    result: RetrievalResult,
    label: str,
) -> str:
    """Format a single retrieval result as a readable citation.

    Args:
        result: The retrieval result with metadata.
        label: The source label (e.g. "S1").

    Returns:
        Formatted citation string like ``[S1] Channel — "Title" @ 12:34``.
    """
    channel = str(result.metadata.get("channel_name", "Unknown"))
    title = str(result.metadata.get("title", ""))
    timestamp_url = str(result.metadata.get("timestamp_url", ""))

    parts = [f"[{label}] {channel}"]
    if title:
        parts.append(f'— "{title}"')

    # Extract MM:SS from timestamp URL when available.
    ts_match = re.search(r"[?&]t=(\d+)", timestamp_url)
    if ts_match:
        total_secs = int(ts_match.group(1))
        minutes, seconds = divmod(total_secs, 60)
        parts.append(f"@ {minutes}:{seconds:02d}")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# AnswerGenerator
# ---------------------------------------------------------------------------


class AnswerGenerator:
    """LLM client for generating dating advice responses.

    Works with any OpenAI-compatible chat completions API.
    Supports both regular and streaming generation.

    When ``provider="nous"`` the API key is resolved dynamically from the
    Hermes operational agent's OAuth token (refreshed on demand), so no static
    key is needed and the same fast flash model Hermes uses is reused.
    """

    def __init__(
        self,
        api_url: str = "https://api.deepseek.com/v1",
        api_key: str = "",
        model: str = "deepseek-v4-flash",
        *,
        timeout: float = 60.0,
        fallback_api_url: str = "",
        fallback_api_key: str = "",
        fallback_model: str = "",
        provider: str = "openai",
        nous_model: str = "stepfun/step-3.7-flash:free",
        nous_auth_path: str = "~/.hermes/auth.json",
    ) -> None:
        """Initialize the generator.

        Args:
            api_url: OpenAI-compatible API base URL.
            api_key: API authentication key.
            model: Model identifier.
            timeout: HTTP request timeout in seconds.
            fallback_api_url: Fallback API base URL (optional).
            fallback_api_key: Fallback API key (optional).
            fallback_model: Fallback model identifier (optional).
            provider: "openai" (static key) or "nous" (Hermes OAuth token).
            nous_model: Model slug when provider == "nous".
            nous_auth_path: Path to Hermes auth.json for Nous token resolution.
        """
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.fallback_api_url = fallback_api_url.rstrip("/") if fallback_api_url else ""
        self.fallback_api_key = fallback_api_key
        self.fallback_model = fallback_model
        self.provider = (provider or "openai").lower()
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
        self._nous: "NousTokenProvider | None" = None
        if self.provider == "nous":
            from dating_rag.generation.nous_client import NousTokenProvider

            self._nous = NousTokenProvider(
                auth_path=nous_auth_path,
                model=nous_model,
            )

    # -- helpers ----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _resolve_endpoint(self) -> tuple[str, str, str]:
        """Return ``(api_url, api_key, model)`` for the next call.

        For the ``nous`` provider the API key is fetched fresh from the Hermes
        OAuth token on every call (cheap, memoized + auto-refreshed inside the
        provider), so it never goes stale.
        """
        if self.provider == "nous" and self._nous is not None:
            creds = self._nous.resolve()
            return creds["base_url"], creds["api_key"], creds["model"]
        return self.api_url, self.api_key, self.model

    def _format_context(self, context: list[RetrievalResult]) -> str:
        """Format retrieval results when no pre-built context is supplied."""
        lines: list[str] = []
        for i, result in enumerate(context):
            label = f"S{i + 1}" if result.source_type == "transcript" else f"C{i + 1}"
            channel = result.metadata.get("channel_name", "Unknown")
            title = result.metadata.get("title", "")
            timestamp_url = result.metadata.get("timestamp_url", "")
            header = f"[{label}] ({channel}"
            if title:
                header += f' — "{title}"'
            header += ")"
            source_line = f"\n  ↳ Source: {timestamp_url}" if timestamp_url else ""
            lines.append(f"{header}\n{result.text}{source_line}")
        return "\n\n".join(lines)

    def _build_messages(
        self,
        question: str,
        context: list[RetrievalResult],
        plan: QueryPlan,
        system_prompt: str | None = None,
        context_text: str | None = None,
    ) -> list[dict[str, str]]:
        """Build messages using the exact labeled context selected by retrieval."""
        ctx_str = context_text if context_text is not None else self._format_context(context)
        user_content = build_prompt(question, ctx_str, plan)
        return [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    # -- non-streaming generation -----------------------------------------

    async def _call_llm(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        *,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """Call primary LLM, fall back to secondary on failure."""
        self._nous_refreshed_for_call = False
        configs = [self._resolve_endpoint()]
        if self.fallback_api_url and self.fallback_api_key:
            configs.append((self.fallback_api_url, self.fallback_api_key, self.fallback_model))

        last_error: Exception | None = None
        for api_url, api_key, model in configs:
            for attempt in range(2):  # max 2 retries per config
                try:
                    headers: dict[str, str] = {"Content-Type": "application/json"}
                    if api_key:
                        headers["Authorization"] = f"Bearer {api_key}"
                    body: dict[str, Any] = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": False,
                    }
                    # DeepSeek V4 flash uses reasoning tokens by default; disable
                    # thinking so max_tokens is spent on visible content/JSON.
                    model_l = (model or "").lower()
                    if "deepseek" in model_l or "deepseek.com" in (api_url or ""):
                        body["thinking"] = {"type": "disabled"}
                    # Nous models (hy3/flash) reject or ignore response_format;
                    # the v2 system-prompt override enforces JSON instead.
                    if response_format and self.provider != "nous":
                        # DeepSeek requires the word "json" in the prompt for
                        # response_format=json_object.
                        if response_format.get("type") == "json_object":
                            msgs = list(messages)
                            if msgs and "json" not in msgs[0].get("content", "").lower():
                                msgs[0] = {
                                    **msgs[0],
                                    "content": (
                                        msgs[0].get("content", "")
                                        + "\n\nRespond with a single JSON object only."
                                    ),
                                }
                            body["messages"] = msgs
                        body["response_format"] = response_format
                    response = await self._client.post(
                        f"{api_url}/chat/completions",
                        headers=headers,
                        json=body,
                    )
                    response.raise_for_status()
                    data: dict[str, Any] = response.json()
                    msg = data["choices"][0]["message"]
                    content = msg.get("content") or ""
                    if not content and msg.get("reasoning_content"):
                        # Fallback if thinking could not be disabled.
                        content = str(msg.get("reasoning_content") or "")
                    return str(content)
                except Exception as exc:
                    last_error = exc
                    # Nous OAuth tokens can expire mid-flight; on 401 force a
                    # token refresh once and retry before giving up.
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if (
                        self.provider == "nous"
                        and self._nous is not None
                        and status == 401
                        and not getattr(self, "_nous_refreshed_for_call", False)
                    ):
                        self._nous_refreshed_for_call = True
                        logger.warning("Nous token rejected (401) — forcing refresh and retry")
                        self._nous.force_refresh()
                        continue
                    # Only retry on rate limit (429) or server errors (5xx)
                    is_retryable = status in (429, 500, 502, 503, 504)
                    if not is_retryable and attempt == 0:
                        # Non-retryable error on first attempt → skip retries, try next config
                        logger.warning("LLM call failed (%s/%s): %s", api_url, model, exc)
                        break
                    if is_retryable and attempt < 2:
                        delay = 2 ** attempt  # 1s, 2s, 4s
                        logger.warning("LLM call failed (%s/%s), retrying in %ds: %s", api_url, model, delay, exc)
                        await asyncio.sleep(delay)
                    else:
                        logger.warning("LLM call failed (%s/%s) after %d attempts: %s", api_url, model, attempt + 1, exc)
        raise last_error  # type: ignore[misc]

    async def generate(
        self,
        question: str,
        context: list[RetrievalResult],
        plan: QueryPlan,
        *,
        system_prompt: str | None = None,
        context_text: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate a response given context and user question."""
        messages = self._build_messages(
            question,
            context,
            plan,
            system_prompt,
            context_text,
        )
        return await self._call_llm(messages, temperature, max_tokens)

    # -- streaming generation ---------------------------------------------

    async def generate_stream(
        self,
        question: str,
        context: list[RetrievalResult],
        plan: QueryPlan,
        *,
        system_prompt: str | None = None,
        context_text: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> AsyncIterator[str]:
        """Stream a response token-by-token.

        Yields partial content strings as they arrive from the API.

        Args:
            question: The user's dating advice question.
            context: Retrieved chunks to use as evidence.
            plan: The query plan controlling strategy hints.
            system_prompt: Override the default system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.

        Yields:
            Incremental content strings.
        """
        messages = self._build_messages(
            question,
            context,
            plan,
            system_prompt,
            context_text,
        )

        api_url, api_key, model = self._resolve_endpoint()
        async with self._client.stream(
            "POST",
            f"{api_url}/chat/completions",
            headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
                **(
                    {"thinking": {"type": "disabled"}}
                    if ("deepseek" in (model or "").lower() or "deepseek.com" in (api_url or ""))
                    else {}
                ),
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload.strip() == "[DONE]":
                    break
                try:
                    chunk: dict[str, Any] = json.loads(payload)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield str(content)
                except (ValueError, KeyError, IndexError):
                    continue

    # -- high-level response builder --------------------------------------

    async def build_response(
        self,
        question: str,
        context: list[RetrievalResult],
        plan: QueryPlan,
        *,
        system_prompt: str | None = None,
        context_text: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> ChatResponse:
        """Build a complete chat response with sources and conflict detection."""
        answer = await self.generate(
            question,
            context,
            plan,
            system_prompt=system_prompt,
            context_text=context_text,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        conflicts = _detect_conflicts(answer, context)

        return ChatResponse(
            answer=answer,
            sources=context,
            conflicts=conflicts,
            plan=plan,
        )

    async def build_v2_response(
        self,
        question: str,
        context: list[RetrievalResult],
        context_text: str,
        plan: QueryPlan,
        registry: CitationRegistry,
        *,
        system_prompt: str | None = None,
        personalization: PersonalizationBlock | None = None,
        cultural_reflection: CulturalReflectionBlock | None = None,
        conversation_context: str = "",
        temperature: float = 0.3,
        max_tokens: int = 3000,
    ) -> ChatV2Answered:
        """Build a v2 JSON-constrained response with citation validation.

        Args:
            question: User's dating advice question.
            context: Raw retrieval results (for fallback context formatting).
            context_text: Pre-formatted evidence context with [S1]/[C1] labels.
            plan: The query plan.
            registry: Citation registry from ContextBuilder.build_context_with_registry.
            system_prompt: Override the default system prompt.
            personalization: Optional personalization metadata to include.
            cultural_reflection: Optional cultural reflection metadata to include.
            temperature: Sampling temperature (lower for JSON reliability).
            max_tokens: Maximum response tokens.

        Returns:
            Validated ChatV2Answered.

        Raises:
            ValueError: If citation validation fails after one retry.
        """
        user_content = build_v2_prompt(
            question, context_text, plan, registry.citation_ids(),
        )
        if conversation_context:
            user_content = conversation_context + "\n\n" + user_content
        messages = [
            {"role": "system", "content": (system_prompt or SYSTEM_PROMPT) + V2_SYSTEM_SUFFIX},
            {"role": "user", "content": user_content},
        ]

        last_raw = ""
        for attempt in range(2):
            raw_content = await self._call_llm(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            last_raw = raw_content

            # Strip markdown code fences if present
            clean = raw_content.strip()
            if clean.startswith("```"):
                lines = clean.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                clean = "\n".join(lines)

            # Extract the outermost JSON object. Flash models sometimes wrap
            # the JSON in prose or emit trailing text, so search for the first
            # "{" and the last "}" rather than requiring a clean object.
            json_start = clean.find("{")
            json_end = clean.rfind("}")
            if json_start != -1:
                # Prefer complete object; if truncated, take from first "{" to end and repair.
                if json_end > json_start:
                    candidate = clean[json_start : json_end + 1]
                else:
                    candidate = clean[json_start:]
                parsed = None
                for cand in (candidate, _repair_json_candidate(candidate) or ""):
                    if not cand:
                        continue
                    try:
                        parsed = json.loads(cand)
                        break
                    except json.JSONDecodeError:
                        continue
            else:
                parsed = None

            if parsed is None:
                logger.warning("No JSON found in LLM response (attempt %d), retrying", attempt)
                messages = list(messages)
                messages.append(
                    {
                        "role": "user",
                        "content": "응답에서 유효한 JSON 객체를 찾을 수 없습니다. 다른 설명 없이 JSON 객체만 반환해주세요.",
                    }
                )
                continue

            # Validate citation IDs
            invalid_ids = self._validate_v2_citations(parsed, registry)
            if not invalid_ids:
                break

            # Retry with error feedback
            if attempt == 0:
                error_feedback = (
                    f"\n\n[SYSTEM FEEDBACK — RETRY]\n"
                    f"The following citation IDs do not exist in the registry: "
                    f"{', '.join(sorted(invalid_ids))}.\n"
                    f"Available IDs: {', '.join(registry.citation_ids())}.\n"
                    f"Regenerate the JSON using only valid citation IDs."
                )
                messages = list(messages)  # copy
                messages.append({"role": "user", "content": error_feedback})
                continue

            raise ValueError(
                f"Citation validation failed after retries. Invalid IDs: {invalid_ids}. "
                f"Last raw (200): {last_raw[:200]}"
            )
        if parsed is None:
            raise ValueError(
                f"Failed to extract valid JSON from LLM after retries. "
                f"Last raw (200): {last_raw[:200]}"
            )

        # Build ChatV2Answered from parsed JSON
        answer = parsed.get("answer", {})
        answer_obj = AnsweredContent(
            empathy=str(answer.get("empathy", "")),
            situation_framing=str(answer.get("situation_framing", "")),
            actions=[
                ActionItem(
                    text=str(a.get("text", "")),
                    basis=str(a.get("basis", "accepted_evidence")),  # type: ignore[arg-type]
                    citation_ids=a.get("citation_ids"),
                    example=a.get("example"),
                    evidence_quote=a.get("evidence_quote"),
                )
                for a in answer.get("actions", [])
            ],
            boundaries=str(answer.get("boundaries", "")),
            summary=str(answer.get("summary", "")),
            narrative=str(answer.get("narrative", "") or ""),
        )

        # Drop evidence claims the model emitted without any citation (flash
        # models sometimes emit empty citation_ids); they are optional
        # enrichment, not part of the core answer.
        evidence_claims = [
            EvidenceClaim(
                claim_id=str(ec.get("claim_id", "")),
                text=str(ec.get("text", "")),
                citation_ids=list(ec.get("citation_ids", [])),
                support_state=str(ec.get("support_state", "supported")) if str(ec.get("support_state", "supported")) in ("supported", "disputed", "conditional") else "supported",  # type: ignore[arg-type]
            )
            for ec in parsed.get("evidence_claims", [])
            if ec.get("citation_ids")
        ]

        # Prefer registry-backed citations so media_kind / excerpts stay accurate
        # even when the model omits or partially fills the citations array.
        registry_by_id = {c.citation_id: c for c in registry.get_all_citations()}
        model_citations = {
            str(c.get("citation_id", "")): c
            for c in parsed.get("citations", [])
            if c.get("citation_id")
        }
        used_ids: list[str] = []
        for a in answer.get("actions", []):
            for cid in a.get("citation_ids") or []:
                if cid not in used_ids:
                    used_ids.append(str(cid))
        for ec in parsed.get("evidence_claims", []):
            for cid in ec.get("citation_ids") or []:
                if cid not in used_ids:
                    used_ids.append(str(cid))
        if not used_ids:
            used_ids = list(registry_by_id.keys())

        # Always surface book + classic sources in citations when present in
        # the retrieval registry, even if the model only cited YouTube IDs.
        # UI 참고 도서 / 이야깃거리 depend on these entries.
        book_ids = [
            cid
            for cid, base in registry_by_id.items()
            if base.media_kind == "book" or (base.source_origin or "").startswith("book")
            or base.source_origin == "classic-literature"
        ]
        # Prefer classics first for narrative empathy, then theory books.
        classic_ids = [
            cid
            for cid in book_ids
            if (registry_by_id[cid].source_origin or "") == "classic-literature"
        ]
        other_book_ids = [cid for cid in book_ids if cid not in classic_ids]
        for cid in classic_ids + other_book_ids:
            if cid not in used_ids:
                used_ids.append(cid)

        citations: list[Citation] = []
        for cid in used_ids:
            if cid not in registry_by_id:
                continue
            base = registry_by_id[cid]
            extra = model_citations.get(cid, {})
            media = str(extra.get("media_kind") or base.media_kind or "youtube")
            if media not in ("youtube", "book", "other"):
                media = base.media_kind
            citations.append(
                Citation(
                    citation_id=base.citation_id,
                    source_type=base.source_type if base.source_type in ("transcript", "claim") else "transcript",  # type: ignore[arg-type]
                    source_id=base.source_id,
                    title=str(extra.get("title") or base.title),
                    creator=str(extra.get("creator") or base.creator),
                    timestamp_url=extra.get("timestamp_url") or base.timestamp_url,
                    start_seconds=extra.get("start_seconds") if extra.get("start_seconds") is not None else base.start_seconds,
                    end_seconds=extra.get("end_seconds") if extra.get("end_seconds") is not None else base.end_seconds,
                    accepted_score=float(extra.get("accepted_score", base.accepted_score)),
                    media_kind=media,  # type: ignore[arg-type]
                    excerpt=extra.get("excerpt") or base.excerpt,
                    source_origin=extra.get("source_origin") or base.source_origin,
                    rights=(
                        str(extra.get("rights") or base.rights or "unknown")
                        if str(extra.get("rights") or base.rights or "unknown")
                        in ("public_domain", "copyrighted_summary", "unknown")
                        else base.rights
                    ),  # type: ignore[arg-type]
                )
            )

        return ChatV2Answered(
            request_id=str(parsed.get("request_id", "unknown")),
            status="answered",
            answer=answer_obj,
            evidence_claims=evidence_claims,
            citations=citations,
            personalization=personalization,
            cultural_reflection=cultural_reflection,
        )

    @staticmethod
    def _validate_v2_citations(
        parsed: dict[str, Any],
        registry: CitationRegistry,
    ) -> set[str]:
        """Return the set of invalid citation IDs found in *parsed*, or empty if valid."""
        allowed = set(registry.citation_ids())
        invalid: set[str] = set()

        # Check evidence_claims
        for ec in parsed.get("evidence_claims", []):
            for cid in ec.get("citation_ids", []):
                if cid not in allowed:
                    invalid.add(cid)

        # Check actions
        for action in parsed.get("answer", {}).get("actions", []):
            for cid in action.get("citation_ids", []) or []:
                if cid not in allowed:
                    invalid.add(cid)

        return invalid

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

Generator = AnswerGenerator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_CONFLICT_SIGNALS = re.compile(
    r"\b(disagree|conflict|contradict|oppos|however|but|on the other hand|"
    r"differ|contrast|unlike|whereas|alternatively)\b",
    re.IGNORECASE,
)


def _detect_conflicts(answer: str, context: list[RetrievalResult]) -> list[str]:
    """Extract conflict descriptions from the generated answer.

    Heuristically looks for sentences that signal disagreement between sources.
    Returns a list of short conflict summaries (may be empty).
    """
    conflicts: list[str] = []
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    for sentence in sentences:
        if _CONFLICT_SIGNALS.search(sentence):
            trimmed = sentence.strip()
            if len(trimmed) > 20:  # skip trivially short matches
                conflicts.append(trimmed[:200])
    return conflicts[:5]  # cap to avoid noise
def _repair_json_candidate(text: str) -> str | None:
    """Best-effort close truncated JSON objects/arrays from LLM output."""
    s = text.strip()
    if not s:
        return None
    # trim trailing incomplete string
    # balance braces/brackets ignoring content in strings
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()
    if in_str:
        s += '"'
    # close remaining in reverse
    while stack:
        opener = stack.pop()
        s += "}" if opener == "{" else "]"
    return s



