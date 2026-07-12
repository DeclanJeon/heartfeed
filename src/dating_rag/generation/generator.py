"""LLM-based response generation with streaming and citation support."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx

from dating_rag.domain.models import ChatResponse, QueryPlan, RetrievalResult
from dating_rag.generation.prompts import SYSTEM_PROMPT, build_prompt


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
    """

    def __init__(
        self,
        api_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "gpt-4o-mini",
        *,
        timeout: float = 60.0,
    ) -> None:
        """Initialize the generator.

        Args:
            api_url: OpenAI-compatible API base URL.
            api_key: API authentication key.
            model: Model identifier.
            timeout: HTTP request timeout in seconds.
        """
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    # -- helpers ----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

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

        response = await self._client.post(
            f"{self.api_url}/chat/completions",
            headers=self._headers(),
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
        )
        response.raise_for_status()

        data: dict[str, Any] = response.json()
        return str(data["choices"][0]["message"]["content"])

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

        async with self._client.stream(
            "POST",
            f"{self.api_url}/chat/completions",
            headers=self._headers(),
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
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
