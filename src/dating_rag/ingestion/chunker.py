"""Transcript chunking with overlap for context preservation."""

from __future__ import annotations

import re

from dating_rag.domain.models import TranscriptChunk, TranscriptSegment

# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

_TS_HEADER_RE = re.compile(
    r"^##\s*\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*$",
    re.MULTILINE,
)

_TS_INLINE_RE = re.compile(
    r"\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]"
    r"(?:\s*-\s*\[(\d{1,2}):(\d{2})(?::(\d{2}))?\])?\s*(.*)",
)


def _to_seconds(a: str, b: str, c: str | None) -> float:
    """Convert timestamp capture groups to seconds.

    2-part ``[MM:SS]`` -> a=MM, b=SS.
    3-part ``[HH:MM:SS]`` -> a=HH, b=MM, c=SS.
    """
    if c is not None:
        return float(int(a) * 3600 + int(b) * 60 + int(c))
    return float(int(a) * 60 + int(b))


def _ts_to_seconds(ts: str) -> float:
    """Convert a ``MM:SS`` or ``HH:MM:SS`` string to seconds."""
    parts = ts.split(":")
    if len(parts) == 3:
        return float(int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]))
    return float(int(parts[0]) * 60 + int(parts[1]))


def parse_timestamp_segments(text: str) -> list[TranscriptSegment]:
    """Parse timestamp-formatted transcript into segments.

    Supports two formats:

    1. **Markdown headers** — ``## [MM:SS]`` or ``## [HH:MM:SS]`` followed by
       transcript text on subsequent lines (the actual data format).
    2. **Inline timestamps** — ``[MM:SS] text`` or
       ``[HH:MM:SS] - [HH:MM:SS] text`` (legacy / test format).

    Format 1 is tried first; if no ``##`` headers are found, format 2 is used.

    Args:
        text: Raw transcript text with timestamps.

    Returns:
        List of parsed transcript segments.
    """
    segments: list[TranscriptSegment] = []

    # ── Try format 1: markdown ``## [TS]`` headers ──────────────────────
    header_matches = list(_TS_HEADER_RE.finditer(text))
    if header_matches:
        for idx, match in enumerate(header_matches):
            start_seconds = _ts_to_seconds(match.group(1))
            text_start = match.end()
            text_end = (
                header_matches[idx + 1].start()
                if idx + 1 < len(header_matches)
                else len(text)
            )
            body = text[text_start:text_end].strip()
            if not body:
                continue

            # End = next header's start timestamp, or start + 30 s fallback
            if idx + 1 < len(header_matches):
                end_seconds = _ts_to_seconds(header_matches[idx + 1].group(1))
            else:
                end_seconds = start_seconds + 30.0

            segments.append(
                TranscriptSegment(
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    text=body,
                )
            )
        return segments

    # ── Fallback: inline ``[TS] text`` format ───────────────────────────
    for match in _TS_INLINE_RE.finditer(text):
        groups = match.groups()
        start_seconds = _to_seconds(groups[0], groups[1], groups[2])

        if groups[3] is not None:
            end_seconds = _to_seconds(groups[3], groups[4], groups[5])
        else:
            end_seconds = start_seconds + 30.0

        segments.append(
            TranscriptSegment(
                start_seconds=float(start_seconds),
                end_seconds=float(end_seconds),
                text=groups[-1].strip(),
            )
        )

    return segments


# ---------------------------------------------------------------------------
# Classic functional API (backward-compatible)
# ---------------------------------------------------------------------------

def chunk_segments(
    segments: list[TranscriptSegment],
    *,
    video_id: str = "",
    channel_id: str = "",
    channel_name: str = "",
    title: str = "",
    max_chars: int = 1000,
    overlap_chars: int = 200,
) -> list[TranscriptChunk]:
    """Chunk transcript segments into overlapping windows.

    Args:
        segments: Parsed transcript segments.
        video_id: Source video identifier.
        channel_id: Source channel identifier.
        channel_name: Source channel name.
        title: Video title.
        max_chars: Maximum characters per chunk.
        overlap_chars: Overlap between consecutive chunks.

    Returns:
        List of transcript chunks with linked list navigation.
    """
    if not segments:
        return []

    chunks: list[TranscriptChunk] = []
    current_text = ""
    current_start = segments[0].start_seconds
    current_end = segments[0].end_seconds
    chunk_index = 0

    for segment in segments:
        candidate = f"{current_text} {segment.text}".strip() if current_text else segment.text

        if len(candidate) > max_chars and current_text:
            # Flush current chunk
            chunks.append(
                TranscriptChunk(
                    video_id=video_id,
                    channel_id=channel_id,
                    channel_name=channel_name,
                    title=title,
                    text=current_text,
                    start_seconds=current_start,
                    end_seconds=current_end,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

            # Overlap: keep tail of previous chunk
            overlap_text = current_text[-overlap_chars:] if len(current_text) > overlap_chars else current_text
            current_text = f"{overlap_text} {segment.text}".strip()
            current_start = segment.start_seconds
            current_end = segment.end_seconds
        else:
            current_text = candidate
            current_end = segment.end_seconds

    # Final chunk
    if current_text:
        chunks.append(
            TranscriptChunk(
                video_id=video_id,
                channel_id=channel_id,
                channel_name=channel_name,
                title=title,
                text=current_text,
                start_seconds=current_start,
                end_seconds=current_end,
                chunk_index=chunk_index,
            )
        )

    # Link chunks
    for i, chunk in enumerate(chunks):
        if i > 0:
            chunk.previous_chunk_id = chunks[i - 1].chunk_id
        if i < len(chunks) - 1:
            chunk.next_chunk_id = chunks[i + 1].chunk_id

    return chunks


# ---------------------------------------------------------------------------
# TimestampChunker (token-aware, class-based API)
# ---------------------------------------------------------------------------

# Sentence-ending punctuation — works for English and Korean
_SENTENCE_END_RE = re.compile(r"[.!?。！？]\s*$")
_WORD_BOUNDARY_RE = re.compile(r"\s+")

# Rough approximation: 1 token ≈ 4 characters (conservative for mixed ko/en)
_CHAR_PER_TOKEN = 4


class TimestampChunker:
    """Token-aware transcript chunker with sentence-boundary splitting.

    Parameters
    ----------
    target_tokens:
        Preferred maximum chunk size in tokens.
    max_tokens:
        Hard ceiling — a chunk MUST be split at or before this size.
    overlap_tokens:
        Number of tokens to carry over from the end of one chunk to the
        start of the next for context continuity.
    """

    def __init__(
        self,
        *,
        target_tokens: int = 360,
        max_tokens: int = 560,
        overlap_tokens: int = 60,
    ) -> None:
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    # ── public API ──────────────────────────────────────────────────────


    def split(
        self,
        segments: list[TranscriptSegment],
        *,
        video_id: str = "",
        channel_id: str = "",
        channel_name: str = "",
        title: str = "",
    ) -> list[TranscriptChunk]:
        """Split transcript segments into chunks.

        1. Merge tiny segments (< 100 tokens ~ 400 chars).
        2. Accumulate until reaching *target_tokens*.
        3. At the target boundary, split at the last sentence-ending
           punctuation (works for English and Korean).
        4. If no sentence boundary exists by *max_tokens*, force-split
           at the last whitespace.
        5. Carry *overlap_tokens* from the tail of each chunk.

        Returns:
            List of :class:`TranscriptChunk` instances.
        """
        if not segments:
            return []

        merged = self._merge_tiny(segments)
        return self._build_chunks(
            merged,
            video_id=video_id,
            channel_id=channel_id,
            channel_name=channel_name,
            title=title,
        )

    # ── internal ────────────────────────────────────────────────────────

    def _merge_tiny(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        """Merge segments shorter than 100 tokens (~400 chars) into neighbours."""
        min_chars = 100 * _CHAR_PER_TOKEN
        merged: list[TranscriptSegment] = []
        buffer = ""

        for seg in segments:
            if not buffer:
                buffer = seg.text
                buf_start = seg.start_seconds
                buf_end = seg.end_seconds
            elif len(buffer) < min_chars:
                buffer = f"{buffer} {seg.text}"
                buf_end = seg.end_seconds
            else:
                merged.append(TranscriptSegment(
                    start_seconds=buf_start,
                    end_seconds=buf_end,
                    text=buffer,
                ))
                buffer = seg.text
                buf_start = seg.start_seconds
                buf_end = seg.end_seconds

        if buffer:
            merged.append(TranscriptSegment(
                start_seconds=buf_start,
                end_seconds=buf_end,
                text=buffer,
            ))

        return merged

    def _build_chunks(
        self,
        segments: list[TranscriptSegment],
        *,
        video_id: str = "",
        channel_id: str = "",
        channel_name: str = "",
        title: str = "",
    ) -> list[TranscriptChunk]:
        """Accumulate merged segments into token-limited chunks."""
        target_chars = self.target_tokens * _CHAR_PER_TOKEN
        max_chars = self.max_tokens * _CHAR_PER_TOKEN
        overlap_chars = self.overlap_tokens * _CHAR_PER_TOKEN

        raw_chunks: list[tuple[str, float, float]] = []
        current_text = ""
        current_start = segments[0].start_seconds
        current_end = segments[0].end_seconds

        for seg in segments:
            candidate = f"{current_text} {seg.text}".strip() if current_text else seg.text
            current_end = seg.end_seconds

            # Under target — keep accumulating
            if len(candidate) <= target_chars:
                current_text = candidate
                continue

            # Over target — try to split current_text at sentence boundary
            if current_text:
                boundary = self._find_sentence_boundary(current_text, target_chars)
                if boundary is not None and boundary > 0:
                    raw_chunks.append((current_text[:boundary].strip(), current_start, current_end))
                    remainder = current_text[boundary:].strip()
                    current_text = f"{remainder} {seg.text}".strip() if remainder else seg.text
                    current_start = seg.start_seconds
                    continue

            # No sentence boundary found (or current_text was empty)
            if len(candidate) > max_chars:
                if current_text:
                    # Split accumulated text at word boundary
                    split_pos = self._find_word_boundary(current_text, max_chars)
                    if split_pos and split_pos > 0:
                        raw_chunks.append((current_text[:split_pos].strip(), current_start, current_end))
                        remainder = current_text[split_pos:].strip()
                    else:
                        raw_chunks.append((current_text, current_start, current_end))
                        remainder = ""
                    current_text = f"{remainder} {seg.text}".strip() if remainder else seg.text
                    current_start = seg.start_seconds
                else:
                    # Single oversized segment — split it directly
                    remaining = seg.text
                    while len(remaining) > max_chars:
                        split_pos = self._find_word_boundary(remaining, max_chars)
                        if split_pos and split_pos > 0:
                            raw_chunks.append((remaining[:split_pos].strip(), seg.start_seconds, seg.end_seconds))
                            remaining = remaining[split_pos:].strip()
                        else:
                            raw_chunks.append((remaining[:max_chars], seg.start_seconds, seg.end_seconds))
                            remaining = remaining[max_chars:]
                    current_text = remaining
                    current_start = seg.start_seconds
            else:
                # Between target and max, no sentence boundary yet
                current_text = candidate

        # Flush remaining
        if current_text:
            raw_chunks.append((current_text, current_start, current_end))

        # Build TranscriptChunk objects with overlap and linking
        chunks: list[TranscriptChunk] = []
        for idx, (text, start, end) in enumerate(raw_chunks):
            if idx > 0 and overlap_chars > 0:
                prev_text = raw_chunks[idx - 1][0]
                tail = prev_text[-overlap_chars:] if len(prev_text) > overlap_chars else prev_text
                text = f"{tail} {text}".strip()

            chunks.append(TranscriptChunk(
                video_id=video_id,
                channel_id=channel_id,
                channel_name=channel_name,
                title=title,
                text=text,
                start_seconds=start,
                end_seconds=end,
                chunk_index=idx,
            ))

        # Link
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.previous_chunk_id = chunks[i - 1].chunk_id
            if i < len(chunks) - 1:
                chunk.next_chunk_id = chunks[i + 1].chunk_id

        return chunks


    @staticmethod
    def _find_sentence_boundary(text: str, before: int) -> int | None:
        """Find the position just after the last sentence-ending punctuation
        at or before *before* characters.  Returns ``None`` if none found."""
        search_region = text[:before]
        # Find all sentence-ending positions
        matches = list(_SENTENCE_END_RE.finditer(search_region))
        if matches:
            return matches[-1].end()
        return None

    @staticmethod
    def _find_word_boundary(text: str, before: int) -> int | None:
        """Find the last whitespace at or before *before* characters."""
        search_region = text[:before]
        matches = list(_WORD_BOUNDARY_RE.finditer(search_region))
        if matches:
            return matches[-1].end()
        return None
