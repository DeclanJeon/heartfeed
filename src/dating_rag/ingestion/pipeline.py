"""Ingestion pipeline orchestrating loading, chunking, and storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from dating_rag.ingestion.chunker import (
    TimestampChunker,
    chunk_segments,
    parse_timestamp_segments,
)
from dating_rag.ingestion.loader import load_all_transcripts, load_directory

if TYPE_CHECKING:
    from dating_rag.domain.models import TranscriptChunk


def ingest_directory(
    directory: Path,
    *,
    max_chars: int = 1000,
    overlap_chars: int = 200,
) -> list[TranscriptChunk]:
    """Ingest all transcript files from a directory into chunks.

    Args:
        directory: Path to directory containing transcript markdown files.
        max_chars: Maximum characters per chunk.
        overlap_chars: Overlap between consecutive chunks.

    Returns:
        List of all chunks from all transcripts.
    """
    all_chunks: list[TranscriptChunk] = []

    for doc in load_directory(directory):
        body = doc.get("body", "")
        segments = parse_timestamp_segments(body)

        if not segments:
            continue
        chunks = chunk_segments(
            segments,
            video_id=doc.get("video_id", ""),
            channel_id=doc.get("channel_id", ""),
            channel_name=doc.get("channel_name", ""),
            title=doc.get("title", ""),
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )
        all_chunks.extend(chunks)

    return all_chunks


def _extract_video_id(url: str) -> str:
    """Extract YouTube video ID from a URL."""
    # https://www.youtube.com/watch?v=VIDEO_ID
    if "v=" in url:
        return url.split("v=")[1].split("&")[0].split("#")[0]
    # https://youtu.be/VIDEO_ID
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0].split("/")[0]
    return url


def _timestamp_url(url: str, start_seconds: float) -> str:
    """Build a source URL anchored to the chunk's first transcript timestamp."""
    if not url:
        return ""
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}t={max(0, int(start_seconds))}"


def run_ingestion(
    data_dir: Path | str,
    output_dir: Path | str,
    *,
    target_tokens: int = 360,
    max_tokens: int = 560,
    overlap_tokens: int = 60,
) -> int:
    """Full ingestion pipeline: load → parse → chunk → save.

    Reads all ``*.md`` transcript files from *data_dir*, parses frontmatter
    and timestamped sections, chunks them with :class:`TimestampChunker`,
    and writes one JSON file per transcript to *output_dir*.

    Args:
        data_dir: Directory containing transcript markdown files.
        output_dir: Directory to write chunk JSON files into.
        target_tokens: Preferred chunk size in tokens.
        max_tokens: Hard chunk size ceiling in tokens.
        overlap_tokens: Token overlap between consecutive chunks.

    Returns:
        Total number of chunks written.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunker = TimestampChunker(
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    total_chunks = 0

    for doc in load_all_transcripts(data_dir):
        body = doc.get("body", "")
        title = doc.get("title", "")
        url = doc.get("url", "")
        uploader = doc.get("uploader", "")

        segments = parse_timestamp_segments(body)
        if not segments:
            continue

        video_id = _extract_video_id(url)
        chunks = chunker.split(
            segments,
            video_id=video_id,
            channel_name=uploader,
            title=title,
        )
        for chunk in chunks:
            chunk.language = "ko" if any(ord(c) > 0xAC00 for c in body[:200]) else "en"
            chunk.timestamp_url = _timestamp_url(url, chunk.start_seconds)
            chunk.category = doc.get("category", "")
            if doc.get("views"):
                chunk.views = int(doc["views"])

        # Derive output filename from title
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in title)
        safe_name = safe_name[:80].rstrip("_")
        if not safe_name:
            safe_name = f"transcript_{total_chunks}"

        out_path = output_dir / f"{safe_name}.json"
        chunk_dicts = [chunk.model_dump(mode="json") for chunk in chunks]
        out_path.write_text(
            json.dumps(chunk_dicts, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        total_chunks += len(chunks)

    return total_chunks
