"""Ingestion pipeline orchestrating loading, chunking, and storage."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_URL, uuid5

from dating_rag.ingestion.chunker import (
    TimestampChunker,
    chunk_segments,
    parse_timestamp_segments,
)
from dating_rag.ingestion.loader import load_directory, load_transcript

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
def _as_bool(value: object) -> bool:
    """Parse common YAML/JSON boolean representations."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}

def _as_list(value: object) -> list[object]:
    """Normalize optional YAML list metadata without dropping the document."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]
def _as_int(value: object, default: int = 0) -> int:
    """Parse an integer without making ingestion fail on malformed metadata."""
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _as_datetime(value: object) -> datetime | None:
    """Parse ISO timestamps while accepting the common trailing ``Z`` form."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None

def _safe_output_name(video_id: str, title: str, fallback: str) -> str:
    """Build a collision-resistant, filesystem-safe chunk filename."""
    safe_video = "".join(
        char if char.isalnum() or char in ("-", "_") else "_" for char in video_id
    )[:32].strip("_") or "unknown-video"
    safe_title = "".join(
        char if char.isalnum() or char in ("-", "_") else "_" for char in title
    )[:80].rstrip("_")
    safe_fallback = "".join(
        char if char.isalnum() or char in ("-", "_") else "_" for char in fallback
    )[:40].strip("_") or "transcript"
    return f"{safe_video}__{safe_title or safe_fallback}.json"


def run_ingestion(
    data_dir: Path | str,
    output_dir: Path | str,
    *,
    target_tokens: int = 360,
    max_tokens: int = 560,
    overlap_tokens: int = 60,
    source_origin: str = "",
    ingestion_run_id: str | None = None,
    manifest_path: Path | str | None = None,
    audit_path: Path | str | None = None,
) -> int:
    """Full ingestion with durable source/chunk reconciliation."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = ingestion_run_id or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    chunk_policy = f"timestamp-v1-{target_tokens}-{max_tokens}-{overlap_tokens}"
    chunker = TimestampChunker(
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    records: list[dict[str, Any]] = []
    total_chunks = 0
    for path in sorted(data_dir.rglob("*.md")):
        read_error: str | None = None
        try:
            raw_bytes = path.read_bytes()
        except OSError as exc:
            raw_bytes = b""
            read_error = f"{type(exc).__name__}: {exc}"
        raw_sha256 = hashlib.sha256(raw_bytes).hexdigest() if raw_bytes else ""
        record: dict[str, Any] = {
            "document_id": "",
            "video_id": "",
            "path": str(path.relative_to(data_dir)),
            "source_origin": source_origin,
            "transcript_status": "unavailable",
            "fallback_used": False,
            "raw_sha256": f"sha256:{raw_sha256}" if raw_sha256 else "",
            "bytes": len(raw_bytes),
            "loader_status": "error" if read_error else "pending",
            "skip_reason": read_error,
            "timestamp_segment_count": 0,
            "chunk_count": 0,
            "chunk_file": None,
        }
        if read_error:
            records.append(record)
            continue
        try:
            doc = load_transcript(path)
            url = str(doc.get("url", "") or "")
            video_id = _extract_video_id(url) if url else str(doc.get("id", "") or "")
            body = str(doc.get("body", "") or "")
            segments = parse_timestamp_segments(body)
            record.update(
                {
                    "video_id": video_id,
                    "document_id": f"doc:youtube:{video_id}" if video_id else "",
                    "title": str(doc.get("title", "") or ""),
                    "url": url,
                    "channel_name": str(doc.get("uploader", "") or ""),
                    "category": str(doc.get("category", "") or ""),
                    "topics": doc.get("topics", []),
                    "loader_status": "ok",
                    "transcript_status": "available" if segments else "unavailable",
                    "fallback_used": _as_bool(doc.get("fallback_used", False)),
                    "timestamp_segment_count": len(segments),
                }
            )
            if not segments:
                record["skip_reason"] = "no_timestamp_segments"
                records.append(record)
                continue
            chunks = chunker.split(
                segments,
                video_id=video_id,
                channel_id=str(doc.get("channel_id", "") or ""),
                channel_name=str(doc.get("uploader", "") or ""),
                title=str(doc.get("title", "") or ""),
            )
            topics = _as_list(doc.get("topics"))
            tags = _as_list(doc.get("tags"))
            merged_tags = list(dict.fromkeys([str(item) for item in [*topics, *tags]]))
            for chunk in chunks:
                chunk.language = "ko" if any(ord(char) > 0xAC00 for char in body[:200]) else "en"
                chunk.timestamp_url = _timestamp_url(url, chunk.start_seconds)
                chunk.category = str(doc.get("category", "") or "")
                chunk.tags = merged_tags
                chunk.views = _as_int(doc.get("views", 0))
                chunk.published_at = _as_datetime(doc.get("published_at"))
                chunk.source_origin = source_origin
                chunk.corpus_type = "transcript"
                chunk.evidence_role = "source_evidence"
                chunk.transcript_status = "available"
                chunk.fallback_used = _as_bool(doc.get("fallback_used", False))
                chunk.raw_document_id = str(record["document_id"])
                chunk.raw_sha256 = str(record["raw_sha256"])
                chunk.ingestion_run_id = run_id
                chunk.chunk_policy_version = chunk_policy
                chunk.content_sha256 = f"sha256:{hashlib.sha256(chunk.text.encode('utf-8')).hexdigest()}"
                chunk.schema_version = 2
                chunk.chunk_id_version = 2

            old_to_new: dict[str, str] = {}
            for chunk in chunks:
                start_ms = int(chunk.start_seconds * 1000)
                stable_id = str(
                    uuid5(
                        NAMESPACE_URL,
                        f"datewise:transcript:{chunk_policy}:{video_id}:{start_ms}:{chunk.chunk_index}:{chunk.content_sha256[:24]}",
                    )
                )
                old_to_new[chunk.chunk_id] = stable_id
                chunk.chunk_id = stable_id
            for chunk in chunks:
                if chunk.previous_chunk_id:
                    chunk.previous_chunk_id = old_to_new.get(
                        chunk.previous_chunk_id, chunk.previous_chunk_id
                    )
                if chunk.next_chunk_id:
                    chunk.next_chunk_id = old_to_new.get(
                        chunk.next_chunk_id, chunk.next_chunk_id
                    )

            output_name = _safe_output_name(video_id, str(doc.get("title", "")), path.stem)
            out_path = output_dir / output_name
            out_path.write_text(
                json.dumps(
                    [chunk.model_dump(mode="json") for chunk in chunks],
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            record["chunk_count"] = len(chunks)
            record["chunk_file"] = output_name
            total_chunks += len(chunks)
        except Exception as exc:  # noqa: BLE001 - record and continue per-file
            record["loader_status"] = "error"
            record["skip_reason"] = f"{type(exc).__name__}: {exc}"
        records.append(record)

    summary = {
        "schema_version": 1,
        "ingestion_run_id": run_id,
        "source_dir": str(data_dir),
        "output_dir": str(output_dir),
        "source_origin": source_origin,
        "chunk_policy_version": chunk_policy,
        "input_files": len(records),
        "loaded_files": sum(record["loader_status"] == "ok" for record in records),
        "skipped_files": sum(record["chunk_count"] == 0 for record in records),
        "timestamp_documents": sum(record["timestamp_segment_count"] > 0 for record in records),
        "total_chunks": total_chunks,
    }
    if manifest_path is not None:
        manifest = Path(manifest_path)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
    if audit_path is not None:
        audit = Path(audit_path)
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_text(
            json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return total_chunks
