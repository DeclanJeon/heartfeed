"""Regression checks for the Flucto-backed target-topic corpus."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.index_target_corpus import _stabilize_chunk_ids
from dating_rag.ingestion.loader import load_all_transcripts, load_transcript
from dating_rag.ingestion.pipeline import run_ingestion

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data/source/flucto-target/corpus"
MANIFEST_PATH = ROOT / "data/source/flucto-target/target-corpus-manifest.json"


def test_target_corpus_is_bounded_unique_and_topic_covered() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    videos = manifest["videos"]

    assert manifest["corpus_total"] == 100
    assert len(videos) == len({item["id"] for item in videos})
    assert manifest["flucto_success"] > 0
    assert manifest["topic_coverage"]["mbti"] >= 20
    assert manifest["topic_coverage"]["texting"] >= 20
    assert manifest["topic_coverage"]["no-contact"] >= 20
    assert manifest["topic_coverage"]["long-distance"] >= 20


def test_flucto_markdown_has_loader_metadata_and_timestamps() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    record = next(item for item in manifest["videos"] if item["source"] == "flucto")
    document = load_transcript(CORPUS_DIR / record["file"])

    assert document["id"] == record["id"]
    assert document["url"].startswith("https://www.youtube.com/watch?v=")
    assert document["uploader"]
    assert "## [" in document["body"]



def test_fallback_markdown_retains_topic_metadata() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    record = next(item for item in manifest["videos"] if item["source"] == "existing-fallback")
    document = load_transcript(CORPUS_DIR / record["file"])

    assert document["category"] in {"mbti", "texting", "breakup", "long-distance"}
    assert record["topics"][0] in document["topics"]


def test_loader_recurses_target_corpus() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    documents = list(load_all_transcripts(CORPUS_DIR))
    expected_transcripts = manifest["flucto_success"] + manifest["fallback_used"]

    assert len(documents) == manifest["corpus_total"]
    assert sum(bool(doc.get("body")) for doc in documents) >= expected_transcripts
def test_reconciled_manifest_artifacts_exist() -> None:
    manifest_path = ROOT / "data/source/flucto-target/manifest.jsonl"
    audit_path = ROOT / "data/source/flucto-target/ingestion-audit.json"
    assert manifest_path.exists()
    assert audit_path.exists()
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert len(rows) == 100
    assert audit["summary"]["input_files"] == 100
    assert audit["summary"]["timestamp_documents"] == 68
    assert audit["summary"]["total_chunks"] == 319

def test_ingestion_writes_typed_manifest_and_audit(tmp_path: Path) -> None:
    source = tmp_path / "source"
    chunks = tmp_path / "chunks"
    source.mkdir()
    (source / "sample.md").write_text(
        "---\n"
        'id: "vid-1"\n'
        'title: "Sample"\n'
        'uploader: "Channel"\n'
        'url: "https://www.youtube.com/watch?v=vid-1"\n'
        "category: texting\n"
        "topics: [texting, conversation]\n"
        "views: 123\n"
        "fallback_used: false\n"
        "---\n"
        "## [00:00]\nHello there.\n"
        "## [00:30]\nHow are you?\n",
        encoding="utf-8",
    )
    (source / "nullable.md").write_text(
        "---\n"
        'id: "vid-2"\n'
        'title: "Nullable"\n'
        'url: "https://www.youtube.com/watch?v=vid-2"\n'
        "topics: null\n"
        "tags: null\n"
        "---\n"
        "## [00:00]\nA transcript without optional lists.\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.jsonl"
    audit = tmp_path / "audit.json"

    assert run_ingestion(
        source,
        chunks,
        source_origin="test",
        ingestion_run_id="run-test",
        manifest_path=manifest,
        audit_path=audit,
    ) > 0

    record = next(
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["video_id"] == "vid-1"
    )
    assert record["loader_status"] == "ok"
    assert record["timestamp_segment_count"] == 2
    assert record["raw_sha256"].startswith("sha256:")
    assert json.loads(audit.read_text(encoding="utf-8"))["summary"]["total_chunks"] > 0
    chunk = next(
        row
        for path in chunks.glob("*.json")
        for row in json.loads(path.read_text(encoding="utf-8"))
        if row["video_id"] == "vid-1"
    )
    assert chunk["tags"] == ["texting", "conversation"]
    assert chunk["views"] == 123
    assert chunk["source_origin"] == "test"
    assert chunk["evidence_role"] == "source_evidence"
    nullable = next(
        row
        for path in chunks.glob("*.json")
        for row in json.loads(path.read_text(encoding="utf-8"))
        if row["video_id"] == "vid-2"
    )
    assert nullable["tags"] == []


def test_target_chunk_ids_are_stable_and_content_aware(tmp_path: Path) -> None:
    chunk_path = tmp_path / "chunks.json"
    base = {
        "chunk_id": "random",
        "video_id": "video-1",
        "chunk_index": 0,
        "start_seconds": 0,
        "content_sha256": "sha256:aaa",
        "chunk_policy_version": "timestamp-v1-360-560-60",
    }
    chunk_path.write_text(json.dumps([base]), encoding="utf-8")

    _stabilize_chunk_ids(tmp_path)
    first = json.loads(chunk_path.read_text(encoding="utf-8"))[0]["chunk_id"]
    chunk_path.write_text(json.dumps([{**base, "chunk_id": "rerun-random"}]), encoding="utf-8")
    _stabilize_chunk_ids(tmp_path)
    rerun = json.loads(chunk_path.read_text(encoding="utf-8"))[0]["chunk_id"]
    assert first == rerun
    changed = {**base, "chunk_id": "different-random", "content_sha256": "sha256:bbb"}
    chunk_path.write_text(json.dumps([changed]), encoding="utf-8")
    _stabilize_chunk_ids(tmp_path)
    second = json.loads(chunk_path.read_text(encoding="utf-8"))[0]["chunk_id"]

    assert first != second
