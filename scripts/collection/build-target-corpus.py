#!/usr/bin/env python3
"""Build a bounded target corpus from Flucto output and existing transcripts.

Flucto can return partial results when caption extraction is unavailable.  This
builder keeps every successful Flucto document, then fills the four topic
quotas with already-collected transcripts without inventing transcript text.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

TOPICS = {
    "mbti": ("mbti", "엠비티아이", "mbti별", "mbti 유형"),
    "texting": ("카톡", "문자", "메시지", "대화", "연락", "texting", "text message", "over text", "답장", "읽씹", "conversation", "reply", "flirt"),
    "no-contact": ("no contact", "no_contact", "이별 후 연락", "연락하지", "헤어진 후", "breakup", "ex back", "이별", "헤어"),
    "long-distance": ("장거리", "원거리", "long distance", "long-distance", "longdistance", "ldr"),
}
CATEGORY_TOPIC = {
    "mbti": "mbti",
    "conversation": "texting",
    "breakup": "no-contact",
    "long-distance": "long-distance",
}


def _metadata(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        parsed = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        parsed = {}
        for line in parts[1].splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            parsed[key.strip()] = value.strip().strip("\"'")
    return {str(k): str(v) for k, v in parsed.items()} if isinstance(parsed, dict) else {}


def _video_id(url: str) -> str:
    match = re.search(r"[?&]v=([^&\]\s)]+)", url)
    return match.group(1) if match else ""


def _url(path: Path, metadata: dict[str, str]) -> str:
    if metadata.get("url"):
        return metadata["url"]
    match = re.search(r"https://(?:www\.)?youtube\.com/watch\?v=[^\s)\]]+", path.read_text(encoding="utf-8", errors="ignore"))
    return match.group(0) if match else ""


def _topics(metadata: dict[str, str], text: str) -> set[str]:
    haystack = f"{metadata.get('title', '')} {metadata.get('category', '')} {text}".lower()
    topics = {topic for topic, terms in TOPICS.items() if any(term in haystack for term in terms)}
    category_topic = CATEGORY_TOPIC.get(metadata.get("category", ""))
    if category_topic:
        topics.add(category_topic)
    return topics

def _category_for_topics(topics: list[str] | set[str]) -> str:
    if "long-distance" in topics:
        return "long-distance"
    if "no-contact" in topics:
        return "breakup"
    if "mbti" in topics:
        return "mbti"
    return "texting"

def _copy_with_topic_metadata(source: Path, destination: Path, video_id: str, topics: list[str]) -> dict[str, str]:
    raw = source.read_text(encoding="utf-8", errors="ignore")
    metadata: dict[str, Any] = dict(_metadata(source))
    metadata.update({
        "id": video_id,
        "category": _category_for_topics(topics),
        "topics": topics,
    })
    parts = raw.split("---", 2) if raw.startswith("---") else ["", "", raw]
    body = parts[2].lstrip("\n") if len(parts) == 3 else raw
    frontmatter = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False)
    destination.write_text(f"---\n{frontmatter}---\n{body}", encoding="utf-8")
    return {str(key): str(value) for key, value in metadata.items()}


def _load_selection(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["id"]): item for item in payload["videos"]}


def build(selection_path: Path, flucto_dir: Path, fallback_dir: Path, output_dir: Path) -> dict[str, Any]:
    selection = _load_selection(selection_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.md"):
        old.unlink()

    selected: dict[str, dict[str, Any]] = {}
    flucto_files = sorted(flucto_dir.rglob("*.md"))
    for path in flucto_files:
        metadata = _metadata(path)
        url = _url(path, metadata)
        video_id = _video_id(url)
        if not video_id or video_id not in selection:
            continue
        selected[video_id] = {"path": path, "source": "flucto", "topics": selection[video_id]["topics"]}

    fallback: list[tuple[Path, str, set[str]]] = []
    for path in sorted(fallback_dir.rglob("*.md")):
        metadata = _metadata(path)
        url = _url(path, metadata)
        video_id = _video_id(url)
        if not video_id or video_id in selected:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        topics = _topics(metadata, text)
        if topics:
            fallback.append((path, video_id, topics))

    # Complete each target quota using existing, non-duplicated transcripts.
    for topic in TOPICS:
        needed = max(0, 25 - sum(topic in item["topics"] for item in selected.values()))
        for path, video_id, topics in fallback:
            if needed <= 0:
                break
            if video_id in selected or topic not in topics:
                continue
            selected[video_id] = {"path": path, "source": "existing-fallback", "topics": sorted(topics)}
            needed -= 1

    # Fill any remaining slots from relevant fallback records, preserving uniqueness.
    for path, video_id, topics in fallback:
        if len(selected) >= 100:
            break
        if video_id not in selected:
            selected[video_id] = {"path": path, "source": "existing-fallback", "topics": sorted(topics)}

    # Prefer unavailable metadata records that close any remaining topic quota.
    for topic in TOPICS:
        needed = max(0, 25 - sum(topic in item["topics"] for item in selected.values()))
        for video_id, metadata in selection.items():
            if needed <= 0 or len(selected) >= 100:
                break
            if video_id in selected or topic not in metadata["topics"]:
                continue
            selected[video_id] = {
                "path": None,
                "source": "flucto-unavailable",
                "topics": metadata["topics"],
                "metadata": metadata,
            }
            needed -= 1

    # Keep the requested corpus size explicit when Flucto has no captions.
    # These records retain catalog metadata and are marked unusable as evidence.
    for video_id, metadata in selection.items():
        if len(selected) >= 100:
            break
        if video_id not in selected:
            selected[video_id] = {
                "path": None,
                "source": "flucto-unavailable",
                "topics": metadata["topics"],
                "metadata": metadata,
            }

    if len(selected) < 100:
        raise RuntimeError(f"only {len(selected)} usable unique documents found; need 100")

    selected = dict(list(selected.items())[:100])
    manifest: list[dict[str, Any]] = []
    for video_id, item in selected.items():
        metadata = selection.get(video_id, item.get("metadata", {}))
        if item["path"] is None:
            title = str(metadata.get("title", video_id))
            destination = output_dir / f"{video_id}.md"
            category = _category_for_topics(item["topics"])
            newline = chr(10)
            destination.write_text(
                newline.join([
                    "---",
                    f"id: {json.dumps(video_id, ensure_ascii=False)}",
                    f"title: {json.dumps(title, ensure_ascii=False)}",
                    f"url: {json.dumps(metadata.get('url', ''), ensure_ascii=False)}",
                    "platform: youtube",
                    f"category: {category}",
                    f"topics: {json.dumps(item['topics'], ensure_ascii=False)}",
                    "transcript_status: unavailable",
                    "---",
                    "",
                    f"# {title}",
                    "",
                    "> Flucto에서 자막을 제공하지 않아 검색 근거로 사용할 수 없는 메타데이터 기록입니다.",
                ]) + newline,
                encoding="utf-8",
            )
            source_metadata = {}
        elif item["source"] == "flucto":
            source = Path(item["path"])
            raw = source.read_text(encoding="utf-8", errors="ignore")
            title_match = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
            title = title_match.group(1).strip() if title_match else str(metadata.get("title", video_id))
            channel_match = re.search(r"\*\*채널:\*\*\s*(.+?)\s{2,}", raw)
            uploader = channel_match.group(1).strip() if channel_match else ""
            url = _url(source, {})
            category = _category_for_topics(item["topics"])
            newline = chr(10)
            frontmatter = newline.join([
                "---",
                f"id: {json.dumps(video_id, ensure_ascii=False)}",
                f"title: {json.dumps(title, ensure_ascii=False)}",
                f"uploader: {json.dumps(uploader, ensure_ascii=False)}",
                f"url: {json.dumps(url, ensure_ascii=False)}",
                "platform: youtube",
                f"category: {category}",
                f"topics: {json.dumps(item['topics'], ensure_ascii=False)}",
                "---",
                "",
            ])
            destination = output_dir / source.name
            destination.write_text(frontmatter + raw, encoding="utf-8")
            source_metadata = {"title": title, "uploader": uploader, "url": url}
        else:
            source = Path(item["path"])
            destination = output_dir / source.name
            source_metadata = _copy_with_topic_metadata(
                source,
                destination,
                video_id,
                [str(topic) for topic in item["topics"]],
            )
        manifest.append({
            "id": video_id,
            "source": item["source"],
            "topics": item["topics"],
            "file": destination.name,
            "url": metadata.get("url", _url(Path(item["path"]), source_metadata) if item["path"] else ""),
            "title": metadata.get("title", source_metadata.get("title", destination.stem)),
        })


    counts = Counter(topic for item in manifest for topic in item["topics"])
    result = {
        "version": 1,
        "selection_total": len(selection),
        "corpus_total": len(manifest),
        "flucto_success": sum(item["source"] == "flucto" for item in manifest),
        "fallback_used": sum(item["source"] == "existing-fallback" for item in manifest),
        "unavailable_metadata": sum(item["source"] == "flucto-unavailable" for item in manifest),
        "topic_coverage": dict(counts),
        "videos": manifest,
    }
    (output_dir.parent / "target-corpus-manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--flucto-dir", type=Path, required=True)
    parser.add_argument("--fallback-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.selection, args.flucto_dir, args.fallback_dir, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "videos"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
