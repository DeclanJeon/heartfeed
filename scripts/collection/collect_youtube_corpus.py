#!/usr/bin/env python3
"""Search YouTube for dating/relationship videos and collect metadata + transcripts.

Usage: python3 collect_youtube_corpus.py --output ./data/source/new-corpus --limit 600
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Categories and search keywords for Korean + English dating content
CATEGORIES = {
    "first_dates": {
        "name": "첫 데이트",
        "keywords": [
            "첫 데이트 대화", "첫 데이트 팁", "first date tips", "first date conversation",
            "데이트 코스 추천", "썸 타는 법", "how to impress on first date",
            "첫 만남 대화 주제", "소개팅 대화법",
        ],
    },
    "conversation": {
        "name": "대화법",
        "keywords": [
            "연애 대화법", "카톡 대화법", "썸 카톡", "dating conversation tips",
            "flirting techniques", "대화 잘하는 법 연애", "연락 잘하는 법",
            "message reply tips", "texting games",
        ],
    },
    "conflict_resolution": {
        "name": "갈등 해결",
        "keywords": [
            "연애 싸움 화해", "커플 갈등 해결", "relationship conflict resolution",
            "couple fight makeup", "연인 다툼 해결", "연애 갈등 대화법",
            "how to resolve argument with partner", "healthy relationship communication",
        ],
    },
    "breakup": {
        "name": "이별",
        "keywords": [
            "이별 극복", "헤어진 후 마음", "breakup recovery", "how to get over breakup",
            "전 애인 마음", "재회 방법", "no contact rule", "이별 후 자기관리",
            "moving on after breakup",
        ],
    },
    "mbti": {
        "name": "MBTI",
        "keywords": [
            "mbti 연애 스타일", "mbti 궁합", "mbti dating compatibility",
            "mbti 연애 유형", "mbti 별 이상형", "mbti relationship advice",
            "mbti 커플 궁합", "mbti 연애 성격",
        ],
    },
    "long_distance": {
        "name": "장거리",
        "keywords": [
            "장거리 연애", "long distance relationship", "장거리 연애 팁",
            "long distance relationship advice", "장거리 커플", "원거리 연애 극복",
            "ldr tips korean",
        ],
    },
    "attraction": {
        "name": "매력",
        "keywords": [
            "매력 어필", "호감 가는 행동", "how to attract someone",
            "dating confidence", "자신감 있는 연애", "first impression tips",
            "매력적인 사람이 되는 법", "charisma dating",
        ],
    },
    "red_flags": {
        "name": "위험 신호",
        "keywords": [
            "연애 red flag", "toxic relationship signs", "나쁜 남자 여자 구별법",
            "relationship red flags", "gaslighting 연애", "관련 조작",
            "healthy relationship signs", "건강한 연애",
        ],
    },
    "attachment_style": {
        "name": "애착 유형",
        "keywords": [
            "애착 유형 연애", "attachment style dating", "회피형 애착",
            "불안형 애착", "anxious avoidant relationship", "애착 이론",
            "secure attachment relationship",
        ],
    },
    "self_improvement": {
        "name": "자기계발",
        "keywords": [
            "연애 자기계발", "자존감 높이는 법", "self confidence dating",
            "single life tips", "혼자 행복하기", "self improvement for dating",
            "자기 사랑", "self love relationship",
        ],
    },
}

# Channels to search from (existing corpus channels + new ones)
TARGET_CHANNELS = [
    "포텐티아",  # User recommended
    "Matthew Hussey",
    "Coach Lee",
    "Psych2Go",
    "Charisma on Command",
    "Jay Shetty",
    "Brad Browning",
    "토모토모TomoTomo",
    "김달 (Moon)",
    "ㄸㅎ Deep",
]


def get_video_info(video_id: str) -> dict | None:
    """Get video metadata using yt-dlp."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-download", f"https://www.youtube.com/watch?v={video_id}"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception:
        pass
    return None


def search_youtube(keyword: str, limit: int = 20) -> list[dict]:
    """Search YouTube using yt-dlp ytsearch."""
    try:
        result = subprocess.run(
            ["yt-dlp", f"ytsearch{limit}:{keyword}", "--flat-playlist", "--dump-json", "--no-warnings"],
            capture_output=True, text=True, timeout=60,
        )
        videos = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                try:
                    videos.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return videos
    except Exception as e:
        print(f"  Search failed for '{keyword}': {e}", file=sys.stderr)
        return []


def get_transcript(video_id: str, languages: list[str] = None) -> list[dict] | None:
    """Get video transcript using youtube-transcript-api."""
    if languages is None:
        languages = ["ko", "en"]
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=languages)
        segments = []
        for snippet in transcript.snippets:
            segments.append({
                "start": snippet.start,
                "duration": snippet.duration,
                "text": snippet.text,
            })
        return segments
    except Exception:
        return None


def video_to_markdown(video: dict, transcript: list[dict] | None, category: str) -> str:
    """Convert video metadata and transcript to markdown format."""
    video_id = video.get("id", "")
    title = video.get("title", "")
    channel = video.get("channel", video.get("uploader", ""))
    url = f"https://www.youtube.com/watch?v={video_id}"
    views = video.get("view_count", 0)
    upload_date = video.get("upload_date", "")
    if upload_date and len(upload_date) == 8:
        upload_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"

    # Frontmatter
    lines = [
        "---",
        f'id: "{video_id}"',
        f'title: "{title.replace(chr(34), chr(39))}"',
        f'uploader: "{channel.replace(chr(34), chr(39))}"',
        f'url: "{url}"',
        f"views: {views}",
        f"category: {category}",
        f"topics: [{category}]",
        f'published_at: "{upload_date}"',
        f"fallback_used: false",
        "---",
        "",
    ]

    # Transcript
    if transcript:
        for seg in transcript:
            mins = int(seg["start"] // 60)
            secs = int(seg["start"] % 60)
            lines.append(f"## [{mins:02d}:{secs:02d}]")
            lines.append(seg["text"].strip())
            lines.append("")
    else:
        lines.append("## [00:00]")
        lines.append("(Transcript unavailable)")
        lines.append("")

    return "\n".join(lines)


def safe_filename(title: str) -> str:
    """Create a safe filename from video title."""
    name = re.sub(r'[^\w가-힣ㄱ-ㅎㅏ-ㅣ\s.-]', '', title)
    name = re.sub(r'\s+', '_', name.strip())
    return name[:80] or "untitled"


def main():
    parser = argparse.ArgumentParser(description="Collect YouTube dating corpus")
    parser.add_argument("--output", default="./data/source/new-corpus", help="Output directory")
    parser.add_argument("--limit", type=int, default=600, help="Target video count")
    parser.add_argument("--search-per-keyword", type=int, default=15, help="Videos per keyword")
    parser.add_argument("--exclude", help="JSON file with existing video IDs to exclude")
    parser.add_argument("--min-views", type=int, default=10000, help="Minimum view count")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between requests (seconds)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load exclude set
    exclude_ids = set()
    if args.exclude and Path(args.exclude).exists():
        with open(args.exclude) as f:
            data = json.load(f)
            if isinstance(data, list):
                exclude_ids = set(data)
            elif isinstance(data, dict) and "video_ids" in data:
                exclude_ids = set(data["video_ids"])

    print(f"Excluding {len(exclude_ids)} existing videos")

    collected = []
    seen_ids = set()
    manifest_entries = []

    for cat_id, cat_info in CATEGORIES.items():
        if len(collected) >= args.limit:
            break

        print(f"\n=== Category: {cat_info['name']} ({cat_id}) ===")

        for keyword in cat_info["keywords"]:
            if len(collected) >= args.limit:
                break

            print(f"  Searching: {keyword}")
            videos = search_youtube(keyword, limit=args.search_per_keyword)
            time.sleep(args.delay)

            for v in videos:
                if len(collected) >= args.limit:
                    break

                video_id = v.get("id", "")
                if not video_id or video_id in seen_ids or video_id in exclude_ids:
                    continue

                views = v.get("view_count", 0) or 0
                if views < args.min_views:
                    continue

                seen_ids.add(video_id)

                # Get transcript
                print(f"    [{len(collected)+1}/{args.limit}] {v.get('title','?')[:50]}... ({views:,} views)")
                transcript = get_transcript(video_id)
                time.sleep(args.delay * 0.5)

                if not transcript:
                    print(f"      SKIP: no transcript")
                    continue

                # Save markdown
                md_content = video_to_markdown(v, transcript, cat_id)
                filename = f"{video_id}__{safe_filename(v.get('title', 'untitled'))}.md"
                md_path = output_dir / filename
                md_path.write_text(md_content, encoding="utf-8")

                collected.append(video_id)
                manifest_entries.append({
                    "video_id": video_id,
                    "title": v.get("title", ""),
                    "channel": v.get("channel", v.get("uploader", "")),
                    "category": cat_id,
                    "views": views,
                    "file": filename,
                    "has_transcript": True,
                })

                print(f"      OK: {len(transcript)} segments, {len(collected)} total")

    # Save manifest
    manifest_path = output_dir.parent / "new-corpus-manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "collected_at": datetime.now().isoformat(),
            "total": len(collected),
            "categories": {cat: len([e for e in manifest_entries if e["category"] == cat]) for cat in CATEGORIES},
            "videos": manifest_entries,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n=== Done: {len(collected)} videos collected ===")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
