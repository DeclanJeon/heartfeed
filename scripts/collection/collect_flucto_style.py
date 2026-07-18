#!/usr/bin/env python3
"""Collect YouTube channel transcripts using yt-dlp (flucto-style).

Downloads auto/manual subtitles for the most recent videos from a channel,
converts them to datewise-rag compatible .md format with YAML frontmatter.

This achieves the same result as `flucto channel to-md` without Electron overhead.

Usage:
    python collect_flucto_style.py --channel-url "https://www.youtube.com/@channel" --limit 15 -o ./output
    python collect_flucto_style.py --channel-url "UCxxxxx" --limit 10 -o ./output -l ko
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse


def get_channel_videos(channel_url: str, limit: int = 15) -> list[dict]:
    """Get recent video metadata from a YouTube channel."""
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-json",
        "--no-download",
        "--playlist-end", str(limit),
        f"{channel_url}/videos",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        videos = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                videos.append({
                    "id": data.get("id", ""),
                    "title": data.get("title", ""),
                    "channel": data.get("channel", ""),
                    "channel_id": data.get("channel_id", ""),
                    "url": f"https://www.youtube.com/watch?v={data.get('id', '')}",
                    "duration": data.get("duration") or 0,
                    "view_count": data.get("view_count") or 0,
                    "upload_date": data.get("upload_date", ""),
                })
            except json.JSONDecodeError:
                continue
        return videos[:limit]
    except Exception as e:
        print(f"  Error getting channel videos: {e}", file=sys.stderr)
        return []


def download_transcript(video: dict, output_dir: Path, lang: str = "ko") -> str | None:
    """Download subtitles for a video and return the transcript text."""
    video_id = video["id"]
    if not video_id:
        return None

    temp_dir = output_dir / f"_temp_{video_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Try to download subtitles (auto + manual)
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-auto-subs",
            "--write-subs",
            "--sub-langs", f"{lang},en",
            "--sub-format", "vtt",
            "--output", str(temp_dir / f"{video_id}.%(ext)s"),
            video["url"],
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        # Find the subtitle file
        vtt_files = list(temp_dir.glob(f"{video_id}*.vtt"))
        if not vtt_files:
            return None

        # Pick the best subtitle (prefer manual over auto, preferred language)
        vtt_file = sorted(vtt_files)[0]  # Take first available

        # Parse VTT to clean text
        content = vtt_file.read_text(encoding="utf-8")
        return parse_vtt(content)
    except Exception as e:
        print(f"  Error downloading transcript for {video_id}: {e}", file=sys.stderr)
        return None
    finally:
        # Cleanup temp files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def parse_vtt(content: str) -> str:
    """Parse VTT subtitle content into timestamped sections."""
    cues = []
    pattern = re.compile(
        r"(?:^|\n)\s*(\d{2}:\d{2}(?::\d{2})?)\.\d{3}\s*-->\s*\S+[\s\S]*?\n([\s\S]*?)(?=\n\s*\n|$)"
    )
    for match in pattern.finditer(content):
        timestamp = match.group(1)
        text = re.sub(r"<[^>]+>", "", match.group(2)).replace("\n", " ").strip()
        if text:
            cues.append(f"## [{timestamp}]\n{text}")
    return "\n\n".join(cues)


def format_duration(seconds: int) -> str:
    """Format seconds to MM:SS or HH:MM:SS."""
    if seconds >= 3600:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}:{m:02d}:{s:02d}"
    m = seconds // 60
    s = seconds % 60
    return f"{m}:{s:02d}"


def format_date(date_str: str) -> str:
    """Format YYYYMMDD to YYYY-MM-DD."""
    if len(date_str) == 8:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    return date_str


def write_markdown(video: dict, transcript: str, output_dir: Path, category: str = "dating") -> Path:
    """Write a datewise-rag compatible .md file."""
    title = video.get("title", "Untitled")
    channel = video.get("channel", "")
    url = video.get("url", "")
    video_id = video.get("id", "")
    duration = video.get("duration", 0)
    upload_date = format_date(video.get("upload_date", ""))
    now = datetime.now().strftime("%Y-%m-%d")

    frontmatter = [
        "---",
        f'id: "{video_id}"',
        f'title: "{title.replace(chr(34), chr(39))}"',
        f'channel: "{channel}"',
        f'url: "{url}"',
        f'platform: "youtube"',
        f"views: {video.get('view_count', 0)}",
        f"duration: {duration}",
        f'uploaded: "{upload_date}"',
        f'collected: "{now}"',
        f'category: "{category}"',
        f'language: "ko"',
        "---",
    ]

    body = f"# {title}\n\n## Transcript\n\n{transcript}" if transcript else f"# {title}\n\n## Transcript\n\n> 자막을 사용할 수 없습니다.\n"

    content = "\n".join(frontmatter) + "\n\n" + body

    safe_name = re.sub(r"[^\w가-힣\-.+]", "_", title)[:80]
    out_path = output_dir / f"{safe_name}.md"
    out_path.write_text(content, encoding="utf-8")
    return out_path


def collect_channel(channel_url: str, limit: int, output_dir: Path, lang: str = "ko", category: str = "dating") -> int:
    """Collect transcripts from a YouTube channel."""
    print(f"\n{'='*60}")
    print(f"Collecting: {channel_url}")
    print(f"Limit: {limit} videos, Language: {lang}")
    print(f"{'='*60}")

    videos = get_channel_videos(channel_url, limit)
    if not videos:
        print("  No videos found!")
        return 0

    print(f"  Found {len(videos)} videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    collected = 0
    for i, video in enumerate(videos, 1):
        title = video.get("title", "?")[:50]
        print(f"\n  [{i}/{len(videos)}] {title}")

        transcript = download_transcript(video, output_dir, lang)
        if transcript:
            out_path = write_markdown(video, transcript, output_dir, category)
            print(f"    ✓ Saved: {out_path.name} ({len(transcript)} chars)")
            collected += 1
        else:
            print(f"    ✗ No subtitles available")

    print(f"\n  Collected: {collected}/{len(videos)} transcripts")
    return collected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel-url", required=True, help="YouTube channel URL or handle")
    parser.add_argument("--limit", type=int, default=15, help="Max videos to collect")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-l", "--lang", default="ko", help="Subtitle language")
    parser.add_argument("--category", default="dating", help="Category tag")
    args = parser.parse_args()

    output_dir = Path(args.output)
    count = collect_channel(args.channel_url, args.limit, output_dir, args.lang, args.category)
    print(f"\nTotal collected: {count} transcripts in {output_dir}")


if __name__ == "__main__":
    main()
