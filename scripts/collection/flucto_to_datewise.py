#!/usr/bin/env python3
"""Convert Flucto CLI Markdown output to datewise-rag ingestion format.

Flucto outputs .md files with blockquote metadata:
    # Title
    > **채널:** Channel
    > **길이:** MM:SS
    > **URL:** [url](url)
    > **추출일:** YYYY-MM-DD HH:MM
    ---
    ## [00:00]
    transcript text...

datewise-rag expects YAML frontmatter:
    ---
    id: "video_id"
    title: "Title"
    channel: "Channel"
    url: "https://..."
    platform: "youtube"
    views: 0
    duration: 789
    uploaded: "2026-07-10"
    collected: "2026-07-16"
    category: "dating"
    language: "ko"
    ---
    # Title
    ## [00:00]
    transcript text...

Usage:
    python flucto_to_datewise.py --input <flucto-output-dir> --output <datewise-source-dir>
    python flucto_to_datewise.py --input data/source/flucto-raw --output data/source/flucto-target/corpus
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse


def parse_flucto_metadata(content: str) -> dict[str, str]:
    """Extract metadata from Flucto's blockquote format."""
    meta: dict[str, str] = {}

    # Title: first H1
    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if title_match:
        meta["title"] = title_match.group(1).strip()

    # Blockquote fields: > **키:** 값  (colon INSIDE bold markers)
    bq_pattern = re.compile(r">\s*\*\*([^*]+?)[:：]?\*\*\s*(.+)", re.MULTILINE)
    for m in bq_pattern.finditer(content):
        key = m.group(1).strip()
        val = m.group(2).strip()
        meta[key] = val

    return meta


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    if not url:
        return ""
    # Handle [text](url) markdown links
    link_match = re.search(r"\]\((https?://[^)]+)\)", url)
    if link_match:
        url = link_match.group(1)
    parsed = urlparse(url)
    if "youtube.com" in parsed.netloc or "youtu.be" in parsed.netloc:
        # youtube.com/watch?v=ID
        if parsed.path == "/watch":
            qs = dict(qc.split("=", 1) for qc in parsed.query.split("&") if "=" in qc)
            return qs.get("v", "")
        # youtu.be/ID
        return parsed.path.lstrip("/")
    return url


def parse_duration_seconds(duration_str: str) -> int:
    """Convert MM:SS or HH:MM:SS to seconds."""
    parts = duration_str.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0


def parse_extract_date(date_str: str) -> str:
    """Extract date portion from 'YYYY-MM-DD HH:MM' format."""
    date_match = re.match(r"(\d{4}-\d{2}-\d{2})", date_str)
    return date_match.group(1) if date_match else ""


def convert_file(src: Path, output_dir: Path, category: str = "dating") -> bool:
    """Convert a single Flucto .md file to datewise-rag format."""
    content = src.read_text(encoding="utf-8")
    meta = parse_flucto_metadata(content)

    if not meta.get("title"):
        print(f"  SKIP (no title): {src.name}")
        return False

    raw_url = meta.get("URL", "")
    # Strip markdown link format [text](url) → plain url
    link_match = re.search(r"\]\((https?://[^)]+)\)", raw_url)
    url = link_match.group(1) if link_match else raw_url.strip()

    video_id = extract_video_id(url)
    channel = meta.get("채널", "")
    duration_str = meta.get("길이", "0:00")
    duration = parse_duration_seconds(duration_str)
    extracted_date = parse_extract_date(meta.get("추출일", ""))

    # Extract transcript body: everything after the first ## [MM:SS] section header
    body_match = re.search(r"(##\s*\[\d{2}:\d{2}(?::\d{2})?\].*)", content, re.DOTALL)
    transcript_body = body_match.group(1) if body_match else content

    # Build YAML frontmatter
    frontmatter_lines = [
        "---",
        f'id: "{video_id}"',
        f'title: "{meta["title"].replace(chr(34), chr(39))}"',
        f'channel: "{channel}"',
        f'url: "{url}"',
        f'platform: "youtube"',
        f"views: 0",
        f"duration: {duration}",
        f'uploaded: "{extracted_date}"',
        f'collected: "{extracted_date}"',
        f'category: "{category}"',
        f'language: "ko"',
        "---",
    ]

    # Reconstruct: frontmatter + title header + transcript body
    output = "\n".join(frontmatter_lines) + "\n\n" + f"# {meta['title']}\n\n" + transcript_body

    safe_name = re.sub(r"[^\w가-힣\-.+]", "_", meta["title"])[:80]
    out_path = output_dir / f"{safe_name}.md"
    out_path.write_text(output, encoding="utf-8")
    print(f"  OK: {out_path.name} ({len(output)} bytes, id={video_id})")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="Flucto output directory containing .md files"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for datewise-rag .md files"
    )
    parser.add_argument(
        "--category", default="dating", help="Category tag for all converted files"
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Recurse into subdirectories"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    glob = "**/*.md" if args.recursive else "*.md"
    md_files = sorted(input_dir.glob(glob))

    if not md_files:
        print(f"No .md files found in {input_dir}")
        sys.exit(1)

    print(f"Converting {len(md_files)} files from {input_dir} → {output_dir}")

    converted = 0
    for md_file in md_files:
        if convert_file(md_file, output_dir, category=args.category):
            converted += 1

    print(f"\nDone: {converted}/{len(md_files)} files converted")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
