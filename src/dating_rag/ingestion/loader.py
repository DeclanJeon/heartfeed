"""Markdown document loader for video transcripts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from collections.abc import Iterator


def load_metadata(path: Path) -> dict[str, str]:
    """Parse YAML frontmatter from a markdown file.

    Args:
        path: Path to the markdown file.

    Returns:
        Dictionary of frontmatter fields. Values are strings except
        ``duration`` which is returned as an ``int`` when parseable.
    """
    content = path.read_text(encoding="utf-8")

    metadata: dict[str, str] = {}
    if not content.startswith("---"):
        return metadata

    # Split on --- delimiters; parts[1] is the YAML block
    parts = content.split("---", 2)
    if len(parts) < 3:
        return metadata

    try:
        parsed = yaml.safe_load(parts[1].strip())
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                metadata[str(key)] = str(value) if value is not None else ""
    except yaml.YAMLError:
        # Fallback: line-by-line key:value parsing
        for line in parts[1].strip().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip().strip('"')

    return metadata


def load_transcript(path: Path) -> dict[str, str]:
    """Load a single markdown transcript file.

    Parses YAML frontmatter and the timestamped transcript body.  The body
    is everything after the frontmatter closing ``---``, excluding the
    ``![thumbnail](…)`` line.

    Args:
        path: Path to the markdown file.

    Returns:
        Parsed transcript with frontmatter fields and ``body`` text.
    """
    content = path.read_text(encoding="utf-8")

    # Frontmatter
    frontmatter: dict[str, str] = {}
    body = content

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                parsed = yaml.safe_load(parts[1].strip())
                if isinstance(parsed, dict):
                    for key, value in parsed.items():
                        frontmatter[str(key)] = str(value) if value is not None else ""
            except yaml.YAMLError:
                for line in parts[1].strip().splitlines():
                    if ":" in line:
                        key, value = line.split(":", 1)
                        frontmatter[key.strip()] = value.strip().strip('"')
            body = parts[2].strip()

    # Drop the thumbnail line if present
    lines = body.splitlines()
    filtered = [ln for ln in lines if not ln.strip().startswith("![thumbnail]")]
    body = "\n".join(filtered).strip()

    frontmatter["body"] = body
    return frontmatter


def load_transcript_segments(body: str) -> list[dict[str, object]]:
    """Parse timestamped sections from a transcript body.

    Handles ``## [MM:SS]`` or ``## [HH:MM:SS]`` markdown headers followed
    by transcript text on subsequent lines up to the next timestamp header.

    Args:
        body: Transcript body text (after frontmatter).

    Returns:
        List of dicts with ``start_seconds`` (float), ``timestamp`` (str),
        and ``text`` (str).
    """
    import re

    # Split on timestamp section headers
    header_pattern = re.compile(
        r"^##\s*\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*$",
        re.MULTILINE,
    )

    sections: list[dict[str, object]] = []

    for match in header_pattern.finditer(body):
        ts_str = match.group(1)
        start_seconds = _ts_to_seconds(ts_str)

        # Collect text from after this header to the start of the next header
        text_start = match.end()
        next_match = header_pattern.search(body, text_start)
        text_end = next_match.start() if next_match else len(body)

        text = body[text_start:text_end].strip()
        if text:
            sections.append({
                "start_seconds": start_seconds,
                "timestamp": ts_str,
                "text": text,
            })

    return sections


def _ts_to_seconds(ts: str) -> float:
    """Convert a ``MM:SS`` or ``HH:MM:SS`` string to seconds."""
    parts = ts.split(":")
    if len(parts) == 3:
        return float(int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]))
    return float(int(parts[0]) * 60 + int(parts[1]))


def load_all_transcripts(data_dir: Path) -> Iterator[dict[str, str]]:
    """Batch-load all markdown transcripts from a directory.

    Searches for ``*.md`` files in *data_dir*.

    Args:
        data_dir: Directory containing transcript markdown files.

    Yields:
        Parsed transcripts as dictionaries (frontmatter + body).
    """
    for path in sorted(data_dir.glob("*.md")):
        try:
            yield load_transcript(path)
        except Exception:
            # Skip files that can't be parsed
            continue


def load_directory(directory: Path, glob: str = "*.md") -> Iterator[dict[str, str]]:
    """Load all transcript files from a directory.

    Args:
        directory: Path to directory containing transcript files.
        glob: Glob pattern for matching files.

    Yields:
        Parsed transcripts as dictionaries.
    """
    for path in sorted(directory.glob(glob)):
        yield load_transcript(path)
