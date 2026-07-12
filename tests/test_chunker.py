"""Tests for the transcript chunker."""

from dating_rag.domain.models import TranscriptSegment
from dating_rag.ingestion.chunker import chunk_segments, parse_timestamp_segments


class TestParseTimestampSegments:
    """Tests for timestamp parsing."""

    def test_simple_timestamps(self) -> None:
        text = "[00:30] Hello there\n[01:00] How are you"
        segments = parse_timestamp_segments(text)
        assert len(segments) == 2
        assert segments[0].start_seconds == 30.0
        assert segments[0].text == "Hello there"
        assert segments[1].start_seconds == 60.0

    def test_range_timestamps(self) -> None:
        text = "[00:00:30] - [00:01:00] Hello there"
        segments = parse_timestamp_segments(text)
        assert len(segments) == 1
        assert segments[0].start_seconds == 30.0
        assert segments[0].end_seconds == 60.0

    def test_empty_text(self) -> None:
        segments = parse_timestamp_segments("")
        assert segments == []


class TestChunkSegments:
    """Tests for segment chunking."""

    def test_single_chunk(self) -> None:
        segments = [
            TranscriptSegment(start_seconds=0, end_seconds=10, text="Hello"),
            TranscriptSegment(start_seconds=10, end_seconds=20, text="World"),
        ]
        chunks = chunk_segments(segments, max_chars=1000)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello World"
        assert chunks[0].previous_chunk_id is None
        assert chunks[0].next_chunk_id is None

    def test_multiple_chunks_with_overlap(self) -> None:
        segments = [
            TranscriptSegment(start_seconds=i * 10, end_seconds=(i + 1) * 10, text=f"Segment {i} " * 50)
            for i in range(5)
        ]
        chunks = chunk_segments(segments, max_chars=200, overlap_chars=50)
        assert len(chunks) > 1
        # Check linking
        assert chunks[0].next_chunk_id == chunks[1].chunk_id
        assert chunks[1].previous_chunk_id == chunks[0].chunk_id
        assert chunks[-1].next_chunk_id is None

    def test_empty_segments(self) -> None:
        chunks = chunk_segments([])
        assert chunks == []

    def test_metadata_preserved(self) -> None:
        segments = [TranscriptSegment(start_seconds=0, end_seconds=10, text="Hello")]
        chunks = chunk_segments(
            segments,
            video_id="vid123",
            channel_id="ch456",
            channel_name="TestChannel",
            title="Test Video",
        )
        assert chunks[0].video_id == "vid123"
        assert chunks[0].channel_id == "ch456"
        assert chunks[0].channel_name == "TestChannel"
        assert chunks[0].title == "Test Video"
