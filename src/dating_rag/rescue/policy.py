"""Rescue product policy helpers."""

from __future__ import annotations

from dating_rag.domain.models import TrackContext
from dating_rag.rescue.guards import is_breakup_related


def is_rescue_mode(product_mode: str) -> bool:
    return (product_mode or "").strip() == "rescue_brt14"


def should_refuse_ood(
    *,
    product_mode: str,
    allow_general: bool,
    track: TrackContext | None,
    question: str,
) -> bool:
    if not is_rescue_mode(product_mode) or allow_general:
        return False
    has_track = track is not None and track.id == "brt14"
    if has_track:
        return False
    return not is_breakup_related(question)


def normalize_track(track: TrackContext | None) -> TrackContext | None:
    if track is None:
        return None
    if track.id != "brt14":
        return None
    # pydantic already clamps day_index 0-13
    return track
