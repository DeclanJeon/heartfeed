"""BRT-14 track config loader."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "tracks" / "brt14.yaml"


@lru_cache(maxsize=4)
def load_brt14_track(path: str | None = None) -> dict[str, Any]:
    p = Path(path) if path else _DEFAULT_PATH
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def track_day(day_index: int, path: str | None = None) -> dict[str, Any] | None:
    data = load_brt14_track(path)
    days = data.get("days") or []
    for d in days:
        if int(d.get("day", -1)) == int(day_index):
            return d
    return None


def track_hints_for_day(day_index: int, path: str | None = None) -> dict[str, Any]:
    data = load_brt14_track(path)
    day = track_day(day_index, path)
    policy = data.get("policy") or {}
    actions = list((day or {}).get("actions") or [])
    return {
        "suggested_day_actions": actions[:3],
        "impulse_protocol": policy.get("impulse_protocol", ""),
        "theme": (day or {}).get("theme", ""),
        "day_index": day_index,
    }
