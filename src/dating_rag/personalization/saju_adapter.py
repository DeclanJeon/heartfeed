"""Deterministic local saju adapter wrapping manse_engine.py.

v1: self-only chart, solar calendar only, no geocoding, no network calls.
Fails closed when engine is unavailable.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dating_rag.domain.models import CulturalReflectionBlock

ADAPTER_VERSION = "saju-adapter-v1.0"

# -- element mappings --------------------------------------------------------
_STEM_ELEMENT: dict[str, str] = {
    "gap": "wood", "eul": "wood",
    "byeong": "fire", "jeong": "fire",
    "gi": "earth", "gyeong": "earth",
    "sin": "metal", "gye": "metal",
    "im": "water", "geo": "water",
}

_BRANCH_ELEMENT: dict[str, str] = {
    "ja": "water",
    "chuk": "earth", "jin": "earth", "mi": "earth",
    "in": "wood", "myo": "wood",
    "sa": "fire", "o": "fire",
    "sin": "metal", "yu": "metal", "sul": "metal",
    "hae": "water",
}

_ZODIAC: dict[str, str] = {
    "ja": "쥐", "chuk": "소", "in": "호랑이", "myo": "토끼",
    "jin": "용", "sa": "뱀", "o": "말", "mi": "양",
    "sin": "원숭이", "yu": "닭", "sul": "개", "hae": "돼지",
}

_ELEMENT_ORDER = ("wood", "fire", "earth", "metal", "water")
_ELEMENT_KO: dict[str, str] = {
    "wood": "목", "fire": "화", "earth": "토", "metal": "금", "water": "수",
}

_GENERATE_MAP: dict[str, str] = {
    "wood": "fire", "fire": "earth", "earth": "metal", "metal": "water", "water": "wood",
}
_CONTROL_MAP: dict[str, str] = {
    "wood": "earth", "earth": "water", "water": "fire", "fire": "metal", "metal": "wood",
}
_GENERATED_BY_MAP: dict[str, str] = {v: k for k, v in _GENERATE_MAP.items()}
_CONTROLLED_BY_MAP: dict[str, str] = {v: k for k, v in _CONTROL_MAP.items()}

_TEN_GOD_KO: dict[str, str] = {
    "비견": "비견", "겁재": "겁재",
    "식신": "식신", "상관": "상관",
    "정재": "정재", "편재": "편재",
    "정관": "정관", "편관": "편관",
    "정인": "정인", "편인": "편인",
}


@dataclass
class SajuResult:
    """Allow-listed saju chart output for v1."""

    day_pillar: str
    zodiac: str
    elements: dict[str, Any]
    ten_gods_summary: str
    strengths: list[str]
    growth_tasks: list[str]
    quality_flags: list[str]
    adapter_version: str


class LocalSajuAdapter:
    """Wraps local manse_engine for self-only chart calculation."""

    def __init__(self, engine_path: str | None = None) -> None:
        self._engine: Any | None = None
        self._load_engine(engine_path)

    # -- public API ----------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the manse engine was imported successfully."""
        return self._engine is not None

    def calculate_chart(
        self,
        birth_date: str,
        birth_time: str | None,
        birthplace: str,
        gender: str,
        calendar_type: str,
        timezone: str,
    ) -> SajuResult:
        """Calculate a self-only saju chart.

        Solar calendar only for v1. Birthplace stored as-is (no geocoding).
        """
        if calendar_type != "solar":
            raise ValueError("v1은 양력만 지원합니다")

        quality_flags: list[str] = []

        if birth_time is None:
            quality_flags.append("birth_time_missing")
            engine_time = ""
        else:
            engine_time = birth_time

        if not self.is_available():
            return self._fallback_result(birth_date, quality_flags)

        try:
            raw: dict = self._engine.calculate_chart(  # type: ignore[union-attr]
                birth_date=birth_date,
                birth_time=engine_time,
                gender=gender,
                birth_place=birthplace,
                timezone=timezone,
                calendar_type="solar",
            )
        except Exception:
            return self._fallback_result(birth_date, quality_flags + ["engine_call_failed"])

        return self._map_result(raw, quality_flags)

    def generate_cultural_reflection(self, result: SajuResult) -> CulturalReflectionBlock:
        """Transform allow-listed SajuResult into CulturalReflectionBlock."""
        disclaimer = (
            "문화적 자기성찰용이며 과학적·의학적 근거나 관계의 운명을 뜻하지 않습니다."
        )

        sections = [
            {
                "title": "일간 분석",
                "content": (
                    f"일간은 {result.day_pillar}이며, "
                    f"띠는 {result.zodiac}입니다."
                ),
            },
            {
                "title": "오행 균형",
                "content": self._format_elements(result.elements),
            },
            {
                "title": "성장 포인트",
                "content": self._format_growth(result),
            },
        ]

        return CulturalReflectionBlock(
            sections=sections,
            quality_flags=list(result.quality_flags),
            disclaimer=disclaimer,
            adapter_version=result.adapter_version,
        )

    # -- internal helpers ----------------------------------------------------

    def _load_engine(self, engine_path: str | None) -> None:
        """Try to import manse_engine; set self._engine or leave None."""
        if engine_path is not None:
            path = Path(engine_path)
            if path.is_file():
                self._engine = self._import_from_path(path)
                return
            # Explicit path given but not found — do not fall through
            self._engine = None
            return

        # Try sibling engine/ directory
        sibling = Path(__file__).resolve().parent.parent.parent.parent / "engine" / "manse_engine.py"
        if sibling.is_file():
            self._engine = self._import_from_path(sibling)
            return

        # Try the known external path
        external = Path.home() / "Documents/Develop/Project/saju-engine-web/discord-saju-engine/manse_engine.py"
        if external.is_file():
            self._engine = self._import_from_path(external)
            return

        self._engine = None

    @staticmethod
    def _import_from_path(path: Path) -> Any:
        """Import a module from an absolute file path."""
        spec = importlib.util.spec_from_file_location("_local_manse_engine", str(path))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        # Register temporarily so relative imports inside the engine resolve
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(spec.name, None)
            return None
        return module

    def _map_result(self, raw: dict, extra_flags: list[str]) -> SajuResult:
        """Map engine output dict to SajuResult."""
        pillars = raw.get("four_pillars", {})
        day = pillars.get("day", {})
        year = pillars.get("year", {})
        month = pillars.get("month", {})
        hour = pillars.get("hour")

        day_pillar = day.get("ko", "알수없음")
        zodiac = _ZODIAC.get(year.get("branch_code", ""), "알수없음")

        all_stems = [
            year.get("stem_code"),
            month.get("stem_code"),
            day.get("stem_code"),
            hour.get("stem_code") if hour else None,
        ]
        all_branches = [
            year.get("branch_code"),
            month.get("branch_code"),
            day.get("branch_code"),
            hour.get("branch_code") if hour else None,
        ]

        elements = self._count_elements(all_stems, all_branches)
        ten_gods_summary = self._ten_gods(day.get("stem_code", ""), all_stems)
        strengths, growth_tasks = self._derive_insights(elements, ten_gods_summary)

        quality_flags = list(raw.get("quality_flags", [])) + extra_flags
        quality_flags = sorted(set(quality_flags))

        return SajuResult(
            day_pillar=day_pillar,
            zodiac=zodiac,
            elements=elements,
            ten_gods_summary=ten_gods_summary,
            strengths=strengths,
            growth_tasks=growth_tasks,
            quality_flags=quality_flags,
            adapter_version=ADAPTER_VERSION,
        )

    @staticmethod
    def _count_elements(
        stems: list[str | None], branches: list[str | None],
    ) -> dict[str, Any]:
        """Count five-element distribution from stem and branch codes."""
        counts: dict[str, int] = {e: 0 for e in _ELEMENT_ORDER}

        for code in stems:
            elem = _STEM_ELEMENT.get(code or "")
            if elem:
                counts[elem] += 1

        for code in branches:
            elem = _BRANCH_ELEMENT.get(code or "")
            if elem:
                counts[elem] += 1

        strongest = max(counts, key=lambda e: counts[e])
        weakest = min(counts, key=lambda e: counts[e])

        return {
            "counts": counts,
            "strongest": strongest,
            "weakest": weakest,
        }

    @staticmethod
    def _ten_gods(day_stem: str, all_stems: list[str | None]) -> str:
        """Compute ten-gods summary relative to the day stem."""
        if not day_stem:
            return "unknown"

        day_elem = _STEM_ELEMENT.get(day_stem, "")
        day_yinyang = _yinyang(day_stem)

        god_counts: dict[str, int] = {}
        for code in all_stems:
            if code is None or code == day_stem:
                continue
            god = _classify_ten_god(day_elem, day_yinyang, code)
            god_counts[god] = god_counts.get(god, 0) + 1

        if not god_counts:
            return "자기 일간만 존재"

        parts = [f"{name}×{cnt}" for name, cnt in sorted(god_counts.items(), key=lambda x: -x[1])]
        return ", ".join(parts)

    @staticmethod
    def _derive_insights(
        elements: dict[str, Any], ten_gods_summary: str,
    ) -> tuple[list[str], list[str]]:
        """Derive strengths and growth tasks from element balance and ten gods."""
        counts: dict[str, int] = elements["counts"]
        total = sum(counts.values()) or 1
        strongest = elements["strongest"]
        weakest = elements["weakest"]

        strengths: list[str] = []
        growth_tasks: list[str] = []

        elem_insights: dict[str, tuple[str, str]] = {
            "wood": ("성장과 창의력이 강점", "고집을 줄이고 유연함을 기르세요"),
            "fire": ("열정과 표현력이 뛰어남", "조급함을 다스리고 인내를 키우세요"),
            "earth": ("안정감과 신뢰감이 강점", "현실에 안주하지 말고 도전하세요"),
            "metal": ("결단력과 원칙이 강점", "융통성을 기르고 관대함을 연습하세요"),
            "water": ("지혜와 적응력이 강점", "우유부단함을 줄이고 실행력을 키우세요"),
        }

        if counts.get(strongest, 0) / total >= 0.3:
            s, g = elem_insights.get(strongest, ("강한 에너지", "균형을 찾으세요"))
            strengths.append(s)
            growth_tasks.append(g)

        if counts.get(weakest, 0) == 0:
            weak_label = _ELEMENT_KO.get(weakest, weakest)
            growth_tasks.append(f"{weak_label} 기운 보충이 필요합니다")

        if "식신" in ten_gods_summary or "상관" in ten_gods_summary:
            strengths.append("창의적 표현력이 두드러짐")

        if not strengths:
            strengths.append("오행이 비교적 균형을 이루고 있습니다")
        if not growth_tasks:
            growth_tasks.append("현재 균형을 유지하며 꾸준히 성장하세요")

        return strengths, growth_tasks

    @staticmethod
    def _fallback_result(birth_date: str, extra_flags: list[str]) -> SajuResult:
        """Return placeholder SajuResult when engine is unavailable."""
        return SajuResult(
            day_pillar="알수없음",
            zodiac="알수없음",
            elements={
                "counts": {e: 0 for e in _ELEMENT_ORDER},
                "strongest": "unknown",
                "weakest": "unknown",
            },
            ten_gods_summary="engine unavailable",
            strengths=["엔진을 사용할 수 없어 분석이 제한됩니다"],
            growth_tasks=["사주 엔진 설치 후 다시 시도하세요"],
            quality_flags=sorted(set(extra_flags + ["engine_unavailable"])),
            adapter_version=ADAPTER_VERSION,
        )

    @staticmethod
    def _format_elements(elements: dict[str, Any]) -> str:
        counts = elements.get("counts", {})
        parts = [f"{_ELEMENT_KO.get(e, e)}: {c}" for e, c in counts.items() if c > 0]
        distribution = ", ".join(parts) if parts else "데이터 없음"
        strongest_ko = _ELEMENT_KO.get(elements.get("strongest", ""), "?")
        weakest_ko = _ELEMENT_KO.get(elements.get("weakest", ""), "?")
        return f"오행 분포: {distribution}. 가장 강한 기운: {strongest_ko}, 가장 약한 기운: {weakest_ko}."

    @staticmethod
    def _format_growth(result: SajuResult) -> str:
        parts = []
        if result.strengths:
            parts.append("강점: " + "; ".join(result.strengths))
        if result.growth_tasks:
            parts.append("성장 과제: " + "; ".join(result.growth_tasks))
        return " ".join(parts) if parts else "분석 정보가 부족합니다."


def _yinyang(stem_code: str) -> str:
    """Return 'yang' or 'yin' for a stem code."""
    yang_stems = {"gap", "byeong", "gi", "sin", "im"}
    return "yang" if stem_code in yang_stems else "yin"


def _classify_ten_god(day_elem: str, day_yinyang: str, target_stem: str) -> str:
    """Classify ten-god relationship between day stem and target stem."""
    target_elem = _STEM_ELEMENT.get(target_stem, "")
    target_yinyang = _yinyang(target_stem)
    same_yy = day_yinyang == target_yinyang

    if target_elem == day_elem:
        return "비견" if same_yy else "겁재"
    if _GENERATE_MAP.get(day_elem) == target_elem:
        return "식신" if same_yy else "상관"
    if _CONTROL_MAP.get(day_elem) == target_elem:
        return "정재" if same_yy else "편재"
    if _CONTROLLED_BY_MAP.get(day_elem) == target_elem:
        return "정관" if same_yy else "편관"
    if _GENERATED_BY_MAP.get(day_elem) == target_elem:
        return "정인" if same_yy else "편인"
    return "unknown"
