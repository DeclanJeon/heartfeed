"""Tests for LocalSajuAdapter (v1: self-only, solar calendar, no network)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from dating_rag.domain.models import CulturalReflectionBlock
from dating_rag.personalization.saju_adapter import (
    ADAPTER_VERSION,
    LocalSajuAdapter,
    SajuResult,
)


@pytest.fixture()
def adapter() -> LocalSajuAdapter:
    """Create an adapter with default engine discovery."""
    return LocalSajuAdapter()


@pytest.fixture()
def adapter_no_engine(tmp_path: Path) -> LocalSajuAdapter:
    """Create an adapter guaranteed to have no engine."""
    return LocalSajuAdapter(engine_path=str(tmp_path / "nonexistent.py"))


# -- calendar type tests ----------------------------------------------------


class TestCalendarType:
    def test_solar_calendar_accepted(self, adapter: LocalSajuAdapter) -> None:
        """Solar calendar should not raise."""
        result = adapter.calculate_chart(
            birth_date="1992-03-01",
            birth_time="14:30",
            birthplace="서울",
            gender="male",
            calendar_type="solar",
            timezone="Asia/Seoul",
        )
        assert isinstance(result, SajuResult)

    def test_lunar_calendar_rejected(self, adapter: LocalSajuAdapter) -> None:
        """Lunar calendar must raise ValueError."""
        with pytest.raises(ValueError, match="양력만 지원"):
            adapter.calculate_chart(
                birth_date="1992-03-01",
                birth_time="14:30",
                birthplace="서울",
                gender="male",
                calendar_type="lunar",
                timezone="Asia/Seoul",
            )


# -- birth_time handling -----------------------------------------------------


class TestBirthTime:
    def test_missing_birth_time_flag(self, adapter: LocalSajuAdapter) -> None:
        """Missing birth_time must appear in quality_flags."""
        result = adapter.calculate_chart(
            birth_date="1990-07-15",
            birth_time=None,
            birthplace="부산",
            gender="female",
            calendar_type="solar",
            timezone="Asia/Seoul",
        )
        assert "birth_time_missing" in result.quality_flags


# -- CulturalReflectionBlock -------------------------------------------------


class TestCulturalReflection:
    EXPECTED_DISCLAIMER = (
        "문화적 자기성찰용이며 과학적·의학적 근거나 관계의 운명을 뜻하지 않습니다."
    )

    def test_returns_cultural_reflection_block(self, adapter: LocalSajuAdapter) -> None:
        result = adapter.calculate_chart(
            birth_date="1988-11-20",
            birth_time="09:15",
            birthplace="대구",
            gender="male",
            calendar_type="solar",
            timezone="Asia/Seoul",
        )
        block = adapter.generate_cultural_reflection(result)
        assert isinstance(block, CulturalReflectionBlock)

    def test_disclaimer_is_correct(self, adapter: LocalSajuAdapter) -> None:
        result = adapter.calculate_chart(
            birth_date="1988-11-20",
            birth_time="09:15",
            birthplace="대구",
            gender="male",
            calendar_type="solar",
            timezone="Asia/Seoul",
        )
        block = adapter.generate_cultural_reflection(result)
        assert block.disclaimer == self.EXPECTED_DISCLAIMER

    def test_sections_present(self, adapter: LocalSajuAdapter) -> None:
        result = adapter.calculate_chart(
            birth_date="1988-11-20",
            birth_time="09:15",
            birthplace="대구",
            gender="male",
            calendar_type="solar",
            timezone="Asia/Seoul",
        )
        block = adapter.generate_cultural_reflection(result)
        titles = [s["title"] for s in block.sections]
        assert "일간 분석" in titles
        assert "오행 균형" in titles
        assert "성장 포인트" in titles

    def test_adapter_version_propagated(self, adapter: LocalSajuAdapter) -> None:
        result = adapter.calculate_chart(
            birth_date="1988-11-20",
            birth_time="09:15",
            birthplace="대구",
            gender="male",
            calendar_type="solar",
            timezone="Asia/Seoul",
        )
        block = adapter.generate_cultural_reflection(result)
        assert block.adapter_version == ADAPTER_VERSION


# -- is_available ------------------------------------------------------------


class TestIsAvailable:
    def test_returns_bool(self, adapter: LocalSajuAdapter) -> None:
        assert isinstance(adapter.is_available(), bool)

    def test_no_engine_returns_false(self, adapter_no_engine: LocalSajuAdapter) -> None:
        assert adapter_no_engine.is_available() is False


# -- engine unavailable graceful fallback ------------------------------------


class TestFallback:
    def test_fallback_when_engine_missing(self, adapter_no_engine: LocalSajuAdapter) -> None:
        result = adapter_no_engine.calculate_chart(
            birth_date="2000-01-01",
            birth_time="12:00",
            birthplace="서울",
            gender="male",
            calendar_type="solar",
            timezone="Asia/Seoul",
        )
        assert "engine_unavailable" in result.quality_flags
        assert result.day_pillar == "알수없음"
        assert result.adapter_version == ADAPTER_VERSION

    def test_fallback_with_missing_time(self, adapter_no_engine: LocalSajuAdapter) -> None:
        result = adapter_no_engine.calculate_chart(
            birth_date="2000-01-01",
            birth_time=None,
            birthplace="서울",
            gender="male",
            calendar_type="solar",
            timezone="Asia/Seoul",
        )
        assert "engine_unavailable" in result.quality_flags
        assert "birth_time_missing" in result.quality_flags


# -- no network calls --------------------------------------------------------


class TestNoNetwork:
    def test_no_requests_or_httpx_imports(self) -> None:
        """Module must not import requests or httpx."""
        module_path = Path(__file__).resolve().parent.parent / "src" / "dating_rag" / "personalization" / "saju_adapter.py"
        source = module_path.read_text(encoding="utf-8")

        tree = ast.parse(source)
        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_names.add(node.module.split(".")[0])

        assert "requests" not in imported_names, "Must not import requests"
        assert "httpx" not in imported_names, "Must not import httpx"
        assert "urllib" not in imported_names, "Must not import urllib"
