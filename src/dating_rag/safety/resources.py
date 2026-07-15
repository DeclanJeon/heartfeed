"""Korean crisis resource registry."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class CrisisResource(BaseModel):
    """A single crisis resource entry."""

    model_config = {"frozen": True}

    name: str
    contact: str
    description: str
    reviewed_at: date = Field(default_factory=date.today)


RESOURCES: dict[str, CrisisResource] = {
    "suicide_prevention": CrisisResource(
        name="자살예방 상담전화",
        contact="109",
        description="24시간 무료 상담",
    ),
    "police": CrisisResource(
        name="경찰청",
        contact="112",
        description="긴급 신고",
    ),
    "emergency": CrisisResource(
        name="소방청",
        contact="119",
        description="응급 의료",
    ),
    "women_emergency": CrisisResource(
        name="여성긴급전화",
        contact="1366",
        description="가정폭력·성폭력 상담",
    ),
    "youth_counsel": CrisisResource(
        name="청소년 상담전화",
        contact="1388",
        description="청소년 상담",
    ),
}
