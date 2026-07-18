"""Concurrency admission control for generation."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

GENERATION_CONCURRENCY = int(os.getenv("GENERATION_CONCURRENCY", "4"))


@dataclass
class GenerationAdmission:
    """Asyncio semaphore wrapper that can refuse when saturated."""

    limit: int = GENERATION_CONCURRENCY

    def __post_init__(self) -> None:
        self._sem = asyncio.Semaphore(self.limit)
        self._in_flight = 0
        self._lock = asyncio.Lock()

    @property
    def in_flight(self) -> int:
        return self._in_flight

    async def try_acquire(self) -> bool:
        async with self._lock:
            if self._in_flight >= self.limit:
                return False
            self._in_flight += 1
        await self._sem.acquire()
        return True

    async def release(self) -> None:
        self._sem.release()
        async with self._lock:
            self._in_flight = max(0, self._in_flight - 1)


generation_admission = GenerationAdmission()
