"""In-memory embedding cache for query vectors."""

from __future__ import annotations

import hashlib
import threading
from typing import Any


class EmbedCache:
    def __init__(self, max_size: int = 512) -> None:
        self._max = max_size
        self._data: dict[str, Any] = {}
        self._lock = threading.Lock()

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def get(self, text: str) -> Any | None:
        with self._lock:
            return self._data.get(self._key(text))

    def set(self, text: str, value: Any) -> None:
        with self._lock:
            if len(self._data) >= self._max:
                # drop arbitrary first key
                try:
                    del self._data[next(iter(self._data))]
                except StopIteration:
                    pass
            self._data[self._key(text)] = value


query_embed_cache = EmbedCache()
