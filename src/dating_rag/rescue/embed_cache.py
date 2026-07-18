"""Query embedding cache: in-memory LRU with optional durable JSON file."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_VERSION = "bge-m3-v1"


class EmbedCache:
    """Thread-safe query embed cache with optional disk persistence."""

    def __init__(
        self,
        max_size: int = 512,
        path: str | Path | None = None,
    ) -> None:
        self._max = max(8, int(max_size))
        self._data: dict[str, Any] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._path = Path(path).expanduser() if path else None
        if self._path is not None:
            self._load_disk()

    @staticmethod
    def _key(text: str) -> str:
        norm = " ".join((text or "").strip().split())
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Any | None:
        k = self._key(text)
        with self._lock:
            val = self._data.get(k)
            if val is not None and k in self._order:
                # refresh LRU
                try:
                    self._order.remove(k)
                except ValueError:
                    pass
                self._order.append(k)
            return val

    def set(self, text: str, value: Any) -> None:
        k = self._key(text)
        with self._lock:
            if k not in self._data and len(self._data) >= self._max:
                # evict oldest
                while self._order and len(self._data) >= self._max:
                    old = self._order.pop(0)
                    self._data.pop(old, None)
            self._data[k] = value
            if k in self._order:
                try:
                    self._order.remove(k)
                except ValueError:
                    pass
            self._order.append(k)
            self._save_disk_unlocked()

    def _load_disk(self) -> None:
        if self._path is None or not self._path.is_file():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or raw.get("version") != CACHE_VERSION:
                logger.info("embed cache version mismatch or invalid; ignoring disk file")
                return
            entries = raw.get("entries") or {}
            if not isinstance(entries, dict):
                return
            # load up to max_size (arbitrary order from file)
            n = 0
            for k, v in entries.items():
                if n >= self._max:
                    break
                if isinstance(k, str) and isinstance(v, dict) and "dense" in v:
                    self._data[k] = v
                    self._order.append(k)
                    n += 1
            logger.info("embed cache loaded from disk n=%s path=%s", n, self._path)
        except Exception:
            logger.warning("embed cache load failed path=%s", self._path, exc_info=True)

    def _save_disk_unlocked(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": CACHE_VERSION,
                "entries": {k: self._data[k] for k in self._order if k in self._data},
            }
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._path)
        except Exception:
            logger.warning("embed cache save failed path=%s", self._path, exc_info=True)


def _default_cache() -> EmbedCache:
    max_size = int(os.environ.get("DATEWISE_EMBED_CACHE_MAX", "512") or 512)
    path = os.environ.get("DATEWISE_EMBED_CACHE_PATH", "").strip()
    if not path:
        # default under data/ when cwd is project root
        path = str(Path("data/cache/query_embeds.json"))
    return EmbedCache(max_size=max_size, path=path)


query_embed_cache = _default_cache()
