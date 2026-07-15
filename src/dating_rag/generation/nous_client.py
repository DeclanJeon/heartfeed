"""NousResearch (Hermes) OAuth token resolver.

Reuses the exact OAuth refresh flow the Hermes agent uses on the operational
server: the Nous Portal issues short-lived access tokens (≈1h) that are
refreshed with a rotating ``refresh_token`` against
``{portal_base_url}/api/oauth/token``. The shared source of truth is the
Hermes ``auth.json`` file (``~/.hermes/auth.json``); writing the refreshed
state back keeps Hermes and HeartFeed in sync.

This is intentionally dependency-light (only ``httpx`` + stdlib) so HeartFeed
does not need the full Hermes agent package installed.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_NOUS_PORTAL_URL = "https://portal.nousresearch.com"
DEFAULT_NOUS_INFERENCE_URL = "https://inference-api.nousresearch.com/v1"
DEFAULT_NOUS_CLIENT_ID = "hermes-cli"
DEFAULT_NOUS_SCOPE = "inference:invoke"

# Default model mirrors Hermes ``model.default`` so HeartFeed answers with the
# same fast flash model the operator already configured.
DEFAULT_NOUS_MODEL = "stepfun/step-3.7-flash:free"

_TOKEN_TTL_GRACE_SECONDS = 120


def _parse_expires_at(value: Any) -> datetime | None:
    """Parse an ISO-8601 expiry string into an aware datetime, or None."""
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _is_expired(state: dict[str, Any], *, now: datetime | None = None) -> bool:
    """Return True when the access token is missing or within the grace window."""
    access = state.get("access_token")
    if not isinstance(access, str) or not access.strip():
        return True
    expires_at = _parse_expires_at(state.get("expires_at"))
    if expires_at is None:
        # No expiry recorded → treat as expired so we force a refresh once.
        return True
    now = now or datetime.now(timezone.utc)
    return expires_at <= (now + timedelta(seconds=_TOKEN_TTL_GRACE_SECONDS))


def load_nous_state(auth_path: str | Path) -> dict[str, Any]:
    """Load the ``nous`` provider block from the Hermes auth.json file."""
    path = Path(auth_path)
    if not path.exists():
        raise FileNotFoundError(f"Hermes auth file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    providers = data.get("providers", {})
    state = providers.get("nous")
    if not isinstance(state, dict):
        raise KeyError("No 'nous' provider block in Hermes auth.json")
    return state


def save_nous_state(auth_path: str | Path, state: dict[str, Any]) -> None:
    """Persist the refreshed ``nous`` state back to Hermes auth.json."""
    path = Path(auth_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("providers", {})["nous"] = state
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def refresh_nous_token(state: dict[str, Any], *, timeout_seconds: float = 15.0) -> dict[str, Any]:
    """Refresh the Nous OAuth access token using the rotating refresh token.

    Mirrors Hermes ``refresh_nous_oauth_pure``: POSTs the refresh token to the
    Nous Portal, rotates both tokens, and returns the updated state dict.
    """
    portal_base_url = str(state.get("portal_base_url") or DEFAULT_NOUS_PORTAL_URL).rstrip("/")
    client_id = str(state.get("client_id") or DEFAULT_NOUS_CLIENT_ID)
    refresh_token = state.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        raise RuntimeError("Nous refresh_token missing — re-run `hermes auth add nous`")

    with httpx.Client(timeout=timeout_seconds) as client:
        resp = client.post(
            f"{portal_base_url}/api/oauth/token",
            headers={"x-nous-refresh-token": refresh_token},
            data={"grant_type": "refresh_token", "client_id": client_id},
        )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Nous token refresh failed ({resp.status_code}): {resp.text[:200]}"
        )
    payload = resp.json()
    if "access_token" not in payload:
        raise RuntimeError(f"Nous refresh response missing access_token: {payload}")

    now = datetime.now(timezone.utc)
    access_ttl = int(payload.get("expires_in", 3600))
    refreshed: dict[str, Any] = dict(state)
    refreshed["access_token"] = payload["access_token"]
    refreshed["refresh_token"] = payload.get("refresh_token") or refresh_token
    refreshed["token_type"] = payload.get("token_type") or state.get("token_type") or "Bearer"
    refreshed["scope"] = payload.get("scope") or state.get("scope") or DEFAULT_NOUS_SCOPE
    refreshed["obtained_at"] = now.isoformat()
    refreshed["expires_in"] = access_ttl
    refreshed["expires_at"] = datetime.fromtimestamp(
        now.timestamp() + access_ttl, tz=timezone.utc
    ).isoformat()
    if payload.get("inference_base_url"):
        refreshed["inference_base_url"] = str(payload["inference_base_url"]).rstrip("/")
    return refreshed


class NousTokenProvider:
    """Lazily resolves a valid Nous inference token, refreshing on demand.

    Thread-safe and memoized: the token is only refreshed when the cached one
    is missing or within the grace window of expiry.
    """

    def __init__(
        self,
        auth_path: str | Path = "~/.hermes/auth.json",
        *,
        model: str = DEFAULT_NOUS_MODEL,
        base_url: str | Path | None = None,
    ) -> None:
        self.auth_path = Path(auth_path).expanduser()
        self.model = model
        self.base_url = (
            str(base_url).rstrip("/")
            if base_url
            else DEFAULT_NOUS_INFERENCE_URL
        )
        self._lock = threading.Lock()
        self._state: dict[str, Any] | None = None

    def _get_state(self) -> dict[str, Any]:
        if self._state is None:
            self._state = load_nous_state(self.auth_path)
        return self._state

    def resolve(self) -> dict[str, Any]:
        """Return ``{'api_key', 'base_url', 'model'}`` with a guaranteed-fresh token."""
        with self._lock:
            state = self._get_state()
            if _is_expired(state):
                refreshed = refresh_nous_token(state)
                try:
                    save_nous_state(self.auth_path, refreshed)
                except OSError as exc:  # pragma: no cover - best effort persist
                    logger.warning("Failed to persist refreshed Nous token: %s", exc)
                self._state = refreshed
                state = refreshed
            api_key = state.get("agent_key") or state.get("access_token")
            if not isinstance(api_key, str) or not api_key.strip():
                raise RuntimeError("Nous token resolved empty — re-auth Hermes")
            base_url = str(state.get("inference_base_url") or self.base_url).rstrip("/")
            return {"api_key": api_key, "base_url": base_url, "model": self.model}

    def force_refresh(self) -> dict[str, Any]:
        """Force a token refresh regardless of cached expiry (used for tests/recovery)."""
        with self._lock:
            state = self._get_state()
            refreshed = refresh_nous_token(state)
            try:
                save_nous_state(self.auth_path, refreshed)
            except OSError as exc:  # pragma: no cover - best effort persist
                logger.warning("Failed to persist refreshed Nous token: %s", exc)
            self._state = refreshed
            return self.resolve()
