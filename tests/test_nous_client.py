"""Tests for the NousResearch (Hermes) OAuth token resolver.

These tests cover the pure logic (expiry parsing/decision) without hitting the
network or requiring a live Hermes auth.json. The refresh + call path is
exercised against the real operational server in deployment.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dating_rag.generation import nous_client


def _state(expires_at: str | None, access_token: str = "tok") -> dict:
    return {
        "access_token": access_token,
        "refresh_token": "rt",
        "client_id": "hermes-cli",
        "portal_base_url": "https://portal.nousresearch.com",
        "inference_base_url": "https://inference-api.nousresearch.com/v1",
        "expires_at": expires_at,
    }


def test_parse_expires_at_iso():
    dt = nous_client._parse_expires_at("2026-07-15T18:27:12+00:00")
    assert dt is not None
    assert dt.tzinfo is not None


def test_parse_expires_at_z_and_naive():
    assert nous_client._parse_expires_at("2026-07-15T18:27:12Z") is not None
    naive = nous_client._parse_expires_at("2026-07-15T18:27:12")
    assert naive is not None and naive.tzinfo is not None


def test_parse_expires_at_invalid():
    assert nous_client._parse_expires_at("not-a-date") is None
    assert nous_client._parse_expires_at("") is None
    assert nous_client._parse_expires_at(None) is None


def test_is_expired_missing_token():
    assert nous_client._is_expired({"access_token": ""}) is True
    assert nous_client._is_expired({}) is True


def test_is_expired_no_expiry():
    assert nous_client._is_expired({"access_token": "tok"}) is True


def test_is_expired_future():
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    assert nous_client._is_expired(_state(future)) is False


def test_is_expired_within_grace():
    soon = (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()
    assert nous_client._is_expired(_state(soon)) is True


def test_is_expired_past():
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    assert nous_client._is_expired(_state(past)) is True


def test_load_nous_state(tmp_path: Path):
    auth = tmp_path / "auth.json"
    auth.write_text(
        json.dumps(
            {"providers": {"nous": _state("2030-01-01T00:00:00+00:00")}}
        ),
        encoding="utf-8",
    )
    state = nous_client.load_nous_state(auth)
    assert state["access_token"] == "tok"


def test_load_nous_state_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        nous_client.load_nous_state(tmp_path / "nope.json")


def test_load_nous_state_no_nous_block(tmp_path: Path):
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"providers": {}}), encoding="utf-8")
    with pytest.raises(KeyError):
        nous_client.load_nous_state(auth)


def test_save_nous_state_persists(tmp_path: Path):
    auth = tmp_path / "auth.json"
    auth.write_text(
        json.dumps({"providers": {"nous": _state("2030-01-01T00:00:00+00:00")}}),
        encoding="utf-8",
    )
    refreshed = _state("2031-01-01T00:00:00+00:00", access_token="newtok")
    nous_client.save_nous_state(auth, refreshed)
    loaded = json.loads(auth.read_text(encoding="utf-8"))
    assert loaded["providers"]["nous"]["access_token"] == "newtok"


def test_resolve_uses_cache_without_refresh(tmp_path: Path, monkeypatch):
    """A valid (future) token must resolve without calling the network."""
    auth = tmp_path / "auth.json"
    future = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    auth.write_text(
        json.dumps({"providers": {"nous": _state(future, access_token="good")}}),
        encoding="utf-8",
    )
    calls = {"refresh": 0}

    def fake_refresh(state, timeout_seconds=15.0):
        calls["refresh"] += 1
        return dict(state)

    monkeypatch.setattr(nous_client, "refresh_nous_token", fake_refresh)
    provider = nous_client.NousTokenProvider(auth_path=str(auth), model="m")
    creds = provider.resolve()
    assert creds["api_key"] == "good"
    assert calls["refresh"] == 0


def test_resolve_refreshes_when_expired(tmp_path: Path, monkeypatch):
    """An expired token must trigger a refresh and persist the new state."""
    auth = tmp_path / "auth.json"
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    auth.write_text(
        json.dumps({"providers": {"nous": _state(past, access_token="old")}}),
        encoding="utf-8",
    )
    new_expiry = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()

    def fake_refresh(state, timeout_seconds=15.0):
        refreshed = dict(state)
        refreshed["access_token"] = "fresh"
        refreshed["expires_at"] = new_expiry
        return refreshed

    monkeypatch.setattr(nous_client, "refresh_nous_token", fake_refresh)
    provider = nous_client.NousTokenProvider(auth_path=str(auth), model="m")
    creds = provider.resolve()
    assert creds["api_key"] == "fresh"
    # persisted back to disk
    saved = json.loads(auth.read_text(encoding="utf-8"))["providers"]["nous"]
    assert saved["access_token"] == "fresh"
