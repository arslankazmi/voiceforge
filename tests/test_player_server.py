"""Tests for the Tier A per-voice server (keyless — synthesis is monkeypatched)."""

from __future__ import annotations

import struct

from fastapi.testclient import TestClient

import voiceforge.player_server as ps

_FAKE_WAV = b"RIFF____WAVEfake-audio-bytes"


def _unframe(body: bytes) -> list[bytes]:
    out, i = [], 0
    while i + 4 <= len(body):
        (n,) = struct.unpack(">I", body[i : i + 4])
        i += 4
        out.append(body[i : i + n])
        i += n
    return out


def test_healthz_and_admin_page():
    client = TestClient(ps.create_app(None))
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["synth_available"] is False
    admin = client.get("/")
    assert admin.status_code == 200
    assert "voice" in admin.text.lower()


def test_config_get_and_update():
    client = TestClient(ps.create_app(None))
    assert client.get("/config").status_code == 200
    r = client.put("/config", json={"temperature": 1.1})
    assert r.status_code == 200
    assert r.json()["temperature"] == 1.1


def test_say_returns_wav(monkeypatch):
    monkeypatch.setattr(ps, "synthesize", lambda engine, text, settings: _FAKE_WAV)
    client = TestClient(ps.create_app(None))
    r = client.post("/say", json={"text": "hello there"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert "X-Synth-Seconds" in r.headers
    assert r.content == _FAKE_WAV


def test_say_empty_text_422():
    client = TestClient(ps.create_app(None))
    assert client.post("/say", json={"text": "   "}).status_code == 422


def test_say_unavailable_503(monkeypatch):
    monkeypatch.setattr(ps, "synthesize", lambda engine, text, settings: None)
    client = TestClient(ps.create_app(None))
    assert client.post("/say", json={"text": "hi"}).status_code == 503


def test_stream_yields_one_frame_per_sentence(monkeypatch):
    monkeypatch.setattr(ps, "synthesize", lambda engine, text, settings: _FAKE_WAV)
    client = TestClient(ps.create_app(None))
    r = client.post("/stream", json={"text": "One sentence. Two sentence!"})
    assert r.status_code == 200
    frames = _unframe(r.content)
    assert len(frames) == 2
    assert all(f == _FAKE_WAV for f in frames)


def test_bearer_token_required(monkeypatch):
    monkeypatch.setattr(ps, "synthesize", lambda engine, text, settings: _FAKE_WAV)
    client = TestClient(ps.create_app(None, token="secret"))
    assert client.get("/healthz").status_code == 200
    assert client.post("/say", json={"text": "hi"}).status_code == 401
    r = client.post("/say", json={"text": "hi"}, headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200
