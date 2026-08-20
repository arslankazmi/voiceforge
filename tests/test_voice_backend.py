"""Tests for the Tier B production backend (keyless — synthesis monkeypatched)."""

from __future__ import annotations

from fastapi.testclient import TestClient

import voiceforge.serve_common as sc
import voiceforge.voice_backend as vb
from voiceforge.voice_artifact import VoiceMeta
from voiceforge.voice_library import VoiceLibrary

_FAKE_WAV = b"RIFF____WAVEfake"


def _client(tmp_path):
    return TestClient(vb.create_app(VoiceLibrary(tmp_path)))


def _seed(tmp_path, name="Announcer", engine="chatterbox-turbo"):
    return VoiceLibrary(tmp_path).save(VoiceMeta(name=name, engine=engine), b"FAKEVOICE")


def test_studio_page(tmp_path):
    c = _client(tmp_path)
    r = c.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    # The recording studio wires the mic to the existing forge + say endpoints.
    assert "VoiceForge Studio" in r.text
    assert "getUserMedia" in r.text
    assert "/api/v1/voices/forge" in r.text


def test_health_ready_metrics(tmp_path):
    c = _client(tmp_path)
    assert c.get("/healthz").json()["status"] == "ok"
    assert c.get("/readyz").json()["status"] == "ready"
    m = c.get("/metrics")
    assert m.status_code == 200
    assert "voiceforge_synth_total" in m.text


def test_list_get_export_delete(tmp_path):
    _seed(tmp_path)
    c = _client(tmp_path)
    voices = c.get("/api/v1/voices").json()["voices"]
    assert voices == [{"slug": "announcer", "name": "Announcer", "engine": "chatterbox-turbo"}]
    assert c.get("/api/v1/voices/announcer").json()["name"] == "Announcer"
    assert c.get("/api/v1/voices/announcer/export").content == b"FAKEVOICE"
    assert c.get("/api/v1/voices/missing").status_code == 404
    assert c.request("DELETE", "/api/v1/voices/announcer").json()["deleted"] is True
    assert c.get("/api/v1/voices").json()["voices"] == []


def test_say_uses_stored_voice(tmp_path, monkeypatch):
    _seed(tmp_path)
    captured = {}

    def fake_synth(engine, text, settings):
        captured["engine"] = engine
        captured["voice_artifact"] = settings.get("voice_artifact")
        return _FAKE_WAV

    monkeypatch.setattr(vb, "synthesize", fake_synth)
    c = _client(tmp_path)
    r = c.post("/api/v1/voices/announcer/say", json={"text": "hello"})
    assert r.status_code == 200
    assert r.content == _FAKE_WAV
    assert captured["engine"] == "chatterbox-turbo"
    assert captured["voice_artifact"].endswith("voice.voice")


def test_say_forwards_expressive_knobs(tmp_path, monkeypatch):
    # A base voice: exaggeration/cfg_weight/temperature from the request reach synthesize.
    VoiceLibrary(tmp_path).save(VoiceMeta(name="Reader", engine="chatterbox"), b"FAKEVOICE")
    captured = {}
    monkeypatch.setattr(
        vb, "synthesize", lambda engine, text, settings: captured.update(settings) or _FAKE_WAV
    )
    c = _client(tmp_path)
    r = c.post(
        "/api/v1/voices/reader/say",
        json={"text": "hi", "exaggeration": 0.9, "cfg_weight": 0.3, "temperature": 1.1},
    )
    assert r.status_code == 200
    assert captured["exaggeration"] == 0.9
    assert captured["cfg_weight"] == 0.3
    assert captured["temperature"] == 1.1


def test_say_missing_voice_404(tmp_path):
    assert _client(tmp_path).post("/api/v1/voices/nope/say", json={"text": "hi"}).status_code == 404


def test_say_unavailable_503(tmp_path, monkeypatch):
    _seed(tmp_path)
    monkeypatch.setattr(vb, "synthesize", lambda engine, text, settings: None)
    c = _client(tmp_path)
    assert c.post("/api/v1/voices/announcer/say", json={"text": "hi"}).status_code == 503


def test_config_update_persists(tmp_path):
    _seed(tmp_path)
    c = _client(tmp_path)
    r = c.put("/api/v1/voices/announcer/config", json={"temperature": 1.2})
    assert r.status_code == 200
    assert r.json()["meta"]["temperature"] == 1.2
    assert _client(tmp_path).get("/api/v1/voices/announcer").json()["meta"]["temperature"] == 1.2


def test_auth_enforced_when_token_set(tmp_path, monkeypatch):
    _seed(tmp_path)
    monkeypatch.setenv("AUTH_TOKEN", "secret")
    monkeypatch.setattr(vb, "synthesize", lambda engine, text, settings: _FAKE_WAV)
    c = _client(tmp_path)
    assert c.get("/healthz").status_code == 200
    assert c.post("/api/v1/voices/announcer/say", json={"text": "hi"}).status_code == 401
    ok = c.post(
        "/api/v1/voices/announcer/say",
        json={"text": "hi"},
        headers={"Authorization": "Bearer secret"},
    )
    assert ok.status_code == 200


def test_split_sentences_helper():
    assert sc.split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert sc.split_sentences("   ") == []


def test_chunk_text_wraps_long_sentences():
    # Short sentences pass through untouched, one chunk each.
    assert sc.chunk_text("One. Two! Three?") == ["One.", "Two!", "Three?"]
    # A run-on sentence with no punctuation is broken on word boundaries under the cap.
    long_sentence = " ".join(["word"] * 200)  # ~1000 chars, no sentence break
    chunks = sc.chunk_text(long_sentence, max_chars=80)
    assert len(chunks) > 1
    assert all(len(c) <= 80 for c in chunks)
    # No words are lost in the wrap.
    assert " ".join(chunks).split() == long_sentence.split()
