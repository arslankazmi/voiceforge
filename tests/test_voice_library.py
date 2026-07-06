"""Tests for the .voice artifact store (torch-free — no model loads)."""

from __future__ import annotations

from voiceforge.voice_artifact import VoiceMeta
from voiceforge.voice_library import StoredVoice, VoiceLibrary, slugify


def test_slugify():
    assert slugify("Retro Hype Announcer") == "retro-hype-announcer"
    assert slugify("  spaces  ") == "spaces"
    assert slugify("!!!") == "voice"


def test_save_get_list_delete_round_trip(tmp_path):
    lib = VoiceLibrary(tmp_path)
    meta = VoiceMeta(name="Announcer", engine="chatterbox-turbo", temperature=0.9)

    stored = lib.save(meta, b"FAKEVOICEBYTES", notes="test", source="my clip")
    assert isinstance(stored, StoredVoice)
    assert stored.slug == "announcer"
    assert stored.engine == "chatterbox-turbo"

    got = lib.get("announcer")
    assert got is not None
    assert got.name == "Announcer"
    assert got.notes == "test"

    s = got.settings()
    assert s["voice_artifact"] == stored.artifact_path
    assert s["engine"] == "chatterbox-turbo"
    assert s["temperature"] == 0.9

    assert [v.slug for v in lib.list()] == ["announcer"]
    assert lib.export("announcer") == b"FAKEVOICEBYTES"

    assert lib.delete("announcer") is True
    assert lib.get("announcer") is None
    assert lib.list() == []
    assert lib.delete("announcer") is False


def test_unique_slug_on_name_collision(tmp_path):
    lib = VoiceLibrary(tmp_path)
    meta = VoiceMeta(name="Same Name", engine="chatterbox")
    a = lib.save(meta, b"a")
    b = lib.save(meta, b"b")
    assert a.slug == "same-name"
    assert b.slug == "same-name-2"


def test_get_missing_returns_none(tmp_path):
    assert VoiceLibrary(tmp_path).get("nope") is None
    assert VoiceLibrary(tmp_path).export("nope") is None
