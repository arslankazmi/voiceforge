"""Tests for the .voice artifact metadata layer (torch-free — no model loads)."""

from __future__ import annotations

import importlib.util

import pytest

from voiceforge.voice_artifact import (
    ENGINES,
    SCHEMA_VERSION,
    VOICE_EXT,
    ArtifactError,
    VoiceMeta,
    ensure_compatible,
    validate_engine,
)

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_meta_round_trips_through_dict():
    meta = VoiceMeta(name="Retro Announcer", engine="chatterbox-turbo", sample_rate=24000)
    restored = VoiceMeta.from_dict(meta.to_dict())
    assert restored == meta
    assert restored.schema_version == SCHEMA_VERSION


def test_meta_from_dict_ignores_unknown_keys():
    meta = VoiceMeta.from_dict({"name": "X", "engine": "chatterbox", "bogus": 123})
    assert meta.name == "X"
    assert meta.engine == "chatterbox"


def test_validate_engine_accepts_known_rejects_unknown():
    for eng in ENGINES:
        validate_engine(eng)
    with pytest.raises(ArtifactError):
        validate_engine("elevenlabs")


def test_ensure_compatible_rejects_cross_variant():
    turbo_voice = VoiceMeta(name="v", engine="chatterbox-turbo")
    ensure_compatible(turbo_voice, "chatterbox-turbo")
    with pytest.raises(ArtifactError):
        ensure_compatible(turbo_voice, "chatterbox")


def test_voice_ext_constant():
    assert VOICE_EXT == ".voice"


@pytest.mark.skipif(_HAS_TORCH, reason="torch present — real forge/load exercised elsewhere")
def test_dumps_requires_torch():
    from voiceforge.voice_artifact import dumps

    with pytest.raises(ImportError):
        dumps(object(), VoiceMeta(name="v", engine="chatterbox-turbo"))
