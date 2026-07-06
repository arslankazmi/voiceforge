"""Shared helpers for the VoiceForge servers (Tier A per-voice + Tier B backend).

``synthesize`` is the single choke point both servers call — it loads the voice's
conditioning and generates under the OOM-safe ``heavy_session``. Kept torch-free at
import; tests monkeypatch ``synthesize`` to exercise HTTP/streaming without a model.
"""

from __future__ import annotations

import importlib.util
import re
import struct

from fastapi import Header, HTTPException

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?", re.S)


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-ish chunks for streaming synthesis."""
    parts = [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]
    if parts:
        return parts
    stripped = text.strip()
    return [stripped] if stripped else []


def synthesize(engine: str, text: str, settings: dict) -> bytes | None:
    """Synthesize *text* for a saved voice via its ``.voice`` artifact.

    ``settings["voice_artifact"]`` (path or bytes) provides the voice; ``temperature``
    tunes it. Returns None (→ HTTP 503) when chatterbox isn't installed.
    """
    if importlib.util.find_spec("chatterbox") is None:
        return None

    from ._chatter_common import supported_generate_kwargs, tensor_to_wav  # noqa: PLC0415
    from ._model_manager import heavy_session  # noqa: PLC0415
    from .engine import loader_for  # noqa: PLC0415
    from .voice_artifact import apply, ensure_compatible, load, loads  # noqa: PLC0415

    artifact = settings.get("voice_artifact")
    temperature = settings.get("temperature")

    with heavy_session(engine, loader_for(engine)) as model:
        if model is None:
            return None
        if artifact is not None:
            cond, meta = (
                loads(bytes(artifact))
                if isinstance(artifact, (bytes, bytearray))
                else load(artifact)
            )
            ensure_compatible(meta, engine)
            apply(model, cond)
        kwargs = (
            supported_generate_kwargs(model.generate, temperature=temperature)
            if temperature is not None
            else {}
        )
        audio = model.generate(text, **kwargs)
        return tensor_to_wav(audio, model.sr)


def frame(wav: bytes) -> bytes:
    """Length-prefixed frame (4-byte big-endian length + WAV) for the stream protocol."""
    return struct.pack(">I", len(wav)) + wav


def make_auth(token: str | None):  # noqa: ANN201
    """Build a FastAPI dependency requiring ``Authorization: Bearer <token>`` (no-op if unset)."""

    async def _dep(authorization: str | None = Header(default=None)) -> None:
        if token and authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="Missing or invalid bearer token")

    return _dep
