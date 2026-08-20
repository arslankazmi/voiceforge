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

#: Per-chunk character ceiling. Chatterbox synthesizes one utterance per ``generate``
#: call, capped at ~1000 speech tokens (~30-40s) and prone to alignment drift on long
#: inputs, so we never feed it more than roughly a sentence at a time.
_MAX_CHUNK_CHARS = 280


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-ish chunks for streaming synthesis."""
    parts = [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]
    if parts:
        return parts
    stripped = text.strip()
    return [stripped] if stripped else []


def chunk_text(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Split *text* into model-sized chunks: by sentence, then wrap over-long sentences.

    A single run-on sentence longer than *max_chars* is broken on word boundaries so no
    chunk exceeds Chatterbox's practical per-call limit.
    """
    chunks: list[str] = []
    for sent in split_sentences(text):
        if len(sent) <= max_chars:
            chunks.append(sent)
            continue
        cur = ""
        for word in sent.split():
            if cur and len(cur) + 1 + len(word) > max_chars:
                chunks.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}" if cur else word
        if cur:
            chunks.append(cur)
    return chunks


def synthesize(engine: str, text: str, settings: dict) -> bytes | None:
    """Synthesize *text* for a saved voice via its ``.voice`` artifact.

    ``settings["voice_artifact"]`` (path or bytes) provides the voice; ``temperature``
    tunes it. Long text is split into sentence-sized chunks (Chatterbox can't hold more
    than ~one utterance per call) and stitched back into a single WAV, with a short
    silence between chunks, under one model load. Returns None (→ HTTP 503) when
    chatterbox isn't installed.
    """
    if importlib.util.find_spec("chatterbox") is None:
        return None

    import torch  # noqa: PLC0415

    from ._chatter_common import supported_generate_kwargs, tensor_to_wav  # noqa: PLC0415
    from ._model_manager import heavy_session  # noqa: PLC0415
    from .engine import loader_for  # noqa: PLC0415
    from .voice_artifact import apply, ensure_compatible, load, loads  # noqa: PLC0415

    artifact = settings.get("voice_artifact")
    chunks = chunk_text(text) or [text.strip()]

    # Expressive knobs. Temperature applies to both engines; exaggeration and cfg_weight
    # are base-only (turbo accepts but warns-and-ignores them), so we only forward those
    # for base to keep the logs clean. supported_generate_kwargs drops any the fn lacks.
    requested = {
        "temperature": settings.get("temperature"),
        "exaggeration": settings.get("exaggeration") if engine == "chatterbox" else None,
        "cfg_weight": settings.get("cfg_weight") if engine == "chatterbox" else None,
    }
    requested = {k: v for k, v in requested.items() if v is not None}

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
        kwargs = supported_generate_kwargs(model.generate, **requested) if requested else {}

        pieces = []
        for chunk in chunks:
            audio = model.generate(chunk, **kwargs)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            pieces.append(audio)
        if not pieces:
            return None

        if len(pieces) == 1:
            full = pieces[0]
        else:
            gap_samples = int(getattr(model, "sr", 24000) * 0.28)
            gap = torch.zeros(
                (pieces[0].shape[0], gap_samples),
                dtype=pieces[0].dtype,
                device=pieces[0].device,
            )
            stitched: list = []
            for i, piece in enumerate(pieces):
                if i:
                    stitched.append(gap)
                stitched.append(piece)
            full = torch.cat(stitched, dim=-1)
        return tensor_to_wav(full, model.sr)


def frame(wav: bytes) -> bytes:
    """Length-prefixed frame (4-byte big-endian length + WAV) for the stream protocol."""
    return struct.pack(">I", len(wav)) + wav


def make_auth(token: str | None):  # noqa: ANN201
    """Build a FastAPI dependency requiring ``Authorization: Bearer <token>`` (no-op if unset)."""

    async def _dep(authorization: str | None = Header(default=None)) -> None:
        if token and authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="Missing or invalid bearer token")

    return _dep
