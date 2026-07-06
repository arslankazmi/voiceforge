"""Tier B — production voice backend service (headless, multi-voice, deployable).

A hardened ASGI app that serves a whole library of ``.voice`` artifacts behind other
apps. Thanks to "voices are cheap, models are expensive", it holds **one** resident
model (via ``heavy_session``) and hot-swaps each voice's ``conds`` per request — dozens
of voices cost one model's worth of RAM, never OOMing the host.

Run: ``voiceforge backend`` (or ``uvicorn voiceforge.voice_backend:app``). Config via env:
``AUTH_TOKEN`` (bearer), ``VOICEFORGE_CORS_ORIGINS`` (comma-separated), ``VOICEFORGE_VOICES_DIR``.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import tempfile
import time

from fastapi import Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from .serve_common import frame, make_auth, split_sentences, synthesize
from .voice_library import get_library

logger = logging.getLogger(__name__)

_SYNTHS = Counter("voiceforge_synth_total", "Synthesis requests", ["voice", "engine"])
_SYNTH_SECONDS = Histogram("voiceforge_synth_seconds", "Synthesis wall-clock seconds", ["engine"])


class SayRequest(BaseModel):
    text: str
    temperature: float | None = None


class ConfigUpdate(BaseModel):
    temperature: float | None = None
    exaggeration: float | None = None
    cfg_weight: float | None = None


def _auth_token() -> str | None:
    return os.environ.get("AUTH_TOKEN") or None


def _cors_origins() -> list[str]:
    raw = os.environ.get("VOICEFORGE_CORS_ORIGINS", "http://127.0.0.1")
    return [o.strip() for o in raw.split(",") if o.strip()]


def _forge_and_store(clip_path: str, engine: str, name: str, exaggeration: float, source: str):  # noqa: ANN201
    """Forge a clip into the library (needs chatterbox). Returns StoredVoice or None."""
    if importlib.util.find_spec("chatterbox") is None:
        return None
    from ._model_manager import heavy_session  # noqa: PLC0415
    from .cli import _chatterbox_version, _now_iso  # noqa: PLC0415
    from .engine import loader_for  # noqa: PLC0415
    from .voice_artifact import VoiceMeta, forge  # noqa: PLC0415

    meta = VoiceMeta(
        name=name,
        engine=engine,
        exaggeration=exaggeration,
        chatterbox_version=_chatterbox_version(),
        created=_now_iso(),
    )
    with heavy_session(engine, loader_for(engine)) as model:
        if model is None:
            return None
        meta.sample_rate = int(getattr(model, "sr", 24000))
        blob = forge(model, clip_path, meta)
    return get_library().save(meta, blob, source=source)


def create_app(library=None) -> FastAPI:  # noqa: ANN001
    app = FastAPI(title="VoiceForge Backend", version="0.1.0")
    auth = make_auth(_auth_token())
    app.add_middleware(
        CORSMiddleware, allow_origins=_cors_origins(), allow_methods=["*"], allow_headers=["*"]
    )
    lib = library if library is not None else get_library()

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict:
        return {"status": "ready", "chatterbox": importlib.util.find_spec("chatterbox") is not None}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/api/v1/voices")
    def list_voices() -> dict:
        return {
            "voices": [{"slug": v.slug, "name": v.name, "engine": v.engine} for v in lib.list()]
        }

    @app.get("/api/v1/voices/{slug}")
    def get_voice(slug: str) -> dict:
        v = lib.get(slug)
        if v is None:
            raise HTTPException(status_code=404, detail="voice not found")
        return {
            "slug": v.slug,
            "name": v.name,
            "engine": v.engine,
            "meta": v.meta,
            "notes": v.notes,
        }

    @app.post("/api/v1/voices", dependencies=[Depends(auth)])
    async def import_voice(artifact: UploadFile = File(...)) -> dict:  # noqa: B008
        data = await artifact.read()
        try:
            stored = lib.import_artifact(data, source="import")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=f"bad .voice artifact: {exc}") from exc
        return {"slug": stored.slug, "name": stored.name, "engine": stored.engine}

    @app.post("/api/v1/voices/forge", dependencies=[Depends(auth)])
    async def forge_voice(
        clip: UploadFile = File(...),  # noqa: B008
        name: str = Form(...),  # noqa: B008
        engine: str = Form(default="chatterbox-turbo"),  # noqa: B008
        exaggeration: float = Form(default=0.5),  # noqa: B008
    ) -> dict:
        suffix = os.path.splitext(clip.filename or "")[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, dir="/tmp", delete=False) as tmp:
            tmp.write(await clip.read())
            tmp_path = tmp.name
        try:
            stored = _forge_and_store(tmp_path, engine, name, exaggeration, source="forge")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        if stored is None:
            raise HTTPException(
                status_code=503, detail="forging unavailable (install [clone] extra)"
            )
        return {"slug": stored.slug, "name": stored.name, "engine": stored.engine}

    @app.get("/api/v1/voices/{slug}/export")
    def export_voice(slug: str) -> Response:
        data = lib.export(slug)
        if data is None:
            raise HTTPException(status_code=404, detail="voice not found")
        return Response(
            data,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{slug}.voice"'},
        )

    @app.delete("/api/v1/voices/{slug}", dependencies=[Depends(auth)])
    def delete_voice(slug: str) -> dict:
        return {"deleted": lib.delete(slug)}

    @app.put("/api/v1/voices/{slug}/config", dependencies=[Depends(auth)])
    def config_voice(slug: str, update: ConfigUpdate) -> dict:
        import json  # noqa: PLC0415
        from dataclasses import asdict  # noqa: PLC0415

        v = lib.get(slug)
        if v is None:
            raise HTTPException(status_code=404, detail="voice not found")
        v.meta.update(update.model_dump(exclude_none=True))
        lib._json_path(slug).write_text(json.dumps(asdict(v), indent=2))  # noqa: SLF001
        return {"slug": slug, "meta": v.meta}

    def _voice_settings(slug: str, temperature: float | None):  # noqa: ANN202
        v = lib.get(slug)
        if v is None:
            raise HTTPException(status_code=404, detail="voice not found")
        settings = v.settings()
        if temperature is not None:
            settings["temperature"] = temperature
        return v, settings

    @app.post("/api/v1/voices/{slug}/say", dependencies=[Depends(auth)])
    def say(slug: str, body: SayRequest) -> StreamingResponse:
        if not body.text.strip():
            raise HTTPException(status_code=422, detail="text is empty")
        v, settings = _voice_settings(slug, body.temperature)
        t0 = time.perf_counter()
        wav = synthesize(v.engine, body.text, settings)
        if wav is None:
            raise HTTPException(
                status_code=503, detail="synthesis unavailable (install [clone] extra)"
            )
        elapsed = time.perf_counter() - t0
        _SYNTHS.labels(voice=slug, engine=v.engine).inc()
        _SYNTH_SECONDS.labels(engine=v.engine).observe(elapsed)
        return StreamingResponse(
            iter([wav]),
            media_type="audio/wav",
            headers={"X-Synth-Seconds": f"{elapsed:.4f}", "X-Engine": v.engine},
        )

    @app.post("/api/v1/voices/{slug}/stream", dependencies=[Depends(auth)])
    def stream(slug: str, body: SayRequest) -> StreamingResponse:
        v, settings = _voice_settings(slug, body.temperature)
        sentences = split_sentences(body.text)

        def gen():  # noqa: ANN202
            for sent in sentences:
                wav = synthesize(v.engine, sent, settings)
                if wav:
                    _SYNTHS.labels(voice=slug, engine=v.engine).inc()
                    yield frame(wav)

        return StreamingResponse(
            gen(),
            media_type="application/octet-stream",
            headers={"X-Format": "framed-wav", "X-Engine": v.engine},
        )

    return app


app = create_app()
