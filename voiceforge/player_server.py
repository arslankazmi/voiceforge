"""Tier A — a self-contained server for **one** ``.voice`` artifact.

``voiceforge serve myvoice.voice`` spins this up: a tiny FastAPI app serving a single
voice, with a **browser admin page** at ``/`` for tuning parameters and hearing the
result, plus a small JSON/stream API. Safe by default (localhost bind, optional bearer
token). Built at import from ``VOICEFORGE_ARTIFACT`` / ``VOICEFORGE_TOKEN`` env vars so
``uvicorn voiceforge.player_server:app`` works; ``create_app`` is used by tests.
"""

from __future__ import annotations

import logging
import os
import time

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .serve_common import frame, make_auth, split_sentences, synthesize

logger = logging.getLogger(__name__)


class _VoiceState:
    """Lazily reads the artifact's meta (name/engine/defaults) without a full model load."""

    def __init__(self, artifact_path: str | None) -> None:
        self.path = artifact_path
        self._loaded = False
        self.name = "voice"
        self.engine = "chatterbox-turbo"
        self.available = False
        self.config: dict = {"temperature": 0.8, "exaggeration": 0.5, "cfg_weight": 0.5}

    def ensure(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path:
            return
        try:
            from .voice_artifact import read_meta  # noqa: PLC0415

            meta = read_meta(self.path)
            self.name = meta.name
            self.engine = meta.engine
            self.config = {
                "temperature": meta.temperature,
                "exaggeration": meta.exaggeration,
                "cfg_weight": meta.cfg_weight,
            }
            self.available = True
        except Exception:  # noqa: BLE001
            logger.warning("Could not read voice meta from %s", self.path, exc_info=True)

    def synth_settings(
        self,
        temperature: float | None = None,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
    ) -> dict:
        s = dict(self.config)
        for key, val in (
            ("temperature", temperature),
            ("exaggeration", exaggeration),
            ("cfg_weight", cfg_weight),
        ):
            if val is not None:
                s[key] = val
        s["voice_artifact"] = self.path
        s["engine"] = self.engine
        return s


class SayRequest(BaseModel):
    text: str
    temperature: float | None = None
    exaggeration: float | None = None  # base only (turbo ignores)
    cfg_weight: float | None = None  # base only (turbo ignores)


class ConfigUpdate(BaseModel):
    temperature: float | None = None
    exaggeration: float | None = None
    cfg_weight: float | None = None


def _admin_html(name: str, engine: str = "chatterbox-turbo") -> str:
    turbo = engine == "chatterbox-turbo"
    dis = "disabled" if turbo else ""
    note = (
        "<p class=note>⚡ Turbo ignores exaggeration &amp; CFG — only temperature applies.</p>"
        if turbo
        else ""
    )
    return f"""<!doctype html><html><head><meta charset=utf-8>
<title>VoiceForge — {name}</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>body{{font-family:system-ui,sans-serif;max-width:640px;margin:2rem auto;padding:0 1rem}}
h1{{font-size:1.3rem}}label{{display:block;margin:.6rem 0 .2rem;font-size:.9rem}}
textarea,input{{width:100%;box-sizing:border-box}}button{{margin-top:1rem;padding:.5rem 1rem}}
.note{{font-size:.8rem;opacity:.7}}#stat{{margin-top:.6rem;font-size:.85rem;color:#555}}</style></head><body>
<h1>🎙 {name}</h1><p>Local voice server ({engine}). Type text and speak it in this voice.</p>
<label>Text</label><textarea id=text rows=4>Hello — this is my cloned voice.</textarea>
<label>Exaggeration <span id=xv>0.50</span></label>
<input id=exag type=range min=0.2 max=1.0 step=0.05 value=0.5 {dis}
       oninput="xv.textContent=(+exag.value).toFixed(2)">
<label>CFG weight <span id=cv>0.50</span></label>
<input id=cfg type=range min=0.0 max=1.0 step=0.05 value=0.5 {dis}
       oninput="cv.textContent=(+cfg.value).toFixed(2)">
<label>Temperature <span id=tv>0.80</span></label>
<input id=temp type=range min=0.3 max=1.5 step=0.05 value=0.8 oninput="tv.textContent=(+temp.value).toFixed(2)">
{note}
<label>Bearer token (if the server requires one)</label><input id=tok type=password placeholder="optional">
<button onclick=speak()>Speak</button><div id=stat></div><audio id=au controls style="width:100%;margin-top:1rem"></audio>
<script>
const TURBO={str(turbo).lower()};
async function speak(){{
  stat.textContent='synthesizing…';
  const h={{'Content-Type':'application/json'}}; if(tok.value)h['Authorization']='Bearer '+tok.value;
  const body={{text:text.value,temperature:+temp.value}};
  if(!TURBO){{body.exaggeration=+exag.value; body.cfg_weight=+cfg.value;}}
  const t0=performance.now();
  const r=await fetch('/say',{{method:'POST',headers:h,body:JSON.stringify(body)}});
  if(!r.ok){{stat.textContent='error '+r.status+': '+await r.text();return;}}
  const rt=((performance.now()-t0)/1000).toFixed(2), ss=r.headers.get('X-Synth-Seconds');
  au.src=URL.createObjectURL(await r.blob()); au.play();
  stat.textContent=`synth ${{ss}}s · round-trip ${{rt}}s`;
}}
</script></body></html>"""


def create_app(artifact_path: str | None = None, token: str | None = None) -> FastAPI:
    app = FastAPI(title="VoiceForge — voice server", version="0.1.0")
    state = _VoiceState(artifact_path)
    auth = make_auth(token)

    @app.get("/", response_class=HTMLResponse)
    def admin() -> str:
        state.ensure()
        return _admin_html(state.name, state.engine)

    @app.get("/healthz")
    def healthz() -> dict:
        state.ensure()
        return {
            "status": "ok",
            "voice": state.name,
            "engine": state.engine,
            "synth_available": state.available,
        }

    @app.get("/config")
    def get_config() -> dict:
        state.ensure()
        return {"name": state.name, "engine": state.engine, **state.config}

    @app.put("/config", dependencies=[Depends(auth)])
    def put_config(update: ConfigUpdate) -> dict:
        state.ensure()
        for k, v in update.model_dump(exclude_none=True).items():
            state.config[k] = v
        return {"name": state.name, "engine": state.engine, **state.config}

    @app.post("/say", dependencies=[Depends(auth)])
    def say(body: SayRequest) -> StreamingResponse:
        state.ensure()
        if not body.text.strip():
            raise HTTPException(status_code=422, detail="text is empty")
        t0 = time.perf_counter()
        wav = synthesize(
            state.engine,
            body.text,
            state.synth_settings(body.temperature, body.exaggeration, body.cfg_weight),
        )
        if wav is None:
            raise HTTPException(
                status_code=503, detail="synthesis unavailable (install [clone] extra)"
            )
        return StreamingResponse(
            iter([wav]),
            media_type="audio/wav",
            headers={
                "X-Synth-Seconds": f"{time.perf_counter() - t0:.4f}",
                "X-Engine": state.engine,
            },
        )

    @app.post("/stream", dependencies=[Depends(auth)])
    def stream(body: SayRequest) -> StreamingResponse:
        state.ensure()
        settings = state.synth_settings(body.temperature, body.exaggeration, body.cfg_weight)
        sentences = split_sentences(body.text)

        def gen():  # noqa: ANN202
            for sent in sentences:
                wav = synthesize(state.engine, sent, settings)
                if wav:
                    yield frame(wav)

        return StreamingResponse(
            gen(),
            media_type="application/octet-stream",
            headers={"X-Format": "framed-wav", "X-Engine": state.engine},
        )

    return app


app = create_app(os.environ.get("VOICEFORGE_ARTIFACT"), os.environ.get("VOICEFORGE_TOKEN"))
