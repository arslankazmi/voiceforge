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
from fastapi.responses import HTMLResponse, StreamingResponse
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
    exaggeration: float | None = None  # base only (turbo ignores)
    cfg_weight: float | None = None  # base only (turbo ignores)


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


_STUDIO_HTML = """<!doctype html><html><head><meta charset=utf-8>
<title>VoiceForge Studio</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:light dark}
body{font-family:system-ui,sans-serif;max-width:680px;margin:1.5rem auto;padding:0 1rem;line-height:1.45}
h1{font-size:1.4rem;margin-bottom:.2rem}h2{font-size:1rem;margin:1.4rem 0 .4rem}
.card{border:1px solid #8884;border-radius:10px;padding:1rem;margin:.8rem 0}
label{display:block;margin:.6rem 0 .2rem;font-size:.85rem;opacity:.85}
input,select,textarea{width:100%;box-sizing:border-box;padding:.4rem;font:inherit}
textarea{min-height:4rem}
button{padding:.5rem .9rem;font:inherit;border-radius:8px;border:1px solid #8886;cursor:pointer;background:#4f46e5;color:#fff}
button.ghost{background:transparent;color:inherit}
button:disabled{opacity:.5;cursor:not-allowed}
.row{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}
.rec{background:#dc2626}
#timer{font-variant-numeric:tabular-nums;font-size:1.5rem;font-weight:600}
.hint{font-size:.8rem;opacity:.7}
.warn{color:#d97706}.err{color:#dc2626}.ok{color:#16a34a}
.vitem{display:flex;justify-content:space-between;align-items:center;gap:.5rem;padding:.4rem 0;border-bottom:1px solid #8883}
.script{background:#8881;border-left:3px solid #4f46e5;border-radius:6px;padding:.7rem .9rem;margin:.6rem 0;font-size:1.05rem;line-height:1.6}
audio{width:100%;margin-top:.5rem}
</style></head><body>
<h1>🎙 VoiceForge Studio</h1>
<p class=hint>Record your voice with the onboard mic and clone it — right here, on-device. Nothing leaves this machine.</p>

<div class=card>
  <h2>1 · Record a sample</h2>
  <p class=hint>Read this aloud at a natural, even pace — it packs nearly every English sound
     into a few sentences, which gives the clone far better coverage than casual talking:</p>
  <blockquote class=script>
    Please call Stella. Ask her to bring these things with her from the store:
    six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack
    for her brother Bob. We also need a small plastic snake and a big toy frog for the kids.
  </blockquote>
  <p class=hint>Stopping after “brother Bob” (~15s) is plenty; reading the whole thing is even better.</p>
  <div class=row>
    <button id=recBtn onclick=toggleRec()>● Record</button>
    <span id=timer>0.0s</span>
    <span id=recHint class=hint>Aim for 10–20s of clear, continuous speech.</span>
  </div>
  <audio id=preview controls hidden></audio>
</div>

<div class=card>
  <h2>2 · Forge the clone</h2>
  <label>Voice name</label><input id=vname placeholder="My Voice" value="My Voice">
  <div class=row style="margin-top:.6rem">
    <div style="flex:1">
      <label>Engine</label>
      <select id=engine onchange=engineChanged()>
        <option value="chatterbox">base — flexible, works with short clips</option>
        <option value="chatterbox-turbo">turbo — faster, needs &gt; 5s</option>
      </select>
    </div>
    <div style="flex:1">
      <label>Exaggeration <span id=exv>0.50</span></label>
      <input id=exag type=range min=0.2 max=1.0 step=0.05 value=0.5
             oninput="exv.textContent=(+exag.value).toFixed(2)">
    </div>
  </div>
  <label>Bearer token (only if this server sets AUTH_TOKEN)</label>
  <input id=tok type=password placeholder="optional">
  <div class=row style="margin-top:.8rem">
    <button id=forgeBtn onclick=createClone() disabled>Create clone</button>
    <span id=forgeStat class=hint></span>
  </div>
</div>

<div class=card>
  <h2>3 · Your voices</h2>
  <div id=voices class=hint>loading…</div>
  <div id=sayBox hidden style="margin-top:1rem">
    <label>Speaking as <b id=activeName></b></label>
    <textarea id=sayText>Hello — this is my cloned voice.</textarea>
    <div class=row style="margin-top:.4rem">
      <div style="flex:1">
        <label>Exaggeration <span id=sxv>0.50</span></label>
        <input id=sxag type=range min=0.2 max=1.0 step=0.05 value=0.5
               oninput="sxv.textContent=(+sxag.value).toFixed(2)">
      </div>
      <div style="flex:1">
        <label>CFG weight <span id=scv>0.50</span></label>
        <input id=scfg type=range min=0.0 max=1.0 step=0.05 value=0.5
               oninput="scv.textContent=(+scfg.value).toFixed(2)">
      </div>
      <div style="flex:1">
        <label>Temperature <span id=stv>0.80</span></label>
        <input id=stmp type=range min=0.3 max=1.5 step=0.05 value=0.8
               oninput="stv.textContent=(+stmp.value).toFixed(2)">
      </div>
    </div>
    <div id=turboNote class=hint hidden>⚡ Turbo ignores exaggeration &amp; CFG — only temperature applies.</div>
    <div class=row style="margin-top:.6rem">
      <button onclick=speak()>Speak</button>
      <span id=sayStat class=hint></span>
    </div>
    <audio id=au controls hidden></audio>
  </div>
</div>

<script>
let rec=null, chunks=[], t0=0, timer=null, lastBlob=null, activeSlug=null, activeEngine='chatterbox';
const $=id=>document.getElementById(id);
function authH(base){const h=base||{};if($('tok').value)h['Authorization']='Bearer '+$('tok').value;return h;}
function engineChanged(){checkTurbo();}
function recSeconds(){return lastBlobDur;}
let lastBlobDur=0;

async function toggleRec(){
  if(rec && rec.state==='recording'){rec.stop();return;}
  let stream;
  try{stream=await navigator.mediaDevices.getUserMedia({audio:true});}
  catch(e){$('recHint').innerHTML='<span class=err>mic blocked: '+e.message+'</span>';return;}
  chunks=[]; rec=new MediaRecorder(stream);
  rec.ondataavailable=e=>{if(e.data.size)chunks.push(e.data);};
  rec.onstop=async()=>{
    stream.getTracks().forEach(t=>t.stop());
    clearInterval(timer);
    $('recBtn').textContent='● Record'; $('recBtn').classList.remove('rec');
    lastBlob=new Blob(chunks,{type:rec.mimeType||'audio/webm'});
    const url=URL.createObjectURL(lastBlob);
    $('preview').src=url; $('preview').hidden=false;
    // measure duration
    try{const ab=await lastBlob.arrayBuffer();
        const ac=new (window.AudioContext||window.webkitAudioContext)();
        const buf=await ac.decodeAudioData(ab); lastBlobDur=buf.duration; ac.close&&ac.close();}
    catch(e){lastBlobDur=0;}
    $('forgeBtn').disabled=false; checkTurbo();
  };
  rec.start(); t0=performance.now();
  $('recBtn').textContent='■ Stop'; $('recBtn').classList.add('rec');
  timer=setInterval(()=>{$('timer').textContent=((performance.now()-t0)/1000).toFixed(1)+'s';},100);
}

function checkTurbo(){
  const turbo=$('engine').value==='chatterbox-turbo';
  $('exag').disabled=turbo;
  if(turbo && lastBlobDur && lastBlobDur<=5.2)
    $('forgeStat').innerHTML='<span class=warn>turbo needs &gt;5s — your clip is '+lastBlobDur.toFixed(1)+'s. Record longer or use base.</span>';
  else $('forgeStat').textContent='';
}

// Decode any recorded blob → mono 16-bit PCM WAV (so the server needs no transcoder).
async function toWav(blob){
  const ab=await blob.arrayBuffer();
  const ac=new (window.AudioContext||window.webkitAudioContext)();
  const b=await ac.decodeAudioData(ab); ac.close&&ac.close();
  const n=b.length, sr=b.sampleRate, mono=new Float32Array(n);
  for(let c=0;c<b.numberOfChannels;c++){const d=b.getChannelData(c);for(let i=0;i<n;i++)mono[i]+=d[i]/b.numberOfChannels;}
  const buf=new ArrayBuffer(44+n*2), v=new DataView(buf);
  const ws=(o,s)=>{for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i));};
  ws(0,'RIFF');v.setUint32(4,36+n*2,true);ws(8,'WAVE');ws(12,'fmt ');
  v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);
  v.setUint32(24,sr,true);v.setUint32(28,sr*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);
  ws(36,'data');v.setUint32(40,n*2,true);
  let o=44;for(let i=0;i<n;i++){let s=Math.max(-1,Math.min(1,mono[i]));v.setInt16(o,s<0?s*0x8000:s*0x7fff,true);o+=2;}
  return new Blob([buf],{type:'audio/wav'});
}

async function createClone(){
  if(!lastBlob){$('forgeStat').innerHTML='<span class=err>record a sample first</span>';return;}
  $('forgeBtn').disabled=true; $('forgeStat').textContent='encoding…';
  let wav; try{wav=await toWav(lastBlob);}catch(e){$('forgeStat').innerHTML='<span class=err>decode failed: '+e.message+'</span>';$('forgeBtn').disabled=false;return;}
  const fd=new FormData();
  fd.append('clip',wav,'mic.wav');
  fd.append('name',$('vname').value||'My Voice');
  fd.append('engine',$('engine').value);
  fd.append('exaggeration',$('exag').value);
  $('forgeStat').textContent='forging (loads the model — first time is slow)…';
  let r; try{r=await fetch('/api/v1/voices/forge',{method:'POST',headers:authH(),body:fd});}
  catch(e){$('forgeStat').innerHTML='<span class=err>'+e.message+'</span>';$('forgeBtn').disabled=false;return;}
  $('forgeBtn').disabled=false;
  if(!r.ok){$('forgeStat').innerHTML='<span class=err>error '+r.status+': '+await r.text()+'</span>';return;}
  const j=await r.json();
  $('forgeStat').innerHTML='<span class=ok>✓ forged '+j.name+'</span>';
  await loadVoices(); selectVoice(j.slug,j.name,j.engine);
}

async function loadVoices(){
  let r; try{r=await fetch('/api/v1/voices');}catch(e){$('voices').textContent='(backend unreachable)';return;}
  const j=await r.json();
  if(!j.voices.length){$('voices').textContent='No voices yet — record one above.';return;}
  $('voices').innerHTML='';
  for(const v of j.voices){
    const row=document.createElement('div'); row.className='vitem';
    row.innerHTML='<span>🔊 <b>'+v.name+'</b> <span class=hint>'+v.engine+'</span></span>';
    const b=document.createElement('button'); b.className='ghost'; b.textContent='Use';
    b.onclick=()=>selectVoice(v.slug,v.name,v.engine);
    row.appendChild(b); $('voices').appendChild(row);
  }
}

function selectVoice(slug,name,engine){
  activeSlug=slug; activeEngine=engine||'chatterbox';
  $('activeName').textContent=name;
  const turbo=activeEngine==='chatterbox-turbo';
  $('sxag').disabled=turbo; $('scfg').disabled=turbo; $('turboNote').hidden=!turbo;
  $('sayBox').hidden=false; $('sayStat').textContent=''; $('au').hidden=true;
  $('sayBox').scrollIntoView({behavior:'smooth',block:'nearest'});
}

async function speak(){
  if(!activeSlug)return;
  $('sayStat').textContent='synthesizing…';
  const body={text:$('sayText').value, temperature:+$('stmp').value};
  if(activeEngine!=='chatterbox-turbo'){body.exaggeration=+$('sxag').value; body.cfg_weight=+$('scfg').value;}
  const t=performance.now();
  let r; try{r=await fetch('/api/v1/voices/'+activeSlug+'/say',
    {method:'POST',headers:authH({'Content-Type':'application/json'}),
     body:JSON.stringify(body)});}
  catch(e){$('sayStat').innerHTML='<span class=err>'+e.message+'</span>';return;}
  if(!r.ok){$('sayStat').innerHTML='<span class=err>error '+r.status+': '+await r.text()+'</span>';return;}
  const ss=r.headers.get('X-Synth-Seconds'), rt=((performance.now()-t)/1000).toFixed(2);
  $('au').src=URL.createObjectURL(await r.blob()); $('au').hidden=false; $('au').play();
  $('sayStat').textContent='synth '+(ss?(+ss).toFixed(2):'?')+'s · round-trip '+rt+'s';
}

loadVoices(); checkTurbo();
</script></body></html>"""


def create_app(library=None) -> FastAPI:  # noqa: ANN001
    app = FastAPI(title="VoiceForge Backend", version="0.1.0")
    auth = make_auth(_auth_token())
    app.add_middleware(
        CORSMiddleware, allow_origins=_cors_origins(), allow_methods=["*"], allow_headers=["*"]
    )
    lib = library if library is not None else get_library()

    @app.get("/", response_class=HTMLResponse)
    def studio() -> str:
        """Browser recording studio: capture mic → forge a clone → test it, all on-device."""
        return _STUDIO_HTML

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

    def _voice_settings(slug: str, body: SayRequest):  # noqa: ANN202
        v = lib.get(slug)
        if v is None:
            raise HTTPException(status_code=404, detail="voice not found")
        settings = v.settings()
        for key in ("temperature", "exaggeration", "cfg_weight"):
            val = getattr(body, key, None)
            if val is not None:
                settings[key] = val
        return v, settings

    @app.post("/api/v1/voices/{slug}/say", dependencies=[Depends(auth)])
    def say(slug: str, body: SayRequest) -> StreamingResponse:
        if not body.text.strip():
            raise HTTPException(status_code=422, detail="text is empty")
        v, settings = _voice_settings(slug, body)
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
        v, settings = _voice_settings(slug, body)
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
