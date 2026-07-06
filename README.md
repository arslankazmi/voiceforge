# VoiceForge

**Drag a voice clip in → get a portable `.voice` artifact → play it or serve it. All on-device, private, no keys.**

VoiceForge turns a few seconds of reference audio into a reusable **voice artifact** you own as a file. Built on [Chatterbox](https://github.com/resemble-ai/chatterbox) (base + Turbo, MIT) — zero-shot cloning, so there's no training step and no per-voice model to manage.

> **Private, local, personal use.** VoiceForge does not publish or share anything. A cloned voice is biometric-adjacent data and, for real people, is protected by publicity/copyright law — only forge voices you own or have permission to use, and keep them on your machine.

## What a `.voice` is

A Chatterbox clone is **zero-shot conditioning, not trained weights**. So the artifact is the model's *speaker conditioning* (the `Conditionals` tensors Chatterbox derives from your clip) plus metadata — a single `~few-hundred-KB` file that reloads and speaks **without the original clip**. Chatterbox's own default voice ships exactly this way.

Artifacts are **engine-locked**: a `base` voice won't load into `turbo` and vice-versa (validated on load).

## Install

```bash
uv sync --extra clone      # pulls torch + chatterbox-tts (~1.5 GB); weights download on first use
```

## Use

```bash
voiceforge forge myclip.wav -o robo.voice        # clip → portable artifact  (Turbo needs a >5s clip)
voiceforge say robo.voice "Anything I type."     # artifact + text → out.wav — no clip needed
voiceforge serve robo.voice                       # Tier A: this voice's own server + browser admin at :8080
voiceforge backend                                # Tier B: headless multi-voice backend at :8090 (/docs, /metrics)
voiceforge export-bundle robo.voice               # a runnable folder that carries the voice + its launcher
```

- `--engine turbo` (default, fast, one-step, needs a >5s clip, temperature-only) or `--engine base` (honors `--exaggeration`).

## Two ways to serve

**Tier A — one voice, its own server.** `voiceforge serve robo.voice` launches a self-contained FastAPI app for that single voice with a browser admin page (temperature slider, text box, speak/stream). Binds `127.0.0.1` by default; `--token` requires a bearer token.

**Tier B — production backend.** `voiceforge backend` serves a whole library of voices behind other apps: `POST /api/v1/voices/{slug}/say`, upload/forge/export/delete, `AUTH_TOKEN` bearer auth, CORS allowlist, `/metrics`, `/healthz`, `/readyz`, Docker. See `docker-compose.yml`.

## Why it won't melt your Mac

**Voices are cheap; models are expensive.** A voice is a few hundred KB; the model is gigabytes. Both servers keep **one** model resident (single-residency manager with a system-RAM preflight) and hot-swap `model.conds` per voice — so dozens of voices cost one model's worth of RAM, and it refuses to load rather than OOM the host.

## Streaming

Chatterbox renders whole clips, so there's no token-level streaming. The `/stream` endpoints split text into sentences and emit each as it finishes (length-prefixed WAV frames), so playback can start before the whole thing renders.

## Dev

```bash
uv run pytest              # keyless — torch/chatterbox lazy-imported; real-model paths skip
uv run ruff check . && uv run ruff format --check . && uv run mypy voiceforge
```

## License & credits

Apache-2.0. Voice engine: [Chatterbox](https://github.com/resemble-ai/chatterbox) by Resemble AI (MIT).
