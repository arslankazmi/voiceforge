# VoiceForge production voice backend (Tier B) — headless, multi-voice.
# Lean base; torch + chatterbox come from the [clone] extra (heavy). Weights download
# to the mounted HF cache on first use. Runs as a non-root user.
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_CACHE_DIR=/tmp/pip \
    HF_HOME=/home/app/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY pyproject.toml README.md ./
COPY voiceforge ./voiceforge
RUN pip install ".[clone]"

USER app
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8090/healthz').status==200 else 1)"

CMD ["uvicorn", "voiceforge.voice_backend:app", "--host", "0.0.0.0", "--port", "8090"]
