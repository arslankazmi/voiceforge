"""Chatterbox model loaders (base + turbo).

Thin wrappers that build a ``ChatterboxTTS`` / ``ChatterboxTurboTTS`` on the best device.
Loading is heavy (torch + ~1.5 GB weights from HuggingFace on first use) and is always
driven through :func:`voiceforge._model_manager.heavy_session`, never directly.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

#: Supported engine variants.
ENGINES = ("chatterbox", "chatterbox-turbo")

#: Test-injection hook: ``{engine_name: fake_model}``. Production leaves this empty —
#: the model manager owns real models so they can be evicted to free memory.
_model_overrides: dict = {}


def _resolve_device() -> str:
    """Pick the inference device: ``$VOICEFORGE_DEVICE`` → CUDA → CPU.

    Apple **MPS is intentionally skipped by default**: Chatterbox's conditioning path
    moves a float64 tensor to the device, and MPS rejects float64
    (``Cannot convert a MPS Tensor to float64``). CPU is correct and safe on Mac; set
    ``VOICEFORGE_DEVICE=mps`` to opt in if your setup handles it, or ``=cuda``/``=cpu``.
    """
    override = os.environ.get("VOICEFORGE_DEVICE")
    if override:
        return override
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _build(engine: str):  # noqa: ANN201
    """Construct the model for *engine*, or return None if chatterbox isn't installed."""
    if engine in _model_overrides:
        return _model_overrides[engine]
    device = _resolve_device()
    try:
        if engine == "chatterbox-turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: PLC0415

            logger.info("ChatterboxTurbo loading on device=%s", device)
            return ChatterboxTurboTTS.from_pretrained(device)
        from chatterbox.tts import ChatterboxTTS  # noqa: PLC0415

        logger.info("Chatterbox loading on device=%s", device)
        return ChatterboxTTS.from_pretrained(device=device)
    except ImportError:
        logger.warning(
            "chatterbox-tts not installed. Install with: pip install 'voiceforge[clone]'"
        )
        return None
    except Exception:
        logger.warning("Failed to load %s model", engine, exc_info=True)
        return None


def loader_for(engine: str):  # noqa: ANN201
    """Return a zero-arg loader for ``heavy_session(engine, loader_for(engine))``."""

    def _loader():  # noqa: ANN202
        return _build(engine)

    return _loader
