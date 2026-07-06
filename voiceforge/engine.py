"""Chatterbox model loaders (base + turbo).

Thin wrappers that build a ``ChatterboxTTS`` / ``ChatterboxTurboTTS`` on the best device.
Loading is heavy (torch + ~1.5 GB weights from HuggingFace on first use) and is always
driven through :func:`voiceforge._model_manager.heavy_session`, never directly.
"""

from __future__ import annotations

import logging

from ._chatter_common import pick_device

logger = logging.getLogger(__name__)

#: Supported engine variants.
ENGINES = ("chatterbox", "chatterbox-turbo")

#: Test-injection hook: ``{engine_name: fake_model}``. Production leaves this empty —
#: the model manager owns real models so they can be evicted to free memory.
_model_overrides: dict = {}


def _build(engine: str):  # noqa: ANN201
    """Construct the model for *engine*, or return None if chatterbox isn't installed."""
    if engine in _model_overrides:
        return _model_overrides[engine]
    device = pick_device()
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
