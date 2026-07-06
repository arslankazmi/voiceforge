"""Shared Chatterbox helpers (vendored, self-contained).

- ``pick_device``                 — MPS→CPU device selection
- ``tensor_to_wav``               — torch tensor → WAV bytes
- ``exaggeration_to_temperature`` — knob (0–1) → sampling temperature
- ``supported_generate_kwargs``   — safely filter kwargs for a generate fn
"""

from __future__ import annotations

import inspect
import logging

logger = logging.getLogger(__name__)


def pick_device() -> str:
    """Return 'mps' if Apple Silicon MPS is available, else 'cpu'."""
    try:
        import torch  # noqa: PLC0415

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def tensor_to_wav(audio_tensor, sample_rate: int) -> bytes | None:  # noqa: ANN001
    """Convert a torch tensor to 16-bit mono PCM WAV bytes, or None on failure."""
    import io  # noqa: PLC0415
    import struct  # noqa: PLC0415

    try:
        arr = audio_tensor.squeeze().detach().cpu().numpy()

        import numpy as np  # noqa: PLC0415

        arr = (arr * 32767).clip(-32768, 32767).astype(np.int16)
        pcm = arr.tobytes()

        n_channels = 1
        sample_width = 2
        byte_rate = sample_rate * n_channels * sample_width
        block_align = n_channels * sample_width
        data_size = len(pcm)

        buf = io.BytesIO()
        buf.write(b"RIFF")
        buf.write(struct.pack("<I", 36 + data_size))
        buf.write(b"WAVE")
        buf.write(b"fmt ")
        buf.write(struct.pack("<I", 16))
        buf.write(struct.pack("<H", 1))
        buf.write(struct.pack("<H", n_channels))
        buf.write(struct.pack("<I", sample_rate))
        buf.write(struct.pack("<I", byte_rate))
        buf.write(struct.pack("<H", block_align))
        buf.write(struct.pack("<H", sample_width * 8))
        buf.write(b"data")
        buf.write(struct.pack("<I", data_size))
        buf.write(pcm)
        return buf.getvalue()
    except Exception:
        logger.warning("tensor_to_wav failed", exc_info=True)
        return None


def exaggeration_to_temperature(exag: float) -> float:
    """Map the exaggeration knob (0–1) to sampling temperature ``clamp(0.5+exag, 0.3, 1.5)``.

    0.0→0.5, 0.5→1.0, 1.0→1.5. Bad input → 0.5.
    """
    try:
        val = float(exag)
    except (TypeError, ValueError):
        return 0.5
    return max(0.3, min(1.5, 0.5 + val))


def supported_generate_kwargs(generate_fn, **kwargs) -> dict:  # noqa: ANN001
    """Return only the kwargs *generate_fn* accepts (pass-through if it takes **kwargs)."""
    try:
        sig = inspect.signature(generate_fn)
        params = sig.parameters
        for p in params.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return dict(kwargs)
        return {k: v for k, v in kwargs.items() if k in params}
    except (ValueError, TypeError):
        return dict(kwargs)
