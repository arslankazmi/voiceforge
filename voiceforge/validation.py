"""Audio upload validation (vendored, self-contained)."""

from __future__ import annotations

import os

MAX_TEXT_LENGTH: int = 5000


def validate_audio(path: str, max_mb: float = 15.0, max_sec: float = 120.0) -> None:
    """Validate audio file size and (if soundfile is available) duration.

    Raises:
        ValueError: if the file exceeds size or duration limits.
    """
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > max_mb:
        raise ValueError(f"File too large: {size_mb:.1f} MB > {max_mb} MB")

    try:
        import soundfile as sf  # noqa: PLC0415

        if sf.info(path).duration > max_sec:
            raise ValueError(f"Audio too long: {sf.info(path).duration:.1f}s > {max_sec}s")
    except ImportError:
        pass
