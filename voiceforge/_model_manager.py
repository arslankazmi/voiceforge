"""Memory-guarded single-residency manager for heavy (torch) TTS models (vendored).

On shared-memory Macs (e.g. 16 GB unified), loading several multi-GB torch models at
once can exhaust unified memory and **crash the whole host**. This makes that impossible:

  * **Single residency** — at most ONE heavy model loaded; a different one evicts the
    current (drop ref + ``gc.collect()`` + ``torch.mps.empty_cache()``).
  * **Serialized heavy work** — a global lock wraps the whole load+synth session.
  * **System-RAM preflight** — refuse (yield ``None``) below a free-memory floor
    (``VOICEFORGE_MIN_FREE_MB``, default 3000) rather than triggering an OS OOM.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import os
import threading
from collections.abc import Callable, Iterator
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MIN_FREE_MB = 3000.0

_LOCK = threading.RLock()
_current_name: str | None = None
_current_model: Any = None


def _free_mb() -> float | None:
    try:
        import psutil  # noqa: PLC0415

        return psutil.virtual_memory().available / (1024 * 1024)
    except Exception:
        return None


def _empty_mps_cache() -> None:
    try:
        import torch  # noqa: PLC0415

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _min_free_mb() -> float:
    try:
        return float(os.environ.get("VOICEFORGE_MIN_FREE_MB", _DEFAULT_MIN_FREE_MB))
    except (TypeError, ValueError):
        return _DEFAULT_MIN_FREE_MB


def evict() -> None:
    """Drop the resident heavy model and reclaim memory. Caller must hold ``_LOCK``."""
    global _current_name, _current_model
    if _current_model is not None:
        logger.info("Evicting heavy model %r to free memory", _current_name)
    _current_model = None
    _current_name = None
    gc.collect()
    _empty_mps_cache()


def resident_model_name() -> str | None:
    """Name of the currently-resident heavy model (tests / introspection)."""
    return _current_name


@contextlib.contextmanager
def heavy_session(name: str, loader: Callable[[], Any]) -> Iterator[Any]:
    """Yield a heavy model under single-residency + memory guards (or ``None`` if unsafe)."""
    global _current_name, _current_model
    with _LOCK:
        if _current_name == name and _current_model is not None:
            yield _current_model
            return

        if _current_model is not None and _current_name != name:
            evict()  # free the other model before allocating the next

        free = _free_mb()
        floor = _min_free_mb()
        if free is not None and free < floor:
            logger.warning(
                "Refusing to load heavy model %r: only %.0f MB free (< %.0f MB floor).",
                name,
                free,
                floor,
            )
            yield None
            return

        try:
            _current_model = loader()
            _current_name = name
        except Exception:
            logger.warning("Heavy model %r failed to load", name, exc_info=True)
            evict()
            yield None
            return

        yield _current_model
