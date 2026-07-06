"""VoiceForge — portable ``.voice`` voice artifacts (forge → play → serve).

Import-time environment hardening (set before any torch/HF import):
  * HuggingFace telemetry off.
  * MPS high-watermark cap + CPU fallback, so the Metal allocator can't grab all of
    unified memory and OOM the host.
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
_os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
_os.environ.setdefault("DISABLE_TELEMETRY", "1")
# Cap the MPS allocator + allow CPU fallback (OOM safety on 16 GB unified memory).
_os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.1.0"
