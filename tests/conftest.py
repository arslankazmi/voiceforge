"""Shared fixtures — reset the single-residency model manager between tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def reset_model_manager(monkeypatch):
    """Evict any resident model and make the RAM preflight pass by default."""
    import voiceforge._model_manager as mm

    mm.evict()
    monkeypatch.setattr(mm, "_free_mb", lambda: 999_999.0)
    yield
    mm.evict()
