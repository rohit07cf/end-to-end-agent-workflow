"""Test config — force mock provider so no network calls happen."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("MODEL_PROVIDER", "mock")


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "mock")
    # Reset cached settings between tests so env overrides take effect.
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
