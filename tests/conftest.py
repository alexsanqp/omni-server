"""Shared fixtures. All tests run in skeleton mode — no torch/GPU required."""

from __future__ import annotations

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from omni_server.config import Settings
from omni_server.main import create_app


def _png_b64(width: int = 64, height: int = 48, color: str = "white") -> str:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture
def png_b64() -> str:
    return _png_b64()


@pytest.fixture
def make_png_b64():
    return _png_b64


@pytest.fixture
def skeleton_settings() -> Settings:
    return Settings(real_model=False, warmup=False, auth_token=None)


@pytest.fixture
def client(skeleton_settings: Settings) -> TestClient:
    return TestClient(create_app(skeleton_settings))
