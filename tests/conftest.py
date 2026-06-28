"""Shared fixtures. Tests inject a ready fake pipeline — no torch/GPU required.

``create_app(settings, pipeline=FakePipeline())`` starts the app in the ``ready``
state with a deterministic stub, so the wire contract (auth, limits, /parse,
/health) is exercised without loading the real OmniParser pipeline and without
any test-only branch living in production code.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from omni_server.config import Settings
from omni_server.main import create_app
from omni_server.schemas import Element


def _png_b64(width: int = 64, height: int = 48, color: str = "white") -> str:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class FakePipeline:
    """A ready, deterministic pipeline double for wire-contract tests."""

    def parse(self, image: Any) -> tuple[tuple[Element, ...], str | None]:
        element = Element(
            label="button",
            bbox=(20.0, 30.0, 220.0, 80.0),
            confidence=0.5,
            tags=("button", "fake"),
            interactivity=True,
            element_id=0,
        )
        return (element,), None


def make_client(settings: Settings | None = None) -> TestClient:
    """A TestClient whose app starts already ``ready`` with a fake pipeline."""
    return TestClient(create_app(settings or Settings(warmup=False), pipeline=FakePipeline()))


@pytest.fixture
def png_b64() -> str:
    return _png_b64()


@pytest.fixture
def make_png_b64() -> Any:
    return _png_b64


@pytest.fixture
def client() -> TestClient:
    return make_client()


@pytest.fixture
def ready_client() -> Any:
    """Factory: ``ready_client(settings)`` -> ready TestClient with a fake pipeline."""
    return make_client
