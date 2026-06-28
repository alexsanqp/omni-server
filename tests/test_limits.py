"""Request guards: size, base64 validity, image validity, megapixels."""

from __future__ import annotations

import base64
from typing import Any

import pytest
from fastapi.testclient import TestClient

from omni_server.config import Settings


def test_invalid_base64_is_400(client: TestClient) -> None:
    resp = client.post("/parse", json={"image_b64": "!!!not base64!!!"})
    assert resp.status_code == 400


def test_non_image_bytes_is_400(client: TestClient) -> None:
    not_an_image = base64.b64encode(b"hello world, definitely not an image").decode()
    resp = client.post("/parse", json={"image_b64": not_an_image})
    assert resp.status_code == 400


def test_oversized_body_is_413(ready_client: Any) -> None:
    c = ready_client(Settings(warmup=False, max_image_bytes=1024))
    big = base64.b64encode(b"x" * 4096).decode()
    resp = c.post("/parse", json={"image_b64": big})
    assert resp.status_code == 413


def test_too_many_megapixels_is_413(ready_client: Any, make_png_b64: Any) -> None:
    c = ready_client(Settings(warmup=False, max_image_megapixels=0.001))
    # 64x48 = 3072 px = 0.003 MP > 0.001 MP limit.
    resp = c.post("/parse", json={"image_b64": make_png_b64(64, 48)})
    assert resp.status_code == 413


def test_decompression_bomb_rejected_from_header(ready_client: Any, make_png_b64: Any) -> None:
    # A small-file / huge-canvas image (the classic decompression bomb) must be
    # rejected from its header — before image.load() decodes the full raster into
    # memory. 6000x6000 = 36 MP exceeds the default 20 MP cap.
    c = ready_client(Settings(warmup=False))  # default 20.0 MP cap
    resp = c.post("/parse", json={"image_b64": make_png_b64(6000, 6000)})
    assert resp.status_code == 413


@pytest.mark.parametrize("missing", [{}, {"image_format": "png"}])
def test_missing_image_b64_is_422(client: TestClient, missing: dict[str, str]) -> None:
    resp = client.post("/parse", json=missing)
    assert resp.status_code == 422
