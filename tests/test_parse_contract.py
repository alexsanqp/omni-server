"""/parse wire contract, exercised against an injected ready fake pipeline.

The fake (conftest.FakePipeline) returns one known Element so these assertions
pin the Element/ParseResponse serialization a downstream client mirrors.
"""

from __future__ import annotations

import base64
import io

from fastapi.testclient import TestClient
from PIL import Image


def _jpeg_b64(width: int = 64, height: int = 48, color: str = "white") -> str:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_parse_returns_elements(client: TestClient, png_b64: str) -> None:
    resp = client.post("/parse", json={"image_b64": png_b64, "image_format": "png"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["parse_time_ms"] >= 0
    assert body["som_image_b64"] is None
    assert len(body["elements"]) == 1
    el = body["elements"][0]
    assert el["label"] == "button"
    assert el["bbox"] == [20.0, 30.0, 220.0, 80.0]  # tuple -> JSON array
    assert 0.0 <= el["confidence"] <= 1.0
    assert el["tags"] == ["button", "fake"]
    assert el["interactivity"] is True
    assert el["element_id"] == 0


def test_parse_defaults_image_format(client: TestClient, png_b64: str) -> None:
    # image_format is advisory and optional.
    resp = client.post("/parse", json={"image_b64": png_b64})
    assert resp.status_code == 200


def test_parse_rejects_unknown_image_format(client: TestClient, png_b64: str) -> None:
    resp = client.post("/parse", json={"image_b64": png_b64, "image_format": "gif"})
    assert resp.status_code == 422  # pydantic pattern violation


def test_parse_ignores_declared_format_jpeg_as_png(client: TestClient) -> None:
    # image_format is advisory: the server sniffs the real format from the bytes.
    # Real JPEG bytes declared as png must still succeed.
    resp = client.post("/parse", json={"image_b64": _jpeg_b64(), "image_format": "png"})
    assert resp.status_code == 200


def test_parse_ignores_declared_format_png_as_jpeg(client: TestClient, png_b64: str) -> None:
    # Real PNG bytes declared as jpeg must still succeed (declared value ignored).
    resp = client.post("/parse", json={"image_b64": png_b64, "image_format": "jpeg"})
    assert resp.status_code == 200
