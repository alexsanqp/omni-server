"""/parse wire contract in skeleton mode (canned element, no GPU)."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_parse_returns_canned_element(client: TestClient, png_b64: str) -> None:
    resp = client.post("/parse", json={"image_b64": png_b64, "image_format": "png"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["parse_time_ms"] >= 0
    assert body["som_image_b64"] is None
    assert len(body["elements"]) == 1
    el = body["elements"][0]
    assert el["label"] == "placeholder-button"
    assert el["bbox"] == [20.0, 30.0, 220.0, 80.0]
    assert 0.0 <= el["confidence"] <= 1.0
    # Optional phase-2 fields carry their documented defaults.
    assert el["interactivity"] is False
    assert el["element_id"] == -1


def test_parse_defaults_image_format(client: TestClient, png_b64: str) -> None:
    # image_format is advisory and optional.
    resp = client.post("/parse", json={"image_b64": png_b64})
    assert resp.status_code == 200


def test_parse_rejects_unknown_image_format(client: TestClient, png_b64: str) -> None:
    resp = client.post("/parse", json={"image_b64": png_b64, "image_format": "gif"})
    assert resp.status_code == 422  # pydantic pattern violation
