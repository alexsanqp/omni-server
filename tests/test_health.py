"""Health endpoint in skeleton mode."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["phase"] == "1-skeleton"
    assert body["detail"] is None


def test_healthz_alias(client: TestClient) -> None:
    assert client.get("/healthz").json()["status"] == "ok"
