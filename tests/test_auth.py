"""Bearer-token auth on /parse (enabled only when OMNI_AUTH_TOKEN is set)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from omni_server.config import Settings
from omni_server.main import create_app

TOKEN = "s3cr3t-token"  # noqa: S105 - test fixture, not a real secret


@pytest.fixture
def auth_client() -> TestClient:
    settings = Settings(real_model=False, warmup=False, auth_token=TOKEN)
    return TestClient(create_app(settings))


def test_parse_requires_token_when_configured(auth_client: TestClient, png_b64: str) -> None:
    resp = auth_client.post("/parse", json={"image_b64": png_b64})
    assert resp.status_code == 401


def test_parse_accepts_valid_token(auth_client: TestClient, png_b64: str) -> None:
    resp = auth_client.post(
        "/parse",
        json={"image_b64": png_b64},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert resp.status_code == 200


def test_parse_rejects_wrong_token(auth_client: TestClient, png_b64: str) -> None:
    resp = auth_client.post(
        "/parse",
        json={"image_b64": png_b64},
        headers={"Authorization": "Bearer nope"},
    )
    assert resp.status_code == 401


def test_health_open_without_token(auth_client: TestClient) -> None:
    # Probes must not need the token.
    assert auth_client.get("/health").status_code == 200


def test_parse_open_when_token_unset(client: TestClient, png_b64: str) -> None:
    # Default fixture has auth_token=None → no auth required.
    assert client.post("/parse", json={"image_b64": png_b64}).status_code == 200


@pytest.mark.parametrize("env_name", ["OMNI_AUTH_TOKEN", "OMNIPARSER_AUTH_TOKEN"])
def test_auth_token_env_aliases(
    monkeypatch: pytest.MonkeyPatch, png_b64: str, env_name: str
) -> None:
    # Both the canonical name and the OMNIPARSER_* alias must enable auth, so a
    # value shared with the YouTube client lines up either way.
    monkeypatch.setenv("OMNI_REAL_MODEL", "0")
    monkeypatch.setenv(env_name, TOKEN)
    c = TestClient(create_app(Settings()))
    assert c.post("/parse", json={"image_b64": png_b64}).status_code == 401
    ok = c.post("/parse", json={"image_b64": png_b64}, headers={"Authorization": f"Bearer {TOKEN}"})
    assert ok.status_code == 200
