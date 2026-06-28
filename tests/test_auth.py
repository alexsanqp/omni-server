"""Bearer-token auth on /parse (enabled only when OMNI_AUTH_TOKEN is set)."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from omni_server.config import Settings
from omni_server.main import _require_auth

TOKEN = "s3cr3t-token"  # noqa: S105 - test fixture, not a real secret


@pytest.fixture
def auth_client(ready_client: Any) -> TestClient:
    return ready_client(Settings(warmup=False, auth_token=TOKEN))


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


@pytest.mark.parametrize(
    "header",
    [
        TOKEN,  # raw token, no "Bearer " prefix
        f"bearer {TOKEN}",  # wrong case
        "Bearer",  # no trailing space
        "Bearer ",  # empty token after the prefix
        f"Basic {TOKEN}",  # wrong scheme
        f"Bearer  {TOKEN}",  # extra space -> leading-space token mismatch
    ],
)
def test_parse_rejects_malformed_authorization(
    auth_client: TestClient, png_b64: str, header: str
) -> None:
    # Only an exact ``Bearer <token>`` is accepted; everything else is 401.
    resp = auth_client.post(
        "/parse", json={"image_b64": png_b64}, headers={"Authorization": header}
    )
    assert resp.status_code == 401


def test_require_auth_rejects_non_ascii_credential_as_401() -> None:
    # A non-ASCII byte in the credential must surface as 401, not a TypeError ->
    # 500 (hmac.compare_digest raises on non-ASCII *str* operands; the server
    # compares on bytes). Driven directly: httpx blocks non-ASCII headers client-side.
    settings = Settings(warmup=False, auth_token=TOKEN)
    with pytest.raises(HTTPException) as exc_info:
        _require_auth(settings, f"Bearer {TOKEN}\xe9")
    assert exc_info.value.status_code == 401


def test_health_open_without_token(auth_client: TestClient) -> None:
    # Probes must not need the token.
    assert auth_client.get("/health").status_code == 200


def test_parse_open_when_token_unset(client: TestClient, png_b64: str) -> None:
    # Default fixture has auth_token=None → no auth required.
    assert client.post("/parse", json={"image_b64": png_b64}).status_code == 200


@pytest.mark.parametrize("env_name", ["OMNI_AUTH_TOKEN", "OMNIPARSER_AUTH_TOKEN"])
def test_auth_token_env_aliases(
    monkeypatch: pytest.MonkeyPatch, ready_client: Any, png_b64: str, env_name: str
) -> None:
    # Both the canonical name and the OMNIPARSER_* alias must enable auth, so a
    # value shared with the YouTube client lines up either way.
    monkeypatch.setenv(env_name, TOKEN)
    c = ready_client(Settings())
    assert c.post("/parse", json={"image_b64": png_b64}).status_code == 401
    ok = c.post("/parse", json={"image_b64": png_b64}, headers={"Authorization": f"Bearer {TOKEN}"})
    assert ok.status_code == 200


def test_blank_auth_token_env_leaves_parse_open(
    monkeypatch: pytest.MonkeyPatch, ready_client: Any, png_b64: str
) -> None:
    # An empty OMNI_AUTH_TOKEN (the .env.example default, and what a compose
    # env_file injects) must mean "no auth" — not a token of "" that 401s every
    # request. Drives the real env path, which Settings(auth_token=None) bypasses.
    monkeypatch.setenv("OMNI_AUTH_TOKEN", "")
    c = ready_client(Settings())
    assert c.post("/parse", json={"image_b64": png_b64}).status_code == 200
