"""Real-model (``real_model=True``) health transitions and /parse availability.

No GPU/torch needed: ``main._load_pipeline`` imports ``omni_server.inference``
lazily, so we inject a fake pipeline into ``sys.modules`` and drive the same
background-loader code path the real model uses. This covers the loading -> ready
/ error states the skeleton-mode suite never reaches.
"""

from __future__ import annotations

import sys
import time
import types
from typing import Any

import pytest
from fastapi.testclient import TestClient

from omni_server.config import Settings
from omni_server.main import create_app


class _ReadyPipeline:
    """A pipeline that loads cleanly and parses to an empty result."""

    def __init__(self, settings: Settings) -> None:
        pass

    def parse(self, image: Any) -> tuple[tuple[Any, ...], None]:
        return (), None


class _BoomPipeline:
    """A pipeline whose load fails, exercising the error path."""

    def __init__(self, settings: Settings) -> None:
        raise RuntimeError("boom")

    def parse(self, image: Any) -> tuple[tuple[Any, ...], None]:  # pragma: no cover
        raise AssertionError("unreachable")


def _inject_pipeline(monkeypatch: pytest.MonkeyPatch, pipeline_cls: type) -> None:
    module = types.ModuleType("omni_server.inference")
    module.OmniParserPipeline = pipeline_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "omni_server.inference", module)


def _wait_until_loaded(client: TestClient, timeout: float = 5.0) -> str:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        status = str(client.get("/health").json()["status"])
        if status != "loading":
            return status
        time.sleep(0.02)
    raise AssertionError("pipeline did not leave the loading state in time")


def test_health_reports_loading_before_pipeline_ready() -> None:
    # No `with` -> lifespan/loader never runs, so state stays "loading".
    client = TestClient(create_app(Settings(real_model=True, warmup=False)))
    body = client.get("/health").json()
    assert body == {"status": "loading", "phase": "2-inference", "detail": None}


def test_parse_returns_503_while_loading(png_b64: str) -> None:
    client = TestClient(create_app(Settings(real_model=True, warmup=False)))
    resp = client.post("/parse", json={"image_b64": png_b64})
    assert resp.status_code == 503
    assert "loading" in resp.json()["detail"]


def test_health_reports_ok_once_pipeline_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    _inject_pipeline(monkeypatch, _ReadyPipeline)
    with TestClient(create_app(Settings(real_model=True, warmup=True))) as client:
        assert _wait_until_loaded(client) == "ok"
        body = client.get("/health").json()
    assert body == {"status": "ok", "phase": "2-inference", "detail": None}


def test_health_reports_error_on_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _inject_pipeline(monkeypatch, _BoomPipeline)
    with TestClient(create_app(Settings(real_model=True, warmup=False))) as client:
        assert _wait_until_loaded(client) == "error"
        body = client.get("/health").json()
    assert body["status"] == "error"
    assert body["phase"] == "2-inference"
    # The open /health endpoint must NOT echo the raw exception text.
    assert body["detail"] == "pipeline failed to load; see server logs"


def test_parse_returns_503_when_pipeline_failed(
    monkeypatch: pytest.MonkeyPatch, png_b64: str
) -> None:
    _inject_pipeline(monkeypatch, _BoomPipeline)
    with TestClient(create_app(Settings(real_model=True, warmup=False))) as client:
        assert _wait_until_loaded(client) == "error"
        resp = client.post("/parse", json={"image_b64": png_b64})
    assert resp.status_code == 503
    # The auth-gated /parse path keeps the verbose detail for operators.
    assert "failed to load" in resp.json()["detail"]
    assert "boom" in resp.json()["detail"]
