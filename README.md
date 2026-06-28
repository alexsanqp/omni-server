# omni-server

Standalone HTTP service exposing **Microsoft OmniParser v2** (YOLOv8 icon
detector + Florence-2 captioner + PaddleOCR/EasyOCR) over a small, stable wire
contract. Give it a screenshot, get back a structured list of UI elements (with
pixel bounding boxes, interactivity, per-element confidence) plus a Set-of-Marks
annotated image.

It is intentionally a **separate project** so the 3+ GB CUDA stack lives only on
a GPU machine, while the client (e.g. the YouTube automation backend) runs
elsewhere and talks to it over the network via `OMNIPARSER_URL`.

```
client machine                         GPU machine
┌────────────────────┐   HTTP /parse   ┌──────────────────────────┐
│ OmniparserClient   │ ──────────────► │ omni-server (FastAPI)    │
│ (OMNIPARSER_URL)   │ ◄────────────── │ YOLOv8 + Florence-2 + OCR│
└────────────────────┘   elements      └──────────────────────────┘
```

## Wire contract

`POST /parse`
```jsonc
// request
{ "image_b64": "<base64 jpeg|png>", "image_format": "jpeg" }
// response
{
  "elements": [
    { "label": "Subscribe", "bbox": [100,50,300,90], "confidence": 0.94,
      "tags": ["icon","box_yolo_content_yolo"], "interactivity": true, "element_id": 42 }
  ],
  "parse_time_ms": 3700,
  "som_image_b64": "<base64 png with numbered boxes>"
}
```
`bbox` is `(left, top, right, bottom)` in **original-image pixels**.

`GET /health` (and `/healthz`) → `{ "status": "ok|loading|error", "phase": "...", "detail": null }`.
During the multi-minute model load the port binds immediately and `status` is
`loading`, so a client can tell "warming" from "down".

Auth: if `OMNI_AUTH_TOKEN` is set, `/parse` requires `Authorization: Bearer <token>`
(`/health` stays open). Requests are bounded by `OMNI_MAX_IMAGE_BYTES` /
`OMNI_MAX_IMAGE_MP`.

## Quick start (no GPU): run the checks

Lint, type-check, and the wire-contract tests need no torch/weights — the suite
injects a fake pipeline, so there is no canned "skeleton" mode baked into the
service itself.

```bash
uv sync --extra dev
uv run ruff check . && uv run mypy omni_server && uv run pytest
```

Actually serving real parses needs a GPU box — see **Full deploy** below.

## Full deploy (GPU box)

### A. Docker (recommended)

```bash
docker compose up -d --build      # needs the NVIDIA Container Toolkit
docker compose logs -f            # weights (~1 GB) download on first start
```

### B. Bare metal (uv)

```bash
# 1. Deps (needs the cu124 torch index reachable):
uv sync --extra gpu --extra dev
# 2. Vendored OmniParser source (clone @ pinned SHA + apply patch):
uv run python scripts/setup_vendor.py
# 3. Model weights (~1 GB from HuggingFace):
uv run python scripts/download_weights.py
# 4. Run (real inference):
uv run omni-server
# /health reports "loading" until the model is up, then "ok".
# 5. Smoke-test it (locally, or against a remote box with OMNI_URL / OMNI_AUTH_TOKEN):
uv run python scripts/smoke_client.py shot.png
```

First cold start fetches the Florence-2 *processor* from HuggingFace
(`trust_remote_code=True`) and ~100 MB of OCR models — set `HF_HOME` to a
persistent dir to cache them across restarts. Subsequent starts are offline.

## Configuration

All settings are env-driven (`omni_server/config.py`). Copy `.env.example` to
`.env` and edit it — the server auto-loads `.env` from its working directory, so
a plain `uv run omni-server` picks up your config with no extra flags. The most
important keys:

| Env | Default | Meaning |
|---|---|---|
| `OMNI_HOST` / `OMNI_PORT` | `0.0.0.0` / `8001` | bind address |
| `OMNI_AUTH_TOKEN` | _(unset)_ | bearer token for `/parse` |
| `OMNI_WARMUP` | `1` | warmup parse at startup |
| `OMNI_MAX_IMAGE_BYTES` | `24 MiB` | request body cap |
| `OMNI_MAX_IMAGE_MP` | `20.0` | max resolution |
| `OMNI_BATCH_SIZE` | `64` | Florence-2 caption batch (lower it on an 8 GB GPU) |
| `OMNI_VENDOR_REF` | `b0d5c9f` | pinned upstream OmniParser SHA |

`OMNIPARSER_WARMUP` / `OMNIPARSER_AUTH_TOKEN` are still accepted as aliases.

## Networking (two machines)

Put the GPU box and the client on the same LAN or a private VPN (Tailscale /
WireGuard). On the GPU box, open inbound TCP `8001` (or bind `0.0.0.0` on a
trusted network) and set `OMNI_AUTH_TOKEN` if it is reachable beyond the LAN.
On the client, set `OMNIPARSER_URL=http://<gpu-host>:8001` (and the matching
token). Note: real parses take seconds — the client's timeout must allow for it
(the reference client defaults to 120 s).

## Performance notes

- **Biggest lever:** `OMNI_USE_LOCAL_SEMANTICS=0` skips Florence-2 captioning
  (the dominant per-frame cost) and doesn't load the caption model at all — a
  much faster, lower-VRAM parse that still returns OCR text + interactive boxes,
  just with generic icon labels instead of rich captions.
- Inference is serialised by an in-process lock (one GPU = one parse at a time).
  Reported `parse_time_ms` is server-side compute.
- Real parse latency depends heavily on screenshot size and element count;
  shrink with `OMNI_IMGSZ_MAX` and the OCR/box thresholds if needed.
- A warmup parse at startup (`OMNI_WARMUP=1`) pays lazy OCR/model init up front
  so the first real request isn't penalised.

## Layout

```
omni_server/      package: config, schemas (wire), inference, main (FastAPI)
scripts/          download_weights, setup_vendor, smoke_client
patches/          UTF-8 patch + README for the vendored OmniParser
tests/            wire-contract tests (fake pipeline injected; run in CI, no GPU)
Dockerfile        GPU image     docker-compose.yml  GPU deploy
```

## Licensing

This repo's code is MIT (`LICENSE`). The vendored OmniParser source and the
downloaded weights carry their **own** licenses and are fetched at setup time,
not redistributed here — read `NOTICE` before any commercial/closed deployment
(the icon detector may be AGPL).
