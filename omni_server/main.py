"""FastAPI app exposing /parse + /health.

Modes (``OMNI_REAL_MODEL``):
  - ``1`` (default)  -> real inference via YOLOv8 + Florence-2 (omni_server.inference).
  - ``0``            -> skeleton (canned response); no GPU/torch needed. Used by
                        CI and no-GPU smoke tests to exercise the wire contract.

The real pipeline loads in a background thread so the port binds immediately and
``/health`` can report ``loading`` during the multi-minute model load (a probing
client can distinguish "warming" from "down"). A warmup parse pays the lazy
PaddleOCR/Florence init at startup instead of on the first real request
(``OMNI_WARMUP=0`` to skip).

Security: when ``OMNI_AUTH_TOKEN`` is set, ``/parse`` requires
``Authorization: Bearer <token>``. ``/health`` is always open for probes.
Request bodies are bounded by ``OMNI_MAX_IMAGE_BYTES`` and decoded images by
``OMNI_MAX_IMAGE_MP``.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hmac
import io
import logging
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from PIL import Image

from omni_server import __version__
from omni_server.config import Settings, get_settings
from omni_server.schemas import Element, HealthResponse, ParseRequest, ParseResponse

logger = logging.getLogger(__name__)

# Pillow raises DecompressionBombError above ~178 MP by default; we enforce our
# own (smaller) cap explicitly in _decode_image, but keep Pillow's guard too.
Image.MAX_IMAGE_PIXELS = None  # our OMNI_MAX_IMAGE_MP check is authoritative


def configure_logging() -> None:
    """Install the project's log format on the root logger unless one exists.

    Called at import time so logs are formatted identically whether the server is
    started via ``omni-server`` / ``python -m omni_server`` or by pointing
    uvicorn/gunicorn straight at ``omni_server.main:app`` — the latter never runs
    ``__main__`` and would otherwise drop the loader thread's INFO lines and
    mangle its load-failure traceback. No-op if a handler is already configured
    (e.g. an operator-supplied ``--log-config``).
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )


def _require_auth(settings: Settings, authorization: str | None) -> None:
    token = settings.auth_token
    if token is None:
        return
    expected = token.get_secret_value()
    prefix = "Bearer "
    presented = (
        authorization[len(prefix) :] if authorization and authorization.startswith(prefix) else ""
    )
    # Constant-time compare on UTF-8 bytes; reject empty/missing too. (Comparing
    # str operands would raise TypeError on a non-ASCII credential byte and 500
    # instead of cleanly returning 401; encoding to bytes never raises.)
    if not presented or not hmac.compare_digest(presented.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="missing or invalid bearer token")


def _decode_image(settings: Settings, image_b64: str) -> Image.Image:
    if len(image_b64) > settings.max_image_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"image_b64 too large ({len(image_b64)} > {settings.max_image_bytes} bytes)",
        )
    try:
        raw = base64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid base64: {exc}") from exc
    if len(raw) > settings.max_image_bytes:
        raise HTTPException(status_code=413, detail="decoded image exceeds size limit")
    # Parse the header first and enforce the megapixel cap *before* decoding the
    # raster: image.load() allocates width*height*channels bytes, so a small-file
    # / huge-canvas decompression bomb would OOM the box before any check that
    # ran after it. With Pillow's own guard disabled above, this ordering is what
    # makes OMNI_MAX_IMAGE_MP authoritative.
    try:
        image = Image.open(io.BytesIO(raw))
    except Exception as exc:  # unrecognized / unsupported format -> bad-image 400
        raise HTTPException(status_code=400, detail="invalid image: unrecognized format") from exc
    megapixels = (image.width * image.height) / 1_000_000
    if megapixels > settings.max_image_megapixels:
        raise HTTPException(
            status_code=413,
            detail=f"image {megapixels:.1f}MP exceeds limit {settings.max_image_megapixels}MP",
        )
    try:
        image.load()  # decode now so truncated / corrupt pixel data fails as 400
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid image: corrupt image data") from exc
    return image


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    use_real_model = settings.real_model
    state: dict[str, Any] = {
        "pipeline": None,
        "status": "loading" if use_real_model else "skeleton",
        "error": None,
    }

    def _load_pipeline() -> None:
        try:
            from omni_server.inference import OmniParserPipeline

            pipeline = OmniParserPipeline(settings)
            if settings.warmup:
                logger.info("Warmup parse (pays lazy OCR/model init up front)...")
                t0 = time.perf_counter()
                pipeline.parse(Image.new("RGB", (320, 200), "white"))
                logger.info("Warmup done in %.1fs", time.perf_counter() - t0)
            state["pipeline"] = pipeline
            state["status"] = "ready"
        except Exception as exc:  # any load failure is surfaced via /health detail
            state["status"] = "error"
            state["error"] = str(exc)
            logger.exception("pipeline load failed")

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if use_real_model:
            logger.info("OMNI_REAL_MODEL=1 -> loading pipeline in background thread")
            threading.Thread(target=_load_pipeline, name="pipeline-loader", daemon=True).start()
        yield

    app = FastAPI(
        title="omni-server",
        version=__version__,
        description="Standalone OmniParser v2 service.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def _limit_body_size(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Reject oversized requests by Content-Length *before* the body is buffered
        # into memory, so a giant POST can't OOM the box before the in-handler
        # image_b64 cap runs. (A client that lies about Content-Length is the front
        # proxy's job; this stops the common honest-but-huge case cheaply.)
        cl = request.headers.get("content-length")
        if cl is not None and cl.isdigit() and int(cl) > settings.max_image_bytes:
            return JSONResponse(status_code=413, content={"detail": "request body too large"})
        return await call_next(request)

    async def auth_dep(authorization: str | None = Header(default=None)) -> None:
        _require_auth(settings, authorization)

    @app.get("/health", response_model=HealthResponse)
    @app.get("/healthz", response_model=HealthResponse)
    async def health() -> HealthResponse:
        phase = "2-inference" if use_real_model else "1-skeleton"
        status = {"skeleton": "ok", "ready": "ok"}.get(str(state["status"]), str(state["status"]))
        return HealthResponse(
            status=status,
            phase=phase,
            # /health is unauthenticated; a raw load-exception string can carry
            # absolute paths / internals. Keep the full text in the logs (and on
            # the auth-gated /parse 503); show anonymous probes only a generic note.
            detail="pipeline failed to load; see server logs" if state["error"] else None,
        )

    @app.post("/parse", response_model=ParseResponse, dependencies=[Depends(auth_dep)])
    async def parse(req: ParseRequest) -> ParseResponse:
        if use_real_model and state["status"] == "loading":
            raise HTTPException(status_code=503, detail="model is loading, retry later")
        if use_real_model and state["status"] == "error":
            raise HTTPException(
                status_code=503, detail=f"pipeline failed to load: {state['error']}"
            )

        # Offload the CPU-bound base64 + raster decode off the event loop (like
        # inference below), so a large screenshot's decode can't stall /health or
        # other requests on the single worker. HTTPExceptions propagate unchanged.
        image = await asyncio.to_thread(_decode_image, settings, req.image_b64)

        t0 = time.perf_counter()
        som_b64: str | None = None
        pipeline = state["pipeline"]
        if pipeline is None:
            elements: tuple[Element, ...] = (
                Element(
                    label="placeholder-button",
                    bbox=(20.0, 30.0, 220.0, 80.0),
                    confidence=0.5,
                    tags=("button", "skeleton"),
                ),
            )
        else:
            try:
                elements, som_b64 = await asyncio.to_thread(pipeline.parse, image)
            except Exception as exc:  # any inference failure is mapped to 500
                logger.exception("pipeline.parse failed")
                raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc

        elapsed = int((time.perf_counter() - t0) * 1000)
        return ParseResponse(elements=elements, parse_time_ms=elapsed, som_image_b64=som_b64)

    return app


configure_logging()
app = create_app()
