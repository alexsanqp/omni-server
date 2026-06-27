"""Server settings, loaded from environment / ``.env``.

All knobs are env-driven so the same image runs on the GPU box, in CI
(skeleton mode), or on a CPU dev laptop without code changes. ``OMNI_*`` is the
canonical prefix; the two original ``OMNIPARSER_*`` names are still accepted for
back-compat with the in-repo service this was extracted from.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PACKAGE_ROOT = Path(__file__).resolve().parent
SERVICE_ROOT = _PACKAGE_ROOT.parent


class Settings(BaseSettings):
    """Process-wide configuration. Source of truth is the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # Accept both the field name (e.g. real_model=, used by tests) and the
        # env aliases (OMNI_REAL_MODEL / OMNIPARSER_REAL_MODEL).
        populate_by_name=True,
    )

    # --- Modes -------------------------------------------------------------
    # real_model=True loads the GPU pipeline (torch + YOLO + Florence-2). False
    # serves a canned skeleton response — used by CI and no-GPU smoke tests so
    # the wire contract can be exercised without 3+ GB of CUDA wheels.
    real_model: bool = Field(
        default=True,
        validation_alias=AliasChoices("OMNI_REAL_MODEL", "OMNIPARSER_REAL_MODEL"),
    )
    # Pay the lazy OCR/model init with one warmup parse at startup instead of on
    # the first real request (which would otherwise blow the client timeout).
    warmup: bool = Field(
        default=True,
        validation_alias=AliasChoices("OMNI_WARMUP", "OMNIPARSER_WARMUP"),
    )

    # --- Network / serving -------------------------------------------------
    host: str = Field(default="0.0.0.0", validation_alias=AliasChoices("OMNI_HOST"))  # noqa: S104
    port: int = Field(default=8001, ge=1, le=65535, validation_alias=AliasChoices("OMNI_PORT"))
    # When set, /parse requires `Authorization: Bearer <token>`. Unset = open
    # (fine on a trusted LAN; set a token when reachable beyond it). Accepts
    # OMNIPARSER_AUTH_TOKEN too, so a single shared value lines up with the
    # YouTube client's OMNIPARSER_AUTH_TOKEN.
    auth_token: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OMNI_AUTH_TOKEN", "OMNIPARSER_AUTH_TOKEN"),
    )

    @field_validator("auth_token", mode="before")
    @classmethod
    def _blank_token_is_open(cls, v: object) -> object:
        # A blank token from the environment (e.g. an empty OMNI_AUTH_TOKEN in a
        # docker-compose env_file) must mean "no auth", not a SecretStr("") that
        # no client can ever present — which would silently lock /parse with 401.
        if isinstance(v, str) and not v.strip():
            return None
        return v

    # --- Request guards ----------------------------------------------------
    # Reject oversized bodies before decoding (screen JPEGs are ~0.1-2 MB).
    max_image_bytes: int = Field(
        default=24 * 1024 * 1024, ge=1024, validation_alias=AliasChoices("OMNI_MAX_IMAGE_BYTES")
    )
    # Reject absurd resolutions that would OOM the GPU (a 4K frame is ~8.3 MP).
    max_image_megapixels: float = Field(
        default=20.0, gt=0, validation_alias=AliasChoices("OMNI_MAX_IMAGE_MP")
    )

    # --- Paths (defaults anchored at the service root; an env override is taken
    # verbatim, so a *relative* override resolves against the process CWD at
    # runtime — prefer an absolute path to avoid CWD ambiguity) -------------
    weights_dir: Path = Field(
        default=SERVICE_ROOT / "weights", validation_alias=AliasChoices("OMNI_WEIGHTS_DIR")
    )
    vendor_dir: Path = Field(
        default=SERVICE_ROOT / "vendor" / "OmniParser",
        validation_alias=AliasChoices("OMNI_VENDOR_DIR"),
    )

    # --- Vendor pinning (used by scripts/setup_vendor.py) ------------------
    vendor_repo: str = Field(
        default="https://github.com/microsoft/OmniParser",
        validation_alias=AliasChoices("OMNI_VENDOR_REPO"),
    )
    vendor_ref: str = Field(default="b0d5c9f", validation_alias=AliasChoices("OMNI_VENDOR_REF"))

    # --- Inference tuning (OmniParser/MSFT gradio-demo defaults) ------------
    box_threshold: float = Field(
        default=0.05, ge=0.0, le=1.0, validation_alias=AliasChoices("OMNI_BOX_THRESHOLD")
    )
    iou_threshold: float = Field(
        default=0.1, ge=0.0, le=1.0, validation_alias=AliasChoices("OMNI_IOU_THRESHOLD")
    )
    imgsz_max: int = Field(
        default=1280, ge=320, le=4096, validation_alias=AliasChoices("OMNI_IMGSZ_MAX")
    )
    caption_prompt: str = Field(
        default="<DETAILED_CAPTION>", validation_alias=AliasChoices("OMNI_CAPTION_PROMPT")
    )
    # Run the Florence-2 captioner for semantic icon labels. This is the single
    # biggest cost per frame; set False for a much faster (and lower-VRAM) parse
    # that returns OCR text + generic icon types but no rich captions. The
    # caption model is then never loaded.
    use_local_semantics: bool = Field(
        default=True, validation_alias=AliasChoices("OMNI_USE_LOCAL_SEMANTICS")
    )
    use_paddleocr: bool = Field(default=True, validation_alias=AliasChoices("OMNI_USE_PADDLEOCR"))
    ocr_text_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, validation_alias=AliasChoices("OMNI_OCR_TEXT_THRESHOLD")
    )
    batch_size: int = Field(
        default=64, ge=1, le=512, validation_alias=AliasChoices("OMNI_BATCH_SIZE")
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
