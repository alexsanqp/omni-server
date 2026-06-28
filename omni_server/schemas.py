"""Wire schemas — the HTTP contract between omni-server and its clients.

These models are the *only* coupling point with any consumer. The reference
client lives in the YouTube project at
``backend/gui_agent/omniparser_client.py``; its ``Element`` / ``ParseResult``
mirror ``Element`` / ``ParseResponse`` below field-for-field. Keep them in sync:
add fields as optional with defaults so old clients keep deserialising.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ParseRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded image (jpeg or png).")
    # Advisory only: the server sniffs the real format from the decoded bytes
    # via Pillow. Kept for forward-compatibility and request logging.
    image_format: str = Field(default="jpeg", pattern=r"^(jpeg|png)$")


class Element(BaseModel):
    model_config = ConfigDict(frozen=True)
    label: str
    # (left, top, right, bottom) in ORIGINAL-image pixels.
    bbox: tuple[float, float, float, float]
    confidence: float = Field(..., ge=0.0, le=1.0)
    tags: tuple[str, ...] = ()
    interactivity: bool = False
    element_id: int = -1


class ParseResponse(BaseModel):
    elements: tuple[Element, ...]
    parse_time_ms: int = Field(..., ge=0)
    som_image_b64: str | None = Field(
        default=None,
        description="Base64 PNG of the screenshot with numbered SoM boxes overlaid. "
        "Designed for VLM consumption (Set-of-Marks prompting). The pipeline returns a "
        "PNG even when nothing is detected (the unannotated frame); null is reserved for "
        "a pipeline that opts out.",
    )


class HealthResponse(BaseModel):
    status: str  # "ok" | "loading" | "error"
    phase: str = "2-inference"  # retained for wire-contract back-compat
    detail: str | None = None
