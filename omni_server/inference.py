"""OmniParser v2 inference pipeline.

Wraps Microsoft OmniParser (YOLOv8 icon detector + Florence-2 captioner +
PaddleOCR/EasyOCR text detection) behind a fixed wire contract. The upstream
code is vendored under ``vendor/OmniParser`` at a pinned SHA with a small patch
applied (see ``scripts/setup_vendor.py`` and ``patches/``); only
``util/utils.py`` (and its sibling ``box_annotator.py``) is imported, via the
``sys.path`` insert below.

Tuning (relative to upstream's first pass; all overridable via env — see
``omni_server.config``):
- iou_threshold 0.1, BOX_THRESHOLD 0.05 (MSFT gradio-demo defaults — fewer false
  dedups so nested elements survive).
- imgsz min(longest_side, 1280) + scale_img=True — better small-icon recall on
  multi-MP screenshots.
- prompt '<DETAILED_CAPTION>' — richer semantic labels.
- use_paddleocr=True (lang='cyrillic' via the vendor patch) — Ukrainian UI text.
- Returns the SoM-numbered annotated image as ParseResponse.som_image_b64 and
  lifts interactivity + element_id + honest per-box confidence to first-class
  Element fields.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from threading import Lock

from PIL import Image

from omni_server.config import Settings, get_settings
from omni_server.schemas import Element

logger = logging.getLogger(__name__)


class OmniParserPipeline:
    """Holds the loaded YOLO + Florence-2 models and runs the full parse pipeline.

    A single ``threading.Lock`` serialises ``parse`` because the underlying GPU
    pipeline is not safe to run concurrently on one device.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._s = settings or get_settings()
        self._vendor_dir = Path(self._s.vendor_dir)
        weights_dir = Path(self._s.weights_dir)
        self._yolo_weights = weights_dir / "icon_detect" / "model.pt"
        self._caption_weights_dir = weights_dir / "icon_caption_florence"

        self._add_vendor_to_path()
        if not self._vendor_dir.is_dir():
            raise FileNotFoundError(
                f"vendored OmniParser missing: {self._vendor_dir} "
                "— run `python scripts/setup_vendor.py`"
            )
        if not self._yolo_weights.is_file():
            raise FileNotFoundError(
                f"YOLO weights missing: {self._yolo_weights} "
                "— run `python scripts/download_weights.py`"
            )

        from util.utils import get_yolo_model

        logger.info("Loading YOLO icon detector from %s", self._yolo_weights)
        self.yolo = get_yolo_model(str(self._yolo_weights))

        self.caption = None
        if self._s.use_local_semantics:
            if not (self._caption_weights_dir / "model.safetensors").is_file():
                raise FileNotFoundError(
                    f"Florence-2 weights missing: {self._caption_weights_dir} "
                    "— run `python scripts/download_weights.py` "
                    "(or set OMNI_USE_LOCAL_SEMANTICS=0 to skip captioning)"
                )
            from util.utils import get_caption_model_processor

            logger.info("Loading Florence-2 caption model from %s", self._caption_weights_dir)
            self.caption = get_caption_model_processor(
                "florence2", model_name_or_path=str(self._caption_weights_dir)
            )
        else:
            logger.info("use_local_semantics=0 -> skipping Florence-2 load (faster, less VRAM)")

        self._lock = Lock()
        logger.info("OmniParser pipeline ready")

    def _add_vendor_to_path(self) -> None:
        p = str(self._vendor_dir)
        if p not in sys.path:
            sys.path.insert(0, p)

    def parse(self, image: Image.Image) -> tuple[tuple[Element, ...], str | None]:
        """Run YOLO + OCR + Florence-2 and return (elements, som_image_b64).

        ``elements``: Element tuple with pixel-coord bboxes (xyxy).
        ``som_image_b64``: base64 PNG of the frame with numbered SoM boxes, ready
        for Set-of-Marks prompting of a VLM (or None if nothing was detected).
        """
        from util.utils import check_ocr_box, get_som_labeled_img

        s = self._s
        rgb = image.convert("RGB")
        w, h = rgb.size
        imgsz = min(max(w, h), s.imgsz_max)

        with self._lock:
            (ocr_text, ocr_bbox), _ = check_ocr_box(
                rgb,
                display_img=False,
                output_bb_format="xyxy",
                use_paddleocr=s.use_paddleocr,
                easyocr_args={"text_threshold": s.ocr_text_threshold, "paragraph": False},
            )
            som_b64, _label_coords, parsed_elems = get_som_labeled_img(
                image_source=rgb,
                model=self.yolo,
                caption_model_processor=self.caption,
                ocr_bbox=ocr_bbox,
                ocr_text=ocr_text,
                BOX_TRESHOLD=s.box_threshold,
                iou_threshold=s.iou_threshold,
                use_local_semantics=s.use_local_semantics,
                prompt=s.caption_prompt,
                draw_bbox_config={
                    "text_scale": 0.5,
                    "text_padding": 4,
                    "text_thickness": 1,
                    "thickness": 2,
                },
                output_coord_in_ratio=False,
                scale_img=True,
                imgsz=imgsz,
                batch_size=s.batch_size,
            )

        out: list[Element] = []
        for idx, p in enumerate(parsed_elems):
            bbox_norm = p["bbox"]
            bbox_px = (
                float(bbox_norm[0]) * w,
                float(bbox_norm[1]) * h,
                float(bbox_norm[2]) * w,
                float(bbox_norm[3]) * h,
            )
            label = str(p.get("content") or p.get("type") or "")
            tags: tuple[str, ...] = (
                str(p.get("type", "unknown")),
                str(p.get("source", "unknown")),
            )
            # YOLO boxes carry their detection score via the vendored patch; OCR
            # text already passed the OCR threshold, so floor it there.
            raw_conf = p.get("conf")
            if raw_conf is not None:
                confidence = _clamp01(float(raw_conf))
            elif p.get("type") == "text":
                confidence = s.ocr_text_threshold
            else:
                confidence = 0.5
            out.append(
                Element(
                    label=label,
                    bbox=bbox_px,
                    confidence=confidence,
                    tags=tags,
                    interactivity=bool(p.get("interactivity", False)),
                    element_id=idx,
                )
            )
        return tuple(out), som_b64


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
