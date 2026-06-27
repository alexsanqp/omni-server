"""Unit test for OmniParserPipeline.parse()'s result-mapping loop.

The mapping (element_id ordering, normalized->pixel bbox scaling by w/h, the
content->type->'' label fallback, the tags tuple, and the conf / ocr-threshold /
0.5 confidence derivation) is the wire contract a downstream client mirrors
field-for-field, and it is the most regression-prone code in inference.py.

No GPU/torch: parse() imports ``util.utils`` lazily, so we inject a fake module
returning hand-built ``parsed_elems`` and drive the loop directly on a pipeline
built with ``__new__`` (bypassing the weight-loading ``__init__``).
"""

from __future__ import annotations

import sys
import types
from threading import Lock
from typing import Any

import pytest
from PIL import Image

from omni_server.config import Settings
from omni_server.inference import OmniParserPipeline


@pytest.fixture
def fake_vendor(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Inject a fake ``util.utils`` and return the parsed_elems it will emit."""
    parsed_elems = [
        {
            "bbox": [0.1, 0.2, 0.5, 0.6],
            "content": "OK",
            "type": "icon",
            "source": "box_yolo_content_yolo",
            "conf": 1.5,  # > 1 -> must clamp to 1.0
            "interactivity": True,
        },
        {  # no content -> label falls back to type; no conf, type=text -> threshold
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "type": "text",
            "source": "box_ocr_content_ocr",
        },
        {  # no content/conf, non-text -> default 0.5 confidence
            "bbox": [0.0, 0.0, 0.0, 0.0],
            "type": "icon",
            "source": "x",
        },
    ]

    def fake_check_ocr_box(*args: Any, **kwargs: Any) -> tuple[tuple[list, list], None]:
        return (["t"], [[0, 0, 1, 1]]), None

    def fake_get_som_labeled_img(*args: Any, **kwargs: Any) -> tuple[str, dict, list]:
        return "SOMB64", {}, parsed_elems

    util_pkg = types.ModuleType("util")
    util_utils = types.ModuleType("util.utils")
    util_utils.check_ocr_box = fake_check_ocr_box  # type: ignore[attr-defined]
    util_utils.get_som_labeled_img = fake_get_som_labeled_img  # type: ignore[attr-defined]
    util_pkg.utils = util_utils  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "util", util_pkg)
    monkeypatch.setitem(sys.modules, "util.utils", util_utils)
    return parsed_elems


def _pipeline() -> OmniParserPipeline:
    obj = OmniParserPipeline.__new__(OmniParserPipeline)
    obj._s = Settings(real_model=False, warmup=False)
    obj._lock = Lock()
    obj.yolo = None
    obj.caption = None
    return obj


def test_parse_maps_elements_field_for_field(fake_vendor: list[dict[str, Any]]) -> None:
    elements, som_b64 = _pipeline().parse(Image.new("RGB", (200, 100), "white"))

    assert som_b64 == "SOMB64"
    assert [e.element_id for e in elements] == [0, 1, 2]  # idx == drawn SoM number

    e0, e1, e2 = elements
    # normalized bbox scaled by (w=200, h=100)
    assert e0.bbox == (20.0, 20.0, 100.0, 60.0)
    assert e1.bbox == (0.0, 0.0, 200.0, 100.0)
    assert e2.bbox == (0.0, 0.0, 0.0, 0.0)
    # label: content -> type -> "" fallback
    assert (e0.label, e1.label, e2.label) == ("OK", "text", "icon")
    # tags == (type, source)
    assert e0.tags == ("icon", "box_yolo_content_yolo")
    assert e1.tags == ("text", "box_ocr_content_ocr")
    # confidence: clamp(conf) / ocr_text_threshold for text / 0.5 default
    assert e0.confidence == 1.0
    assert e1.confidence == _pipeline()._s.ocr_text_threshold
    assert e2.confidence == 0.5
    # interactivity passthrough
    assert (e0.interactivity, e1.interactivity, e2.interactivity) == (True, False, False)
