"""Standalone smoke client for a running omni-server.

Posts an image to ``/parse`` and prints the parsed elements. Depends only on the
stdlib + Pillow.

Usage::

    python scripts/smoke_client.py                       # generated test image
    python scripts/smoke_client.py path/to/screenshot.png
    OMNI_URL=http://gpu-box:8001 OMNI_AUTH_TOKEN=secret python scripts/smoke_client.py shot.png
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

URL = os.environ.get("OMNI_URL", "http://127.0.0.1:8001").rstrip("/")
TOKEN = os.environ.get("OMNI_AUTH_TOKEN")


def _make_test_image() -> bytes:
    img = Image.new("RGB", (640, 400), "white")
    d = ImageDraw.Draw(img)
    d.rectangle((40, 40, 240, 100), outline="black", width=2)
    d.text((60, 60), "Search", fill="black")
    d.rectangle((40, 140, 200, 200), fill="#cc0000")
    d.text((60, 160), "Subscribe", fill="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _load(path: str | None) -> tuple[bytes, str]:
    if path:
        data = Path(path).read_bytes()
        fmt = "png" if path.lower().endswith(".png") else "jpeg"
        return data, fmt
    return _make_test_image(), "png"


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    image_bytes, fmt = _load(arg)
    body = json.dumps(
        {"image_b64": base64.b64encode(image_bytes).decode("ascii"), "image_format": fmt}
    ).encode()
    req = urllib.request.Request(f"{URL}/parse", data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:500]}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"cannot reach {URL}: {exc}", file=sys.stderr)
        return 1
    wall = (time.perf_counter() - t0) * 1000

    elements = data.get("elements", [])
    print(f"OK: {len(elements)} elements, server={data.get('parse_time_ms')}ms wall={wall:.0f}ms")
    for e in elements[:25]:
        print(
            f"  #{e.get('element_id'):>3} conf={e.get('confidence'):.2f} "
            f"interactive={e.get('interactivity')} {e.get('tags')} {e.get('label')!r}"
        )
    if data.get("som_image_b64"):
        out = Path("smoke_som.png")
        out.write_bytes(base64.b64decode(data["som_image_b64"]))
        print(f"SoM image -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
