"""Bootstrap the vendored OmniParser source.

Clones https://github.com/microsoft/OmniParser at a pinned SHA into
``vendor/OmniParser`` and applies ``patches/omniparser-vendor-<ref>.patch``
(lazy/Cyrillic OCR, empty-screen guards, honest YOLO confidence — see the patch
README). Idempotent: re-running on an already-set-up tree is a no-op.

Only ``util/utils.py`` + ``util/box_annotator.py`` are imported at runtime, but
we keep the full checkout so upstream imports resolve. The directory is
git-ignored; this script (plus the patch) is the persistent record.

Usage::

    python scripts/setup_vendor.py            # clone + patch
    python scripts/setup_vendor.py --force    # wipe and re-clone

Override the repo/ref via ``OMNI_VENDOR_REPO`` / ``OMNI_VENDOR_REF``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = Path(os.environ.get("OMNI_VENDOR_DIR") or (SERVICE_ROOT / "vendor" / "OmniParser"))
REPO = os.environ.get("OMNI_VENDOR_REPO", "https://github.com/microsoft/OmniParser")
REF = os.environ.get("OMNI_VENDOR_REF", "b0d5c9f")
PATCH = SERVICE_ROOT / "patches" / f"omniparser-vendor-{REF}.patch"
PATCH_SENTINEL = "PATCHED: lazy init paddle_ocr"


def _run(args: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def _already_patched() -> bool:
    utils = VENDOR_DIR / "util" / "utils.py"
    if not utils.is_file():
        return False
    return PATCH_SENTINEL in utils.read_text(encoding="utf-8", errors="ignore")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="wipe vendor/ and re-clone")
    args = ap.parse_args()

    if not PATCH.is_file():
        print(f"ERROR: patch not found: {PATCH}", file=sys.stderr)
        return 1

    if args.force and VENDOR_DIR.exists():
        print(f"Removing {VENDOR_DIR}", flush=True)
        shutil.rmtree(VENDOR_DIR)

    if _already_patched():
        print(f"vendor already present and patched at {VENDOR_DIR} — nothing to do.")
        return 0

    if VENDOR_DIR.exists() and any(VENDOR_DIR.iterdir()):
        print(
            f"ERROR: {VENDOR_DIR} exists but is not patched. Re-run with --force.",
            file=sys.stderr,
        )
        return 1

    VENDOR_DIR.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--no-checkout", REPO, str(VENDOR_DIR)])
    _run(["git", "checkout", REF], cwd=VENDOR_DIR)
    # Apply with the patch relative to the vendor root (patch paths are a/util/...).
    _run(["git", "apply", "--whitespace=nowarn", str(PATCH)], cwd=VENDOR_DIR)

    if not _already_patched():
        print("ERROR: patch applied but sentinel missing — verify manually.", file=sys.stderr)
        return 1
    print(f"\nVendor ready at {VENDOR_DIR} (ref {REF}, patched).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
