"""Download Microsoft OmniParser v2 weights from HuggingFace.

Pulls the YOLO icon detector + Florence-2 caption model into ``weights/`` (or
``$OMNI_WEIGHTS_DIR``) and renames ``icon_caption`` -> ``icon_caption_florence``
to match the layout ``omni_server.inference`` expects.

Note: the Florence-2 *processor* (tokenizer + remote modeling code) is fetched
separately from ``microsoft/Florence-2-base`` by transformers at first model
load, so the GPU box needs HuggingFace reachable on its first cold start. Set
``HF_HOME`` to a persistent dir to cache it.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

WEIGHTS_DIR = Path(
    os.environ.get("OMNI_WEIGHTS_DIR") or (Path(__file__).resolve().parent.parent / "weights")
)
REPO = os.environ.get("OMNI_WEIGHTS_REPO", "microsoft/OmniParser-v2.0")

FILES = [
    "icon_detect/train_args.yaml",
    "icon_detect/model.pt",
    "icon_detect/model.yaml",
    "icon_caption/config.json",
    "icon_caption/generation_config.json",
    "icon_caption/model.safetensors",
]


def main() -> int:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        print(f"Downloading {f}...", flush=True)
        path = hf_hub_download(repo_id=REPO, filename=f, local_dir=str(WEIGHTS_DIR))
        size_mb = Path(path).stat().st_size / 1024 / 1024
        print(f"  -> {path} ({size_mb:.1f} MB)", flush=True)

    icon_caption = WEIGHTS_DIR / "icon_caption"
    icon_caption_florence = WEIGHTS_DIR / "icon_caption_florence"
    if icon_caption.is_dir() and not icon_caption_florence.exists():
        print(f"Renaming {icon_caption.name} -> {icon_caption_florence.name}", flush=True)
        shutil.move(str(icon_caption), str(icon_caption_florence))

    print("\nFinal layout:")
    for p in sorted(WEIGHTS_DIR.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(WEIGHTS_DIR)}  ({p.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
