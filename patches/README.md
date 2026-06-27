# Vendored OmniParser patch

`vendor/OmniParser` is a git clone of https://github.com/microsoft/OmniParser at
commit **`b0d5c9f`** (Merge pull request #337). It is **git-ignored** (not
committed) — only `util/utils.py` (+ `util/box_annotator.py`) is imported at
runtime. This patch is the persistent record of our local changes to it.

## Reproduce `vendor/` (recommended: one command)

```bash
python scripts/setup_vendor.py        # clone @ b0d5c9f + apply this patch
python scripts/setup_vendor.py --force  # wipe and re-do
```

## Reproduce manually

```bash
git clone --no-checkout https://github.com/microsoft/OmniParser vendor/OmniParser
cd vendor/OmniParser
git checkout b0d5c9f
git apply ../../patches/omniparser-vendor-b0d5c9f.patch
```

> The patch is UTF-8. (The version inherited from the old in-repo service was
> UTF-16/CRLF, which made `git apply` fail — that has been fixed here.)

## What the patch changes (`util/utils.py`)

1. **Lazy PaddleOCR** with `lang='cyrillic'` (was module-level, `en`) — Cyrillic
   text on the Ukrainian YouTube UI.
2. **Lazy EasyOCR reader** `['uk','en'], verbose=False` — avoids loading ~100 MB
   of GPU models the PaddleOCR path never uses; `verbose=False` dodges a U+2588
   crash on Windows cp1251 stdout during first-run model download.
3. **Empty-OCR guard**: PaddleOCR 2.x returns `[None]` when no text is found
   (previously a `TypeError` → 500 on every text-free screenshot, e.g. fullscreen
   video); `ocr_bbox = []` instead of `None` (upstream `zip(None, ...)` bug).
4. **Empty-screen guard**: zero detections produced a 0-dim tensor that crashed
   `box_convert`; now returns the original frame with no elements.
5. **YOLO confidence pass-through** (`conf`) through `get_som_labeled_img` →
   `remove_overlap_new` → elements, so the server reports honest per-element
   confidence instead of a hardcoded 1.0.
6. Removed eager `AzureOpenAI` and `matplotlib` imports (unused at
   `display_img=False`).

## Refresh the patch after editing `vendor/`

```bash
cd vendor/OmniParser
git diff > ../../patches/omniparser-vendor-b0d5c9f.patch
```

Keep the output UTF-8 (on Windows PowerShell use
`git diff | Out-File -Encoding utf8 ...`, not `>`).
