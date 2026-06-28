#!/usr/bin/env bash
# Container entrypoint: ensure vendor + weights exist (downloading weights into
# the mounted volume on first start), then exec the server.
set -euo pipefail

cd /app

# Vendor is baked at build time, but re-create it if the layer was stripped or
# OMNI_VENDOR_DIR points at an empty mounted volume.
if [ ! -f "${OMNI_VENDOR_DIR:-/app/vendor/OmniParser}/util/utils.py" ]; then
  echo "[entrypoint] vendor missing — bootstrapping..."
  python3.12 scripts/setup_vendor.py
fi

weights_dir="${OMNI_WEIGHTS_DIR:-/app/weights}"
yolo="${weights_dir}/icon_detect/model.pt"
florence="${weights_dir}/icon_caption_florence/model.safetensors"
# Gate on the LAST artifact (the renamed Florence-2 safetensors) as well as the
# YOLO model, so a first download interrupted between the two is repaired on the
# next start instead of leaving the pipeline permanently FileNotFoundError.
if [ ! -f "$yolo" ] || [ ! -f "$florence" ]; then
  echo "[entrypoint] weights missing/incomplete — downloading (~1 GB)..."
  python3.12 scripts/download_weights.py
fi

exec "$@"
