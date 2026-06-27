#!/usr/bin/env bash
# Container entrypoint: ensure vendor + weights exist (downloading weights into
# the mounted volume on first start), then exec the server.
set -euo pipefail

cd /app

# Vendor is baked at build time, but re-create it if the layer was stripped or
# OMNI_VENDOR_DIR points at an empty mounted volume.
if [ "${OMNI_REAL_MODEL:-1}" = "1" ]; then
  if [ ! -f "${OMNI_VENDOR_DIR:-/app/vendor/OmniParser}/util/utils.py" ]; then
    echo "[entrypoint] vendor missing — bootstrapping..."
    python3.12 scripts/setup_vendor.py
  fi
  if [ ! -f "${OMNI_WEIGHTS_DIR:-/app/weights}/icon_detect/model.pt" ]; then
    echo "[entrypoint] weights missing — downloading (~1 GB)..."
    python3.12 scripts/download_weights.py
  fi
fi

exec "$@"
