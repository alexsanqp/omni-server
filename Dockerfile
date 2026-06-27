# omni-server GPU image.
#
# Build (needs network for the cu124 torch index + the vendored OmniParser clone):
#   docker build -t omni-server .
# Run (needs the NVIDIA Container Toolkit on the host):
#   docker run --gpus all -p 8001:8001 \
#       -v omni-weights:/app/weights -v hf-cache:/root/.cache/huggingface omni-server
#
# Weights (~1 GB) are NOT baked in — the entrypoint downloads them into the
# mounted /app/weights volume on first start. The vendored OmniParser source IS
# baked (it is small and pins the upstream SHA).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:/root/.local/bin:$PATH"

# System deps: python 3.12 (deadsnakes), git (vendor clone), and the shared libs
# OpenCV / PaddleOCR / EasyOCR load at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        build-essential libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv (self-contained installer).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# Resolve deps first for layer caching.
COPY pyproject.toml README.md LICENSE ./
RUN uv venv --python 3.12 && uv pip install --python /app/.venv .[gpu]

# App source + vendor bootstrap.
COPY omni_server ./omni_server
COPY scripts ./scripts
COPY patches ./patches
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN python3.12 scripts/setup_vendor.py && chmod +x /usr/local/bin/entrypoint.sh

ENV OMNI_HOST=0.0.0.0 \
    OMNI_PORT=8001 \
    OMNI_REAL_MODEL=1

EXPOSE 8001
# Generous start-period: a fresh-volume first boot downloads ~1 GB of weights +
# HF models and then does a multi-minute model load + warmup before /health is
# "ok"; a shorter grace window would flap the container to "unhealthy" meanwhile.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${OMNI_PORT}/health" | grep -q '"status":"ok"' || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["omni-server"]
