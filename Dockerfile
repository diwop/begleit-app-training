# --- Stage 1: Build & Compile ---
ARG CUDA_VERSION=13.0.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install compiler prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    python3.12-full \
    python3.12-dev \
    libibverbs-dev \
    libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set up relocatable virtual environment
ENV VIRTUAL_ENV=/workspace/axolotl-venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install PyTorch with CUDA 13.0 support
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cu130

# Install build dependencies (needed for setuptools-rust or ninja-based compilations)
RUN uv pip install packaging ninja setuptools-rust

# Install Axolotl with flash-attn and deepspeed (this compiles custom kernels)
# Note: For now, sglang is omitted as requested.
RUN uv pip install "axolotl[flash-attn,deepspeed] @ git+https://github.com/axolotl-ai-cloud/axolotl.git"
RUN uv pip install liger-kernel

# Compile project-specific requirements
COPY pyproject.toml /tmp/
RUN uv pip compile /tmp/pyproject.toml -o /tmp/requirements.txt && \
    uv pip install -r /tmp/requirements.txt

# Clean package caches to minimize folder size before copy
RUN rm -rf /root/.cache /root/.cargo

# --- Stage 2: Runtime (Clean Option A) ---
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install standard runtime and system utilities (excluding heavy build-only packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12-full \
    git \
    curl \
    ca-certificates \
    libibverbs1 \
    libnuma1 \
    ffmpeg \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Activate pre-baked virtualenv
ENV VIRTUAL_ENV=/workspace/axolotl-venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy virtual environment from builder stage
COPY --from=builder /workspace/axolotl-venv /workspace/axolotl-venv

# Define default HuggingFace / cache locations
ENV HF_HOME="/app/huggingface_cache" \
    HF_HUB_CACHE="/app/huggingface_cache/hub" \
    HF_XET_CACHE="/app/huggingface_cache/xet" \
    HUGGINGFACE_HUB_CACHE="/app/huggingface_cache/hub" \
    TRANSFORMERS_CACHE="/app/huggingface_cache/hub" \
    XDG_CACHE_HOME="/app/xdg_cache" \
    UV_CACHE_DIR="/app/uv_cache" \
    TMPDIR="/app/tmp" \
    TMP="/app/tmp" \
    TEMP="/app/tmp"

WORKDIR /runner

COPY runner/entrypoint.sh /runner/entrypoint.sh
RUN chmod +x /runner/entrypoint.sh

ENTRYPOINT ["/bin/bash", "/runner/entrypoint.sh"]
CMD ["/bin/bash", "/runner/repo/train.sh"]