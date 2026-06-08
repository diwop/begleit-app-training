#!/bin/bash
# --- spike.sh ---
set -e

echo "📦 Mapping Axolotl Virtual Environment paths..."
export VIRTUAL_ENV="/workspace/axolotl-venv"
export PATH="/workspace/axolotl-venv/bin:$PATH"

echo "🐳 Activating Dependency Override Shield for CUDA 12.8..."

# 1. Capture the exact working pre-baked PyTorch configuration inside the venv
SYSTEM_TORCH=$(python -c "import torch; print(torch.__version__)")
echo "Detected Native PyTorch inside VENV: $SYSTEM_TORCH"

# 2. Purge the accidental CUDA 13 packages to ensure a pristine slate
echo "🧹 Purging mismatched CUDA 13 runtime artifacts..."
uv pip uninstall vllm flashinfer-python flashinfer-cubin compressed-tensors tilelang xgrammar nvidia-cuda-runtime nvidia-cuda-nvrtc nvidia-cuda-cccl nvidia-cuda-crt nvidia-cuda-nvcc nvidia-nvvm || true

# 3. Create the temporary override directive for Torch
echo "torch==$SYSTEM_TORCH" > /tmp/overrides.txt

# 4. Install the explicit CUDA 12.8 compiled wheel from the release index
# We pass the cu128 extra-index-url to force sub-dependencies to lock onto CUDA 12 variants
VLLM_VERSION="0.22.1"
echo "📥 Fetching and installing true cu128 wheel for vLLM v$VLLM_VERSION..."
uv pip install \
  "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu128-cp38-abi3-manylinux_2_35_x86_64.whl" \
  transformers \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --override /tmp/overrides.txt

# 5. Suppress asynchronous multiprocess deadlocks across the multi-GPU setup
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

echo "🎯 Environment locked down with true CUDA 12.8 wheel. Running spike..."
python src/spike.py