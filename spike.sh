#!/bin/bash
# --- spike.sh ---
set -e

echo "📦 Mapping Axolotl Virtual Environment paths..."
export VIRTUAL_ENV="/workspace/axolotl-venv"
export PATH="/workspace/axolotl-venv/bin:$PATH"

echo "🐳 Activating Unified CUDA 12 Dependency Shield..."

# 1. Capture the exact pre-baked PyTorch configuration inside the venv
SYSTEM_TORCH=$(python -c "import torch; print(torch.__version__)")
echo "Detected Native PyTorch inside VENV: $SYSTEM_TORCH"

# 2. Ensure any lingering CUDA 13 packages are completely wiped
echo "🧹 Purging mismatched CUDA 13 runtime artifacts..."
uv pip uninstall vllm flashinfer-python flashinfer-cubin compressed-tensors tilelang xgrammar nvidia-cuda-runtime nvidia-cuda-nvrtc nvidia-cuda-cccl nvidia-cuda-crt nvidia-cuda-nvcc nvidia-nvvm || true

# 3. Create the temporary override directive for Torch
echo "torch==$SYSTEM_TORCH" > /tmp/overrides.txt

# 4. Fetch the official unified CUDA 12 release wheel (+cu129)
VLLM_VERSION="0.22.1"
echo "📥 Downloading unified CUDA 12 wheel (v$VLLM_VERSION)..."
uv pip install \
  "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
  transformers \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  --override /tmp/overrides.txt

# 5. Suppress asynchronous multiprocess deadlocks across the multi-GPU setup
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

echo "🎯 Environment locked down with unified CUDA 12 wheel. Running spike..."
python src/spike.py