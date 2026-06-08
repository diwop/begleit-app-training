#!/bin/bash
# --- spike.sh ---
set -e

echo "📦 Mapping Axolotl Python 3.12 Virtual Environment..."
export VIRTUAL_ENV="/workspace/axolotl-venv"
export PATH="/workspace/axolotl-venv/bin:$PATH"

# Eliminate PyTorch management thread bloat
export OMP_NUM_THREADS=1

# --- 🛠️ NCCL & ENGINE CONTROLS ---
# Force NCCL to communicate via safe host memory channels instead of direct hardware loops
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

# CRITICAL FIX: Disable the alpha V1 engine to bypass shared memory timeouts.
# This forces vLLM to use the ultra-stable, industry-standard V0 core.
export VLLM_USE_V1=0
# ---------------------------------

echo "📥 Ensuring mainstream vLLM stack from PyPI..."
uv pip install vllm transformers

# Prevent multi-GPU worker initialization deadlocks
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

echo "🎯 Environment fully optimized. Launching text generation spike..."
python src/spike.py