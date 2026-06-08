#!/bin/bash
# --- spike.sh ---
set -e

echo "📦 Mapping Axolotl Python 3.12 Virtual Environment..."
export VIRTUAL_ENV="/workspace/axolotl-venv"
export PATH="/workspace/axolotl-venv/bin:$PATH"

# OPTIMIZATION: Eliminate PyTorch management thread bloat and clear CPU lanes
export OMP_NUM_THREADS=1

echo "📥 Ensuring mainstream vLLM stack from PyPI..."
uv pip install vllm transformers

# Prevent multi-GPU worker initialization deadlocks
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

echo "🎯 Environment fully optimized. Launching text generation spike..."
python src/spike.py