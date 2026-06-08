#!/bin/bash
# --- spike.sh ---
set -e

echo "📦 Mapping Axolotl Python 3.12 Virtual Environment..."
export VIRTUAL_ENV="/workspace/axolotl-venv"
export PATH="/workspace/axolotl-venv/bin:$PATH"

echo "🐍 Running Pre-Install Hardware Handshake..."
python src/check_gpu.py

echo "📥 Installing standard production vLLM from PyPI..."
# No overrides, no custom wheels. Native, clean compilation.
uv pip install vllm transformers

echo "🐍 Running Post-Install Hardware Handshake..."
python src/check_gpu.py

# Prevent multi-GPU worker initialization deadlocks
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

echo "🎯 Environment fully unified. Launching text generation spike..."
python src/spike.py