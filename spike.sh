#!/bin/bash
# --- spike.sh ---
set -e

echo "📦 Mapping Virtual Environment paths on CUDA 13..."
export VIRTUAL_ENV="/workspace/axolotl-venv"
export PATH="/workspace/axolotl-venv/bin:$PATH"

echo "📥 Installing mainstream vLLM stack from PyPI..."
# No overrides, no github wheels, no extra-index URLs. Pure automation.
uv pip install vllm transformers

# Suppress asynchronous multiprocess deadlocks across the multi-GPU setup
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

echo "🚀 Environment natively unified. Launching inference spike..."
python src/spike.py