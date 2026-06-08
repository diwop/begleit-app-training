#!/bin/bash
# --- spike.sh ---
set -e

echo "🐳 Initializing minimal environment variables..."

# Force an in-place installation of vllm into the system scope
pip install --no-cache-dir vllm transformers

# Crucial environmental variable to prevent multi-GPU worker deadlocks when vLLM initializes
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

echo "🚀 Environment secured. Launching inference spike application..."
python src/spike.py