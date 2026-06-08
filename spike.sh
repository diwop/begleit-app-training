#!/bin/bash
# --- spike.sh ---
set -e

echo "📦 Mapping Axolotl Python 3.12 Virtual Environment..."
export VIRTUAL_ENV="/workspace/axolotl-venv"
export PATH="/workspace/axolotl-venv/bin:$PATH"


# 2. Redirect all global OS temporary staging tracks to the volume
export TMPDIR="/app/tmp"
export TMP="/app/tmp"
export TEMP="/app/tmp"

# 3. Force the final assembled file cache to stay on the volume
export HF_HOME="/app/huggingface_cache"
export HF_HUB_CACHE="/app/huggingface_cache/hub"
export TRANSFORMERS_CACHE="/app/huggingface_cache/hub"

# Eliminate CPU management thread bloat
export OMP_NUM_THREADS=1

# Safe virtual host network routing blocks (Essential for RunPod multi-GPU handshakes)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=INFO # Activate for more information

echo "📥 Upgrading environment dependencies for Mistral Small 4..."
# Ensure your virtual environment pulls the necessary tokenization engine extensions
uv pip install vllm transformers "mistral_common>=1.11.0"

export VLLM_WORKER_MULTIPROC_METHOD="spawn"

echo "🎯 Environment locked down. Launching Mistral Small 4 Inference..."
python src/spike.py