#!/bin/bash
# --- spike.sh ---
set -e

echo "📦 Mapping Axolotl Virtual Environment paths..."
# 1. Force the non-interactive script to use the workspace venv
export VIRTUAL_ENV="/workspace/axolotl-venv"
export PATH="/workspace/axolotl-venv/bin:$PATH"

echo "Current VIRTUAL_ENV: $VIRTUAL_ENV"
echo "Current PATH: $PATH"

python src/check_gpu.py

echo "🐳 Activating Dependency Override Shield..."

# 2. Capture the exact working pre-baked PyTorch configuration inside the venv
# This now correctly uses /workspace/axolotl-venv/bin/python
SYSTEM_TORCH=$(python -c "import torch; print(torch.__version__)")
echo "Detected Native PyTorch inside VENV: $SYSTEM_TORCH"

# 3. Generate the temporary override directive
echo "torch==$SYSTEM_TORCH" > /tmp/overrides.txt

# 4. Install vllm and transformers directly INTO the virtual environment
# Note: We removed '--system' because UV auto-targets $VIRTUAL_ENV now!
uv pip install vllm transformers --override /tmp/overrides.txt

# 5. Suppress asynchronous multiprocess deadlocks across the multi-GPU setup
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

python src/check_gpu.py

echo "🎯 Environment locked down. Initiating token generation spike..."
python src/spike.py