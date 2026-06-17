#!/bin/bash
# --- spike.sh ---
set -e

# Eliminate CPU management thread bloat
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Safe virtual host network routing blocks (Essential for RunPod multi-GPU handshakes)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Enable debugging
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

echo "Pulling dataset from DVC..."
python -m dvc pull

echo "🎯 Environment locked down. Launching Evaluation..."
python src/evaluation.py