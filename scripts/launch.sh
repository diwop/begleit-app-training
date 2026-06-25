#!/bin/bash
set -e

DEFAULT_REPO="https://github.com/diwop/begleit-app-training.git"

# Define default environment variables
BRANCH=${BRANCH:-"main"}
REPO_URL=${REPO_URL:-"$DEFAULT_REPO"}
MODE=${MODE:-"eval"} # "train" or "eval"

echo "=== Environment Diagnostics ==="
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: GPU environment is missing! nvidia-smi not found."
    exit 1
else
    echo "✅ GPU support detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi

if [ "$MODE" = "train" ]; then
    if [ ! -d "/workspace/axolotl-venv" ]; then
        echo "❌ ERROR: /workspace/axolotl-venv not found! You must run training on the Axolotl docker image."
        exit 1
    else
        echo "✅ Correct Axolotl training image detected."
    fi
elif [ "$MODE" = "eval" ] || [ "$MODE" = "evaluation" ]; then
    if [ -d "/workspace/axolotl-venv" ]; then
        echo "⚠️ WARNING: You are running evaluation mode on the Axolotl training image."
        echo "It is highly recommended to run evaluation in a dedicated SGLang/PyTorch container to avoid library conflicts."
    else
        echo "✅ Separate evaluation container detected."
    fi
fi


# --- 💾 CRITICAL IMAGE-LEVEL STORAGE PROTECTION ---
export HF_HOME="/app/huggingface_cache"
export HF_HUB_CACHE="/app/huggingface_cache/hub"
export HF_XET_CACHE="/app/huggingface_cache/xet"
export HUGGINGFACE_HUB_CACHE="/app/huggingface_cache/hub"
export TRANSFORMERS_CACHE="/app/huggingface_cache/hub"
export XDG_CACHE_HOME="/app/xdg_cache"
export UV_CACHE_DIR="/app/uv_cache"
export TMPDIR="/app/tmp"
export TMP="/app/tmp"
export TEMP="/app/tmp"

echo "=== Initializing Worker Node ==="

# Ensure high-capacity directory structures are present on the volume mount
echo "Initializing structural directories on the /app volume..."
mkdir -p /app/huggingface_cache/hub
mkdir -p /app/huggingface_cache/xet
mkdir -p /app/xdg_cache
mkdir -p /app/uv_cache
mkdir -p /app/tmp

if [ "${START_JUPYTER:-false}" = "true" ]; then
    # Start JupyterLab Sidecar in the background
    echo "Starting JupyterLab..."

    # Safely inject Jupyter into the system environment utilizing our secure cache paths
    if ! command -v jupyter &> /dev/null; then
        echo "Installing JupyterLab..."
        uv pip install --system --break-system-packages jupyterlab
    fi

    # Run JupyterLab Bound cleanly to your custom port parameters
    jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser \
      --IdentityProvider.token="${JUPYTER_PASSWORD:-}" \
      --ServerApp.password="" &
fi

echo "Cloning branch '$BRANCH' from $REPO_URL..."
rm -rf /runner/repo
git clone -b "$BRANCH" "$REPO_URL" /runner/repo

cd /runner/repo

if [ "$MODE" = "eval" ] || [ "$MODE" = "evaluation" ]; then
    echo "Starting evaluation phase..."
    bash scripts/eval.sh
else
    echo "Starting training phase..."
    bash scripts/train.sh
fi
