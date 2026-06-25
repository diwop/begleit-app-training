#!/bin/bash
set -e

DEFAULT_REPO="https://github.com/diwop/begleit-app-training.git"

# Define default environment variables
BRANCH=${BRANCH:-"main"}
REPO_URL=${REPO_URL:-"$DEFAULT_REPO"}


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

# Start JupyterLab Sidecar in the background
echo "Starting JupyterLab..."

# Safely inject Jupyter into the system environment utilizing our secure cache paths
if ! command -v jupyter &> /dev/null; then
    echo "Installing JupyterLab..."
    uv pip install --system jupyterlab
fi

# Run JupyterLab Bound cleanly to your custom port parameters
jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser \
  --IdentityProvider.token="${JUPYTER_PASSWORD:-}" \
  --ServerApp.password="" &

echo "Cloning branch '$BRANCH' from $REPO_URL..."
rm -rf /runner/repo
git clone -b "$BRANCH" "$REPO_URL" /runner/repo

cd /runner/repo

MODE=${MODE:-"train"} # "train" or "eval"

if [ "$MODE" = "eval" ] || [ "$MODE" = "evaluation" ]; then
    echo "Starting evaluation phase..."
    bash scripts/eval.sh
else
    echo "Starting training phase..."
    bash scripts/train.sh
fi
