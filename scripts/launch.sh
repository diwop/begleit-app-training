#!/bin/bash
set -e

DEFAULT_REPO="https://github.com/diwop/begleit-app-training.git"

# Define default environment variables
BRANCH=${BRANCH:-"main"}
REPO_URL=${REPO_URL:-"$DEFAULT_REPO"}

# 1. PERMANENTLY activate Axolotl's pre-built master environment!
export VIRTUAL_ENV="/workspace/axolotl-venv"
export PATH="/workspace/axolotl-venv/bin:$PATH"

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

# Safely inject Jupyter into the active axolotl-venv utilizing our secure cache paths
if ! command -v jupyter &> /dev/null; then
    echo "Installing JupyterLab into master runtime..."
    uv pip install jupyterlab
fi

# Run JupyterLab Bound cleanly to your custom port parameters
jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser \
  --IdentityProvider.token="${JUPYTER_PASSWORD:-}" \
  --ServerApp.password="" &

echo "Cloning branch '$BRANCH' from $REPO_URL..."
rm -rf /runner/repo
git clone -b "$BRANCH" "$REPO_URL" /runner/repo

cd /runner/repo

echo "Installing training dependencies into axolotl-venv..."
uv pip compile src-train/pyproject.toml -o src-train/requirements.txt
uv pip install -r src-train/requirements.txt

if [ "${SKIP_TRAIN:-false}" != "true" ]; then
    echo "Starting training phase..."
    bash scripts/train.sh
else
    echo "Skipping training phase..."
fi

if [ "${SKIP_EVAL:-false}" != "true" ]; then
    echo "Starting evaluation phase..."
    bash scripts/eval.sh
else
    echo "Skipping evaluation phase..."
fi
