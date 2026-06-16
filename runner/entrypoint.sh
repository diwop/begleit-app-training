#!/bin/bash
set -e

DEFAULT_REPO="https://github.com/diwop/begleit-app-training.git"

# Define default environment variables
BRANCH=${BRANCH:-"main"}
REPO_URL=${REPO_URL:-"$DEFAULT_REPO"}

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

# Install (optional) dependency delta
cd /runner/repo
uv pip compile pyproject.toml -o requirements.txt
uv pip install -r requirements.txt
uv pip install vllm

# Prevent loop of death and manage script hand-offs cleanly
if [ $# -eq 0 ] || [[ "$*" == *"/runner/entrypoint.sh"* ]]; then
    echo "Defaulting execution to repository training script..."
    exec bash /runner/repo/train.sh
else
    echo "Handing off execution to: $@"
    exec "$@"
fi