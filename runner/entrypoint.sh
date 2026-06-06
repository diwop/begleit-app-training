#!/bin/bash
set -e

DEFAULT_REPO="https://github.com/diwop/begleit-app-training.git"

# Define default environment variables
BRANCH=${BRANCH:-"main"}
REPO_URL=${REPO_URL:-"$DEFAULT_REPO"}

echo "=== Initializing Worker Node ==="

# Start JupyterLab Sidecar in the background
echo "Starting JupyterLab..."

if ! command -v jupyter &> /dev/null; then
    echo "Jupyter not found. Installing into the active transparent venv..."
    uv pip install jupyterlab
fi

jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser \
  --IdentityProvider.token="${JUPYTER_PASSWORD:-}" \
  --ServerApp.password="" &

echo "Cloning branch '$BRANCH' from $REPO_URL..."
rm -rf /runner/repo
git clone -b "$BRANCH" "$REPO_URL" /runner/repo

chmod +x /runner/repo/train.sh
chmod +x /runner/repo/eval.sh

# Prevent loop of death
if [ $# -eq 0 ] || [[ "$*" == *"/runner/entrypoint.sh"* ]]; then
    echo "Defaulting execution to repository training script..."
    exec bash /runner/repo/train.sh
else
    echo "Handing off execution to: $@"
    exec "$@"
fi