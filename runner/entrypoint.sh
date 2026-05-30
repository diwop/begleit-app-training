#!/bin/bash
set -e

DEFAULT_REPO="https://github.com/diwop/begleit-app-training.git"

# Define default environment variables (overridable at runtime)
BRANCH=${BRANCH:-"main"}
TRAIN=${TRAIN:-"train"}
REPO_URL=${REPO_URL:-"$DEFAULT_REPO"}

echo "=== Initializing Worker Node ==="
if [ "$BRANCH" != "main" ]; then
    echo "Target Branch: $BRANCH"
fi
echo "Target Config: config/$TRAIN.yml"
if [ "$REPO_URL" != "$DEFAULT_REPO" ]; then
    echo "Custom repository: $REPO_URL"
fi

# Clone the requested branch into a fresh workspace
rm -rf /runner/repo
git clone -b "$BRANCH" "$REPO_URL" /runner/repo
cd /runner/repo

# Pull the dataset
echo "Pulling dataset from DVC..."
python -m dvc pull

# Configure a directory that is mounted into container if its state is persistent (avoids re-downloading models)
export HF_HOME="/app/huggingface_cache"

# Hand off to the dynamic hardware launcher
echo "Executing dynamic hardware launcher..."
python src/launcher.py --config "config/${TRAIN}.yml"