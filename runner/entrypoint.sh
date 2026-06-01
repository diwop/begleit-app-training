#!/bin/bash
set -e

DEFAULT_REPO="https://github.com/diwop/begleit-app-training.git"

# Define default environment variables
BRANCH=${BRANCH:-"main"}
REPO_URL=${REPO_URL:-"$DEFAULT_REPO"}

echo "=== Initializing Worker Node ==="
echo "Cloning branch '$BRANCH' from $REPO_URL..."

rm -rf /runner/repo
git clone -b "$BRANCH" "$REPO_URL" /runner/repo
cd /runner/repo

echo "Handing off execution to repository logic..."

chmod +x train.sh

exec bash ./train.sh
