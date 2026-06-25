#!/bin/bash
set -e

cd /runner/repo/


echo "Creating isolated sglang evaluation environment..."
uv venv /app/sglang-venv
export VIRTUAL_ENV="/app/sglang-venv"
export PATH="/app/sglang-venv/bin:$PATH"

echo "Installing evaluation dependencies..."
uv pip compile src-eval/pyproject.toml -o src-eval/requirements.txt
uv pip install -r src-eval/requirements.txt

echo "Running evaluation script..."
python -u src-eval/evaluation.py
