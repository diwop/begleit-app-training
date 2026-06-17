#!/bin/bash
set -e

cd /runner/repo/

# Check if both S3 variables are provided for standalone evaluation
if [ -n "${S3_BUCKET:-}" ] && [ -n "${S3_ADAPTER_RUN:-}" ]; then
    echo "Downloading adapters from s3://${S3_BUCKET}/${S3_ADAPTER_RUN}..."
    
    # We are still in the axolotl-venv here, which is fine for awscli
    uv pip install awscli 
    
    mkdir -p /app/output/adapter
    aws s3 sync "s3://${S3_BUCKET}/${S3_ADAPTER_RUN}" /app/output/adapter
    echo "✅ Adapter successfully downloaded to /app/output/adapter!"
    echo ""
fi

echo "Creating isolated sglang evaluation environment..."
uv venv /app/sglang-venv
export VIRTUAL_ENV="/app/sglang-venv"
export PATH="/app/sglang-venv/bin:$PATH"

echo "Installing evaluation dependencies..."
uv pip compile src-eval/pyproject.toml -o src-eval/requirements.txt
uv pip install -r src-eval/requirements.txt

echo "Running evaluation script..."
python -u src-eval/evaluation.py
