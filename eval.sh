#!/bin/bash
set -e

echo "🚀 INITIATING EVALUATION-ONLY MODE"

cd /runner/repo/

# Check if both S3 variables are provided
if [ -n "${S3_BUCKET:-}" ] && [ -n "${S3_ADAPTER:-}" ]; then
    echo "📥 S3_BUCKET and S3_ADAPTER detected!"
    echo "Downloading adapter from s3://${S3_BUCKET}/${S3_ADAPTER}..."

    uv add awscli # make sure aws cli is available

    # Ensure the target directory exists before syncing
    mkdir -p /app/output/adapter
    
    # Sync the remote S3 folder directly into the local evaluation directory
    aws s3 sync "s3://${S3_BUCKET}/${S3_ADAPTER}" /app/output/adapter
    
    echo "✅ Adapter successfully downloaded to /app/output/adapter!"
    echo ""
else
    echo "If a valid adapter is found, the training phase will be automatically bypassed."
    echo ""

fi

# Set the environment variable for this execution
export EVAL=true

# Trigger the main pipeline script, passing along any arguments
bash ./train.sh "$@"
