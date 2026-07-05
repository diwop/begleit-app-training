#!/bin/bash
set -e

# Navigate to the repository root relative to the script location
cd "$(dirname "$0")/.."

LOG_FILE="/app/evaluation_run.log"

# Create and activate a temporary virtualenv to avoid system package conflicts
if [ ! -d "/tmp/eval-venv" ]; then
    echo "Creating temporary virtual environment in /tmp/eval-venv..."
    python3 -m venv /tmp/eval-venv
fi
echo "Activating /tmp/eval-venv..."
source /tmp/eval-venv/bin/activate

echo "Installing evaluation dependencies..."
uv pip install --upgrade textstat boto3 "dvc[s3]"

export TP_SIZE=${TP_SIZE:-2}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export HF_HOME=${HF_HOME:-/app/huggingface_cache}

echo "Pulling dataset from DVC..."
python3 -m dvc pull

echo "Running evaluation script..."
set +e
python3 -u src-eval/evaluation.py 2>&1 | tee "$LOG_FILE"
EVAL_EXIT_CODE=${PIPESTATUS[0]}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ -n "${S3_BUCKET:-}" ]; then
    echo "S3_BUCKET is set to '${S3_BUCKET}'. Copying logs..."
    FILENAME="logs/${TIMESTAMP}_evaluation.log"
    python -u -c "import boto3; boto3.client('s3').upload_file('$LOG_FILE', '$S3_BUCKET', '$FILENAME')"

    if [ $? -eq 0 ]; then
        echo "Logs copied to S3 as $FILENAME."
    else
        echo "WARNING: Could not copy logs to S3."
        sleep 60
    fi
fi

set -e

# Handle lifecycle & (optional) RunPod shutdown
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully!"
else
    echo "[FATAL] Evaluation failed with exit code $EVAL_EXIT_CODE."
    sleep 60
fi

if [ "${KEEP_ALIVE:-false}" = "true" ]; then
    echo "KEEP_ALIVE flag is active. Bypassing RunPod shutdown API."
    echo "Pipeline complete. Returning control to terminal."
elif [ -n "$RUNPOD_POD_ID" ]; then
    echo "RunPod environment detected. Shutting down pod to save costs..."
    curl -s --request POST "https://api.runpod.io/graphql" \
    --header "Authorization: Bearer $RUNPOD_API_KEY" \
    --header "Content-Type: application/json" \
    --data "{\"query\": \"mutation { podStop(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) { id } }\"}"
else
    echo "Other environment detected. Keeping container alive."
    sleep infinity
fi
