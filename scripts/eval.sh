#!/bin/bash
set -e

# Navigate to the repository root relative to the script location
cd "$(dirname "$0")/.."

LOG_FILE="/app/evaluation_run.log"

# Isolate DVC in its own virtualenv to avoid system package conflicts
if [ ! -d "/tmp/dvc-venv" ]; then
    echo "Creating isolated DVC virtual environment in /tmp/dvc-venv..."
    python3 -m venv /tmp/dvc-venv
    /tmp/dvc-venv/bin/pip install --upgrade pip
    /tmp/dvc-venv/bin/pip install "dvc[s3]>=3.50.0"
fi

echo "Pulling dataset from DVC (isolated)..."
/tmp/dvc-venv/bin/python3 -m dvc pull

echo "Installing evaluation dependencies (system)..."
uv pip install --system --break-system-packages --upgrade "textstat>=0.7.13" boto3 "vllm==0.7.3" "transformers @ git+https://github.com/huggingface/transformers.git"

export TP_SIZE=${TP_SIZE:-2}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export HF_HOME=${HF_HOME:-/app/huggingface_cache}

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
