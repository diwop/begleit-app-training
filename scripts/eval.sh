#!/bin/bash
set -e

cd /runner/repo/

LOG_FILE="/app/evaluation_run.log"

echo "Installing evaluation dependencies..."
uv pip install --system --break-system-packages textstat

echo "Running evaluation script..."
set +e
python -u src-eval/evaluation.py 2>&1 | tee "$LOG_FILE"
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
