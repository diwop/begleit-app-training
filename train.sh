#!/bin/bash
set -e

cd /runner/repo/
uv sync # Install (optional) dependency delta

TRAIN=${TRAIN:-"train"}

export HF_HOME="/workspace/huggingface_cache"
LOG_FILE="/workspace/training_run.log"


echo "=== Repository Execution Started ==="
echo "Target Config: config/${TRAIN}.yml"
echo "Logs will be saved to: $LOG_FILE"

echo "Pulling dataset from DVC..."
python -m dvc pull

echo "Executing dynamic hardware launcher..."

# Temporarily disable 'set -e' so a crash doesn't kill the script
set +e

# Use 'tee' to print logs to the screen AND save them to the persistent disk.
# 2>&1 captures both standard output and error messages
# -u enforces unbuffered output by python
python -u src/launcher.py --config "config/${TRAIN}.yml" 2>&1 | tee "$LOG_FILE"

TRAIN_EXIT_CODE=${PIPESTATUS[0]} # Gets the exit code of python, not tee!

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ -n "${S3_BUCKET:-}" ]; then
    echo "S3_BUCKET is set to '${S3_BUCKET}'. Copying logs..."
    aws s3 cp "$LOG_FILE" "s3://${S3_BUCKET}/${TIMESTAMP}_training_run.log"

    if [ $? -eq 0 ]; then
        echo "=== S3 Copy Successful! ==="
    else
        echo "=== WARNING: S3 Copy Failed! ==="
        sleep 60 # Keep the pod alive for log download
    fi
fi

set -e

# Handle lifecycle, S3 sync & (optional) RunPod shutdown

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "[FATAL] Training failed with exit code $TRAIN_EXIT_CODE."
    sleep 60 # Keep the pod alive for log download
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
