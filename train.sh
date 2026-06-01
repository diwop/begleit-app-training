#!/bin/bash
set -e

TRAIN=${TRAIN:-"train"}

export HF_HOME="/workspace/huggingface_cache"
LOG_FILE="/workspace/training_run.log"

echo "=== Repository Execution Started ==="
echo "Target Config: config/${TRAIN}.yml"
echo "Logs will be saved to: $LOG_FILE"

# Install (optional) delta between docker image and current state
echo "Syncing package dependencies..."
uv export --no-emit-project --format requirements-txt > requirements.txt
uv pip install --system -r requirements.txt

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
set -e

# Handle lifecycle & (optional) RunPod shutdown

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "[FATAL] Training failed with exit code $TRAIN_EXIT_CODE."
fi

if [ -n "$RUNPOD_POD_ID" ]; then
    echo "RunPod environment detected. Shutting down pod to save costs..."
    curl -s --request POST "https://api.runpod.io/graphql" \
    --header "Authorization: Bearer $RUNPOD_API_KEY" \
    --header "Content-Type: application/json" \
    --data "{\"query\": \"mutation { podStop(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) { id } }\"}"
else
    echo "Other environment detected. Keeping container alive."
    sleep infinity
fi
