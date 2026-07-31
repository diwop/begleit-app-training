#!/bin/bash
# Train, on any supported platform. scripts/lib/platform.sh holds the differences.
#
#   darwin-arm64  Metal, the tiny stand-in config, artefacts under .local/
#   linux-cuda    the container's CUDA stack, DeepSpeed ZeRO-3, artefacts under /app
#
# src-train/train.py detects the accelerator itself and drops DeepSpeed and
# FlashAttention-2 when there is no CUDA, so the command below is the same everywhere.
set -e

cd "$(dirname "$0")/.."
# shellcheck source=lib/platform.sh
source scripts/lib/platform.sh

bash scripts/setup.sh

LOG_FILE="${LOG_FILE:-$OUTPUT_ROOT/training_run.log}"
mkdir -p "$(dirname "$LOG_FILE")"

if [ ! -f data/train/dataset.jsonl ]; then
    echo "==> dataset missing, pulling via DVC"
    # The default remote is the local data/s3-mock directory, so this needs no AWS
    # credentials -- see .dvc/config.
    "$(dirname "$PY_TRAIN")/dvc" pull 2>/dev/null || "$PY_TRAIN" -m dvc pull
fi

if [ "$IS_LOCAL" = "0" ]; then
    # Diagnostics that only mean anything on a real cluster.
    export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
    export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
fi

echo "==> training ($PLATFORM, config=$TRAIN_CONFIG)"
# src-train/train.py pre-stages the base weights before the first step, which is the same
# silent multi-GB wait the eval has.
# shellcheck source=lib/progress.sh
source scripts/lib/progress.sh
progress_start "$LOG_FILE"
set +e
"${NICE[@]}" "$PY_TRAIN" -u src-train/train.py 2>&1 | tee "$LOG_FILE"
TRAIN_EXIT_CODE=${PIPESTATUS[0]}
set -e
progress_stop

# shellcheck source=lib/finish.sh
source scripts/lib/finish.sh
finish "$TRAIN_EXIT_CODE" "$LOG_FILE" "training"
