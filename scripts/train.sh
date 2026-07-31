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

# Named explicitly, never a bare `dvc pull`: that fetches every out in the pipeline,
# including data/eval/holdout.jsonl -- the one file this container must not have.
TRAIN_DATA=(data/train/dataset.jsonl data/train/validation.jsonl data/split_manifest.json)
if [ ! -f data/train/dataset.jsonl ] || [ ! -f data/train/validation.jsonl ]; then
    echo "==> training data missing, pulling via DVC"
    # The default remote is s3://diwop-analysis/dvc, so this needs AWS credentials that can
    # read that bucket -- not only the S3_BUCKET the adapters are published to.
    "$(dirname "$PY_TRAIN")/dvc" pull "${TRAIN_DATA[@]}" 2>/dev/null || \
        "$PY_TRAIN" -m dvc pull "${TRAIN_DATA[@]}"
fi

# The holdout score is only unbiased if this container never held the holdout. On a laptop
# the same checkout produced all three splits, so warn; in the container, refuse.
if [ -f data/eval/holdout.jsonl ]; then
    if [ "$IS_LOCAL" = "1" ]; then
        echo "⚠️  data/eval/holdout.jsonl exists locally. Axolotl never reads it, but on"
        echo "    RunPod its presence aborts the run."
    else
        echo "❌ data/eval/holdout.jsonl is present in the training container."
        echo "   That file is what the eval container scores. A training run that could"
        echo "   have seen it makes every number computed from it meaningless."
        exit 1
    fi
fi

# Axolotl caches the tokenised dataset under `dataset_prepared_path` and keys that cache on
# the dataset *config block*, not on the contents of the file it names. Growing
# data/train/dataset.jsonl from 8 documents to 533 therefore did not invalidate it: the run
# silently kept training on yesterday's three samples while reporting the new validation
# set, and the only visible symptom was `epoch: 20` at step 60. Key it on the data instead.
#
# A content hash, not mtime: `dvc checkout` rewrites these files and would otherwise force
# a pointless re-tokenisation of the whole corpus every time.
PREPARED_DIR="${PREPARED_DIR:-last_run_prepared}"
FINGERPRINT_FILE="$PREPARED_DIR/.data-fingerprint"
DATA_FINGERPRINT="$(cat data/train/dataset.jsonl data/train/validation.jsonl 2>/dev/null \
    | shasum -a 256 | cut -d' ' -f1)"
if [ -d "$PREPARED_DIR" ] && [ "$(cat "$FINGERPRINT_FILE" 2>/dev/null)" != "$DATA_FINGERPRINT" ]; then
    echo "==> training data changed since the last run; clearing $PREPARED_DIR"
    rm -rf "${PREPARED_DIR:?}"
fi
mkdir -p "$PREPARED_DIR"
echo "$DATA_FINGERPRINT" > "$FINGERPRINT_FILE"

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
# shellcheck source=lib/s3_sync.sh
source scripts/lib/s3_sync.sh
progress_start "$LOG_FILE"
# train.py publishes the finished artefacts once, at the end. This publishes the
# TensorBoard events and the log while the run is still going, so progress is visible
# without SSH and survives a pod that dies mid-run.
s3_sync_start "$OUTPUT_ROOT/adapter/$(basename "$TRAIN_CONFIG" .yml)" \
    "$(basename "$TRAIN_CONFIG" .yml)" "$LOG_FILE"
set +e
"${NICE[@]}" "$PY_TRAIN" -u src-train/train.py 2>&1 | tee "$LOG_FILE"
TRAIN_EXIT_CODE=${PIPESTATUS[0]}
set -e
s3_sync_stop
progress_stop

# shellcheck source=lib/finish.sh
source scripts/lib/finish.sh
finish "$TRAIN_EXIT_CODE" "$LOG_FILE" "training"
