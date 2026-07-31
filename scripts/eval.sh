#!/bin/bash
# Serve a trained adapter and check it actually changes the output, on any platform.
# scripts/lib/platform.sh holds the differences; src-eval/smoke_adapter.py is identical
# everywhere and reads its settings from the environment.
set -e

cd "$(dirname "$0")/.."
# shellcheck source=lib/platform.sh
source scripts/lib/platform.sh

bash scripts/setup.sh

LOG_FILE="${LOG_FILE:-$OUTPUT_ROOT/evaluation_run.log}"
mkdir -p "$(dirname "$LOG_FILE")"

if [ ! -f data/train/dataset.jsonl ]; then
    echo "==> dataset missing, pulling via DVC"
    # Default remote is the local data/s3-mock directory, so no AWS credentials needed.
    # src-eval declares dvc[s3], so this should succeed -- but it stays non-fatal on
    # purpose: smoke_adapter.py already drops the training-sample case when the dataset is
    # absent and still exercises the adapter, so a failed pull is not worth losing a
    # started pod over. It cost exactly that once.
    "$(dirname "$PY_TRAIN")/dvc" pull 2>/dev/null || "$PY_TRAIN" -m dvc pull || \
        echo "⚠️  no DVC on this image -- continuing without the training sample."
fi

export SMOKE_ENGINE="${SMOKE_ENGINE:-vllm}"
export SMOKE_BASE="${SMOKE_BASE:-$BASE_MODEL}"
export SMOKE_ADAPTER="${SMOKE_ADAPTER:-$OUTPUT_ROOT/adapter/$(basename "$TRAIN_CONFIG" .yml)}"
export SMOKE_ADAPTER_S3="${SMOKE_ADAPTER_S3:-}"
# Eager skips torch.compile and CUDA graph capture: ~20 min saved, irrelevant for 3 prompts.
export SMOKE_EAGER="${SMOKE_EAGER:-1}"
export TP_SIZE SMOKE_MAX_MODEL_LEN SMOKE_MAX_TOKENS

if [ "$IS_LOCAL" = "0" ]; then
    # Derived from TP_SIZE, not hardcoded to "0,1": with TP_SIZE=1 the old default exposed
    # a second card the engine never used, which made free-VRAM checks and any co-tenant
    # accounting misleading.
    # `seq | paste`, not `seq -s,`: the latter leaves a trailing comma.
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq 0 $((TP_SIZE - 1)) | paste -sd, -)}"
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    export TORCH_NCCL_BLOCKING_WAIT=1

    # /app is MooseFS, a network filesystem. torch.compile writes thousands of small kernel
    # files and fails there with OSError: [Errno 5]. launch.sh points TMPDIR at /app, which
    # inductor inherits, so the cache dirs must be forced back to local disk.
    export TMPDIR=/tmp
    export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/root/.cache/inductor}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/.cache/triton}"
    export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/root/.cache/vllm}"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

    # A failed vLLM run orphans its EngineCore, which holds the whole card and makes every
    # later run fail on memory instead of on whatever is being tested. Match on
    # 'VLLM::EngineCore', not 'vllm' -- the latter matches the shell's own cwd.
    echo "Reclaiming GPU from any orphaned engine processes..."
    pkill -9 -f 'VLLM::EngineCore' 2>/dev/null || true
    pkill -9 -f 'smoke_adapter\.py' 2>/dev/null || true
    sleep 5

    FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    echo "  free VRAM: ${FREE_MIB} MiB"
    if [ "${FREE_MIB:-0}" -lt 40000 ]; then
        echo "[FATAL] Only ${FREE_MIB} MiB free; the model needs far more."
        echo "        Check: nvidia-smi ; ps aux | grep -E 'EngineCore|python'"
        exit 1
    fi
fi

echo "==> adapter smoke test ($PLATFORM)"
echo "  base   : ${SMOKE_BASE}"
echo "  adapter: ${SMOKE_ADAPTER}"
# shellcheck source=lib/progress.sh
source scripts/lib/progress.sh
progress_start "$LOG_FILE"
set +e
"${NICE[@]}" "$PY_EVAL" -u src-eval/smoke_adapter.py 2>&1 | tee "$LOG_FILE"
EVAL_EXIT_CODE=${PIPESTATUS[0]}
set -e
progress_stop

# shellcheck source=lib/finish.sh
source scripts/lib/finish.sh
finish "$EVAL_EXIT_CODE" "$LOG_FILE" "evaluation"
