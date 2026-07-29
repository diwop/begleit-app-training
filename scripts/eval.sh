#!/bin/bash
set -e

# Runs on the SGLang image (eval_image in README.md), which already provides sglang and
# torch. Nothing here installs an inference engine: mixing vLLM into this image is what
# produced the transformers / huggingface_hub breakages in the July 5 runs.

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

echo "Installing evaluation dependencies..."
pip install --no-cache-dir --upgrade boto3

export TP_SIZE=${TP_SIZE:-2}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export TORCH_NCCL_BLOCKING_WAIT=1
export HF_HOME=${HF_HOME:-/app/huggingface_cache}

# /app is MooseFS, a network filesystem. Model weights read fine from it, but torch.compile
# writes thousands of small kernel files and fails there with OSError: [Errno 5]. launch.sh
# points TMPDIR at /app, which inductor inherits, so the cache dirs must be set explicitly
# back to the local container disk.
export TMPDIR=/tmp
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/root/.cache/inductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/root/.cache/triton}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/root/.cache/vllm}
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

# Compiling LoRA-specialised CUDA graphs at 51 batch sizes costs ~20 min and is pointless
# for a 3-prompt smoke test. Set SMOKE_EAGER=0 to exercise the compiled production path.
export SMOKE_EAGER=${SMOKE_EAGER:-1}

# 'auto' picks whichever engine the image provides. SMOKE_ADAPTER_S3 pins the S3 prefix:
# without it the newest '*_run/' wins, which has silently tested the wrong adapter before.
export SMOKE_ENGINE=${SMOKE_ENGINE:-auto}
export SMOKE_ADAPTER_S3=${SMOKE_ADAPTER_S3:-}
export SMOKE_BASE=${SMOKE_BASE:-google/gemma-4-26b-a4b-it}

# A failed vLLM run orphans its EngineCore, which holds the whole card (~75 GB) and makes
# every later run fail on memory instead of on whatever is actually being tested.
echo "Reclaiming GPU from any orphaned engine processes..."
pkill -9 -f 'VLLM::EngineCore' 2>/dev/null || true
pkill -9 -f 'smoke_adapter\.py' 2>/dev/null || true
pkill -9 -f 'sglang.*scheduler' 2>/dev/null || true
sleep 5

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "  free VRAM: ${FREE_MIB} MiB"
if [ "${FREE_MIB:-0}" -lt 40000 ]; then
    echo "[FATAL] Only ${FREE_MIB} MiB free; the model needs far more."
    echo "        Something outside this script holds the GPU. Check:"
    echo "          nvidia-smi ; ps aux | grep -E 'EngineCore|python'"
    exit 1
fi

echo "=== adapter smoke test ==="
echo "  engine : ${SMOKE_ENGINE}"
echo "  base   : ${SMOKE_BASE}"
echo "  adapter: ${SMOKE_ADAPTER_S3:-<newest *_run/ prefix>}"
# No engine source patches are installed: a failure here is an upstream capability gap
# and the traceback names it directly.
set +e
python3 -u src-eval/smoke_adapter.py 2>&1 | tee "$LOG_FILE"
EVAL_EXIT_CODE=${PIPESTATUS[0]}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ -n "${S3_BUCKET:-}" ]; then
    echo "S3_BUCKET is set to '${S3_BUCKET}'. Copying logs..."
    FILENAME="logs/${TIMESTAMP}_evaluation.log"
    python3 -u -c "import boto3; boto3.client('s3').upload_file('$LOG_FILE', '$S3_BUCKET', '$FILENAME')"

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
