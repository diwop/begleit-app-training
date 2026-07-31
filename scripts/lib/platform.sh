# shellcheck shell=bash
# Sourced, never executed. The ONE place that knows how the platforms differ.
#
# Everything else -- setup.sh, train.sh, eval.sh -- reads these values and contains no
# platform conditionals of its own. Adding a platform should mean editing this file only.
#
#   darwin-arm64  a laptop: local venvs under .local/, Metal for training, CPU for serving
#   linux-cuda    a container with the CUDA stack already baked in (RunPod or any host
#                 running the images named in README.md). We never provision torch /
#                 flash-attention / DeepSpeed ourselves -- that is the image's job.

case "$(uname -sm)" in
    "Darwin arm64") PLATFORM="darwin-arm64" ;;
    Linux*)         PLATFORM="linux-cuda" ;;
    *)              echo "❌ unsupported platform: $(uname -sm)" >&2; return 1 2>/dev/null || exit 1 ;;
esac
export PLATFORM

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO_ROOT

# `pixi`-free consistency guard. vLLM is named in two places that must agree:
#   README.md            eval_image:  vllm/vllm-openai:vX.Y.Z-...   (the container)
#   src-eval/pyproject.toml           # vllm-release: vX.Y.Z        (the source build)
# Comparing the declared release strings needs no network and no git, so it is cheap
# enough to run on every invocation. A mismatch means local and prod would silently serve
# different engines -- exactly the class of drift that cost us a day.
assert_vllm_pin_matches_image() {
    local image_tag pinned_tag
    image_tag="$(sed -n 's/^eval_image:.*vllm-openai:\(v[0-9][^-]*\).*/\1/p' "$REPO_ROOT/README.md" | head -1)"
    pinned_tag="$(sed -n 's/^# vllm-release: *\(v[0-9.]*\).*/\1/p' "$REPO_ROOT/src-eval/pyproject.toml" | head -1)"

    # Fail, do not warn. A guard that cannot read its inputs and still reports success is
    # worse than no guard -- it was silently passing while README still named the SGLang
    # image, which is exactly the drift it exists to catch.
    if [ -z "$image_tag" ] || [ -z "$pinned_tag" ]; then
        echo "❌ could not read both vLLM release markers:" >&2
        echo "     README.md            eval_image: vllm/vllm-openai:<tag>  -> '${image_tag:-MISSING}'" >&2
        echo "     src-eval/pyproject.toml  '# vllm-release: <tag>'         -> '${pinned_tag:-MISSING}'" >&2
        return 1
    fi
    if [ "$image_tag" != "$pinned_tag" ]; then
        echo "❌ vLLM version drift:" >&2
        echo "     README.md eval_image      -> $image_tag" >&2
        echo "     src-eval/pyproject.toml   -> $pinned_tag" >&2
        echo "   Local and prod would run different engines. Update both, and set the" >&2
        echo "   direct-reference SHA in src-eval/pyproject.toml to that release's commit." >&2
        return 1
    fi
}

if [ "$PLATFORM" = "darwin-arm64" ]; then
    IS_LOCAL=1
    LOCAL_DIR="${LOCAL_DIR:-$REPO_ROOT/.local}"
    PY_TRAIN="$LOCAL_DIR/venv-train/bin/python"
    PY_EVAL="$LOCAL_DIR/venv-eval/bin/python"

    # The 5.4 MB stand-in, not the 26B. See docs/local-pipeline.md.
    TRAIN_CONFIG="${TRAIN_CONFIG:-config/train-gemma4-tiny.yml}"

    OUTPUT_ROOT="${OUTPUT_ROOT:-$LOCAL_DIR/output}"
    export HF_HOME="${HF_HOME:-$LOCAL_DIR/huggingface}"

    # One CPU worker; the vocabulary, not the model, drives memory (docs/local-pipeline.md).
    TP_SIZE="${TP_SIZE:-1}"
    SMOKE_MAX_MODEL_LEN="${SMOKE_MAX_MODEL_LEN:-12288}"
    SMOKE_MAX_TOKENS="${SMOKE_MAX_TOKENS:-32}"
    export VLLM_CPU_KVCACHE_SPACE="${VLLM_CPU_KVCACHE_SPACE:-2}"

    # Leave the machine usable: someone is working on it.
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
    export PYTORCH_MPS_LOW_WATERMARK_RATIO="${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.3}"
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.5}"
    NICE=(nice -n 10)
else
    IS_LOCAL=0
    # The container owns the interpreter. Training runs inside the Axolotl image's venv;
    # eval runs inside the vLLM image, which has no separate venv.
    PY_TRAIN="${PY_TRAIN:-/workspace/axolotl-venv/bin/python}"
    [ -x "$PY_TRAIN" ] || PY_TRAIN="$(command -v python3)"
    PY_EVAL="${PY_EVAL:-$(command -v python3)}"

    TRAIN_CONFIG="${TRAIN_CONFIG:-config/train-gemma4.yml}"

    OUTPUT_ROOT="${OUTPUT_ROOT:-/app/output}"
    export HF_HOME="${HF_HOME:-/app/huggingface_cache}"

    # Ask the hardware, do not assume two cards. The pipeline has run on 1x RTX PRO 6000
    # (96 GB) as well as on 2x L40S, and eval.sh derives CUDA_VISIBLE_DEVICES from this --
    # a hardcoded 2 makes vLLM claim a second card that a single-GPU pod does not have.
    if [ -z "${TP_SIZE:-}" ]; then
        TP_SIZE="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
        [ "${TP_SIZE:-0}" -ge 1 ] 2>/dev/null || TP_SIZE=1
    fi
    SMOKE_MAX_MODEL_LEN="${SMOKE_MAX_MODEL_LEN:-8192}"
    SMOKE_MAX_TOKENS="${SMOKE_MAX_TOKENS:-512}"
    NICE=()
fi

# The model comes from the config, never from the platform. Hardcoding a per-platform
# default meant that pointing TRAIN_CONFIG at a different model trained one base and then
# served a completely different one -- vLLM died with
#   RuntimeError: The size of tensor a (8) must match the size of tensor b (1536)
# i.e. the tiny model's hidden size against another model's adapter. The config already
# declares base_model; that is the single source of truth.
if [ -z "${BASE_MODEL:-}" ]; then
    BASE_MODEL="$(sed -n 's/^base_model: *//p' "$REPO_ROOT/$TRAIN_CONFIG" | head -1 | tr -d '"'"'"' ')"
fi
if [ -z "$BASE_MODEL" ]; then
    BASE_MODEL="$(sed -n 's/^base_model: *//p' "$REPO_ROOT/config/base.yml" | head -1 | tr -d '"'"'"' ')"
fi
if [ -z "$BASE_MODEL" ]; then
    echo "❌ no base_model declared in $TRAIN_CONFIG or config/base.yml" >&2
    return 1 2>/dev/null || exit 1
fi
export BASE_MODEL

# RunPod is a property of the host, not of the platform: a plain Linux+CUDA box running the
# same image is linux-cuda but not RunPod, and must not try to call the shutdown API.
[ -n "${RUNPOD_POD_ID:-}" ] && IS_RUNPOD=1 || IS_RUNPOD=0

export IS_LOCAL IS_RUNPOD PY_TRAIN PY_EVAL TRAIN_CONFIG OUTPUT_ROOT
export TP_SIZE SMOKE_MAX_MODEL_LEN SMOKE_MAX_TOKENS
