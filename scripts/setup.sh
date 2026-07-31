#!/bin/bash
# Prepare an environment to train or evaluate in. Same entry point on every platform;
# scripts/lib/platform.sh holds the differences.
#
#   darwin-arm64  builds two local venvs under .local/ from the pyproject files, including
#                 the pinned vLLM source build (no macOS wheel exists).
#   linux-cuda    the container image already provides torch, CUDA, flash-attention and
#                 DeepSpeed. We only add the thin delta this repo needs. We deliberately do
#                 NOT provision the heavy stack -- see docs/local-pipeline.md for why that
#                 was tried and rejected.
#
# Idempotent: every step is skipped when already satisfied, so re-running costs seconds.
set -e

cd "$(dirname "$0")/.."
# shellcheck source=lib/platform.sh
source scripts/lib/platform.sh

# Refuse to run if local and prod would serve different vLLM builds.
assert_vllm_pin_matches_image

command -v uv >/dev/null || { echo "❌ uv not found: https://docs.astral.sh/uv/"; exit 1; }

if [ "$IS_LOCAL" = "1" ]; then
    xcode-select -p >/dev/null 2>&1 || { echo "❌ Xcode Command Line Tools missing: xcode-select --install"; exit 1; }
    command -v ccache >/dev/null || echo "ℹ️  ccache not installed; 'brew install ccache' makes rebuilds much faster."

    # The only two settings pyproject.toml cannot carry: both are build-time environment
    # variables read by vLLM's setup.py.
    #   VLLM_TARGET_DEVICE - vLLM has no GPU backend on macOS; serving runs on CPU.
    #   MAX_JOBS           - vLLM reads this, NOT CMAKE_BUILD_PARALLEL_LEVEL, and would
    #                        otherwise use every core.
    export VLLM_TARGET_DEVICE=cpu
    export MAX_JOBS="${MAX_JOBS:-6}"

    # Both, not just axolotl: the two are installed by separate commands below, and a venv
    # with axolotl but without this repo's own dependencies passed the old check happily.
    # src-train/validation_metrics.py then failed at the first evaluation on a missing
    # textstat -- an hour into a run, which is a bad place to find out.
    if "$PY_TRAIN" -c "import axolotl, textstat" 2>/dev/null; then
        echo "==> [1/2] training venv already good, skipping"
    else
        echo "==> [1/2] training venv (Axolotl on Metal)"
        # Guarded like venv-eval below: `uv venv` refuses to touch a directory that already
        # exists, so an install that died half way left a venv that could never be repaired
        # -- the next run failed on the venv creation instead of retrying the install.
        [ -d "$LOCAL_DIR/venv-train" ] || uv venv --python 3.12 "$LOCAL_DIR/venv-train"
        # The one dependency that cannot live in src-train/pyproject.toml: axolotl pins
        # antlr4-python3-runtime==4.13.2, which contradicts the hydra-core that dvc[s3]
        # needs. `uv pip install` resolves per-platform and installs both fine; the
        # UNIVERSAL lock that `uv run --project` builds (dvc.yaml, CI) cannot. Declaring it
        # there breaks dataset prep, so it is installed here instead. Everything else --
        # including the pinned vLLM build -- stays declarative.
        # TWO commands, and in this order. One command asks uv to satisfy both at once,
        # which is unsatisfiable: axolotl pins antlr4-python3-runtime==4.13.2 and the
        # hydra-core that dvc[s3] needs contradicts it, so uv walks the whole axolotl
        # version history and gives up. Installing sequentially lets axolotl's pins land
        # last and win, which is what we want -- it owns the training stack. Doing it the
        # other way round (axolotl first) leaves a venv where importing axolotl dies inside
        # kernels/huggingface_hub.
        VIRTUAL_ENV="$LOCAL_DIR/venv-train" uv pip install "src-train/[local]"
        VIRTUAL_ENV="$LOCAL_DIR/venv-train" uv pip install axolotl
        # Then repair the three pins axolotl's install walks over. Each of these was an
        # import error at run time, not a warning:
        #
        #   antlr4  axolotl requires ==4.13.2, but omegaconf 2.3.1 requires 4.9.* and its
        #           grammar lexer is GENERATED against 4.9. With 4.13.2 installed, merely
        #           importing omegaconf raises "Could not deserialize ATN with version 3
        #           (expected 4)" -- which takes down src-train/train.py and axolotl too,
        #           since both import it. Nothing here calls antlr directly.
        #   boto3   axolotl pins botocore back below the boto3 that dvc[s3] and awscli
        #           brought, and boto3 then cannot import DocumentModifiedShape from it.
        #   torchvision  needed by Gemma4Processor, and ABI-bound to the torch it was
        #           built against. Against axolotl's torch it fails with "operator
        #           torchvision::nms does not exist"; upgrading realigns the pair.
        VIRTUAL_ENV="$LOCAL_DIR/venv-train" uv pip install --upgrade \
            "antlr4-python3-runtime==4.9.3" boto3 botocore awscli torchvision
    fi

    if "$PY_EVAL" -c "import vllm" 2>/dev/null; then
        echo "==> [2/2] eval venv already has vLLM, skipping the build"
    else
        echo "==> [2/2] eval venv (vLLM CPU backend, built from source -- the slow part)"
        [ -d "$LOCAL_DIR/venv-eval" ] || uv venv --python 3.12 "$LOCAL_DIR/venv-eval"
        export VIRTUAL_ENV="$LOCAL_DIR/venv-eval"
        # Build deps first: vLLM builds with isolation disabled (see [tool.uv] in
        # src-eval/pyproject.toml), so torch must already be importable.
        uv pip install "src-eval/[build]"
        uv pip install "src-eval/[local]"
    fi
else
    echo "==> installing the repo's dependencies on top of the container image"
    # No venv: the image owns the interpreter, and its prebuilt CUDA stack is the whole
    # reason we use it. `src-train/` without the [local] extra skips everything the image
    # already provides.
    if [ -x "/workspace/axolotl-venv/bin/python" ]; then
        VIRTUAL_ENV=/workspace/axolotl-venv uv pip install src-train/
    else
        # vLLM image: eval only. It ships neither boto3 nor DVC.
        uv pip install --system --no-cache src-eval/ || uv pip install --system --break-system-packages src-eval/
    fi
fi

echo
echo "✅ Setup complete ($PLATFORM)."
