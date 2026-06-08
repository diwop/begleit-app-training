# Optimized for L40S GPUs
FROM axolotlai/axolotl-cloud-uv:main-py3.12-cu130-2.10.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 1. PERMANENTLY activate Axolotl's pre-built master environment!
ENV VIRTUAL_ENV="/workspace/axolotl-venv"
ENV PATH="/workspace/axolotl-venv/bin:$PATH"

# --- 💾 CRITICAL IMAGE-LEVEL STORAGE PROTECTION ---
ENV HF_HOME="/app/huggingface_cache" \
    HF_HUB_CACHE="/app/huggingface_cache/hub" \
    HF_XET_CACHE="/app/huggingface_cache/xet" \
    HUGGINGFACE_HUB_CACHE="/app/huggingface_cache/hub" \
    TRANSFORMERS_CACHE="/app/huggingface_cache/hub" \
    XDG_CACHE_HOME="/app/xdg_cache" \
    UV_CACHE_DIR="/app/uv_cache" \
    TMPDIR="/app/tmp" \
    TMP="/app/tmp" \
    TEMP="/app/tmp"

WORKDIR /runner

COPY pyproject.toml ./

# Dynamically compile the pyproject.toml dependencies against the container's environment state.
# This ensures clearml, dvc, etc. are bolted on without touching or downgrading the core torch layers.
RUN --mount=type=cache,target=/app/uv_cache \
    uv pip compile pyproject.toml -o requirements.txt && \
    uv pip install -r requirements.txt

# Install vllm separately
RUN --mount=type=cache,target=/app/uv_cache \
    uv pip install vllm

COPY runner/entrypoint.sh /runner/entrypoint.sh
RUN chmod +x /runner/entrypoint.sh

# Always run the setup and start Jupyter lab
ENTRYPOINT ["/bin/bash", "/runner/entrypoint.sh"]

# Set the default action to execute the current script in the repository
CMD ["/bin/bash", "/runner/repo/train.sh"]