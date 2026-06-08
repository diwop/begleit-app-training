# Use your optimized foundational L40S + CUDA 13 + PyTorch 2.10 image
FROM axolotlai/axolotl-cloud-uv:main-py3.12-cu130-2.10.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Permanently activate Axolotl's pre-built master environment
ENV VIRTUAL_ENV="/workspace/axolotl-venv"
ENV PATH="/workspace/axolotl-venv/bin:$PATH"

# Force all data, cache, and temporary file systems onto your high-capacity volume mount at /app
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
# --------------------------------------------------

WORKDIR /runner

# Copy dependency definition files
COPY pyproject.toml uv.lock ./

# Use export + pip install to SAFELY BOLT ON evaluation packages without deleting torch layers
# Utilizing cache mounts prevents re-downloading identical wheels during iterative image builds
RUN --mount=type=cache,target=/app/uv_cache \
    uv export --no-emit-project --format requirements-txt > requirements.txt && \
    uv pip install -r requirements.txt

COPY runner/entrypoint.sh /runner/entrypoint.sh
RUN chmod +x /runner/entrypoint.sh

# Always run the setup and start Jupyter lab
ENTRYPOINT ["/bin/bash", "/runner/entrypoint.sh"]

# Set the default action to execute your model evaluation pipeline sequence
CMD ["/bin/bash", "/runner/repo/train.sh"]