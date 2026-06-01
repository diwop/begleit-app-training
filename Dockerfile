# Pick base image with CUDA 12.1 which is optimized for HSUper A100 and L40S GPUs
FROM axolotlai/axolotl-cloud:main-20250129-py3.11-cu121-2.3.1

# Make Axolotl available globally
ENV PATH="/root/miniconda3/envs/py3.11/bin:$PATH"
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /runner

# Copy dependency definition files
COPY pyproject.toml uv.lock ./

# Pre-install heavy dependencies so the runtime install is fast
RUN uv export --no-emit-project --format requirements-txt > requirements.txt && \
    uv pip install --system --no-cache -r requirements.txt

COPY runner/entrypoint.sh /runner/entrypoint.sh
RUN chmod +x /runner/entrypoint.sh

CMD ["/bin/bash", "/runner/entrypoint.sh"]