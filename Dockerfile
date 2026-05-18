# Pick base image with CUDA 12.1 which is optimized for HSUper A100 and L40S GPUs
FROM axolotlai/axolotl-cloud:main-20250129-py3.11-cu121-2.3.1

# Make Axolotl available globally
ENV PATH="/root/miniconda3/envs/py3.11/bin:$PATH"
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Pre-install heavy dependencies so the runtime install is fast
RUN pip install --no-cache-dir clearml pytest datasets

WORKDIR /runner

COPY runner/entrypoint.sh /runner/entrypoint.sh
RUN chmod +x /runner/entrypoint.sh

ENTRYPOINT ["/runner/entrypoint.sh"]