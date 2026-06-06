# Optimized for L40S GPUs
FROM axolotlai/axolotl-uv:main-py3.11-cu128-2.9.1

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Permanently activate the environment for all subsequent commands and scripts
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

WORKDIR /runner

# Copy dependency definition files
COPY pyproject.toml uv.lock ./

# Create the transparent environment and pre-install the heavy dependencies
# The --system-site-packages flag allows pass-through to the base PyTorch installation.
RUN uv venv --system-site-packages /opt/venv && \
    uv sync --active

COPY runner/entrypoint.sh /runner/entrypoint.sh
RUN chmod +x /runner/entrypoint.sh

# Always run the set up and start Jupyter lab
ENTRYPOINT ["/bin/bash", "/runner/entrypoint.sh"]

CMD ["/bin/bash", "/runner/repo/train.sh"]