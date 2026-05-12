# Use an official Axolotl compatible base image
FROM axolotlai/axolotl-cloud:main-20260512-py3.11-cu128-2.9.1

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

# Set working directory to avoid overwriting axolotl's /workspace
WORKDIR /app

# Copy project files
COPY data/ /app/data/
COPY src/ /app/src/
COPY tests/ /app/tests/
COPY config/ /app/config/
COPY pyproject.toml /app/

# Prepare data before training
RUN python src/prepare_data.py

# Install testing/data prep dependencies natively using uv into the active python env
RUN uv pip install --python $(which python) \
    datasets \
    pytest

# Create output directory
RUN mkdir -p /app/output

# Run unit tests to validate the environment during build
ENV IN_DOCKER=true
RUN python -m pytest tests/

# Set entrypoint to run axolotl
ENTRYPOINT ["accelerate", "launch", "-m", "axolotl.cli.train", "config/train.yml"]
