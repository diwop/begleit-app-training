# Use an official Axolotl compatible base image
FROM winglian/axolotl:main-py3.10-cu121-2.3.1

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

# Set working directory
WORKDIR /workspace

# Copy project files
COPY data/ /workspace/data/
COPY src/ /workspace/src/
COPY tests/ /workspace/tests/
COPY config/ /workspace/config/
COPY pyproject.toml /workspace/

# Prepare data before training
RUN python src/prepare_data.py

# Install testing/data prep dependencies natively using uv
RUN uv pip install --system \
    datasets \
    pytest

# Create output directory
RUN mkdir -p /workspace/output

# Run unit tests to validate the environment during build
ENV IN_DOCKER=true
RUN python -m pytest tests/

# Set entrypoint to run axolotl
ENTRYPOINT ["accelerate", "launch", "-m", "axolotl.cli.train", "config/train.yml"]
