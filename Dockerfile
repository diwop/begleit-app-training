# Use an Unsloth compatible base image
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install required system packages
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

# Set working directory
WORKDIR /workspace

# Copy project files
COPY data/ /workspace/data/
COPY src/ /workspace/src/
COPY tests/ /workspace/tests/
COPY pyproject.toml /workspace/

# Install dependencies using uv
RUN uv pip install --system \
    "unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git" \
    transformers \
    trl \
    peft \
    datasets \
    pytest \
    bitsandbytes

# Create output directory
RUN mkdir -p /workspace/output

# Set entrypoint
ENTRYPOINT ["python", "src/train.py"]
