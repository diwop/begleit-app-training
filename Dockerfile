# Optimized for L40S GPUs
FROM axolotlai/axolotl-uv:main-py3.11-cu128-2.9.1

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 1. PERMANENTLY activate Axolotl's pre-built master environment!
ENV VIRTUAL_ENV="/workspace/axolotl-venv"
ENV PATH="/workspace/axolotl-venv/bin:$PATH"

WORKDIR /runner

# Copy dependency definition files
COPY pyproject.toml uv.lock ./

# Use export + pip install to SAFELY BOLT ON packages without deleting torch
RUN uv export --no-emit-project --format requirements-txt > requirements.txt && \
    uv pip install -r requirements.txt

COPY runner/entrypoint.sh /runner/entrypoint.sh
RUN chmod +x /runner/entrypoint.sh

# Always run the set up and start Jupyter lab
ENTRYPOINT ["/bin/bash", "/runner/entrypoint.sh"]

# Set the default action to execute the current script in the repository
CMD ["/bin/bash", "/runner/repo/train.sh"]