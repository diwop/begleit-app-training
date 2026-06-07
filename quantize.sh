#!/bin/bash
set -e

cd /runner/repo/

# Install (optional) dependency delta
uv export --no-emit-project --format requirements-txt > requirements.txt
uv pip install -r requirements.txt

# dynamically bolt on bitsandbytes lib without triggering a global dependency resolution
uv pip install "bitsandbytes>=0.49.1" # to be in sync with transformers

python src/quantize.py