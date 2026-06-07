#!/bin/bash
set -e

cd /runner/repo/

# Install (optional) dependency delta
uv export --no-emit-project --format requirements-txt > requirements.txt
uv pip install -r requirements.txt

uv pip install autoawq

python src/quantize.py