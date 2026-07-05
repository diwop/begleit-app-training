---
axolotl_image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
vllm_image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
---
# DiWop Begleit-App Training

Welcome to the `begleit-app-training` repository.

See [docs/implementation-details.md](docs/implementation-details.md) for technical information.

## Getting Started

Use the agentic workflow [`/onboarding`](.agent/workflows/onboarding.md) to get started.

See [docs/data.md](docs/data.md) how to add raw data and create the dataset for training.

See [docs/pipeline.md](docs/pipeline.md) how to fine-tune and evaluate models.