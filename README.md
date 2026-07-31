---
train_image: axolotlai/axolotl-cloud-uv:main-py3.12-cu130-2.10.0
eval_image: vllm/vllm-openai:v0.26.0-cu129-ubuntu2404
---
# DiWop Begleit-App Training

Welcome to the `begleit-app-training` repository.

See [docs/implementation-details.md](docs/implementation-details.md) for technical information.

## Getting Started

Use the agentic workflow [`/onboarding`](.agent/workflows/onboarding.md) to get started.

See [docs/data.md](docs/data.md) how to add raw data and create the dataset for training.

See [docs/pipeline.md](docs/pipeline.md) how to fine-tune and evaluate models.

See [docs/local-pipeline.md](docs/local-pipeline.md) to run the whole train → adapter →
serve handoff locally on Apple Silicon in about a minute, against a 5.4 MB stand-in model
that reproduces Gemma 4's MoE, vision-tower and missing-`v_proj` quirks.

> When bumping `eval_image` above, bump the `# vllm-release:` marker and pinned SHA in `src-eval/pyproject.toml` to the same
> release so the local build keeps matching the container. It currently pins `v0.26.0` (`568afb3a`), and
> `scripts/lib/platform.sh` refuses to run if the two drift apart.