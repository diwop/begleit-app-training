# Implementation Details

- **Dataset Preparation:** The raw dataset consists of paired Markdown or text files under `data/raw/` (e.g., `<id>_Standardsprache` and `<id>_Leichte_Sprache`) tracked via DVC. A Python compiler (`src/prepare_dataset.py`) pairs these files, combines them using a global system prompt (`data/system-prompt.md`) and a user prompt template (`data/prompt-template.md`), and compiles the training dataset into `data/train/dataset.jsonl`. For local testing, S3 remote storage is mocked as a local folder (`data/s3-mock/`).
- **Testing:** Unit tests are configured via `pytest` to verify the execution and behavior of the dynamic hardware launcher (`tests/test_launcher.py`). The dataset pipeline structure is validated via DVC status checks on PRs.
- **Training Configuration:** `config/train.yml` utilizes Axolotl to load `unsloth/Mixtral-8x7B-Instruct-v0.1-bnb-4bit` in 4-bit quantization and configure a QDoRA/QLoRA adapter. It targets the compiled dataset `data/train/dataset.jsonl`. Checkpoints and final adapters are saved to `/app/output`.
- **Containerization:** The environment is containerized using the `axolotlai/axolotl-cloud:main-20250129-py3.11-cu121-2.3.1` base image. The code and data are decoupled from the image; instead, `runner/entrypoint.sh` clones the repository at runtime, pulls the dataset using `dvc pull` (with `dvc` pre-installed in the Docker image), and hands off execution to the launcher.
- **CI/CD:** GitHub Actions are configured to run Python tests, execute DVC checks (`validate-data` job using `dvc status` to verify that the compiled training dataset is up to date), run Docker build dry-runs on pull requests, and push the production image to the GitHub Container Registry (GHCR) upon merges to `main`. Dependabot is also configured to track updates.

## Technical Choices

- **Local Package Manager:** `uv` is used for fast and deterministic dependency resolution during local development and testing (via `uv.lock`). Inside the container, standard `pip` is used to pre-install heavy dependencies quickly.
- **Base Image:** We chose `axolotlai/axolotl-cloud:main-20250129-py3.11-cu121-2.3.1` to leverage an official Axolotl environment where flash-attention and deepspeed are already pre-compiled and configured for PyTorch and CUDA.
- **Dynamic Hardware Launcher:** `src/launcher.py` acts as a dynamic entry point. It automatically detects the number of GPUs and their VRAM, adjusts `micro_batch_size` and `gradient_accumulation_steps` to prevent Out-Of-Memory errors, and enables DeepSpeed ZeRO-3 if multiple GPUs are present.
- **Decoupled Execution:** To iterate faster, the codebase and dataset are not embedded in the container. The `ENTRYPOINT` clones the repository at runtime and pulls the dataset via DVC (`dvc pull`), allowing code and data changes to run immediately on platforms like RunPod without rebuilding the 10GB+ Docker image.

## Model Configurations & Hardware Scaling

### Supported Model Configurations
Beyond the default `train.yml` (which uses Mixtral-8x7B), there are alternative configuration files available for different architectures to suit varying hardware footprints or experiment needs:
- `config/train.yml`: The default configuration loading `unsloth/Mixtral-8x7B-Instruct-v0.1-bnb-4bit` (4-bit quantized).
- `config/train-gemma4.yml`: A configuration tailored for the Gemma architecture in 4-bit quantization.
- `config/train-mistral4small.yml`: A configuration designed for smaller Mistral/Mixtral variants, offering a more lightweight footprint.

You can select a different model by passing the `TRAIN` environment variable in RunPod (e.g., `TRAIN=train-gemma4`).

### Dynamic Reconfiguration (launcher.py)
To ensure the pipeline remains resilient across various cloud GPU instances (e.g., swapping between 24GB RTX 3090s/4090s and 80GB A100s), `src/launcher.py` applies dynamic hardware scaling prior to execution:
- **VRAM Scaling:** It detects the total memory of the primary GPU. If the VRAM is below 30GB, it safely overrides execution with `micro_batch_size: 1` and `gradient_accumulation_steps: 8` to prevent Out-Of-Memory (OOM) errors. For GPUs with >30GB VRAM, it optimizes throughput using `micro_batch_size: 4` and `gradient_accumulation_steps: 2`.
- **Multi-GPU Parallelism:** It automatically detects the number of available CUDA GPUs. If more than one GPU is present, it injects DeepSpeed ZeRO-3 optimization (`--deepspeed config/zero3.json`) via `accelerate launch`, enabling efficient memory sharding across devices. For single GPUs, it runs natively in PyTorch to avoid DeepSpeed overhead.
- **Memory Fragmentation Fix:** It sets the `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` environment variable globally to mitigate memory fragmentation issues common during long-running LLM fine-tuning.

## Running on RunPod

Because the fine-tuning pipeline is completely containerized, it can be deployed seamlessly to Cloud GPU providers like RunPod.

### Step-by-Step Guide for RunPod

1. **Choose an Instance:** Navigate to your RunPod console and click on **Deploy > Deploy from Custom Image**. You will need a GPU with sufficient VRAM. For a 4-bit quantized Mixtral 8x7B model, an A100 (80GB) is highly recommended.
2. **Container Image:** Enter the GitHub Container Registry image path. This automatically pulls the latest image built by the CI/CD pipeline:
   ```text
   ghcr.io/diwop/begleit-app-training:main
   ```
   > **Note:** By default, new packages on GitHub Container Registry might be private. Make sure to change the package visibility to "Public" in the GitHub Package settings, or configure RunPod with your GitHub Container Registry credentials.
3. **Container Arguments (Optional):** By default, the script fine-tunes using `config/train.yml`. You can pass overrides via Environment Variables in RunPod (e.g., `BRANCH` to run a specific branch or `TRAIN` to change the config file).
4. **Volume Mounts:** To ensure your fine-tuned LoRA adapters are saved permanently, mount a RunPod Network Volume (or a persistent pod volume) to `/app/output`. This ensures that even if you terminate the container, the trained adapters remain accessible.
5. **Execution:** Once the Pod is booted, the container's `ENTRYPOINT` will automatically clone the latest code, prepare the data, and execute `launcher.py`. You can monitor the dynamic hardware scaling and the training progress via the RunPod Web Terminal or Container Logs.
