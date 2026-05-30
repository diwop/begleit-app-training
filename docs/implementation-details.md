# Implementation Details

- **Dataset Preparation & Validation:** The raw dataset consists of paired Markdown or text files under `data/raw/` (e.g., `<id>_Standardsprache` and `<id>_Leichte_Sprache`) tracked via DVC. A Python compiler (`src/prepare_dataset.py`) pairs these files, combines them using a global system prompt (`data/system-prompt.md`) and a user prompt template (`data/prompt-template.md`), and compiles the training dataset into `data/train/dataset.jsonl`. During compilation, it validates the token count of each processed text pair against the `sequence_len` limit configured in `config/base.yml`. It calculates and prints a token distribution histogram showing the 50%, 75%, 90%, and Max token counts for both the total sequence length and the assistant's responses separately, along with their respective averages. To ensure fast subsequent runs, the validation dynamically tries to load the tokenizer from local cache first before checking the Hugging Face Hub. If any sample exceeds the token limit, the data preparation script raises a warning and terminates with an error code, preventing invalid/oversized training data from entering the training pipeline. For local testing, S3 remote storage is mocked as a local folder (`data/s3-mock/`).
- **Testing:** Unit tests are configured via `pytest` to verify the execution, behavior, and config merging logic of the dynamic hardware launcher (`tests/test_launcher.py`). The dataset pipeline structure is validated via DVC status checks on PRs.
- **Training Configuration:** Baseline hyperparameters (such as LoRA adapter settings, dataset paths, and a default sequence length of 4096) are defined in `config/base.yml`. Model-specific configuration files (e.g., `config/train.yml` for Mixtral, `config/train-gemma4.yml` for Gemma, and `config/train-mistral4small.yml` for smaller variants) inherit from `base.yml` and only define model-specific overrides (like `base_model` and quantization settings).
- **Containerization:** The environment is containerized using the `axolotlai/axolotl-cloud:main-20250129-py3.11-cu121-2.3.1` base image. The code and data are decoupled from the image; `runner/entrypoint.sh` cleans the repository folder, clones the repository at runtime, performs a fast dependency sync using `uv` to install any new delta package requirements, pulls the dataset using DVC (`python -m dvc pull`), sets up the HF cache path (`HF_HOME`) to prevent re-downloading model weights across restarts, and hands off execution to the launcher. Python dependencies are dynamically compiled from `pyproject.toml`/`uv.lock` and pre-installed during the image build using `uv` to ensure consistency.
- **CI/CD:** GitHub Actions are configured to run Python tests and execute DVC checks (`validate-data` job using `dvc status` to verify that the compiled training dataset is up to date) on pull requests. Upon merges to `main`, the build workflow cleans runner disk space, builds the Docker image, executes an environment smoke test inside the container (ensuring PyTorch and Transformers compatibility), and pushes the production image to the GitHub Container Registry (GHCR). Dependabot is also configured to track updates.

## Technical Choices

- **Local Package Manager:** `uv` is used for fast and deterministic dependency resolution during local development and testing (via `uv.lock`). Inside the container, standard `pip` is used to pre-install heavy dependencies quickly.
- **Base Image:** We chose `axolotlai/axolotl-cloud:main-20250129-py3.11-cu121-2.3.1` to leverage an official Axolotl environment where flash-attention and deepspeed are already pre-compiled and configured for PyTorch and CUDA.
- **Dynamic Hardware Launcher:** `src/launcher.py` acts as a dynamic entry point. It resolves configuration properties by merging `config/base.yml` with the specified model config override using OmegaConf. It automatically detects the number of GPUs and their VRAM, adjusts `micro_batch_size` and `gradient_accumulation_steps` to prevent Out-Of-Memory errors, and enables DeepSpeed ZeRO-3 if multiple GPUs are present.
- **Decoupled Execution:** To iterate faster, the codebase, package dependencies, and dataset are not embedded in the container. The `ENTRYPOINT` clones the repository at runtime, installs any package delta updates, and pulls the dataset via DVC (`python -m dvc pull`), allowing code, dependency, and data changes to run immediately on platforms like RunPod without rebuilding the 10GB+ Docker image.

## Model Configurations & Hardware Scaling

### Supported Model Configurations
To reduce duplication across configurations, a base layout is defined in `config/base.yml` containing standard parameters (such as the default sequence length of 4096). Model-specific files only specify the differences:
- `config/base.yml`: The baseline Axolotl configuration containing shared parameters (sequence length, dataset configs, optimizer, etc.).
- `config/train.yml`: The default configuration loading `unsloth/Mixtral-8x7B-Instruct-v0.1-bnb-4bit` (4-bit quantized) and defining Mixtral-specific training variables.
- `config/train-gemma4.yml`: A configuration tailored to override the model for the Gemma architecture in 4-bit quantization.
- `config/train-mistral4small.yml`: A configuration tailored to override the model for smaller Mistral/Mixtral variants.

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
4. **Volume Mounts:** To ensure your fine-tuned LoRA adapters and Hugging Face model cache are saved permanently, mount a RunPod Network Volume (or a persistent pod volume) to `/app`. This ensures that even if you terminate the container, the downloaded model weights (stored in `/app/huggingface_cache`) and trained adapters (stored in `/app/output`) remain accessible.
5. **Execution:** Once the Pod is booted, the container's `ENTRYPOINT` will automatically clone the latest code, sync any new package dependency changes, prepare the data, and execute `launcher.py`. You can monitor the dynamic hardware scaling and the training progress via the RunPod Web Terminal or Container Logs.
