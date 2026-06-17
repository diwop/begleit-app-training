# Implementation Details

- **Dataset Preparation & Validation:** The raw dataset consists of paired Markdown or text files under `data/raw/` (e.g., `<id>_Standardsprache` and `<id>_Leichte_Sprache`) tracked via DVC. A Python compiler (`src/prepare_dataset.py`) pairs these files, combines them using a global system prompt (`data/system-prompt.md`) and a user prompt template (`data/prompt-template.md`), and compiles the training dataset into `data/train/dataset.jsonl`. During compilation, it validates the token count of each processed text pair against the `sequence_len` limit configured in `config/base.yml`. It calculates and prints a token distribution histogram showing the configured percentile distribution and Max token counts for both the total sequence length and the assistant's responses separately, along with their respective averages. To ensure fast subsequent runs, the validation dynamically tries to load the tokenizer from local cache first before checking the Hugging Face Hub. If any sample exceeds the token limit, the data preparation script raises a warning and terminates with an error code, preventing invalid/oversized training data from entering the training pipeline. For local testing, S3 remote storage is mocked as a local folder (`data/s3-mock/`).
- **Testing:** Unit tests are configured via `pytest` to verify the execution, behavior, and config merging logic of the dynamic hardware launcher (`tests/test_train.py`), and to ensure that the container image version specified in the PR workflow matches the Dockerfile (`tests/test_ci_sync.py`). A dedicated container smoke test (`tests/container_smoke_test.py`) verifies core library compatibility (PyTorch, Transformers, and Axolotl) inside the built container. The dataset pipeline structure is validated via DVC status checks on PRs.
- **Training Configuration:** Baseline hyperparameters (such as LoRA adapter settings, dataset paths, and a default sequence length) are defined in `config/base.yml`. Model-specific configuration files (e.g., `config/train.yml` for Mixtral, `config/train-gemma4.yml` for Gemma, and `config/train-mistral4small.yml` for smaller variants) inherit from `base.yml` and only define model-specific overrides (like `base_model` and quantization settings).
- **Containerization:** The environment is containerized using the `axolotlai/axolotl-cloud-uv:main-py3.12-cu130-2.10.0` base image. The code and data are decoupled from the image; `scripts/launch.sh` clones the repository at runtime, performs a fast dependency sync using `uv` to install any new delta package requirements, pulls the dataset using DVC (`python -m dvc pull`), sets up the HF cache path (`HF_HOME`) to prevent re-downloading model weights across restarts, and hands off execution to the training pipeline. Python dependencies are dynamically compiled from `src-train/pyproject.toml` and installed at runtime using `uv` to ensure consistency.
- **CI/CD:** GitHub Actions are configured to run Python tests and execute DVC checks (`validate-data` job using `dvc status` to verify that the compiled training dataset is up to date) on pull requests. Additionally, a container smoke test workflow executes `tests/container_smoke_test.py` inside the base container to catch dependency or compatibility issues early. Dependabot is also configured to track updates.

## Technical Choices

- **Local Package Manager:** `uv` is used for fast and deterministic dependency resolution during local development and testing (via `uv.lock`). Inside the container, standard `pip` is used to pre-install heavy dependencies quickly.
- **Base Image:** We chose `axolotlai/axolotl-cloud-uv:main-py3.12-cu130-2.10.0` to leverage an official Axolotl environment where flash-attention and deepspeed are already pre-compiled and configured for PyTorch and CUDA 13.0.
- **Dynamic Hardware Launcher:** `src-train/train.py` acts as a dynamic entry point. It resolves configuration properties by merging `config/base.yml` with the specified model config override using OmegaConf. It automatically detects the number of GPUs to determine which models can be trained, and enables DeepSpeed ZeRO-3 across devices.
- **Decoupled Execution:** To iterate faster, the codebase, package dependencies, and dataset are not embedded in the container. The `scripts/launch.sh` bootstrap script clones the repository at runtime, installs any package delta updates, and pulls the dataset via DVC (`python -m dvc pull`), allowing code, dependency, and data changes to run immediately on platforms like RunPod without requiring a custom Docker image.

## Model Configurations & Hardware Scaling

### Supported Model Configurations
To reduce duplication across configurations, a base layout is defined in `config/base.yml` containing standard parameters (such as the default sequence length). Model-specific files only specify the differences:
- `config/base.yml`: The baseline Axolotl configuration containing shared parameters (sequence length, dataset configs, optimizer, etc.).
- `config/train.yml`: The default configuration loading `unsloth/Mixtral-8x7B-Instruct-v0.1-bnb-4bit` (4-bit quantized) and defining Mixtral-specific training variables.
- `config/train-gemma4.yml`: A configuration tailored to override the model for the Gemma architecture in 4-bit quantization.
- `config/train-mistral4small.yml`: A configuration tailored to override the model for smaller Mistral/Mixtral variants.

The training script automatically executes a predefined pipeline of models sequentially depending on the available hardware (e.g., Mistral requires 8 GPUs, while Gemma runs on 2 or more).

### Dynamic Reconfiguration (train.py)
To ensure the pipeline remains resilient across various cloud GPU instances, `src-train/train.py` applies dynamic scaling prior to execution:
- **Multi-GPU Parallelism:** It automatically detects the number of available CUDA GPUs. It injects DeepSpeed ZeRO-3 optimization (`--deepspeed config/zero3.json`) via `accelerate launch`, enabling efficient memory sharding across devices.
- **Memory Fragmentation Fix:** It sets the `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` environment variable globally to mitigate memory fragmentation issues common during long-running LLM fine-tuning.

### Pipeline Execution Control (`SKIP_TRAIN` / `SKIP_EVAL`)

To allow for modular execution of the pipeline, `SKIP_TRAIN` and `SKIP_EVAL` act as shell-level toggles enforced directly within the `scripts/launch.sh` orchestrator script. 

Because we decoupled the code from the container, the environment isn't tightly bound to a single monolithic task. When the container boots in RunPod:
1. `scripts/launch.sh` clones the repository and sets up the root cache structure.
2. It checks the `SKIP_TRAIN` environment variable. If it is **not** set to `true`, it calls `bash scripts/train.sh` (which installs the `src-train` dependencies and runs `train.py`). If it **is** `true`, it bypasses the training workflow entirely.
3. It then checks `SKIP_EVAL`. If it is **not** `true`, it calls `bash scripts/eval.sh` (which isolates and installs the `src-eval` dependencies and runs `evaluation.py`). 

By elevating this control flow logic out of Python and into the bash orchestrator, you avoid downloading and installing massive dependency trees (like `sglang` or `liger-kernel`) if you are only running one half of the pipeline. If you want to use previously generated adapters to run an isolated evaluation spike, you can just pass `SKIP_TRAIN=true` and the container spins up purely for inference testing.
