# Implementation Details

- **Dataset Preparation:** The conversational data is stored in `data/sample_dataset.jsonl`. `src/prepare_data.py` handles loading and formatting the dataset into an Axolotl-compatible ShareGPT JSONL format (`data/axolotl_dataset.jsonl`).
- **Testing:** Unit tests have been written in `tests/test_data.py` and are verified via `pytest` to ensure structural integrity of the dataset mappings.
- **Training Configuration:** `config/train.yml` utilizes Axolotl to load `unsloth/mixtral-8x7b-v0.1-bnb-4bit` in 4-bit quantization. It configures a QDoRA/QLoRA adapter. Checkpoints and final adapters are saved to `/workspace/output`.
- **Containerization:** The entire environment is containerized using `winglian/axolotl:main-py3.10-cu121-2.1.2` base image. The `uv` package manager installs additional prep/test dependencies natively to ensure rapid builds. The dataset is prepared and embedded during the Docker build.
- **CI/CD:** GitHub Actions have been configured to automatically test Python code, verify the Docker build via a dry run on pull requests, and push the image to the GitHub Container Registry (GHCR) upon merges to `main`. Dependabot is also fully configured to track both Python (`pip`) and `docker` updates.

## Technical Choices

- **`uv` Package Manager:** Used to guarantee fast and deterministic dependency resolution inside the Docker container.
- **Base Image:** We chose `winglian/axolotl:main-py3.10-cu121-2.1.2` to leverage an official Axolotl environment where flash-attention and deepspeed are already pre-compiled and configured for PyTorch and CUDA.
- **Embedded Dataset:** To simplify the initial iterations, the data is bundled directly into the container. Future steps could decouple this and mount datasets dynamically.

## Running on RunPod

Because the fine-tuning pipeline is completely containerized, it can be deployed seamlessly to Cloud GPU providers like RunPod.

### Step-by-Step Guide for RunPod

1. **Choose an Instance:** Navigate to your RunPod console and click on **Deploy > Deploy from Custom Image**. You will need a GPU with sufficient VRAM. For a 4-bit quantized Mixtral 8x7B model, an A100 (80GB) is highly recommended.
2. **Container Image:** Enter the GitHub Container Registry image path. This automatically pulls the latest image built by the CI/CD pipeline:
   ```text
   ghcr.io/diwop/begleit-app-training:main
   ```
   > **Note:** By default, new packages on GitHub Container Registry might be private. Make sure to change the package visibility to "Public" in the GitHub Package settings, or configure RunPod with your GitHub Container Registry credentials.
3. **Container Arguments (Optional):** By default, the script fine-tunes using `config/train.yml`. You can pass overrides via the container command if needed.
4. **Volume Mounts:** To ensure your fine-tuned LoRA adapters are saved permanently, mount a RunPod Network Volume (or a persistent pod volume) to `/workspace/output`. This ensures that even if you terminate the container, the trained adapters remain accessible.
5. **Execution:** Once the Pod is booted, the container's `ENTRYPOINT` will automatically execute `accelerate launch -m axolotl.cli.train config/train.yml`, parsing the config and executing the trainer. You can monitor the training progress via the RunPod Web Terminal or Container Logs.
