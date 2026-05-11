# Implementation Details

- **Dataset Preparation:** The conversational data is stored in `data/sample_dataset.jsonl` using a ChatML-compatible format. `src/prepare_data.py` handles loading and formatting the dataset using Hugging Face's `datasets` library.
- **Testing:** Unit tests have been written in `tests/test_data.py` and are verified via `pytest` to ensure structural integrity of the dataset mappings.
- **Training Script:** `src/train.py` utilizes Unsloth to load `mistralai/Mixtral-8x7B-v0.1` (or a custom model via arguments) in 4-bit quantization. It configures a QDoRA/QLoRA adapter and fine-tunes it using `SFTTrainer`. Checkpoints and final adapters are saved to `/workspace/output`.
- **Containerization:** The entire environment is containerized using `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel`. The `uv` package manager installs dependencies natively to ensure rapid builds. The dataset is embedded in the Docker image.
- **CI/CD:** GitHub Actions have been configured to automatically test Python code, verify the Docker build via a dry run on pull requests, and push the image to the GitHub Container Registry (GHCR) upon merges to `main`. Dependabot is also fully configured to track both Python (`pip`) and `docker` updates.

## Technical Choices

- **`uv` Package Manager:** Used to guarantee fast and deterministic dependency resolution inside the Docker container.
- **Base Image:** We chose `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel` to accommodate `unsloth`'s dependencies, which require PyTorch 2.4.0 or greater. The Unsloth extras tag `[cu121-torch240]` was mapped to correctly install the optimized `xformers` version.
- **Embedded Dataset:** To simplify the initial iterations, the data is bundled directly into the container. Future steps could decouple this and mount datasets dynamically.

## Running on RunPod

Because the fine-tuning pipeline is completely containerized, it can be deployed seamlessly to Cloud GPU providers like RunPod.

### Step-by-Step Guide for RunPod

1. **Choose an Instance:** Navigate to your RunPod console and click on **Deploy > Deploy from Custom Image**. You will need a GPU with sufficient VRAM. For a 4-bit quantized Mixtral 8x7B model, an A100 (40GB or 80GB) is highly recommended.
2. **Container Image:** Enter the GitHub Container Registry image path. This automatically pulls the latest image built by the CI/CD pipeline:
   ```text
   ghcr.io/diwop/begleit-app-training:main
   ```
   > **Note:** By default, new packages on GitHub Container Registry might be private. Make sure to change the package visibility to "Public" in the GitHub Package settings, or configure RunPod with your GitHub Container Registry credentials.
3. **Container Arguments (Optional):** By default, the script fine-tunes `mistralai/Mixtral-8x7B-v0.1`. You can pass a different model ID by overriding the container command, e.g., `python src/train.py --model_id your/custom-model`.
4. **Volume Mounts:** To ensure your fine-tuned LoRA adapters are saved permanently, mount a RunPod Network Volume (or a persistent pod volume) to `/workspace/output`. This ensures that even if you terminate the container, the trained adapters remain accessible.
5. **Execution:** Once the Pod is booted, the container's `ENTRYPOINT` will automatically execute `python src/train.py`, downloading the model and executing the SFT trainer. You can monitor the training progress via the RunPod Web Terminal or Container Logs.
