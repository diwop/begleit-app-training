# Training & Evaluation Pipeline

The training pipeline automatically manages and runs all active model configurations defined in `src-train/train.py` sequentially.

It currently runs:
1. `config/train-gemma4.yml` (requires GPU count >= 2)
2. `config/train-mistral4small.yml` (Skipped if < 8 GPUs) **DEACTIVATED FOR NOW***.

The training and evaluation phases are split and run on separate containers:
1. **Training Phase (`MODE=train`)**: Runs on the Axolotl container, compiles datasets, fine-tunes the model, and uploads the trained LoRA adapters to S3.
2. **Evaluation Phase (`MODE=eval`)**: Runs on the SGLang container, downloads the latest adapters from S3, and executes the evaluation suite.

By default, the launch script runs in `MODE=eval`.

## Persistent Caching and Output

To avoid downloading heavy model weights on every run, and to save your training outputs, make sure to mount a persistent network volume to `/app`. 
* Hugging Face cache will be stored at `/app/huggingface_cache`
* Training checkpoints and adapters will be saved to `/app/output`

## Sequence Length Validation

To avoid runtime out-of-memory (OOM) errors during fine-tuning, the data preparation step (`src-train/prepare_dataset.py`) automatically validates the token count of each compiled conversation pair against `sequence_len` in `config/base.yml`.

- The tokenizer used for token count validation is `cyankiwi/Mistral-Small-4-119B-2603-AWQ-4bit` (which optimizes local cache access).
- If any conversation pair exceeds the token limit, the dataset compilation fails with an exit code of `1`.
- If your dataset contains longer sequences, you will need to increase `sequence_len` in `config/base.yml`. Note that this will increase the GPU VRAM requirements during training, and you may need to reduce `micro_batch_size` or upgrade your GPU tier to compensate.

## Running on RunPod

Because the fine-tuning pipeline is completely container based, it can be deployed seamlessly to Cloud GPU providers like RunPod.
The training and evaluation pipeline is optimized for NVIDIA L40S GPUs.

### Step-by-Step Guide for RunPod

1. **Choose an Instance:** Navigate to your RunPod console and click on **Deploy > Deploy from Custom Image**. You will need a GPU with sufficient VRAM:

* For **Gemma 4** training you'll need **2x L40S à 48 GB**. The training won't start with just a single GPU.
* For **Mistral Small 4** training you'll need **8x L40S à 48 GB**. The training will skip Mistral if less than 8 GPUs are available.

2. **Container Image:** Enter the appropriate Docker image. The recommended and tested image versions are tracked at the top of the [README.md](../README.md) file:
   * For training (`MODE=train`), use `axolotl_image` (e.g. `axolotlai/axolotl-cloud-uv:main-py3.12-cu130-2.10.0`).
   * For evaluation (`MODE=eval`), use `sglang_image` (e.g. `lmsysorg/sglang:latest`).
3. **Docker Command:** Set the RunPod Docker Command to bootstrap the repository:
   ```bash
   bash -c "wget -qO- https://raw.githubusercontent.com/diwop/begleit-app-training/${BRANCH:-main}/scripts/launch.sh | bash"
   ```

4. **Container Arguments (required):**

* You need to pass the `HF_TOKEN` (HuggingFace token to download guarded base models).

5. **Container Arguments (optional):**

You can pass overrides via Environment Variables in RunPod:

* `MODE` to toggle pipeline stages (`train` or `eval`)
* `S3_BUCKET` to store logs, evaluation results, and adapters in an S3 bucket (requires `AWS_DEFAULT_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` to be set as well)

6. **Execution:** Once the Pod is booted, the bootstrap command will automatically clone the latest code, sync any new package dependency changes, prepare the data, and execute `launch.sh`. You can monitor the dynamic hardware scaling and the training progress via the RunPod Web Terminal or Container Logs.
