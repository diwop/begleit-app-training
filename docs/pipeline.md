# Training & Evaluation Pipeline

The training pipeline automatically manages and runs all active model configurations defined in `src-train/train.py` sequentially.

It currently runs:
1. `config/train-gemma4.yml` (requires GPU count >= 2)
2. `config/train-mistral4small.yml` (Skipped if < 8 GPUs)

After training the pipeline automatically runs evaluation jobs from `src-eval/evaluation.py` with and without reasoning/adapters.

Use `SKIP_TRAIN=true` to only run the evaluation (with the currently available adapters).
Use `SKIP_EVAL=true` to only run the training without evaluation.

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

2. **Container Image:** Enter the official Axolotl image. The currently recommended and tested image version is tracked in the `axolotl_image` field at the top of the [README.md](../README.md) file.
   ```text
   axolotlai/axolotl-cloud-uv:main-py3.12-cu130-2.10.0
   ```
3. **Docker Command:** Set the RunPod Docker Command to bootstrap the repository:
   ```bash
   bash -c "wget -qO- https://raw.githubusercontent.com/diwop/begleit-app-training/${BRANCH:-main}/scripts/launch.sh | bash"
   ```

4. **Container Arguments (required):**

* You need to pass the `HF_TOKEN` (HuggingFace token to download guarded base models).

5. **Container Arguments (optional):**

You can pass overrides via Environment Variables in RunPod:

* `BRANCH` to run a specific branch
* `SKIP_TRAIN=true`/`SKIP_EVAL=true` to toggle pipeline stages
* `S3_BUCKET` to store logs and results in an S3 bucket (requires `AWS_DEFAULT_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` to be set as well)

6. **Volume Mounts:** To ensure your fine-tuned LoRA adapters and Hugging Face model cache are saved permanently, mount a RunPod Network Volume (or a persistent pod volume) to `/app`. This ensures that even if you terminate the container, the downloaded model weights (stored in `/app/huggingface_cache`) and trained adapters (stored in `/app/output`) remain accessible.

7. **Execution:** Once the Pod is booted, the bootstrap command will automatically clone the latest code, sync any new package dependency changes, prepare the data, and execute `launch.sh`. You can monitor the dynamic hardware scaling and the training progress via the RunPod Web Terminal or Container Logs.
