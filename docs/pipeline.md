# Training & Evaluation Pipeline

The training pipeline automatically manages and runs all active model configurations defined in `src-train/train.py` sequentially.

**Please note:** It currently just runs `config/train-gemma4.yml`, which needs roughly 96 GB
of VRAM in total but no particular number of cards -- `src-train/train.py` runs Gemma at any
GPU count and only skips Mistral below 8.

The training and evaluation phases are split and run on separate containers:
1. **Training Phase (`MODE=train`)**: Runs on a training container, compiles datasets, fine-tunes the model, and uploads the trained LoRA adapters to S3.
2. **Evaluation Phase (`MODE=eval`)**: Runs on an evaluation container, downloads the latest adapters from S3, and executes the evaluation suite.

By default, the launch script runs in `MODE=eval`.

## Persistent Caching and Output

To avoid downloading heavy model weights on every run, and to save your training outputs, make sure to mount a persistent network volume to `/app`. 
* Hugging Face cache will be stored at `/app/huggingface_cache`
* Training checkpoints and adapters will be saved to `/app/output`

## Sequence Length Validation

To avoid runtime out-of-memory (OOM) errors during fine-tuning, the data preparation step (`src-train/prepare_dataset.py`) automatically validates the token count of each compiled conversation pair against `sequence_len` in `config/base.yml`.

If any conversation pair exceeds the token limit, the dataset compilation fails with an exit code of `1`.
If your dataset contains longer sequences, you will need to increase `sequence_len` in `config/base.yml`. Note that this will increase the GPU VRAM requirements during training, and you may need to reduce `micro_batch_size` or upgrade your GPU tier to compensate.

## Running on RunPod

Because the fine-tuning pipeline is completely container based, it can be deployed seamlessly to Cloud GPU providers like RunPod.
The training and evaluation pipeline is optimized for NVIDIA L40S GPUs.

### The short way

    bash scripts/start_runpod.sh train
    bash scripts/start_runpod.sh eval
    bash scripts/start_runpod.sh train attach   # just watch a run that is already going

`scripts/start_runpod.sh` reuses an existing GPU pod if there is one, otherwise it lists the
secure-cloud offers that clear all three filters and creates one only after you confirm:

* **an explicit allow-list of FP8-capable cards** (`FP8_GPUS` in the script). FP8 needs
  compute capability 8.9 or newer -- Ada, Hopper, Blackwell -- and the deployment target is
  the FP8 base model. Every Ampere card is excluded, which matters because the two cheapest
  offers that otherwise qualify (2x A40 at $0.88, 2x RTX A6000 at $1.06) are Ampere.
* at least `MIN_TOTAL_VRAM_GB` (80) of VRAM **in total**, over the card counts in
  `GPU_COUNTS` ("1 2").
* no more than `MAX_USD_PER_HR` (3) for the whole pod.

The offers are ranked by preference rather than by price, because one card avoids
cross-GPU communication altogether -- no NCCL, no P2P workarounds, no tensor parallelism:

1. a single GPU under `PREFER_USD_PER_HR` (2)
2. several GPUs under `PREFER_USD_PER_HR`
3. anything else up to `MAX_USD_PER_HR`, as a last resort

so a single card at $1.99 is offered ahead of two cards at $1.68. What that leaves, in that
order, at the prices of 2026-07-31:

| | GPU | VRAM total | $/hr | stock |
|---|---|---|---|---|
| 1 | 1x RTX PRO 6000 Blackwell Workstation | 96 GB | 1.89 | Low |
| 2 | 1x RTX PRO 6000 Blackwell Server | 96 GB | 1.99 | **High** |
| 3 | 2x RTX 6000 Ada | 96 GB | 1.68 | Low |
| 4 | 2x L40S | 96 GB | 1.98 | Low |
| 5 | 1x H100 PCIe | 80 GB | 2.89 | Low |
| 6 | 1x H100 SXM | 80 GB | 2.99 | **High** |

H100 NVL ($3.19), H200 and B200 are on the allow-list but over budget.

Reuse is deliberately **not** filtered by the allow-list -- an existing pod is worth more
than a perfect one -- but a pre-Ada pod gets a warning, so an FP8 result from an Ampere card
cannot be believed by accident.

It sets the image and the start command, waits for the pod, prints the SSH command and tails
the run's log -- Ctrl+C there detaches without touching the run.

It needs `RUNPOD_API_KEY` (see `.envrc`) and passes `HF_TOKEN`, `S3_BUCKET` and the `AWS_*`
variables through to the pod if they are set locally. The pod runs `origin/<branch>`, not
your working tree, so push first; the script refuses to start otherwise. By default the pod
stops itself when the run ends -- `KEEP_ALIVE=true` keeps it warm for the next run.

### Step-by-Step Guide for RunPod (manual)

1. **Choose an Instance:** Navigate to your RunPod console and click on **Deploy > Deploy from Custom Image**. You will need a GPU with sufficient VRAM:

* For **Gemma 4** training you need about **96 GB of VRAM in total**, in any number of cards. Runs that worked: **1x RTX PRO 6000** (96 GB), **2x L40S à 48 GB**, and **1x H100 SXM** (80 GB) as the pricier option.
* For **Mistral Small 4** training you'll need **8x L40S à 48 GB**. The training will skip Mistral if less than 8 GPUs are available.

2. **Container Image:** Enter the appropriate Docker image. The recommended and tested image versions are tracked at the top of the [README.md](../README.md) file:
   * For training (`MODE=train`), use `train_image`.
   * For evaluation (`MODE=eval`), use `eval_image`.
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
