# Training Different Models

Axolotl's flexible configuration allows you to easily switch the underlying model you are fine-tuning by pointing to a different YAML configuration file. 

To reduce duplication, we use a base configuration file at [config/base.yml](config/base.yml) that defines baseline parameters (such as the default sequence length, datasets, and LoRA adapter details). The model-specific configuration files only specify model-specific overrides (like `base_model` and quantization).

At runtime, the dynamic hardware launcher [src/launcher.py](src/launcher.py) merges the base configuration with the selected override file using OmegaConf to produce a resolved `.merged-train.yml` configuration before starting Axolotl.

## How to use custom YAML configs

Instead of overriding the container's `ENTRYPOINT` directly, you should select the configuration file using the `TRAIN` environment variable. The entrypoint script (`runner/entrypoint.sh`) will read this variable and hand it over to the launcher.

### Persistent Caching and Output

To avoid downloading heavy model weights on every run, and to save your training outputs, make sure to mount a persistent network volume to `/app`. 
* Hugging Face cache will be stored at `/app/huggingface_cache`
* Training checkpoints and adapters will be saved to `/app/output`

### 1. Mistral 4 Small 4-bit (`train-mistral4small.yml`)

`cyankiwi/Mistral-Small-4-119B-2603-AWQ-4bit` (the 119B parameter model) requires a massive amount of VRAM. Utilizing this pre-quantized AWQ model avoids massive CPU RAM bottlenecks during loading (because the weights are natively 4-bit), but you will still need a multi-GPU setup (e.g., 2x or 4x A100 80GB/H100) to train effectively. Note that `load_in_4bit: true` is omitted since the model is already quantized.

To fine-tune using this configuration:
```bash
docker run -e HF_TOKEN="your_token" -e TRAIN="train-mistral4small" -v /path/to/persistent/volume:/app ghcr.io/diwop/begleit-app-training:main
```

### 2. Gemma 4-bit (`train-gemma4.yml`)

The `google/gemma-4-26B-A4B-it` model requires slightly different learning rates and specific target modules compared to Mixtral, and standard QLoRA without DoRA works best. Like Mistral, it is dynamically quantized using bitsandbytes on load.

To fine-tune using this configuration:
```bash
docker run -e HF_TOKEN="your_token" -e TRAIN="train-gemma4" -v /path/to/persistent/volume:/app ghcr.io/diwop/begleit-app-training:main
```

### 3. Creating Your Own Configurations

You can create custom configurations within the `config/` directory (e.g., `config/train-my-model.yml`). Any configuration you pass will automatically be merged with `config/base.yml` by the launcher.

To run with your custom configuration:
```bash
docker run -e TRAIN="train-my-model" -v /path/to/persistent/volume:/app ghcr.io/diwop/begleit-app-training:main
```

## Sequence Length Validation

To avoid runtime out-of-memory (OOM) errors during fine-tuning, the data preparation step (`src/prepare_dataset.py`) automatically validates the token count of each compiled conversation pair against `sequence_len` in `config/base.yml` (default `4096`).

- The tokenizer used for token count validation is `cyankiwi/Mistral-Small-4-119B-2603-AWQ-4bit` (which optimizes local cache access).
- If any conversation pair exceeds the token limit, the dataset compilation fails with an exit code of `1`.
- If your dataset contains longer sequences, you will need to increase `sequence_len` in `config/base.yml`. Note that this will increase the GPU VRAM requirements during training, and you may need to reduce `micro_batch_size` or upgrade your GPU tier to compensate.

