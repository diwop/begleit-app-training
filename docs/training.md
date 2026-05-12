# Training Different Models

Axolotl's flexible configuration allows you to easily switch the underlying model you are fine-tuning by pointing to a different YAML configuration file.

We have included two specific examples in the `config/` directory for **4-bit quantized** versions of popular models: **Gemma 4 (4-bit)** and **Mistral 4 Small (4-bit)**. 

## How to use custom YAML configs

You can run these specific models by simply overriding the container's default `ENTRYPOINT` when executing it (for example, on RunPod) to point to the respective file.

### 1. Mistral 4 Small 4-bit (`train-mistral4small.yml`)

`cyankiwi/Mistral-Small-4-119B-2603-AWQ-4bit` (the 119B parameter model) requires a massive amount of VRAM. Utilizing this pre-quantized AWQ model avoids massive CPU RAM bottlenecks during loading (because the weights are natively 4-bit), but you will still need a multi-GPU setup (e.g., 2x or 4x A100 80GB/H100) to train effectively. Note that `load_in_4bit: true` is omitted since the model is already quantized.

To fine-tune using this configuration (note: you may need to pass your HF token if the model is gated):
```bash
docker run -e HF_TOKEN="your_token" --entrypoint "accelerate launch -m axolotl.cli.train config/train-mistral4small.yml" ghcr.io/diwop/begleit-app-training:main
```

### 2. Gemma 4-bit (`train-gemma4.yml`)

The `google/gemma-4-26B-A4B-it` model requires slightly different learning rates and specific target modules compared to Mixtral, and standard QLoRA without DoRA works best. Like Mistral, it is dynamically quantized using bitsandbytes on load.

To fine-tune using this configuration:
```bash
docker run -e HF_TOKEN="your_token" --entrypoint "accelerate launch -m axolotl.cli.train config/train-gemma4.yml" ghcr.io/diwop/begleit-app-training:main
```

### 3. Creating Your Own Configurations

You can create unlimited YAML configurations (e.g., `config/my-custom-model.yml`) within the repository. Axolotl automatically parses the configuration passed as the final argument to its CLI. Alternatively, you can use CLI arguments to override any values on the fly without making new files:

```bash
docker run ghcr.io/diwop/begleit-app-training:main --learning_rate 0.0001 --num_epochs 3
```
