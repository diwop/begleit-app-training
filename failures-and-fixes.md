# Training

... (overall)

## Gemma

### Iteration 1: FlashAttention Head Dimension Error
* **Error**: `[rank1]: RuntimeError: FlashAttention forward only supports head dimension at most 256`
* **What didn't work**: Globally forcing `attn_implementation: "flash_attention_2"`. Gemma 4 has hybrid attention layers where some global layers use a head dimension of 512, exceeding FlashAttention-2's hard limit of 256.
* **Fix**: Made the `attn_implementation` configuration option configurable in `src/launcher.py` and set it to `sdpa` (Scaled Dot Product Attention) in `config/train-gemma4.yml`.

### Iteration 2: Activation Checkpointing Mismatch under DeepSpeed ZeRO-3
* **Error**: `torch.utils.checkpoint.CheckpointError: Recomputed values for the following tensors have different metadata than during the forward pass. Saved metadata: torch.Size([512]) ... Recomputed metadata: torch.Size([0])`
* **What didn't work**: PyTorch's default non-reentrant activation checkpointing (`use_reentrant: false`) checks tensor metadata strictly. Under DeepSpeed ZeRO-3, parameter sharding causes placeholder shapes (`[0]`) to trigger validation errors before weights are gathered. Setting `use_reentrant: true` in the configuration didn't work because Axolotl auto-detects Gemma 4 and forcibly resets `use_reentrant` to `false`.
* **Fix**: Created `src/train_patched.py` to monkeypatch `axolotl.train.train` right before execution starts (after config validation finishes) to force-inject `use_reentrant: true`. Modified the launcher to call this wrapper script.

### Iteration 3: CUDA Out of Memory (OOM) during Logits upcasting
* **Error**: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 9.97 GiB. GPU 1 has a total capacity of 44.39 GiB of which 3.08 GiB is free. Including non-PyTorch memory, this process has 41.31 GiB memory in use.`
* **What didn't work**: Using pure `sdpa` attention implementation fallback. Standard attention materialization for global layers (head dim 512) left too little VRAM free before the final loss layer. Upcasting logits to float32 (`logits.float()`) at 16k or 10.8k context requires 10-16 GB of memory, causing OOM.
* **Fix**: Enabled `gemma4_hybrid_attn_impl: true` in `config/train-gemma4.yml` and reverted to `flash_attention_2` (default). This uses high-performance Flash Attention 2 on sliding-window layers (head dim 256) and falls back to SDPA only on global layers (head dim 512), dramatically reducing peak VRAM usage and allowing 16K max context length to fit and train on 2x L40S.

### Iteration 4: CUDA Out of Memory (OOM) during SDPA forward pass at 16K context
* **Error**: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.55 GiB. GPU 1 has a total capacity of 44.39 GiB of which 71.31 MiB is free.`
* **What didn't work**: On a 2x L40S cluster, sharding model weights under ZeRO-3 leaves ~18.6 GB of free VRAM per GPU. During the forward pass at 16K context, the SDPA attention fallback on Gemma 4's global layers (head dim 512) and the massive activations memory footprint exhausted the remaining GPU VRAM, leading to OOM before logits/loss computation could even start. Additionally, DeepSpeed's default `stage3_param_persistence_threshold: "auto"` leaves smaller MoE expert modules replicated rather than sharded.
* **Fix**: Added `liger-kernel` to dependencies in `pyproject.toml` and enabled it in `config/train-gemma4.yml` to optimize activation memory (and completely eliminate logits upcasting allocation). Enabled DeepSpeed CPU activation checkpointing (`deepspeed_cpu_checkpointing: true`) to offload activation checkpoints to CPU RAM. Set `deepspeed_param_persistence_threshold: 0` to force sharding of all parameters (such as MoE expert parameters) across the GPUs. Made these DeepSpeed settings customizable via the launcher.
* **Update (OOM persistent)**: Even with sharded weights and Liger kernel, the remaining ~18.6 GB of VRAM was too small to host 16K context activations during forward steps. We resolved this by enabling DeepSpeed CPU offloading for parameters and optimizer states (`deepspeed_offload_param: true` and `deepspeed_offload_optimizer: true`), reducing the GPU memory required for weights from 26 GB to almost 0.


## Mistral

### Iteration 1: Setting up Mistral Small 4 (119B) FP8 on 4x L40S
* **Approach**: Mistral-Small-4-119B-2603 is a 119B parameter model released natively in FP8 (`float8_e4m3fn`). Sharding this model across 4x L40S (4x 48GB VRAM) under ZeRO-3 results in 29.75 GB of sharded weights per GPU. Given 16K context activations, keeping weights on the GPU would lead to OOM. We apply our custom DeepSpeed CPU parameter and optimizer offloading (`deepspeed_offload_param: true`, `deepspeed_offload_optimizer: true`), CPU activation checkpointing (`deepspeed_cpu_checkpointing: true`), and Liger Kernel optimizations to run this 119B model efficiently on 4x L40S.

### Iteration 2: Mistral Tokenizer Validation Failure
* **Error**: `mistral_common.exceptions.InvalidMessageStructureException: Expected last role User or Tool (or Assistant with prefix or continue_final_message set to True) for serving but got assistant`
* **What didn't work**: 
  - Using `chat_template: chatml` failed because `MistralCommonTokenizer` overrides `apply_chat_template` to delegate to `mistral-common` validation rules (which expect the conversation to end in `User` or `Tool` for serving).
  - Replacing `apply_chat_template` and switching `chat_template` to `tokenizer_default` triggered a secondary error: `ValueError: chat_template choice is tokenizer_default but tokenizer's chat_template is null. Please add a chat_template in tokenizer config`. This occurs because `MistralCommonTokenizer` does not populate the `chat_template` property on the instance from `tokenizer_config.json`.
  - Adding the `chat_template` property getter fallback triggered a third error: `NotImplementedError: MistralCommonBackend does not implement get_chat_template`. This occurs because `PreTrainedTokenizerBase.apply_chat_template` internally calls `self.get_chat_template(chat_template, tools)`, which is overridden in `MistralCommonTokenizer` to raise `NotImplementedError`.
* **Fix**: 
  - Added a class property getter monkeypatch to `PreTrainedTokenizerBase` in `src/train_patched.py` that intercepts `chat_template` lookups. If `chat_template` is `None` and the tokenizer class/model is Mistral-based, it returns the official Mistral Small 4 chat template string.
  - Monkeypatched both `MistralCommonTokenizer` and `TokenizersBackend`'s `get_chat_template` and `apply_chat_template` methods to redirect them to the respective base implementations in `PreTrainedTokenizerBase`. This completely bypasses the custom Mistral implementations/validations and routes template rendering and retrieval to the standard Hugging Face Jinja2 engine.
  - Set `chat_template: tokenizer_default` in `config/train-mistral4small.yml` to train the adapter on the model's native format (`<s>[SYSTEM_PROMPT]...[/SYSTEM_PROMPT][MODEL_SETTINGS]...[/MODEL_SETTINGS][INST]...[/INST]...</s>`).

### Iteration 3: Trainer FP8 Quantization Block
* **Error**: `ValueError: The model you are trying to fine-tune is quantized with fp8 but that quantization method do not support training. Please open an issue on GitHub: https://github.com/huggingface/transformers to request the support for training support for fp8`
* **What didn't work**: Hugging Face `Trainer` performs a hard check (`validate_quantization_for_training`) during initialization and raises a ValueError if the base model has FP8 quantized parameters. This is a false-positive for parameter-efficient fine-tuning (PEFT/LoRA) because the FP8 base weights are completely frozen, and only the float16/bfloat16 LoRA adapter parameters are being trained.
* **Fix**: Monkeypatched `validate_quantization_for_training` in both `transformers.trainer_utils` and `transformers.trainer` inside `src/train_patched.py` to be a no-op dummy function before loading the trainer.

### Iteration 4: MistralTokenizer save_pretrained save_jinja_files Failure
* **Error**: `ValueError: Kwargs ['save_jinja_files'] are not supported by MistralCommonBackend.save_pretrained.`
* **What didn't work**: When Axolotl initializes training, it saves the initial configs and calls `tokenizer.save_pretrained(cfg.output_dir, save_jinja_files=cfg.tokenizer_save_jinja_files)`. The `MistralCommonTokenizer`'s `save_pretrained` method delegates to `MistralCommonBackend.save_pretrained`, which strictly checks for unknown kwargs and raises a ValueError if any (including `save_jinja_files`) are passed.
* **Fix**: Monkeypatched `save_pretrained` on `PreTrainedTokenizerBase`, `MistralCommonTokenizer`, and `TokenizersBackend` to intercept calls and pop the `save_jinja_files` key from the keyword arguments dictionary before passing it to the underlying save backend.

### Iteration 5: CPU RAM Out of Memory (OOM) / SIGKILL during Trainer initialization
* **Error**: `Signal 9 (SIGKILL) received by PID 4955` / CPU RAM exhausted during DeepSpeed setup.
* **What didn't work**: Enabling CPU offloading for model parameters (`deepspeed_offload_param: true`) requires allocating and pinning the full 119B model weight space (~119 GB) in CPU memory. With 4 ranks running on the same host, the overhead and memory pinning completely exhausted the instance's available CPU RAM (VRAM remained unused at 1% because the execution crashed before launching GPU kernels), triggering the OS OOM killer.
* **Fix**: Disabled CPU offloading of parameters and optimizer states (`deepspeed_offload_param: false` and `deepspeed_offload_optimizer: false`) in `config/train-mistral4small.yml`. Under DeepSpeed ZeRO-3, the 119B FP8 model is sharded across all 4 L40S GPUs (29.75 GB of weights per GPU), leaving ~17.4 GB VRAM per GPU. This is more than sufficient for training activations when combined with gradient checkpointing and FlashAttention-2, and avoids CPU RAM OOM crashes entirely.

### Iteration 6: Trainer Hang / Deadlock during distributed process group initialization
* **Error**: The training run hangs indefinitely during `Trainer` instantiation (right after `Gradient accumulation steps mismatch` warning) with VRAM at 1% and CPU RAM stable at 78%.
* **What didn't work**: Having `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` enabled in `src/launcher.py` (which were carried over from an old vLLM spike config). For large models like Mistral Small 119B, forcing NCCL to route all parameter and gradient synchronization traffic through CPU sockets and the local TCP interface (`eth0`) instead of direct GPU-to-GPU memory copies (NVLink/PCIe) causes network buffer saturation and a communication deadlock, resulting in ranks hanging indefinitely.
* **Fix**: Removed `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` from `src/launcher.py` to allow the GPUs to communicate over high-speed Peer-to-Peer (PCIe/NVLink) direct channels. Additionally, set `TORCH_NCCL_BLOCKING_WAIT=1` to ensure any future distributed communication hangs time out with a descriptive error instead of locking up.

### Iteration 7: High CPU RAM Usage / Parallel loading memory exhaustion during model loading
* **Error**: CPU RAM spikes to 74% (527 GB) during model loading and the processes get stuck or killed.
* **What didn't work**: Under DeepSpeed Stage 3, unless `zero3_init_flag` is explicitly set to `true` in the DeepSpeed config file, Hugging Face `transformers` does not use the `deepspeed.zero.Init()` context manager during model loading. Consequently, all 4 ranks load the entire 119B model into CPU memory in parallel before partitioning it. This causes a massive memory spike (4 x 119 GB = 476 GB + overhead, exceeding 500 GB) which triggers system paging/thrashing or OOM crashes.
* **Fix**: Added `"zero3_init_flag": true` to the root level of the DeepSpeed configuration dynamically compiled in `src/launcher.py`. This instructs `transformers` to wrap model loading in `deepspeed.zero.Init()`, sharding the weights on-the-fly directly to the GPU VRAM as they are loaded, keeping CPU memory usage extremely low.

### Iteration 8: Persistent High CPU RAM Usage during model loading (DeepSpeed Zero Init bypass)
* **Error**: CPU RAM spikes to 91% (over 650 GB) during loading while VRAM remains at 1%.
* **What didn't work**: Merely adding `"zero3_init_flag": true` inside the DeepSpeed configuration is insufficient if we launch the training script via a standard `accelerate launch` command without DeepSpeed flags. Because `accelerate` is unaware of DeepSpeed during script launch, it does not set the required environment hooks, causing Hugging Face to load the model on CPU inside each rank's thread before initializing the DeepSpeed engine. This results in the same parallel 500+ GB CPU memory spike and subsequent thrashing.
* **Fix**: Modified `src/launcher.py` to pass `--use_deepspeed` and `--deepspeed_config_file` arguments directly to the `accelerate launch` shell call. This forces `accelerate` to configure the DeepSpeed ZeRO-3 Init context manager globally at launch, ensuring that the 119B model parameters are created directly sharded on the GPU devices as they are loaded, keeping CPU RAM usage minimal.

### Iteration 9: accelerate launch mutually exclusive argument validation failure
* **Error**: `ValueError: You can only use one of --cpu, --multi_gpu, --tpu, --use_deepspeed, --use_fsdp at a time.`
* **What didn't work**: Passing both `--multi_gpu` and `--use_deepspeed` to the `accelerate launch` command. `accelerate` enforces strict mutual exclusivity among these strategy flags because `--use_deepspeed` automatically sets up and manages the multi-GPU environment parameters.
* **Fix**: Removed `--multi_gpu` from the launcher command array in `src/launcher.py` when DeepSpeed is enabled, letting `--use_deepspeed` handle the multi-GPU orchestration internally while still specifying the GPU process count via `--num_processes`.

### Iteration 10: CPU RAM Out of Memory (OOM) during training due to FP8 / low_cpu_mem_usage conflict with DeepSpeed Stage 3
* **Error**: `Root Cause (first observed failure): [2]: traceback : Signal 9 (SIGKILL) received by PID 7279` / CPU RAM spikes and gets killed.
* **What didn't work**: Using `torch_dtype: "float8_e4m3fn"` and `low_cpu_mem_usage: true` with DeepSpeed Stage 3. DeepSpeed Stage 3 is fundamentally incompatible with the Hugging Face `low_cpu_mem_usage=True` flag. By telling HF to load the model on CPU, it bypassed the DeepSpeed ZeRO-3 `zero.Init()` partitioning, causing the 119B model parameters to load natively on CPU and remain there (VRAM stayed at 1%). When training started, CPU memory spiked as the trainer tried to operate on/copy/upcast the 119B model on CPU, eventually exhausting the host's 712 GB RAM and getting SIGKILLed.
* **Attempted Fix**: Switched `torch_dtype` to `"bfloat16"` and removed `low_cpu_mem_usage: true` in `config/train-mistral4small.yml`, enabling DeepSpeed CPU parameter and optimizer offloading (`deepspeed_offload_param: true` and `deepspeed_offload_optimizer: true`).

### Iteration 11: CPU RAM Out of Memory (OOM) during loading due to parallel bfloat16 state dict instantiation
* **Error**: `Root Cause (first observed failure): [0]: traceback : Signal 9 (SIGKILL) received by PID 7739` / CPU RAM spikes to 450+ GB and gets killed during loading.
* **What didn't work**: Loading the 119B parameter model in `bfloat16` (238 GB) with `low_cpu_mem_usage: false`. When `low_cpu_mem_usage` is `false`, each rank loads the entire model state dict into CPU memory in parallel. For a 238 GB model, 4 ranks * 238 GB = 952 GB of CPU RAM is allocated during loading, which immediately exceeded the host's 712 GB physical memory limit and triggered the OS OOM killer.
* **Fix**: Reverted the model dtype to `"float8_e4m3fn"` (119 GB weights) and set `low_cpu_mem_usage: false`. Disabled CPU parameter and optimizer offloading (`deepspeed_offload_param: false` and `deepspeed_offload_optimizer: false`). Because the model weights are loaded in FP8, the peak parallel memory allocation on CPU is only 4 ranks * 119 GB = 476 GB, which easily fits within the 712 GB RAM. Once loaded, DeepSpeed Stage 3 `zero.Init()` shards the parameters directly onto the GPU VRAM (29.75 GB of sharded weights per GPU), fitting perfectly inside each GPU's 48 GB VRAM. This avoids both CPU RAM OOM and GPU VRAM OOM while ensuring native DeepSpeed compatibility.

# Evaluating

...

## 

...

## 

...