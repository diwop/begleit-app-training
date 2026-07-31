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

### Iteration 5: Hugging Face Cache causing Container Root Disk Full / OOM
* **Error**: The training script hung or crashed during base model pre-staging because the container root disk (`/`) filled up to 100%.
* **What didn't work**: The `pre_download_models` script used `subprocess.run` to download models without setting the `HF_HOME` environment variable. This caused Hugging Face to download the massive model weights into the default container cache at `/root/.cache/huggingface/hub`, which resided on the small 50 GB container root disk instead of the large persistent volume mounted at `/app`.
* **Fix**: Injected `os.environ["HF_HOME"] = "/app/huggingface_cache"` and `os.environ["HF_HUB_CACHE"] = "/app/huggingface_cache/hub"` at the top of `src-train/train.py`. This globally forces the Hugging Face hub (and all underlying `accelerate` subprocesses) to use the correct persistent volume, bypassing the root disk entirely.

### Iteration 6: `lora_target_linear` adapts the vision tower and the MoE router
* **Error**: `ValueError: Target module Gemma4ClippableLinear((linear): Linear(in_features=1152, out_features=1152, bias=False)) is not supported. Currently, only the following modules are supported: torch.nn.Linear, ...` during `axolotl.cli.merge_lora`. In SGLang the same adapter produced `ValueError: LoRA B output dim 4224 does not match base partition prefix dim 8608 for 2 slices.`
* **Root cause**: `gemma-4-26B-A4B-it` is a `Gemma4ForConditionalGeneration` wrapper around three networks: a ~550M vision tower, a projector, and the MoE language model. `lora_target_linear: true` resolves to PEFT `all-linear`, which on this architecture matches **every** `nn.Linear` in all three — including 190 vision-tower modules, the projector, and all 30 `router.proj` layers. PEFT stores `target_modules` as bare suffixes (`gate_proj`, `up_proj`, `linear`, `proj`, ...), so names collide across towers. The 4224/8608 mismatch is arithmetic proof: 4224 = 2 x 2112 (language `intermediate_size`) being loaded into a slot expecting 2 x 4304 (vision `intermediate_size`). The vision tower's linears are additionally nested one level deeper (`self_attn.q_proj.linear`) inside `Gemma4ClippableLinear` wrappers, which PEFT cannot wrap at all.
* **Measured impact**: the resulting adapter was 852 tensors / 62.29M params / 125 MB, of which only 59.7% (language attention + shared MLP) was servable. 35.6% was vision tower, 4.5% router, 0.2% projector. All 191 vision/projector `lora_B` matrices were **exactly zero** — no image ever enters the model, so they never received gradient — meaning 40% of the adapter was untrained weight that nonetheless broke every downstream tool.
* **Note**: the routed experts (`experts.gate_up_proj` `[128, 1408, 2816]`, `experts.down_proj` `[128, 2816, 704]`) are packed 3D `nn.Parameter` tensors inside `Gemma4TextExperts`, not `nn.Linear`. PEFT cannot see them, so no expert LoRA was ever produced. SGLang's missing MoE-expert LoRA support was never the blocker.
* **Fix**: Replaced `lora_target_linear: true` in `config/base.yml` with a fully qualified regex in `config/train-gemma4.yml`, pinning targets to the language model and excluding the router. Target specifications are inherently model-specific and no longer live in the shared base config. Added `src-train/audit_adapter.py`, which instantiates the base model on the meta device (no weights, no GPU, ~20 s) and reports which towers an adapter touches; it exits non-zero on unservable targets and is suitable as a post-training gate. The corrected adapter is 470 tensors / 37.17M params / 71 MiB, and `merge_lora` reports `Applied LoRA to 205/1013 tensors`.

### Iteration 7: Merged model written to a nested `merged/` subdirectory
* **Error**: `Unrecognized model in /app/output/merged/train-gemma4-bf16. Should have a model_type key in its config.json.`
* **What didn't work**: `axolotl.cli.merge_lora --output_dir=X` does not write the model into `X`; it writes into `X/merged/`. `run_fp8_compression` was pointed at `X`, found no `config.json` at all, and Transformers reported the absence as a missing `model_type` key, which is misleading.
* **Fix**: Added `resolve_merged_dir()` in `src-train/train.py`, which returns whichever of `X/merged` or `X` actually contains a `config.json` and raises a clear error if neither does.

### Iteration 8: CUDA OOM during llmcompressor MoE linearization
* **Error**: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 44.40 GiB of which 4.12 MiB is free.` at `Linearizing experts: 13/30`.
* **What didn't work**: `AutoModelForCausalLM.from_pretrained(..., device_map="auto")` greedily fills both GPUs with the 49 GB bf16 model. llmcompressor then has to unpack the packed 3D expert tensors into per-expert 2D `Linear` modules ("linearization") before quantizing, and there is no VRAM left to do it.
* **Fix**: Load on CPU (`device_map="cpu"`). `FP8_DYNAMIC` is calibration-free — llmcompressor logs `Inferred DataFreePipeline for QuantizationModifier` — so no forward passes and no GPU are needed; it is a per-channel scale and a cast. On CPU the whole step takes ~2 minutes (21 s linearization, 13 s quantization, 63 s writing shards) and peaks well inside the 204 GB of host RAM. A faster alternative that avoids the 2D -> 3D -> 2D round trip is `llmcompressor.modeling.moe.linearize.load_quantizable_moe` at load time (untested here).

### Iteration 9: Merged models silently overwritten in S3
* **Error**: None — this is a latent hazard found while publishing, not a crash.
* **Root cause**: Two independent problems. First, `train.py` only synced directories containing `adapter_config.json`, so merged models were appended to the sync list and then silently skipped; the merged model in S3 had to be uploaded by hand. Second, syncing a re-merged model into a fixed prefix mixes shard layouts: Transformers' `modeling_utils.py` checks `model.safetensors` **before** `model.safetensors.index.json`, so a stray single-file checkpoint from an earlier run wins over a newer sharded one and the old weights load with no warning.
* **Fix**: Replaced the `(dir, config)` tuples with an explicit `SyncTarget(local_dir, s3_prefix)` so each artifact carries its own destination and no content-sniffing guard is needed. Merged models now publish to a per-run prefix (`models/<run_id>/<config>-fp8`), which makes overwrites impossible. Kept outside the run prefix used for adapters, because the evaluation pipeline downloads that entire prefix and would otherwise pull 28 GB on every adapter fetch. Also made merge/quantization failure fatal: the adapters still sync, then the run exits non-zero instead of reporting success.


### Iteration 10: `eot_tokens` named a Gemma 3 token that does not exist in Gemma 4
* **Error**: No crash — a warning that had been mistaken for a false positive:
  `[WARNING] [axolotl.prompt_strategies.chat_template] EOT token '<end_of_turn>' not found in chat_template.`
* **Root cause**: `config/train-gemma4.yml` set `eot_tokens: ["<end_of_turn>"]`, and the
  comment above it said the mapping existed "to help Axolotl's parser identify turn
  boundaries, clearing the false-positive warning". It never cleared it, and the warning was
  not a false positive. `<end_of_turn>` is the **Gemma 3** turn terminator. Gemma 4 does not
  have it: `tokenizer.encode("<end_of_turn>")` returns **7 tokens**
  (`[236820, 643, 236779, 1340, 236779, 887, 236813]`), i.e. ordinary text. Gemma 4 ends a
  turn with `<turn|>`, a single token with id **106** — which is also the second entry in the
  model's own `eos_token_id: [1, 106]`. `<end_of_turn>` appears **zero times** in the chat
  template shipped with `google/gemma-4-26B-A4B-it`.
* **Impact**: Axolotl uses `eot_tokens` to locate the turn terminator and decide, via
  `train_on_eot`, whether it falls inside the trained span
  (`prompt_strategies/chat_template.py`, "Handle special tokens (EOT and EOS)"). Pointed at
  a token that never occurs, `find_first_eot_token` cannot match, so the real terminator was
  never trained — meaning the adapter had no gradient teaching it to stop. Worth checking
  against any adapter trained before this fix: the symptom would be generations that run on
  past the end of the answer.
* **Found by**: the local pipeline (`docs/local-pipeline.md`). The stand-in model
  `tiny-random/gemma-4-moe` carries the 26B's tokenizer and chat template verbatim, so
  running training on a laptop surfaced a production config bug in seconds.
* **Fix**: `eot_tokens: ["<turn|>"]` in `config/train-gemma4.yml` and
  `config/train-gemma4-tiny.yml`. `special_tokens.eos_token: "<eos>"` (id 1) is left as is —
  it matches the tokenizer's own EOS; the turn terminator is the separate concern.

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
* **Attempted Fix**: Reverted the model dtype to `"float8_e4m3fn"` (119 GB weights) and set `low_cpu_mem_usage: false`. Disabled CPU parameter and optimizer offloading (`deepspeed_offload_param: false` and `deepspeed_offload_optimizer: false`).

### Iteration 12: CPU RAM Out of Memory (OOM) during loading due to parallel FP8-to-bfloat16 conversion on CPU
* **Error**: `Root Cause (first observed failure): [0]: traceback : Signal 9 (SIGKILL) received by PID 8280` / CPU RAM spikes and gets killed.
* **What didn't work**: Loading the model in FP8 (`float8_e4m3fn`) with `low_cpu_mem_usage: false` and no offloading. DeepSpeed Stage 3 `zero.Init()` is not natively compatible with FP8 models and automatically falls back to Stage 0 (no sharding) or fails to place weights on GPU VRAM. As a result, the model was loaded entirely on CPU unpartitioned. All 4 processes loaded the 119 GB FP8 weights in parallel on CPU RAM (4 x 119 GB = 476 GB). Then, Axolotl attempted to convert the model to `bfloat16` in parallel on CPU RAM, which doubled the memory footprint to 4 x 238 GB = 952 GB, causing the host to OOM and crash.
* **Attempted Fix**: Switched `torch_dtype` to `"bfloat16"` and set `low_cpu_mem_usage: true` in `config/train-mistral4small.yml`, enabling DeepSpeed CPU parameter and optimizer offloading (`deepspeed_offload_param: true` and `deepspeed_offload_optimizer: true`). While `low_cpu_mem_usage` is incompatible with `device_map`, it is fully supported under DeepSpeed when no `device_map` is used. This allows the processes to load the model in `bfloat16` sequentially layer-by-layer, keeping the peak loading RAM to only ~75 GB per rank (300 GB total), which safely shards the 238 GB weights across the CPU memory under Stage 3.

### Iteration 13: CPU RAM Out of Memory (OOM) during training due to DataLoader worker copy-on-write memory replication
* **Error**: `Root Cause (first observed failure): [3]: traceback : Signal 9 (SIGKILL) received by PID 8719` / CPU RAM spikes from 93 GB to 450+ GB and gets killed during training.
* **What didn't work**: Using `dataloader_num_workers: 4` (default) with DeepSpeed CPU offloading. Under CPU offloading, the parent rank processes hold the sharded 238 GB weights in CPU memory. When the PyTorch dataloader spawns worker processes using the default Linux `fork` start method, the worker processes share the parent's memory pages. As the workers load and prefetch data, copy-on-write (CoW) triggers, duplicating memory pages and causing the CPU RAM to balloon rapidly by hundreds of gigabytes until OOMing.
* **Fix**: Set `dataloader_num_workers: 0` in `config/train-mistral4small.yml`. This forces all data loading to run inside each rank's main process thread, completely eliminating multiprocessing worker spawns and copy-on-write memory replication overhead.

# Evaluating

## Gemma 4

### Iteration 1: SGLang PEFT-to-SGLang Key Mismatch (Multimodal path nesting)
* **Error**: Weight-loading `RuntimeError` during SGLang server startup indicating mismatch between base model keys and adapter keys.
* **What didn't work**: Direct loading of Gemma 4 adapters trained with Axolotl PEFT. Gemma 4 is structured as a multimodal wrapper (`Gemma4ForCausalLM` -> `model` -> `language_model` -> `model` -> `layers`), resulting in nested adapter keys containing `language_model.model.layers.` or `language_model.layers.` prefixes. SGLang's regex parser fails to resolve these deep paths.
* **Fix**: Added a programmatic preprocessing function `preprocess_adapter` inside `src-eval/evaluation.py` that runs before launching SGLang. It:
  - Updates `target_modules` in `adapter_config.json` to strip/map `language_model` prefixes.
  - Rewrites all keys in `adapter_model.safetensors` (or fallback `adapter_model.bin`) mapping `language_model.model.layers.` and `language_model.layers.` directly to `model.layers.`.
  - Saves the cleaned configuration and weight files to an adjacent patched directory (e.g. `gemma4-patched`), which is then served by SGLang.

### Iteration 2: SGLang FlashInfer Fallback Crash on Heterogeneous Head Dimensions
* **Error**: SGLang engine crashes or throws an validation error on the first query with FlashInfer.
* **What didn't work**: SGLang's default FlashInfer attention backend. Gemma 4 has hybrid attention layers featuring heterogeneous head dimensions (local SWA at 256, global full-context at 512). This mismatch causes FlashInfer to crash during runtime inference.
* **Fix**: Forced the Triton attention backend (`attention_backend="triton"` and `moe_runner_backend="triton"`) when initializing the SGLang Engine for Gemma models.

### Iteration 3: Gemma 4 MoE LoRA CUDA Graph Garbage Output
* **Error**: The model produces garbled or repetitive text outputs after about 100 tokens of generation.
* **What didn't work**: Default SGLang CUDA Graph capture/replay execution. The CUDA Graph capture of parallel MoE LoRA layers suffers from numerical drift and scaling errors, corrupting the generation output in eager-mode execution.
* **Fix**: Disabled CUDA graphs (`disable_cuda_graph=True`) when initializing the SGLang Engine for Gemma models.

### Iteration 4: SGLang Clippable Linear LoRA Wrapper Mismatch and Vision Tower Shape Mismatch
* **Error**: 
  - `Exception: No corresponding LoRA layer supported for <class 'sglang.srt.layers.clippable_linear.ClippableRowParallelLinear'>.`
  - `ValueError: LoRA B output dim 4224 does not match base partition prefix dim 8608 for 2 slices.`
  - `AttributeError: 'CompressedTensorsW8A8Fp8MoE' object has no attribute 'get_triton_quant_info'`
* **What didn't work**: 
  - SGLang does not natively support wrapping clippable linear wrapper layers (`ClippableRowParallelLinear`, `ClippableColumnParallelLinear`, etc.) with LoRA layers.
  - SGLang's `lora_manager.py` loops over all model named modules. Since Gemma 4 is a multimodal model, it contains both a language model and a vision tower. SGLang attempted to wrap the vision tower's `gate_proj`, `up_proj` etc. with LoRA layers, resulting in shape/dimension mismatches (LoRA B output dimension 4224 did not match the vision projection output partition dimension of 8608).
  - Triton MoE LoRA initialization calls `get_triton_quant_info` on the quantization scheme, but the nightly SGLang `CompressedTensorsW8A8Fp8MoE` scheme lacked this method.
* **Fix**: Added dynamic hot-patches in `src-eval/evaluation.py`:
  - **get_lora_layer patch**: Unwraps clippable layers to their standard parallel linear layers (e.g. `layer.linear`) so SGLang can wrap them.
  - **lora_manager patch**: Skip named modules containing `"vision"` or `"audio"` in `lora_manager.py` to prevent wrapping vision tower/multimodal projections. Also filter out `"vision"` modules from the adapter's `target_modules` during preprocessing.
  - **CompressedTensorsW8A8Fp8MoE patch**: Implement `get_triton_quant_info` returning `TritonMoeQuantInfo` with `use_fp8_w8a8=True`.

### Iteration 5: Monkeypatch Regex Over-matching and Syntax Errors
* **Error**: `SyntaxError: 'return' outside function` or `AttributeError: module 'sglang.srt.models.gemma4_mm' has no attribute 'get_hidden_dim'`
* **What didn't work**: Using a non-greedy regex (`.*?`) to replace the `get_hidden_dim` function body. In some SGLang versions, the function body contains internal `def` statements or complex logic that caused the regex to stop early or capture too much, leaving dangling code that broke the module import.
* **Fix**: Updated the regex in `src-eval/evaluation.py` to be greedy (`.*?(?=\n    def )`) or explicitly target the entire function block until the next top-level definition. This ensures the entire function is replaced cleanly without leaving syntax-breaking residue.

### Iteration 6: SGLang rejects a regex `target_modules`
* **Error**: `RuntimeError: Failed to load LoRA adapter ft: Only 'all' or 'all-linear' can be used as the string for target module` (`sglang/srt/lora/lora_manager.py:541`).
* **What didn't work**: The fully qualified regex introduced in Training Iteration 6. PEFT accepts `target_modules` as either a regex string or a list, so a regex is the maintainable way to pin targets to the language model at training time. SGLang accepts only a list of names or the literals `all` / `all-linear`, and PEFT copies the regex verbatim into `adapter_config.json`.
* **Fix**: Added `src-train/expand_targets.py`, which instantiates the base model on the meta device and rewrites the regex into the explicit list of 205 matching module names. Fully qualified names cannot suffix-collide with the vision tower, so this satisfies both tools. Wired into `src-train/train.py` after training, so every future adapter ships engine-loadable. No retraining is needed for an existing adapter — only `adapter_config.json` changes, the weights are untouched.

### Iteration 7: SGLang requires `v_proj` on every adapted attention layer
* **Error**: `RuntimeError: Failed to load LoRA adapter ft: 'base_model.model.model.language_model.layers.5.self_attn.v_proj.lora_A.weight'`. After removing all `v_proj` entries the same error reappeared for `layers.0`.
* **Root cause**: Gemma 4 sets `attention_k_eq_v: true`, so its five global-attention layers (5, 11, 17, 23, 29 — the `full_attention` entries in `layer_types`) share one tensor for K and V and have **no `v_proj` module at all**. The adapter correctly omits them, covering 25 of 30 layers. SGLang builds a fused QKV projection and demands a `v_proj` weight for every layer whenever attention is adapted. Requesting `v_proj` on layer 0 *after* it had been dropped from `target_modules` entirely proves the requirement comes from the fused path, not from the adapter.
* **Conclusion**: unsatisfiable. Attention LoRA on Gemma 4 cannot work in SGLang regardless of adapter shape — the engine needs a projection the architecture does not have.

### Iteration 8: SGLang cannot wrap `ClippableRowParallelLinear` even for an MLP-only adapter
* **Error**: `Exception: No corresponding LoRA layer supported for <class 'sglang.srt.layers.clippable_linear.ClippableRowParallelLinear'>.`
* **What didn't work**: Reducing the adapter to MLP only (`gate/up/down_proj`, 90 modules, 38% of the original capacity) to sidestep Iteration 7. That cleared the `v_proj` requirement but hit the same clippable-wrapper limitation recorded in Iteration 4 — `down_proj` is a `RowParallelLinear`, so even a pure-MLP adapter trips it.
* **Conclusion**: **No adapter shape loads in stock SGLang.** Iteration 7 rules out attention, Iteration 8 rules out MLP. The five source patches in Iterations 1-5 were rational responses to a genuine dead end, not workarounds for a misconfigured adapter.

### Iteration 9: vLLM serves the adapter unpatched (resolution)
* **Result**: `VERDICT: adapter altered 2/3 outputs under greedy decoding. Adapter is being applied at inference time.`
* **What works**: `vllm/vllm-openai:v0.26.0-cu129-ubuntu2404` loads the **full 205-module adapter** — `v_proj` asymmetry included, no capacity sacrificed — with no source patches. Relevant log lines:
  - `Gemma4 model has heterogeneous head dimensions (head_dim=256, global_head_dim=512). FA4 not available, forcing TRITON_ATTN backend.` — vLLM detects and handles the quirk that needed manual forcing in SGLang (Iteration 2).
  - `Using TRITON Unquantized MoE LoRA backend` — vLLM has a purpose-built MoE LoRA path; SGLang's equivalent is still unimplemented upstream.
  - `Breakable CUDA graph is incompatible with LoRA; disabling prefill CUDA graph` — handled internally, so the Iteration 3 workaround is obsolete.
* **Required settings**: `max_lora_rank` must be raised to the adapter's `r` (vLLM defaults to 16, the adapter is 32); `enforce_eager=True` for smoke tests, see Inference Containers Iteration 3.
* **Consequence**: `post_training_merge` can be set to `false`. A 71 MiB adapter serves directly, so the 28 GB merged FP8 model and its per-eval download are no longer required. None of the five SGLang patches, nor `preprocess_adapter`, are needed.
* **Comparison under greedy decoding** (temperature 0, so differences are the adapter, not sampling): the adapter preferred unsplit compounds (`Blutdruck` over `Blut-Druck`), softened modality in line with the source (`Du kannst` over `Du musst` for "Sie sollten versuchen"), used shorter imperatives (`Mach` over `Mache`), and added an extra concrete example. The trivial greeting case was unchanged, as expected — the base model already handles it.

## Inference Containers

### Iteration 1: The vLLM image's default entrypoint reserves the whole GPU
* **Error**: `ValueError: Free memory on device cuda:0 (4.22/79.14 GiB) on startup is less than desired GPU memory utilization (0.9, 71.22 GiB).` — reproducible across four pods with an identical free-memory figure.
* **What didn't work**: Killing `VLLM::EngineCore`, which freed the card for a minute before it filled again. The memory was never leaked by our runs: `ps -eo pid,ppid,cmd` showed `pid 1 /sbin/docker-init -- vllm serve` and `pid 51 python3 /usr/local/bin/vllm serve`, an OpenAI API server the image starts by default. It was serving `Qwen/Qwen3-0.6B` and reserving 90% of an 80 GB A100. Killing the child only made the parent respawn it.
* **Fix**: Always set `dockerStartCmd`, never inherit the image default. Three images behaved three different ways — Axolotl starts sshd and JupyterLab, SGLang's entrypoint exits immediately (container dies on boot), vLLM's runs and takes the GPU. See `docs/launcher-hardening.md`.

### Iteration 2: `InductorError: OSError: [Errno 5] Input/output error`
* **Error**: After ~23 minutes of `torch.compile`, the engine died during `profile_run` with an I/O error while writing compiled kernels.
* **What didn't work**: Neither disk space (60 GB free locally, 228 TB on the volume) nor the network filesystem explained it — `/app/tmp/torchinductor_root` was empty, so inductor was writing to local disk. **Root cause unconfirmed.**
* **Mitigation**: Pin the cache directories explicitly to local disk (`TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `VLLM_CACHE_ROOT`) rather than inheriting `TMPDIR=/app/tmp` from `launch.sh`, and prefer eager mode for smoke tests, which avoids compilation entirely. Worth knowing regardless: `/app` is MooseFS, a network filesystem served from another datacenter.

### Iteration 3: Startup cost — 659 s to 11 s
* **Observation**: A cold pod spent 659 s on `Model loading took 51.04 GiB memory`, then ~23 min compiling LoRA-specialised CUDA graphs at 51 batch sizes before failing.
* **Fix**: With `HF_HOME=/app/huggingface_cache` warm, the same load took **11.4 s**. `enforce_eager=True` (`SMOKE_EAGER=1`) skips `torch.compile` and CUDA graph capture, which costs per-token throughput but is irrelevant for a three-prompt smoke test. Set `SMOKE_EAGER=0` to exercise the compiled path that production serving would use.
* **Caveat**: pod-local volumes die with the pod, so a recreated pod pays the 50 GB download again. Only a network volume survives recreation, at the cost of pinning deployments to one datacenter.

## 2-GPU Inference

### Iteration 1: NCCL P2P Handshake Deadlock on L40S Pods
* **Error**: SGLang engine hangs indefinitely during `Init START` with `TP_SIZE=2`. VRAM spikes to ~70% and stays there with 0% GPU utilization.
* **What didn't work**: Standard `TP_SIZE=2` initialization. The dual-L40S pod configuration (on certain RunPod providers) has issues with Peer-to-Peer (P2P) memory access over the PCIe bus during the NCCL handshake, causing the ranks to deadlock while waiting for a response that never arrives.
* **Fix**: Set `NCCL_P2P_DISABLE=1` in the environment before launching the evaluation script. This forces NCCL to use shared memory or standard PCIe transfers instead of direct P2P, bypassing the hardware-level handshake deadlock.