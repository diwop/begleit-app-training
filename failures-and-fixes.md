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



## Mistral

...

# Evaluating

...

## 

...

## 

...