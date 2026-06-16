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
* **Error**: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 9.97 GiB. GPU 1 has a total capacity of 44.39 GiB ...`
* **What didn't work**: Using the base `sequence_len: 16384`. Since Gemma models have a large vocabulary size (256,000), representing and casting the logits tensor to `float32` (via `logits.float()`) during the loss computation layer requires ~25 GB of VRAM at 16k context, leading to OOM on L40S.
* **Fix**: Reduced `sequence_len` to `10880` in `config/train-gemma4.yml`. Since the dataset's maximum sequence length is `10853`, this saves over 8.4 GB of VRAM at peak while preserving all tokens without truncation.

## Mistral

...

# Evaluating

...

## 

...

## 

...