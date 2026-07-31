# Local Pipeline (Apple Silicon)

Runs the whole train → adapter → serve handoff on a MacBook in about a minute, with no
pod, no S3 and no 50 GB download. It exists because that handoff is where every real bug
in this project has been (see `failures-and-fixes.md`), and iterating on it in the cloud
costs a pod boot each time.

The Python **and the scripts** are the same ones RunPod runs: `scripts/train.sh` and
`scripts/eval.sh` work on both, with `scripts/lib/platform.sh` holding every difference.
Only the environment and the config change.

## Setup (once, ~8 minutes)

```bash
bash scripts/setup.sh
```

Builds two virtualenvs under `.local/`. Training and serving cannot share one — Axolotl and
vLLM disagree on `transformers`, the same split the RunPod containers enforce. vLLM has no
macOS wheel and must be compiled from source, which is most of the eight minutes.

Requires `uv` and the Xcode Command Line Tools. `brew install ccache` is optional but makes
rebuilds substantially faster.

The script is **idempotent and resumable**: each step is skipped when already satisfied, so
a second run takes seconds and an interrupted build picks up where it left off. To force a
vLLM rebuild, delete `.local/venv-eval`.

### The vLLM revision is pinned, and matches RunPod

The pin lives in **`src-eval/pyproject.toml`**, as a PEP 508 direct reference:

```toml
# vllm-release: v0.26.0
local = ["vllm @ git+https://github.com/vllm-project/vllm.git@568afb3a... ; sys_platform == 'darwin'"]
```

That is the exact commit behind the **v0.26.0** release, the same revision as
`vllm/vllm-openai:v0.26.0-cu129-ubuntu2404` in `eval_image`, so local and cloud run the
same engine. uv fetches and caches the checkout by commit — there is no manual clone.

It is pinned rather than tracking `main` because vLLM's `main` does not always build on
macOS, and an unpinned clone gives every developer a different day's luck: the first
version of this setup cloned `main` and broke within a day of being written.

**Bump `eval_image` and the `# vllm-release:` marker together.**
`scripts/lib/platform.sh` compares them on every run and refuses to start if they drift,
so the two cannot silently diverge. Also update the SHA to that release's commit, and the
`torch` pin in the `build` extra (see the segfault note below).

## Run

```bash
bash scripts/train.sh && bash scripts/eval.sh
```

`train.sh` trains on Metal and expands the adapter's regex `target_modules` into explicit
names; `eval.sh` serves it. Success looks like:

```
VERDICT: adapter altered 3/3 outputs under greedy decoding.
✅ Adapter is being applied at inference time.
```

## Troubleshooting

**`cmake --build ... returned non-zero exit status 1` during setup.** Almost always the
vLLM build against a revision that does not compile on macOS. The real compiler error is
far above the Python traceback in the output; the traceback tail only says "a subprocess
failed". Try another revision by editing the direct reference in `src-eval/pyproject.toml`.

**The build eats the machine.** vLLM reads `MAX_JOBS` (not `CMAKE_BUILD_PARALLEL_LEVEL`)
and defaults to every core. The script sets 6; lower it with `MAX_JOBS=4`.

**`CMake Error ... 'ninja' '--version' failed with: no such file or directory`.** A stale
CMake cache inside uv's cached git checkout, pinning an absolute path into a build
environment that has since been deleted. `uv cache clean vllm` does *not* clear it — the
checkout lives separately. Remove it and rebuild:

```bash
rm -rf ~/.cache/uv/git-v0 && rm -rf .local/venv-eval && bash scripts/setup.sh
```

**`Segmentation fault: 11` in `xgrammar::__TVMFFIStaticInitFunc0` when serving.** vLLM's
native extensions were built against a different torch than the one installed. The `torch`
pin in `src-eval/pyproject.toml`'s `build` extra must equal the darwin pin in
`requirements/cpu.txt` **at the pinned vLLM revision** — different revisions want different
torch (v0.26.0 wants 2.11.0; main wanted 2.13.0). There is no resolution error and no
Python traceback, so check the pin first:

```bash
curl -s https://raw.githubusercontent.com/vllm-project/vllm/<sha>/requirements/cpu.txt | grep ^torch
```

**`ModuleNotFoundError: No module named 'vllm'` when running the pipeline.** The eval venv
exists but its build did not finish. Re-run `scripts/setup.sh` — it resumes.

**Training is killed, or the machine swaps.** `sequence_len` is the knob; see the memory
note below. Do not raise it toward `base.yml`'s 16384 on a laptop.

## What is and is not exercised

**Is:** the Gemma 4 MoE architecture and all its quirks, the LoRA target regex, Axolotl's
prompt masking, the regex→list expansion, the adapter audit, and vLLM serving a LoRA
adapter onto a MoE base with heterogeneous attention.

**Is not:** output quality. `tiny-random/gemma-4-moe` is randomly initialised, so
generations are gibberish by construction. The smoke test asserts that the adapter
*changes* the output, not that it improves it. Nor does this say anything about whether
16K context fits on 2x L40S — that remains a RunPod question.

## Why the stand-in model works

`tiny-random/gemma-4-moe` is `google/gemma-4-26B-A4B-it`'s own config with the dimensions
shrunk, so at 5.4 MB it still reproduces every quirk that has broken things:

| Quirk | Real 26B-A4B | Tiny |
|---|---|---|
| `Gemma4ForConditionalGeneration` wrapper | ✓ | ✓ |
| Packed 3D `Gemma4TextExperts` (invisible to PEFT) | 128 experts | 128 experts |
| `attention_k_eq_v` — full-attn layers with **no `v_proj`** | 5 of 30 | 2 of 4 |
| Mixed head dims (sliding / global) | 256 / 512 | 32 / 64 |
| Vision tower in `Gemma4ClippableLinear` wrappers | ✓ | ✓ |

Its tokenizer and chat template are copied verbatim from the 26B, so prompt formatting is
the real thing.

## Platform notes

**Training uses the GPU, serving does not.** PyTorch's Metal backend (MPS) runs training on
the GPU cores. vLLM on macOS is CPU-only — `VLLM_TARGET_DEVICE` is forced to `cpu` and
there is no Metal support. Fine at this size.

**M1 and M2 have no hardware bfloat16.** Axolotl converts bf16→fp16 on Metal; this config
sets both false and trains in fp32.

**Memory is dominated by the vocabulary, not the model.** The stand-in is 2.7M parameters,
but the vocabulary is the real 262144, so the logits tensor (`seq_len × 262144 × 4 bytes`)
is the largest allocation by far. Measured peak on an M1 Max at `sequence_len: 2048`:
9.75 GiB over 2 steps, 13.98 GiB over 30. **Do not inherit `base.yml`'s 16384** — that
extrapolates to something that will swap a laptop. `config/train-gemma4-tiny.yml` sets its
own, and `scripts/train.sh / scripts/eval.sh` caps threads and bounds Metal so an unexpected
allocation errors out rather than pushing the machine into swap.

## The three Apple Silicon patches

All in `src-train/mps_patch.py`, all no-ops off Metal, so RunPod is unaffected.

1. **`device_map`.** `ModelLoader._set_device_map_config` unconditionally overwrites
   device_map with `"mps:0"`, ignoring the config — and that value hangs. Measured on an
   M1 Max: `"auto"` hangs, `"mps"` segfaults, `"mps:0"` hangs, `{"": "mps"}` hangs,
   `{"": 0}` works. The patch rewrites it to `{"": 0}`. This single line is also why plain
   `from_pretrained(..., device_map="auto")` dies on Metal in *any* framework.
2. **Triton.** `axolotl/monkeypatch/trainer/__init__.py` eagerly imports a module with a
   top-level `import triton`; Triton has no macOS wheel. Its two functions are used only by
   the reinforcement-learning trainer, so the patch stubs that one leaf module. Stubbing
   `triton` itself does not work — torch probes `triton.backends` and the failure cascades.
3. **`torchao`.** Plain missing dependency; `scripts/setup.sh` installs it.

Plus config: `lora_mlp_kernel` / `lora_qkv_kernel` / `lora_o_kernel: false`, since Axolotl's
fused LoRA kernels are Triton-only too.

## The CPU-only LoRA restriction

`smoke_adapter.py` passes `lora_target_modules` to vLLM **only when the backend is CPU**.

At engine start — before any adapter is read — vLLM wraps every supported module for LoRA,
including Gemma 4's `FusedMoE` layers. On CPU those use a monolithic kernel and the wrapper
asserts `Monolithic kernels are not supported for Fused MoE LoRA`. The decision comes from
vLLM's own `lora_config.target_modules` and never from `adapter_config.json`, so no
training-side setting avoids it — it is an engine-side knob that happens to share a name
with the Axolotl one.

On CUDA the fused path works (it logs `Using TRITON Unquantized MoE LoRA backend`) and is
what would apply expert LoRA weights if an adapter ever carried them. PEFT cannot produce
those today — Gemma 4's experts are packed 3D `nn.Parameter`s it cannot see — but
suppressing the path globally would silently skip them if that changes, so the workaround
stays scoped to CPU and the GPU arguments are unchanged.

Not to be confused with `TRITON_ATTN`, the attention backend vLLM auto-selects for Gemma
4's heterogeneous head dimensions. That one is load-bearing and untouched.

## Gotcha: a false "not being applied"

PEFT initialises every `lora_B` to exactly zero, so a freshly trained adapter is
mathematically the identity until `B` moves. A 2-step run at the production learning rate
reached `max|lora_B| = 3e-4` — far too small to change any argmax over 262144 tokens — and
the smoke test reported *"it loaded but is not being applied"*, which sends you after
entirely the wrong bug. 30 steps at `1e-2` reach `0.14` and alter every prompt.

`smoke_adapter.py` now inspects `max|lora_B|` on failure and says which case it is. The
tiny config is deliberately tuned for a measurable delta rather than for quality.
