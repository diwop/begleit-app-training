# --- src/launcher.py ---
import os
import json
import time
import hashlib
import shutil
import datetime
import torch
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from omegaconf import DictConfig, OmegaConf

# Force Hugging Face to use the persistent volume cache directory to prevent downloading to
# container root disk. `setdefault`, not assignment: on a laptop these point into .local/
# (scripts/lib/platform.sh), and an unconditional write also fired on plain `import train`
# under pytest.
os.environ.setdefault("HF_HOME", "/app/huggingface_cache")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))

# Everything this run writes. Overridden per platform so the same code serves a RunPod
# volume and a laptop checkout.
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "/app/output")

# TRAIN_CONFIG lets scripts/lib/platform.sh point a laptop at the tiny stand-in config
# without editing this list. Unset, the production Gemma 4 config runs, as before.
TRAINING_PIPELINE = [
    os.environ.get("TRAIN_CONFIG", "config/train-gemma4.yml"),
    # "config/train-mistral4small.yml" Mistral is not feasible on RunPod (OOM at 4x L40S)
]

# Where src-eval/evaluation.py expects to find merged models, locally and in S3.
# The S3 prefix is versioned per run: overwriting a merged model in place is unsafe,
# because transformers prefers a stray model.safetensors over a newer shard index and
# would silently load the older weights.
MERGED_DIR = os.path.join(OUTPUT_ROOT, "merged")
MERGED_S3_PREFIX = "models"


@dataclass
class SyncTarget:
    """A directory to publish and the S3 prefix it belongs under."""
    local_dir: str
    s3_prefix: str


def _merge_by_step(entries: Iterable[dict]) -> List[dict]:
    """One row per evaluation, not two.

    Trainer logs `eval_loss` when it finishes evaluating, and the callback in
    src-train/validation_metrics.py logs its `eval_ls_*` numbers immediately afterwards.
    Both carry the same `step`, so the history holds two half-rows per evaluation -- which
    makes the file annoying to read and every naive plot of it wrong.
    """
    merged: Dict[int, dict] = {}
    for entry in entries:
        merged.setdefault(entry.get("step"), {}).update(entry)
    return [merged[step] for step in sorted(merged)]


def write_eval_metrics(output_dir: str) -> None:
    """Lift the metric history out of the checkpoints and into one readable file.

    HF Trainer keeps every logged number in `trainer_state.json` under `log_history`, but
    only inside `checkpoint-*/`. That is an awkward place for it: `save_total_limit` rotates
    checkpoints away, `save_strategy: "no"` would drop the history entirely, and
    src-eval/smoke_adapter.py deliberately skips `checkpoint-*` when fetching an adapter
    from S3 -- so the eval container could never see the curve that produced the weights it
    is serving. Written next to the adapter, it travels with it.

    This is `eval_metrics.json` from docs/train-eval-review.md P1-1.
    """
    checkpoints = sorted(Path(output_dir).glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[-1]))
    if not checkpoints:
        print(f"⚠️ [WARNING] No checkpoint under {output_dir}; no metric history to save.")
        return

    # The last checkpoint's state contains the whole run, not just its own step.
    state = json.loads((checkpoints[-1] / "trainer_state.json").read_text(encoding="utf-8"))
    history: List[dict] = state.get("log_history", [])

    metrics = {
        "source": str(checkpoints[-1].name),
        # Which metric this is comes from the config; run_manifest.json pins that by hash.
        "best_metric": state.get("best_metric"),
        "best_model_checkpoint": (Path(state["best_model_checkpoint"]).name
                                  if state.get("best_model_checkpoint") else None),
        "global_step": state.get("global_step"),
        # Split by kind because the two are read for different reasons: the evaluations are
        # the result, the training entries are how you tell overfitting from underfitting.
        "evaluations": _merge_by_step(e for e in history if any(k.startswith("eval_") for k in e)),
        "training": [e for e in history if "loss" in e],
    }
    target = Path(output_dir) / "eval_metrics.json"
    target.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(f"📊 Wrote {target} ({len(metrics['evaluations'])} evaluations)", flush=True)


def write_run_manifest(output_dir: str, config_path: str, merged_cfg: DictConfig,
                       run_id: str) -> None:
    """Record what this run trained on, next to the adapter it produced.

    Evaluation happens in a different container on a different pod, so nothing physically
    ties a score to the run that earned it. This file is that tie: it names the base model,
    hashes the resolved config, and copies the split manifest, so the eval container can
    show that the documents it scored are the ones this adapter never saw.
    """
    target = Path(output_dir) / "run_manifest.json"
    if not target.parent.is_dir():
        # Axolotl created this directory to write the adapter into. If it is not there,
        # the run did not produce one and there is nothing to describe.
        print(f"⚠️ [WARNING] {target.parent} does not exist; skipping the run manifest.")
        return

    split_manifest_path = Path("data/split_manifest.json")
    split_manifest = (json.loads(split_manifest_path.read_text(encoding="utf-8"))
                      if split_manifest_path.exists() else None)
    if split_manifest is None:
        print(f"⚠️ [WARNING] {split_manifest_path} is missing; the run manifest cannot "
              f"record which ids were held out.")

    resolved = OmegaConf.to_yaml(merged_cfg)
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "config": config_path,
        "config_sha256": hashlib.sha256(resolved.encode("utf-8")).hexdigest(),
        "base_model": str(merged_cfg.get("base_model", "")),
        "num_epochs": merged_cfg.get("num_epochs"),
        "sequence_len": merged_cfg.get("sequence_len"),
        "split_manifest": split_manifest,
    }
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(f"📝 Wrote {target}", flush=True)

def detect_accelerator() -> Tuple[str, int]:
    """Returns (accelerator, worker_count).

    "cuda" is the production path: DeepSpeed ZeRO-3 across the visible GPUs. "mps" is a
    laptop running the tiny stand-in config, where DeepSpeed, FlashAttention-2 and NCCL all
    do not exist. Returning a count of 1 for mps keeps `accelerate launch --num_processes`
    honest without pretending Metal is a GPU cluster.
    """
    if torch.cuda.is_available():
        return "cuda", torch.cuda.device_count()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", 1
    return "cpu", 1


def merge_configs(base_path: str, override_path: str):
    """Loads and merges a base YAML and an override YAML. Override values take precedence."""
    base_cfg = OmegaConf.load(base_path)
    override_cfg = OmegaConf.load(override_path)
    return OmegaConf.merge(base_cfg, override_cfg)

def pre_download_models(pipeline_configs):
    """
    Sequentially pre-stages base models in a single-process environment.
    CRITICAL: Immediately aborts execution if HF_TOKEN is missing or empty.
    """
    # Deliberately NOT a pre-flight token check. Gating is a property of the repo, not of
    # the name: google/gemma-4-26b-a4b-it is gated, google/gemma-4-E2B-it is not, and a
    # prefix heuristic wrongly blocked the ungated one. Attempt the download and let the
    # failure explain itself.
    
    print("\n" + "="*60)
    print("📥 PRE-STAGING BASE MODELS (Single-Process Cache Warmup)")
    print("="*60, flush=True)
    
    processed_models = set()
    
    for config_path in pipeline_configs:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌ Configuration matrix error: Target file missing '{config_path}'")
            
        merged_cfg = merge_configs("config/base.yml", config_path)
        base_model_str = str(merged_cfg.get("base_model", "")).strip()
        
        if base_model_str and base_model_str not in processed_models:
            print(f"📦 Invoking native hf engine for: '{base_model_str}'...", flush=True)
            
            try:
                # The Python API, not the `hf` console script: callers invoke this file by
                # interpreter path without activating a venv, so the script is not
                # necessarily on PATH. Same downloader underneath, and it still picks up
                # HF_TOKEN and HF_HOME from the environment.
                from huggingface_hub import snapshot_download

                # The 51 GB base model lives in a dozen large shards, so the default
                # 8 download workers give no parallelism worth having -- the reporter
                # showed "1 file(s) in flight" for most of every run, and a single
                # connection dropping to 9 MB/s stalled everything behind it while
                # another run of the same download peaked at 481 MB/s. hf_transfer
                # issues parallel range requests WITHIN one file, which is the shape
                # of this problem. Opt in only when it is importable: huggingface_hub
                # raises if the flag is set without it.
                try:
                    import hf_transfer  # noqa: F401

                    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
                    print("   using hf_transfer for the download", flush=True)
                except ImportError:
                    print("   hf_transfer not installed; falling back to the default "
                          "downloader (expect a slow, serial transfer)", flush=True)

                snapshot_download(base_model_str)

                print(f"✅ Weight cache successfully validated for: {base_model_str}\n", flush=True)
                processed_models.add(base_model_str)
            except Exception as e:
                print(f"\n❌ CRITICAL: failed to download {base_model_str}!")
                print(f"   {type(e).__name__}: {e}")
                if not os.environ.get("HF_TOKEN", "").strip():
                    print("\n💡 HF_TOKEN is not set. If this repo is gated (as")
                    print("   google/gemma-4-26b-a4b-it and Mistral Small are), you need one:")
                    print('       export HF_TOKEN="hf_your_token_here"')
                sys.exit(1)
                
    print("="*60 + "\n🏁 All base model weights are cached locally. Ready for distributed execution.\n")

def generate_runtime_deepspeed(
    output_json_path: str,
    cpu_checkpointing: bool = False,
    offload_optimizer: bool = False,
    offload_param: bool = False,
    param_persistence_threshold: str = "auto"
) -> str:
    """
    Reads the base Axolotl ZeRO-3 template and injects a high-performance, 
    local VRAM policy or offloads states to CPU if requested to avoid OOM.
    """
    source_ds_path = "/workspace/axolotl/deepspeed_configs/zero3_bf16.json"
    
    if os.path.exists(source_ds_path):
        with open(source_ds_path, "r", encoding="utf-8") as f:
            ds_dict = json.load(f)
    else:
        # Fallback structural configuration blueprint
        ds_dict = {
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {"device": "none"},
                "offload_param": {"device": "none"},
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto"
            }
        }

    # Enforce strict Stage 3 parameter sharding across distributed nodes
    ds_dict["zero_optimization"]["stage"] = 3
    ds_dict["zero3_init_flag"] = True
    
    # LoRA optimization: Do not gather full base model weights during saves to prevent massive VRAM OOM spikes on rank 0
    ds_dict["zero_optimization"]["gather_16bit_weights_on_model_save"] = False
    
    # Optimizer and parameter offloading configurations
    ds_dict["zero_optimization"]["offload_optimizer"] = {"device": "cpu" if offload_optimizer else "none"}
    ds_dict["zero_optimization"]["offload_param"] = {"device": "cpu" if offload_param else "none"}
    
    # Parameter sharding threshold configuration (useful for MoE expert models)
    if param_persistence_threshold == "0" or param_persistence_threshold == 0:
        ds_dict["zero_optimization"]["stage3_param_persistence_threshold"] = 0
    else:
        try:
            val = int(param_persistence_threshold)
            ds_dict["zero_optimization"]["stage3_param_persistence_threshold"] = val
        except (ValueError, TypeError):
            ds_dict["zero_optimization"]["stage3_param_persistence_threshold"] = "auto"

    # Inject Long-Context Activation Protection
    ds_dict["activation_checkpointing"] = {
        "partition_activations": True,
        "contiguous_memory_optimization": True,
        "cpu_checkpointing": cpu_checkpointing
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(ds_dict, f, indent=2)
        
    print(f"✅ DeepSpeed Stage 3 configuration compiled successfully at: {output_json_path} (cpu_checkpointing={cpu_checkpointing}, offload_optimizer={offload_optimizer}, offload_param={offload_param}, param_persistence_threshold={ds_dict['zero_optimization']['stage3_param_persistence_threshold']})")
    return output_json_path

def run_training_job(config_path: str, num_gpus: int, accelerator: str = "cuda") -> Tuple[str, DictConfig, str]:
    """
    Loads YAML parameters, binds unified runtime DeepSpeed assets, 
    and launches the distributed training engine without model-specific hardcodes.
    """
    print("\n" + "="*60)
    print(f"🎬 INITIATING PIPELINE TRAINING JOB: {config_path}")
    print("="*60, flush=True)

    # Ingest core configuration parameters across the inheritance layer
    merged_cfg = merge_configs("config/base.yml", config_path)
    config_filename = os.path.basename(config_path).replace(".yml", "").replace(".yaml", "")
    temp_yaml_path = f".merged-{config_filename}.yml"
    runtime_ds_path = f".ds-config-{config_filename}.json"

    # NATIVE FLOAT8 DATATYPE RECONCILIATION:
    # Pull the native datatype to the root level to prevent Axolotl from 
    # forcing a conflicting BF16 upcast on pretrained FP8 model layers.
    if "extra_model_config_kwargs" in merged_cfg and "torch_dtype" in merged_cfg["extra_model_config_kwargs"]:
        merged_cfg["torch_dtype"] = merged_cfg["extra_model_config_kwargs"]["torch_dtype"]

    # FlashAttention-2 is CUDA-only. On Metal the config supplies sdpa instead, so only
    # default it in when we actually have CUDA.
    if "attn_implementation" not in merged_cfg and accelerator == "cuda":
        merged_cfg["attn_implementation"] = "flash_attention_2"

    # Attention overrides from the environment, so bisecting a failure on a GPU pod costs a
    # relaunch rather than a commit, a push and a wait.
    #
    # THESE TWO GO TOGETHER. Gemma 4's five global layers have head_dim 512 and
    # FlashAttention-2 refuses anything above 256 (failures-and-fixes.md, Training
    # Iteration 1). `gemma4_hybrid_attn_impl` exists to route exactly those five to SDPA
    # while the other 25 keep FA2, which is what made 16k context fit (Iteration 3).
    # Turning it off therefore requires attn_implementation=sdpa everywhere -- and buys
    # back the memory problem Iteration 3 solved. Setting one without the other just
    # reproduces a failure we already understand.
    for env_key, cfg_key, parse in (
        ("ATTN_IMPLEMENTATION", "attn_implementation", str),
        ("GEMMA4_HYBRID_ATTN", "gemma4_hybrid_attn_impl", lambda v: v == "1"),
        ("EVAL_STRATEGY", "eval_strategy", str),
        ("TF32", "tf32", lambda v: v == "1"),
    ):
        raw = os.environ.get(env_key, "")
        if raw:
            merged_cfg[cfg_key] = parse(raw)
            print(f"⚙️  {env_key}={raw} overrides {cfg_key} -> {merged_cfg[cfg_key]}", flush=True)

    # EVAL_STRATEGY=no is the escape hatch while evaluation is broken: it gets a trained
    # adapter out of a pod that would otherwise die at step 0. It has to drag
    # load_best_model_at_end with it -- HF refuses the combination, since there would be no
    # metric to choose a checkpoint by. Nothing is measured in this mode; the run produces
    # weights and no evidence that they are any good.
    if str(merged_cfg.get("eval_strategy", "")).lower() in ("no", "none"):
        # test_datasets has to go, and it is the ONLY thing that actually works.
        # axolotl/core/builders/base.py:515 decides evaluation like this:
        #
        #   if not self.eval_dataset and self.cfg.val_set_size == 0:
        #       training_args_kwargs["eval_strategy"] = "no"     # the only real off switch
        #   elif self.cfg.eval_steps:      ... eval_on_start = True
        #   elif self.cfg.eval_strategy:   ... eval_on_start = True
        #
        # `self.cfg.eval_strategy` is the STRING "no", which is truthy, so asking for
        # eval_strategy: no lands in the third branch and switches eval_on_start ON. HF
        # then evaluates once before training regardless of the strategy
        # (trainer.py:1514, `if args.eval_on_start`), which is why a run with evaluation
        # supposedly disabled still died in prediction_step at step 0.
        #
        # eval_steps goes too, or axolotl's schema rejects the pair. save_strategy and
        # save_steps stay: checkpoints feed eval_metrics.json and the published adapter,
        # and have nothing to do with evaluating.
        merged_cfg.pop("test_datasets", None)
        merged_cfg.pop("eval_steps", None)
        merged_cfg["load_best_model_at_end"] = False
        print("⚠️  eval_strategy=no: test_datasets and eval_steps dropped, "
              "load_best_model_at_end forced off. This run produces NO validation "
              "numbers at all.", flush=True)

    # Extract DeepSpeed tuning settings from Axolotl YAML if configured
    cpu_checkpointing = merged_cfg.get("deepspeed_cpu_checkpointing", False)
    offload_optimizer = merged_cfg.get("deepspeed_offload_optimizer", False)
    offload_param = merged_cfg.get("deepspeed_offload_param", False)

    # DEEPSPEED_OFFLOAD=0 keeps parameters and optimizer state on the GPU.
    #
    # The evaluation crash is not a bad matmul: with ROPE_DEBUG the reporter ran
    # `ones(2,2) @ ones(2,2)` on the same device immediately beforehand and got the same
    # CUBLAS_STATUS_INVALID_VALUE, so cuBLAS is unusable by then and the rotary embedding
    # is a bystander. The same report showed only 24 MiB allocated on a card holding a 26B
    # model -- the weights are not resident. With offload on, ZeRO-3 keeps them in CPU
    # memory and gathers them through the DeepSpeed engine, and the engine's frame is
    # absent from every eval traceback: HF's prediction_step calls the module directly.
    #
    # These settings were written for 2x L40S (2x44 GB), where a 52 GB bf16 model has to be
    # offloaded. On a single 96 GB card it fits, so this is worth having as a switch
    # regardless of whether it turns out to be the fix.
    if os.environ.get("DEEPSPEED_OFFLOAD") == "0":
        offload_param = offload_optimizer = False
        print("⚙️  DEEPSPEED_OFFLOAD=0: parameters and optimizer stay on the GPU", flush=True)
    param_persistence_threshold = merged_cfg.get("deepspeed_param_persistence_threshold", "auto")

    # DeepSpeed is CUDA-only. On Metal it is not merely unnecessary, it cannot load, so
    # neither the config nor the accelerate flags may be produced.
    use_deepspeed = accelerator == "cuda"
    if use_deepspeed:
        generate_runtime_deepspeed(
            runtime_ds_path,
            cpu_checkpointing=bool(cpu_checkpointing),
            offload_optimizer=bool(offload_optimizer),
            offload_param=bool(offload_param),
            param_persistence_threshold=param_persistence_threshold
        )
        merged_cfg["deepspeed"] = runtime_ds_path
    else:
        merged_cfg.pop("deepspeed", None)
        for key in [k for k in merged_cfg if str(k).startswith("deepspeed_")]:
            merged_cfg.pop(key, None)

    if not merged_cfg.get("output_dir"):
        merged_cfg["output_dir"] = os.path.join(OUTPUT_ROOT, "adapter", config_filename)

    # Save the resolved, finalized configuration path for Axolotl to consume
    OmegaConf.save(config=merged_cfg, f=temp_yaml_path)
    
    # Extract the string for local launcher asset checks
    output_dir = str(merged_cfg["output_dir"])

    # Formulate the launch execution array command with DeepSpeed integration
    # Note: --multi_gpu is omitted because it is mutually exclusive with --use_deepspeed in accelerate launch
    # `sys.executable -m accelerate.commands.launch`, not the bare `accelerate` binary:
    # callers invoke this script by interpreter path without activating a venv, so the
    # console script is not necessarily on PATH. This also guarantees the launcher and the
    # training process share one interpreter.
    cmd = [sys.executable, "-m", "accelerate.commands.launch",
           "--num_machines", "1", "--num_processes", str(num_gpus)]
    if use_deepspeed:
        cmd += ["--use_deepspeed", "--deepspeed_config_file", runtime_ds_path]
    cmd += ["src-train/train_patched.py", temp_yaml_path]

    print(f"\n🚀 Launching Axolotl Training Engine:\n{' '.join(cmd)}\n", flush=True)
    try:
        subprocess.run(cmd, check=True)
        print(f"\n🎉 [Success] Job completed successfully for {config_path}!")
        return output_dir, merged_cfg, temp_yaml_path
    except subprocess.CalledProcessError as e:
        print(f"\n❌ [FATAL ERROR] Axolotl core process crashed on {config_path} with exit code {e.returncode}")
        sys.exit(1)

def merge_gemma4_lora(config_path: str, adapter_dir: str, output_dir: str):
    """
    Phase 1: High-Precision Weights Merge
    Uses Axolotl's native CLI to consolidate LoRA adapter weights into the base BF16 model.
    This ensures that architecture-specific patches (like Gemma4ClippableLinear) are correctly handled.
    """
    print("\n" + "="*60)
    print(f"🧬 MERGING ADAPTER WEIGHTS (via Axolotl CLI): {adapter_dir}")
    print("="*60, flush=True)

    # We use the same config used for training, but override output_dir and provide lora_model_dir
    env = os.environ.copy()
    env.setdefault("MASTER_ADDR", "localhost")
    env.setdefault("MASTER_PORT", "12345")
    env.setdefault("WORLD_SIZE", "1")
    env.setdefault("RANK", "0")
    env.setdefault("LOCAL_RANK", "0")

    cmd = [
        "python3", "-m", "axolotl.cli.merge_lora",
        config_path,
        f"--lora_model_dir={adapter_dir}",
        f"--output_dir={output_dir}"
    ]
    
    print(f"🚀 Executing merge command: {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True, env=env)
        print(f"✅ Weight merging completed successfully at {output_dir}")
        return output_dir
    except subprocess.CalledProcessError as e:
        print(f"❌ Axolotl merge failed with exit code {e.returncode}")
        raise e

def resolve_merged_dir(output_dir: str) -> str:
    """
    Axolotl's merge_lora writes the model into a nested `merged/` subdirectory rather
    than into --output_dir itself. Returns whichever level actually holds the model.
    """
    for candidate in (os.path.join(output_dir, "merged"), output_dir):
        if os.path.exists(os.path.join(candidate, "config.json")):
            return candidate
    raise FileNotFoundError(f"❌ No config.json found under '{output_dir}' or its 'merged/' subdirectory")

def run_fp8_compression(merged_bf16_dir: str, output_fp8_dir: str):
    """
    Phase 2: Post-Training FP8 Quantization
    Compresses the merged BF16 model using FP8_DYNAMIC scheme.
    """
    from transformers import AutoProcessor, AutoModelForCausalLM
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    print("\n" + "="*60)
    print(f"📉 APPLYING FP8 QUANTIZATION: {merged_bf16_dir}")
    print("="*60, flush=True)

    # FP8_DYNAMIC is calibration-free, so no GPU is needed. Loading onto CPU also leaves
    # room for llmcompressor to unpack the packed 3D MoE expert tensors into 2D Linears,
    # which OOMs on a GPU that device_map="auto" has already filled.
    print(f"Loading merged BF16 model: {merged_bf16_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        merged_bf16_dir,
        torch_dtype="auto",
        device_map="cpu",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(merged_bf16_dir, trust_remote_code=True)

    # Configure the FP8_DYNAMIC scheme targeting linear projections
    # CRITICAL: We target Linear layers but let llmcompressor handle 
    # the MoE-specific structure through target filtering if needed.
    # We skip vision, norm, and embeddings to preserve stability.
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["re:.*vision.*", "re:.*norm.*", "re:.*embed.*", "re:.*router.*", "re:.*gate$"]
    )

    print("Applying post-training FP8 quantization via one-shot API...")
    oneshot(model=model, recipe=recipe)

    os.makedirs(output_fp8_dir, exist_ok=True)
    print(f"Saving quantized weights in compressed-tensors format to: {output_fp8_dir}")
    model.save_pretrained(output_fp8_dir, save_compressed=True)
    processor.save_pretrained(output_fp8_dir)
    print("✅ FP8 Quantization process successfully completed!")
    return output_fp8_dir

def main():

    # Eliminate CPU management thread bloat across multi-GPU ranks
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Apply memory segmentation allocations globally before execution hooks begin
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # deprecated but still mentioned in error messages

    # Prevent vLLM multi-GPU deadlocks caused by master process CUDA leaks
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn" # CRITICAL for parent-child CUDA isolation

    # Enable blocking waits for NCCL to help diagnose hangs/timeouts
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"

    accelerator, num_gpus = detect_accelerator()
    if accelerator == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n[Hardware] cuda: {num_gpus} GPUs Online | ~{vram_gb:.1f} GB VRAM per GPU\n")
    elif accelerator == "mps":
        print("\n[Hardware] Apple Metal (mps): single process, no DeepSpeed, no FlashAttention.")
        print("           Local smoke run against the stand-in model -- see docs/local-pipeline.md.\n")
    else:
        print("\n❌ ERROR: no CUDA and no Metal device found; nothing to train on.")
        sys.exit(1)

    # On 2-GPU instances, PCIe P2P is frequently broken/unsupported on cloud providers (causing deadlocks).
    # We default to disabling P2P to ensure robust execution unless explicitly overridden.
    if accelerator == "cuda" and num_gpus == 2:
        if "NCCL_P2P_DISABLE" not in os.environ:
            print("ℹ️ 2-GPU cluster detected. Auto-disabling NCCL P2P to prevent virtualized PCIe deadlocks (NCCL_P2P_DISABLE=1).", flush=True)
            os.environ["NCCL_P2P_DISABLE"] = "1"
        if "NCCL_IB_DISABLE" not in os.environ:
            os.environ["NCCL_IB_DISABLE"] = "1"


    
    # Filter pipeline based on GPU count constraints:
    # Mistral requires at least 8 GPUs, Gemma can run on any count.
    active_pipeline = []
    for config_yaml_path in TRAINING_PIPELINE:
        if "mistral" in config_yaml_path.lower():
            if num_gpus < 8:
                print(f"\n⚠️ [SKIP] '{config_yaml_path}' requires at least 8 GPUs, but {num_gpus} are online. Skipping...")
                continue
        active_pipeline.append(config_yaml_path)
    
    if not active_pipeline:
        print("\n🏁 No active training jobs in the pipeline after applying hardware constraints. Exiting.")
        return

    # Pre-download models in a single process to build out local disk structures smoothly
    pre_download_models(active_pipeline)
    
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_run"
    sync_targets: List[SyncTarget] = []
    merge_failed = False

    print(f"🎬 Starting Pipeline Master Loop ({len(active_pipeline)} jobs registered)...")

    for config_yaml_path in active_pipeline:
        output_path, merged_config_data, resolved_config_path = run_training_job(config_yaml_path, num_gpus, accelerator)
        config_name = os.path.basename(config_yaml_path).replace(".yml", "").replace(".yaml", "")
        sync_targets.append(SyncTarget(output_path, f"{run_id}/{config_name}"))

        write_run_manifest(output_path, config_yaml_path, merged_config_data, run_id)
        write_eval_metrics(output_path)

        # Post-training Merge and Quantization
        base_model_id = str(merged_config_data.get("base_model") or "")

        # PEFT accepts a regex target_modules, SGLang only a list. Expand it so the
        # published adapter loads in an inference engine as well as in merge_lora.
        try:
            from expand_targets import expand
            print(f"\n🔧 Expanding adapter target_modules for engine compatibility: {output_path}", flush=True)
            print(f"✅ target_modules now lists {expand(Path(output_path), base_model_id)} module names", flush=True)
        except Exception as e:
            print(f"⚠️ [WARNING] Could not expand target_modules: {e}")
            print("   The adapter still merges, but SGLang will refuse to load it.")
        is_gemma4 = "gemma-4" in base_model_id.lower()
        wants_merge = bool(merged_config_data.get("post_training_merge", True))

        if is_gemma4 and not wants_merge:
            print("\n⏭️  post_training_merge is disabled; publishing the adapter only.", flush=True)

        if is_gemma4 and wants_merge:
            merged_bf16_dir = os.path.join(MERGED_DIR, f"{config_name}-bf16")
            merged_fp8_dir = os.path.join(MERGED_DIR, f"{config_name}-fp8")

            try:
                # Phase 1: Merge
                merge_gemma4_lora(resolved_config_path, output_path, merged_bf16_dir)

                # Phase 2: Quantize
                run_fp8_compression(resolve_merged_dir(merged_bf16_dir), merged_fp8_dir)

                sync_targets.append(SyncTarget(merged_fp8_dir, f"{MERGED_S3_PREFIX}/{run_id}/{config_name}-fp8"))

                # Cleanup BF16 merged model to save disk space
                print(f"🧹 Cleaning up intermediate BF16 merged model at {merged_bf16_dir}")
                shutil.rmtree(merged_bf16_dir, ignore_errors=True)

            except Exception as e:
                # Sync the adapter first so the run is not a total loss, then fail below.
                print(f"❌ Error during post-training merge/quantization: {e}")
                merge_failed = True

    # Cloud Sync Layer
    s3_bucket = os.environ.get("S3_BUCKET", "")
    if s3_bucket:
        print("\n" + "="*60 + "\n📤 INITIATING MASTER CLOUD SYNCHRONIZATION TO S3\n" + "="*60, flush=True)
        for target in sync_targets:
            if not os.path.isdir(target.local_dir):
                print(f"⚠️ [WARNING] Nothing to sync, directory missing: {target.local_dir}")
                continue
            s3_target = f"s3://{s3_bucket}/{target.s3_prefix}"
            print(f"Syncing directory: {target.local_dir} -> {s3_target} ...", flush=True)
            try:
                subprocess.run(["aws", "s3", "sync", target.local_dir, s3_target], check=True)
                print(f"✅ Synchronization successful for {target.s3_prefix}!")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ [WARNING] S3 Sync failed for {target.local_dir} with exit code {e.returncode}!")
                time.sleep(60)

    if merge_failed:
        print("\n❌ [FATAL] Merge/quantization failed. Adapters were synced, but no merged model was produced.")
        sys.exit(1)

    # TODO: Activate when evaluation can process adapters
    # Evaluation Layer
    # print("\n" + "="*60 + "\n🎬 LAUNCHING POST-TRAINING METRICS EVALUATION PIPELINE\n" + "="*60, flush=True)
    # if os.path.exists("src-eval/evaluation.py"):
        # try:
            # subprocess.run(["python", "src-eval/evaluation.py"], check=True)
            # print("\n🎉 [Success] Post-training validation and evaluation pipeline finished!")
        # except subprocess.CalledProcessError as e:
            # print(f"\n❌ [ERROR] Evaluation phase terminated with non-zero exit code {e.returncode}")
            # sys.exit(1)

if __name__ == "__main__":
    main()