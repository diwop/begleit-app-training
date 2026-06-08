# --- src/launcher.py ---
import os
import json
import time
import datetime
import torch
import subprocess
import sys
from omegaconf import OmegaConf
from huggingface_hub import snapshot_download

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
    token = os.environ.get("HF_TOKEN")
    if not token or not token.strip():
        print("\n" + "❌"*30)
        print("❌ CRITICAL ENVIRONMENT VIOLATION: HF_TOKEN is missing or empty!")
        print("❌ Gated models (Gemma, Mistral Small, Llama) require active authentication.")
        print("❌ Please export your token before running the script:")
        print("❌     export HF_TOKEN=\"hf_your_token_here\"")
        print("❌"*30 + "\n", flush=True)
        sys.exit(1)
    
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
            print(f"📦 Verifying local file mapping for: '{base_model_str}'...", flush=True)
            try:
                snapshot_download(
                    repo_id=base_model_str,
                    token=token,
                    ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.pt"]
                )
                print(f"✅ Weight cache successfully validated for: {base_model_str}\n", flush=True)
                processed_models.add(base_model_str)
            except Exception as e:
                print(f"\n❌ CRITICAL: Failed to cache weights for {base_model_str}!")
                print(f"Error Source: {e}")
                sys.exit(1)
                
    print("="*60 + "\n🏁 All base model weights are cached locally. Ready for distributed execution.\n")

def generate_runtime_deepspeed(stage: int, output_json_path: str):
    """
    Dynamically compiles a specialized DeepSpeed configuration profile based on the architecture stage.
    Stage 2 is utilized for 8-bit LoRA compatibility; Stage 3 is utilized for 4-bit parameter sharding.
    """
    if stage == 2:
        # High-performance pure-VRAM Stage 2 Blueprint for 8-bit LoRA bases
        ds_dict = {
            "bf16": {"enabled": True},
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 200000000,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 200000000,
                "contiguous_gradients": True
            }
        }
    else:
        # Pure-VRAM Stage 3 Blueprint for massive 4-bit QLoRA sharding
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
            },
            "activation_checkpointing": {
                "partition_activations": True,
                "contiguous_memory_optimization": True,
                "cpu_checkpointing": False
            }
        }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(ds_dict, f, indent=2)
        
    print(f"✅ DeepSpeed Stage {stage} configuration compiled successfully at: {output_json_path}")
    return output_json_path

def run_training_job(config_path: str, num_gpus: int, run_id: str):
    """
    Isolates configuration merging, custom runtime deepspeed building,
    and execution tracking for an individual model training run.
    """
    print("\n" + "="*60)
    print(f"🎬 INITIATING PIPELINE TRAINING JOB: {config_path}")
    print("="*60, flush=True)

    # 1. Merge target configuration parameters
    merged_cfg = merge_configs("config/base.yml", config_path)

    config_filename = os.path.basename(config_path).replace(".yml", "").replace(".yaml", "")
    temp_yaml_path = f".merged-{config_filename}.yml"
    runtime_ds_path = f".ds-config-{config_filename}.json"

    # Enforce long-prompt parameters directly
    merged_cfg["micro_batch_size"] = 1
    merged_cfg["gradient_accumulation_steps"] = 8
    merged_cfg["sample_packing"] = True
    
    # 2. DYNAMIC ARCHITECTURE ROUTING LAYER
    if "gemma" in config_path.lower():
        print("🚀 [Pipeline Orchestrator] Configuring DeepSpeed Stage 2 for Gemma 8-bit LoRA...")
        stage = 2
    else:
        print("🚀 [Pipeline Orchestrator] Configuring DeepSpeed Stage 3 + Sharded Loading for Mistral 4-bit QLoRA...")
        stage = 3
        # Forces Axolotl to shard the weights layer-by-layer during download/load streams
        merged_cfg["qlora_sharded_model_loading"] = True
    
    # Compile and bind the dynamic deepspeed blueprint
    generate_runtime_deepspeed(stage, runtime_ds_path)
    merged_cfg["deepspeed"] = runtime_ds_path

    # Save the finalized temporary configuration for Axolotl
    OmegaConf.save(config=merged_cfg, f=temp_yaml_path)

    output_dir = str(merged_cfg.get("output_dir", f"/app/output/adapter/{config_filename}"))
    is_eval_mode = os.environ.get("EVAL", "false").lower() == "true"
    adapter_exists = os.path.exists(os.path.join(output_dir, "adapter_config.json"))

    if is_eval_mode and adapter_exists:
        print(f"\n[SKIP] EVAL=true and valid adapter discovered at '{output_dir}'. Bypassing training pass.")
        return output_dir, merged_cfg

    # 3. Formulate the multi-GPU launch execution command
    cmd = [
        "accelerate", "launch",
        "--multi_gpu",
        "--num_machines", "1",
        "--num_processes", str(num_gpus),
        "-m", "axolotl.cli.train",
        temp_yaml_path
    ]

    print(f"\n🚀 Launching Axolotl Training Engine:\n{' '.join(cmd)}\n", flush=True)
    try:
        subprocess.run(cmd, check=True)
        print(f"\n🎉 [Success] Job completed successfully for {config_path}!")
        return output_dir, merged_cfg
    except subprocess.CalledProcessError as e:
        print(f"\n❌ [FATAL ERROR] Axolotl core process crashed on {config_path} with exit code {e.returncode}")
        sys.exit(1)

def main():
    # Apply memory allocations globally
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["VLLM_USE_V1"] = "0"

    # Enforce homogeneous communication paths to prevent topology locks
    print("🛡️  Enforcing uniform NCCL distributed communication transport paths...")
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"

    if not torch.cuda.is_available():
        print("❌ ERROR: No CUDA devices identified on the host cluster!")
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"\n[Hardware Cluster Configuration] {num_gpus} GPUs Online | ~{vram_gb:.1f} GB VRAM per GPU\n")

    TRAINING_PIPELINE = [
        "config/train-gemma4.yml",
        "config/train-mistral4small.yml"
    ]
    
    # Pre-download models in a single process to ensure clean local caching
    pre_download_models(TRAINING_PIPELINE)
    
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_run"
    completed_output_dirs = []

    print(f"🎬 Starting Pipeline Master Loop ({len(TRAINING_PIPELINE)} jobs registered)...")
    
    for config_yaml_path in TRAINING_PIPELINE:
        output_path, merged_config_data = run_training_job(config_yaml_path, num_gpus, run_id)
        completed_output_dirs.append((output_path, config_yaml_path))

    # Cloud Sync Layer
    s3_bucket = os.environ.get("S3_BUCKET", "")
    if s3_bucket:
        print("\n" + "="*60 + "\n📤 INITIATING MASTER CLOUD SYNCHRONIZATION TO S3\n" + "="*60, flush=True)
        for output_dir, config_path in completed_output_dirs:
            if os.path.exists(os.path.join(output_dir, "adapter_config.json")):
                model_target_dirname = os.path.basename(os.path.normpath(output_dir))
                s3_target = f"s3://{s3_bucket}/{run_id}/{model_target_dirname}"
                print(f"Syncing directory: {output_dir} -> {s3_target} ...", flush=True)
                try:
                    subprocess.run(["aws", "s3", "sync", output_dir, s3_target], check=True)
                    print(f"✅ Synchronization successful for {model_target_dirname}!")
                except subprocess.CalledProcessError as e:
                    print(f"⚠️ [WARNING] S3 Sync failed for {output_dir} with exit code {e.returncode}!")
                    time.sleep(60)

    # Evaluation Layer
    print("\n" + "="*60 + "\n🎬 LAUNCHING POST-TRAINING METRICS EVALUATION PIPELINE\n" + "="*60, flush=True)
    if os.path.exists("src/evaluation.py"):
        try:
            subprocess.run(["python", "src/evaluation.py"], check=True)
            print("\n🎉 [Success] Post-training validation pipeline finished!")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ [ERROR] Evaluation phase terminated with exit code {e.returncode}")
            sys.exit(1)

if __name__ == "__main__":
    main()