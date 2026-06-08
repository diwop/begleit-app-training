# --- src/launcher.py ---
import os
import json
import time
from datetime import datetime, UTC
import torch
import subprocess
import sys
from omegaconf import OmegaConf

def merge_configs(base_path: str, override_path: str):
    """Loads and merges a base YAML and an override YAML. Override values take precedence."""
    base_cfg = OmegaConf.load(base_path)
    override_cfg = OmegaConf.load(override_path)
    return OmegaConf.merge(base_cfg, override_cfg)

def generate_runtime_deepspeed(large: bool, output_json_path: str):
    """
    Reads the base Axolotl ZeRO-3 template, injects long-prompt activation partitioning,
    and dynamically applies CPU offloading for large models to prevent OOMs.
    """
    source_ds_path = "/workspace/axolotl/deepspeed_configs/zero3_bf16.json"
    
    if os.path.exists(source_ds_path):
        with open(source_ds_path, "r", encoding="utf-8") as f:
            ds_dict = json.load(f)
    else:
        # Fallback hardcoded blueprint if the native file is absent
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

    # Inject Activation Partitioning (Crucial to process 10k+ token sequences)
    ds_dict["activation_checkpointing"] = {
        "partition_activations": True,
        "contiguous_memory_optimization": True,
        "cpu_checkpointing": False
    }

    # Apply Conditional CPU Offloading for large models
    if large:
        print("⚙️ [DeepSpeed Engine] Activating CPU Optimizer Offloading for large model...")
        ds_dict["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True
        }
    else:
        print("⚙️ [DeepSpeed Engine] Maximizing VRAM execution speed...")
        ds_dict["zero_optimization"]["offload_optimizer"] = {"device": "none"}

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(ds_dict, f, indent=2)
        
    print(f"✅ DeepSpeed configuration compiled successfully at: {output_json_path}")
    return output_json_path

def run_training_job(config_path: str, num_gpus: int, run_id: str, large: bool):
    """
    A modular function that isolates configuration merging, runtime deepspeed building,
    and execution tracking for an individual model training loop run.
    """
    print("\n" + "="*60)
    print(f"🎬 INITIATING PIPELINE TRAINING JOB: {config_path}")
    print("="*60, flush=True)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Target training configuration file missing: '{config_path}'")

    # 1. Merge selected target configuration with base parameters
    merged_cfg = merge_configs("config/base.yml", config_path)
    base_model_str = str(merged_cfg.get("base_model", ""))

    # Unique filenames to prevent overlapping cache clashes inside sequential cycles
    config_filename = os.path.basename(config_path).replace(".yml", "").replace(".yaml", "")
    temp_yaml_path = f".merged-{config_filename}.yml"
    runtime_ds_path = f".ds-config-{config_filename}.json"

    # 2. Enforce long-prompt safety overrides directly to the configuration block
    merged_cfg["micro_batch_size"] = 1
    merged_cfg["gradient_accumulation_steps"] = 8
    merged_cfg["sample_packing"] = True
    
    # FIXED BUG: Pass the boolean flag 'large' and target file path to generate_runtime_deepspeed
    generate_runtime_deepspeed(large, runtime_ds_path)
    merged_cfg["deepspeed"] = runtime_ds_path

    # Save the resolved, finalized configuration path for Axolotl to consume
    OmegaConf.save(config=merged_cfg, f=temp_yaml_path)

    output_dir = str(merged_cfg.get("output_dir", f"/app/output/adapter/{config_filename}"))
    is_eval_mode = os.environ.get("EVAL", "false").lower() == "true"
    adapter_exists = os.path.exists(os.path.join(output_dir, "adapter_config.json"))

    if is_eval_mode and adapter_exists:
        print(f"\n[SKIP] EVAL=true and valid adapter discovered at '{output_dir}'. Bypassing training pass.")
        return output_dir, merged_cfg

    if is_eval_mode and not adapter_exists:
        print(f"\n[WARNING] EVAL=true is set, but no adapter was found at '{output_dir}'. Commencing training...")

    # 3. Formulate the multi-GPU launch execution array command
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
    # Apply memory segmentation allocations globally before execution hooks begin
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if not torch.cuda.is_available():
        print("❌ ERROR: No CUDA devices identified on the host cluster!")
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"\n[Hardware Cluster Configuration] {num_gpus} GPUs Online | ~{vram_gb:.1f} GB VRAM per GPU\n")

    # -------------------------------------------------------------------------
    # THE SEQUENTIAL TRAINING PIPELINE MATRIX
    # Register your training configurations here to step through them in a loop
    # -------------------------------------------------------------------------
    TRAINING_PIPELINE = [
        ["config/train-gemma4.yml", False],
        ["config/train-mistral4small.yml", True]
    ]
    
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    run_id = f"{timestamp}_run"
    
    # Track output directory paths to synchronize everything to S3 at the end
    completed_output_dirs = []

    print(f"🎬 Starting Pipeline Master Loop ({len(TRAINING_PIPELINE)} jobs registered)...")
    
    for config_yaml_path, large in TRAINING_PIPELINE:
        # FIXED BUG: Correctly pass the large boolean flag to the training job orchestrator
        output_path, merged_config_data = run_training_job(config_yaml_path, num_gpus, run_id, large)
        completed_output_dirs.append((output_path, config_yaml_path))

    # -------------------------------------------------------------------------
    # FINALIZATION: BULK CLOUD SYNCHRONIZATION TO AWS S3
    # -------------------------------------------------------------------------
    s3_bucket = os.environ.get("S3_BUCKET", "")
    if s3_bucket:
        print("\n" + "="*60)
        print("📤 INITIATING MASTER CLOUD SYNCHRONIZATION TO S3")
        print("="*60, flush=True)
        
        for output_dir, config_path in completed_output_dirs:
            if os.path.exists(os.path.join(output_dir, "adapter_config.json")):
                # Extract the final sub-folder directory name to keep your bucket clean
                model_target_dirname = os.path.basename(os.path.normpath(output_dir))
                s3_target = f"s3://{s3_bucket}/{run_id}/{model_target_dirname}"
                
                print(f"Syncing directory: {output_dir} -> {s3_target} ...", flush=True)
                try:
                    subprocess.run(["aws", "s3", "sync", output_dir, s3_target], check=True)
                    print(f"✅ Synchronization successful for {model_target_dirname}!")
                except subprocess.CalledProcessError as e:
                    print(f"⚠️ [WARNING] S3 Sync failed for {output_dir} with exit code {e.returncode}!")
                    print("Pausing script for 60 seconds to allow for manual cluster debugging...")
                    time.sleep(60)
            else:
                print(f"ℹ️ Skipping S3 sync for '{output_dir}' - No valid adapter assets found.")
    else:
        print("\n⚠️ Note: S3_BUCKET environment variable is missing. Skipping final cloud sync phase.")

    # -------------------------------------------------------------------------
    # LAUNCH POST-TRAINING MATRIX EVALUATION PHASE
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("🎬 LAUNCHING POST-TRAINING METRICS EVALUATION PIPELINE")
    print("="*60, flush=True)
    
    if os.path.exists("src/evaluation.py"):
        print("Running batch evaluation matrix sequence via evaluation.py...")
        try:
            subprocess.run(["python", "src/evaluation.py"], check=True)
            print("\n🎉 [Success] Post-training validation and evaluation pipeline finished!")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ [ERROR] Evaluation phase terminated with non-zero exit code {e.returncode}")
            sys.exit(1)
    else:
        print("❌ [ERROR] 'src/evaluation.py' script was not found. Bypassing metrics verification.")
        sys.exit(1)

    print("\n🏁 ALL CONFIGURED PIPELINE TRAINING & EVALUATION RUNS COMPLETED SUCCESSFULLY!\n")

if __name__ == "__main__":
    main()