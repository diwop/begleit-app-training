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
            print(f"📦 Invoking native hf engine for: '{base_model_str}'...", flush=True)
            try:
                # Build the native shell execution array
                cmd = ["hf", "download", base_model_str]
                
                # Execute the standalone downloader. It automatically 
                # picks up the HF_TOKEN from the environment variables.
                subprocess.run(cmd, check=True)
                
                print(f"✅ Weight cache successfully validated for: {base_model_str}\n", flush=True)
                processed_models.add(base_model_str)
            except subprocess.CalledProcessError as e:
                print(f"\n❌ CRITICAL: Native 'hf' tool failed to download {base_model_str}!")
                print(f"Exit Code: {e.returncode}")
                sys.exit(1)
                
    print("="*60 + "\n🏁 All base model weights are cached locally. Ready for distributed execution.\n")

def generate_runtime_deepspeed(output_json_path: str) -> str:
    """
    Reads the base Axolotl ZeRO-3 template and injects a high-performance, 
    zero-latency local VRAM policy. All parameters, activations, and optimizer 
    states are kept local to the GPU to maximize throughput.
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
    
    # MAXIMUM SPEED: Keep all parameters and optimizer states in GPU VRAM
    ds_dict["zero_optimization"]["offload_optimizer"] = {"device": "none"}
    ds_dict["zero_optimization"]["offload_param"] = {"device": "none"}

    # Inject Long-Context Activation Protection without slow PCIe-to-CPU swaps
    ds_dict["activation_checkpointing"] = {
        "partition_activations": True,
        "contiguous_memory_optimization": True,
        "cpu_checkpointing": False
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(ds_dict, f, indent=2)
        
    print(f"✅ DeepSpeed Stage 3 configuration compiled successfully at: {output_json_path}")
    return output_json_path

def run_training_job(config_path: str, num_gpus: int, run_id: str) -> tuple[str, dict]:
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

    # Enforce high-performance FlashAttention-2 backend globally
    merged_cfg["attn_implementation"] = "flash_attention_2"

    # Generate and link the unified VRAM-centric DeepSpeed configuration file
    generate_runtime_deepspeed(runtime_ds_path)
    merged_cfg["deepspeed"] = runtime_ds_path

    # Save the resolved, finalized configuration path for Axolotl to consume
    OmegaConf.save(config=merged_cfg, f=temp_yaml_path)

    output_dir = str(merged_cfg.get("output_dir", f"/app/output/adapter/{config_filename}"))
    is_eval_mode = os.environ.get("EVAL", "false").lower() == "true"
    adapter_exists = os.path.exists(os.path.join(output_dir, "adapter_config.json"))

    if is_eval_mode and adapter_exists:
        print(f"\n[SKIP] EVAL=true and valid adapter discovered at '{output_dir}'. Bypassing training pass.")
        return output_dir, merged_cfg

    # Formulate the multi-GPU launch execution array command
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
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # deprecated but still mentioned in error messages

    # Deactivate the experimental vLLM V1 graph compiler for child evaluation tasks
    os.environ["VLLM_USE_V1"] = "0"

    # Enforce homogeneous communication paths across the 4x L40S cluster nodes.
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
        "config/train-mistral4small.yml",
        # "config/train-gemma4.yml",
    ]
    
    # Pre-download models in a single process to build out local disk structures smoothly
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
            print("\n🎉 [Success] Post-training validation and evaluation pipeline finished!")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ [ERROR] Evaluation phase terminated with non-zero exit code {e.returncode}")
            sys.exit(1)

if __name__ == "__main__":
    main()