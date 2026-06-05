import os
import time
import datetime
import torch
import subprocess
import argparse
import sys
from omegaconf import OmegaConf

def merge_configs(base_path: str, override_path: str):
    """
    Loads and merges a base YAML and an override YAML.
    Override values take precedence.
    """
    base_cfg = OmegaConf.load(base_path)
    override_cfg = OmegaConf.load(override_path)
    return OmegaConf.merge(base_cfg, override_cfg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to Axolotl YAML config")
    args = parser.parse_args()

    # Apply the memory fragmentation fix globally inside the script
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Detect the GPUs and their VRAM
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPUs detected!")
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024**3)

    print(f"\n[Hardware Detected] {num_gpus} GPUs | ~{vram_gb:.1f} GB VRAM per GPU\n")

    # Merge the selected config file with the base
    merged_cfg = merge_configs("config/base.yml", args.config)

    print(f"\n[DEBUG] Found base_model: {merged_cfg.get('base_model')}")
    print(f"[DEBUG] Found inference_model: {merged_cfg.get('inference_model')}\n")

    # Extract custom values
    inference_model = str(merged_cfg.get("inference_model", merged_cfg.get("base_model")))

    # Sanitize custom values
    if "inference_model" in merged_cfg:
        del merged_cfg["inference_model"]

    # Save the finalized, resolved config for Axolotl to read
    temp_config_path = ".merged-train.yml"
    OmegaConf.save(config=merged_cfg, f=temp_config_path)

    # Extract configuration variables early to use in our skip logic
    output_dir = str(merged_cfg.get("output_dir", "/workspace/output"))
    seq_len = str(merged_cfg.get("sequence_len", 2048))
    lora_rank = str(merged_cfg.get("lora_r", 64))

    # --- CONDITIONAL TRAINING LOGIC ---
    is_eval_mode = os.environ.get("EVAL", "false").lower() == "true"
    adapter_exists = os.path.exists(os.path.join(output_dir, "adapter_config.json"))

    if is_eval_mode and adapter_exists:
        print(f"\n[SKIP] EVAL=true detected and adapter found at '{output_dir}'. Bypassing training phase.")
    else:
        if is_eval_mode and not adapter_exists:
            print(f"\n[WARNING] EVAL=true is set, but no valid adapter was found at '{output_dir}'. Proceeding with training!")

        # Construct the base command for the training
        cmd = [
            "accelerate", "launch",
            "--num_processes", str(num_gpus),
            "-m", "axolotl.cli.train",
            ".merged-train.yml"
        ]

        # Add overrides based on VRAM
        if vram_gb < 30:
            print("[Override] <30GB VRAM detected. Injecting safety limits...")
            cmd.extend([
                "--micro_batch_size", "1",
                "--gradient_accumulation_steps", "8"
            ])
        else:
            print("[Override] >30GB VRAM detected. Optimizing for high-throughput...")
            cmd.extend([
                "--micro_batch_size", "4",
                "--gradient_accumulation_steps", "2"
            ])

        # Add DeepSpeed toggle based on GPU count
        if num_gpus > 1:
            print("[Override] Multiple GPUs detected. Injecting DeepSpeed ZeRO-3...")
            cmd.extend(["--deepspeed", "config/zero3.json"])
        else:
            print("[Override] Single GPU detected. Running native PyTorch (No DeepSpeed).")

        # Execute Training
        print(f"\nExecuting: {' '.join(cmd)}\n")
        try:
            subprocess.run(cmd, check=True)
            print("\n[Success] Training completed successfully!")
            
            s3_bucket = os.environ.get("S3_BUCKET", "")
            if s3_bucket:
                print(f"\nS3_BUCKET '{s3_bucket}' configured. Backing up adapter and checkpoints...")
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                s3_target = f"s3://{s3_bucket}/{timestamp}_run"
                
                try:
                    # Sync the entire output directory recursively
                    print(f"Syncing local {output_dir} -> {s3_target} ...")
                    subprocess.run(["aws", "s3", "sync", output_dir, s3_target], check=True)
                    print("=== S3 Backup Successful! ===")
                    
                except subprocess.CalledProcessError as e:
                    print(f"=== WARNING: S3 Backup Failed with exit code {e.returncode}! ===")
                    print("Sleeping for 60 seconds to allow manual debugging before continuing...")
                    time.sleep(60)

        except subprocess.CalledProcessError as e:
            print(f"\n[FATAL ERROR] Training failed with exit code {e.returncode}")
            sys.exit(1)

    # --- EVALUATION LOGIC ---
    print("\nStarting post-training evaluation...")
    
    os.makedirs(output_dir, exist_ok=True)

    eval_cmd = [
        "python", "src/evaluation.py",
        "--base_model", inference_model,  # Restored mapping to the AWQ variable
        "--adapter_path", output_dir,
        "--dataset_path", str(merged_cfg.datasets[0].path),
        "--seq_length", seq_len,
        "--lora_rank", lora_rank,
        "--output_file", f"{output_dir}/evaluation_results.md"
    ]

    print(f"Executing Evaluation: {' '.join(eval_cmd)}\n")
    try:
        subprocess.run(eval_cmd, check=True)
        print(f"\n[Success] Pipeline finished! Results saved to {output_dir}/evaluation_results.md")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Evaluation failed with exit code {e.returncode}")
        sys.exit(1)        

if __name__ == "__main__":
    main()