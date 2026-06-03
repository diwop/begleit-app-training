import os
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

    # Save the finalized, resolved config for Axolotl to read
    temp_config_path = ".merged-train.yml"
    OmegaConf.save(config=merged_cfg, f=temp_config_path)

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
    except subprocess.CalledProcessError as e:
        print(f"\n[FATAL ERROR] Training failed with exit code {e.returncode}")
        sys.exit(1)

    # Run Evaluation
    print("\nStarting post-training evaluation...")
    
    # Extract values with safe fallbacks
    output_dir = str(merged_cfg.get("output_dir", "/workspace/output"))
    seq_len = str(merged_cfg.get("sequence_len", 2048))
    lora_rank = str(merged_cfg.get("lora_r", 64))
    
    os.makedirs(output_dir, exist_ok=True)

    eval_cmd = [
        "python", "src/evaluation.py",
        "--base_model", str(merged_cfg.base_model),
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