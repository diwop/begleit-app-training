import os
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
    import torch

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
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        sys.exit(1)        

if __name__ == "__main__":
    main()