from aiohttp import client_middleware_digest_auth
import torch
import subprocess
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to Axolotl YAML config")
    args = parser.parse_args()

    # Detect the GPUs and their VRAM
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPUs detected!")
        sys.exit(1)

    num_gpus = torch.cuda.device_count()
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024**3)

    print(f"\n[Hardware Detected] {num_gpus} GPUs | ~{vram_gb:.1f} GB VRAM per GPU\n")

    # Construct the base command for the training
    cmd = [
        "accelerate", "launch",
        "--num_processes", str(num_gpus),
        "-m", "axolotl.cli.train",
        args.config
    ]

    # Add overrides based on VRAM and GPU Count
    if vram_gb < 30:
        print("[Override] <30GB VRAM detected. Injecting 24GB safety limits...")
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

    # Execute Training
    print(f"\nExecuting: {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()