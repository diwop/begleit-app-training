"""Find the smallest thing that makes cuBLAS fail on this machine.

Every training and evaluation run on 2026-07-31 died the same way:

    RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE
                  when calling `cublasSgemm(handle, opa, opb, m, n, k, ...)`

always at the first fp32 matmul of the first forward pass. A probe inserted just
before it showed that `ones(2,2) @ ones(2,2)` fails there too, so the arguments were
never the problem -- cuBLAS is simply unusable by that point. What is NOT known is
whether it is unusable in a *fresh* process, or whether something in the stack breaks
it on the way. This answers that, without the 51 GB base model or any data.

Each stage runs in its own subprocess, so a poisoned CUDA context cannot leak into the
next one and turn a single failure into a cascade of meaningless ones.

    python src-train/cuda_smoke.py            # run every stage, report the first failure
    python src-train/cuda_smoke.py --only 2   # one stage, full traceback

Read the FIRST failing stage: it names the layer that breaks cuBLAS.
"""
import argparse
import subprocess
import sys
import traceback

# Ordered from "the GPU works at all" to "the full training stack is loaded". Each entry
# is (name, what it proves, source). The source runs in a fresh interpreter.
STAGES = [
    ("alloc", "the GPU accepts an allocation", """
import torch
x = torch.ones(2, 2, device="cuda")
torch.cuda.synchronize()
print("allocated", tuple(x.shape), x.device, torch.cuda.get_device_name(0))
"""),

    ("matmul_fp32", "cublasSgemm works in a bare process -- THE failing op", """
import torch
x = torch.ones(2, 2, device="cuda", dtype=torch.float32)
print("result", (x @ x).sum().item())
"""),

    ("matmul_bf16", "whether the breakage is fp32-specific (the model runs bf16)", """
import torch
x = torch.ones(2, 2, device="cuda", dtype=torch.bfloat16)
print("result", (x @ x).float().sum().item())
"""),

    ("matmul_fp32_tf32_off", "whether TF32 is implicated", """
import torch
torch.backends.cuda.matmul.allow_tf32 = False
x = torch.ones(2, 2, device="cuda", dtype=torch.float32)
print("result", (x @ x).sum().item())
"""),

    ("after_transformers", "importing transformers does not break it", """
import transformers, torch
print("transformers", transformers.__version__)
x = torch.ones(2, 2, device="cuda", dtype=torch.float32)
print("result", (x @ x).sum().item())
"""),

    ("after_deepspeed_import", "importing deepspeed does not break it", """
import deepspeed, torch
print("deepspeed", deepspeed.__version__)
x = torch.ones(2, 2, device="cuda", dtype=torch.float32)
print("result", (x @ x).sum().item())
"""),

    ("after_deepspeed_dist", "initialising the DeepSpeed process group does not break it", """
import os, deepspeed, torch
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29555")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
deepspeed.init_distributed(dist_backend="nccl")
x = torch.ones(2, 2, device="cuda", dtype=torch.float32)
print("result", (x @ x).sum().item())
"""),

    ("under_zero_init", "ZeRO-3's zero.Init() does not break it", """
import os, deepspeed, torch
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29556")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
deepspeed.init_distributed(dist_backend="nccl")
cfg = {"train_batch_size": 1, "bf16": {"enabled": True},
       "zero_optimization": {"stage": 3, "stage3_param_persistence_threshold": 0}}
with deepspeed.zero.Init(config_dict_or_path=cfg):
    layer = torch.nn.Linear(8, 8)
x = torch.ones(2, 2, device="cuda", dtype=torch.float32)
print("result", (x @ x).sum().item())
"""),
]


def run_stage(index: int, verbose: bool) -> bool:
    name, proves, source = STAGES[index]
    print(f"\n[{index}] {name}\n    proves: {proves}", flush=True)
    completed = subprocess.run([sys.executable, "-c", source],
                               capture_output=True, text=True, timeout=600)
    if completed.returncode == 0:
        print(f"    PASS  {completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ''}")
        return True

    print("    FAIL")
    detail = (completed.stderr or "").strip().splitlines()
    # The exception line is what matters; the rest is import machinery.
    for line in (detail if verbose else [ln for ln in detail if "Error" in ln or "error" in ln][-3:]):
        print(f"      {line}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Bisect what breaks cuBLAS on this host.")
    parser.add_argument("--only", type=int, help="run a single stage by index, verbosely")
    args = parser.parse_args()

    try:
        import torch
        print(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
              f"devices {torch.cuda.device_count()}")
    except Exception:
        traceback.print_exc()
        sys.exit("torch is not importable; nothing here can run")

    if args.only is not None:
        sys.exit(0 if run_stage(args.only, verbose=True) else 1)

    for index in range(len(STAGES)):
        if not run_stage(index, verbose=False):
            print(f"\n>>> FIRST FAILURE: stage {index} ({STAGES[index][0]}).")
            print(f">>> Re-run it alone for the full traceback:")
            print(f"      {sys.executable} {sys.argv[0]} --only {index}")
            if index <= 1:
                print(">>> A failure this early means the GPU itself cannot do a 2x2 matmul.")
                print(">>> That is a host or driver fault, not anything this repo controls.")
            sys.exit(1)

    print("\n>>> All stages passed: cuBLAS is healthy here, including under ZeRO-3 init.")
    print(">>> The fault therefore needs the model or the data to reproduce.")


if __name__ == "__main__":
    main()
