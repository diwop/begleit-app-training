"""Find the smallest thing that makes cuBLAS fail on this machine.

Written for the 2026-07-31 failure, where every run died at the first fp32 matmul of the
first forward pass:

    RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE
                  when calling `cublasSgemm(handle, opa, opb, m, n, k, ...)`

That one turned out to be a mismatched cuBLAS pair -- the wheel's libcublas against the
image's libcublasLt -- and `scripts/lib/platform.sh` now prevents it. The header below
prints the loaded library paths, which is the quickest way to recognise a recurrence.

The ladder still earns its place for anything else that makes cuBLAS unusable: it runs in
seconds, needs neither the 51 GB base model nor any data, and each stage runs in its own
subprocess so a poisoned CUDA context cannot cascade into the ones after it.

    python src-train/cuda_smoke.py            # run every stage, report the first failure
    python src-train/cuda_smoke.py --only 2   # one stage, full traceback

Read the FIRST failing stage: it names the layer that breaks cuBLAS. Stages 1-3 together
tell library from kernel path -- a broken library fails through every entry point, a
broken kernel path fails selectively.
"""
import argparse
import subprocess
import sys
from pathlib import Path
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

    # ---- Stages 0-7 pass on a healthy pod. Everything below closes the remaining gap to
    # the real pipeline, one element at a time: the launcher, OUR generated DeepSpeed
    # config, and train_patched.py's monkeypatches. The first rung that fails is the cause.

    ("under_accelerate", "`accelerate launch` itself does not break it", """
import os, subprocess, sys, tempfile
body = "import torch\\nx = torch.ones(2, 2, device=torch.device(0))\\nprint('result', x.matmul(x).sum().item())\\n"
with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
    f.write(body); path = f.name
subprocess.run([sys.executable, "-m", "accelerate.commands.launch",
                "--num_machines", "1", "--num_processes", "1", path], check=True)
"""),

    ("under_accelerate_deepspeed", "accelerate + OUR generated ZeRO-3 config does not break it", """
import os, subprocess, sys, tempfile
sys.path.insert(0, "@SRC_TRAIN@")
from train import generate_runtime_deepspeed
# Exactly what config/train-gemma4.yml asks for.
ds = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False).name
generate_runtime_deepspeed(ds, cpu_checkpointing=True, offload_optimizer=True,
                           offload_param=True, param_persistence_threshold=0)
body = "import torch\\nx = torch.ones(2, 2, device=torch.device(0))\\nprint('result', x.matmul(x).sum().item())\\n"
with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
    f.write(body); path = f.name
subprocess.run([sys.executable, "-m", "accelerate.commands.launch",
                "--num_machines", "1", "--num_processes", "1",
                "--use_deepspeed", "--deepspeed_config_file", ds, path], check=True)
"""),

    ("with_production_ds_config", "deepspeed.initialize with OUR exact config does not break it", """
import os, sys, torch, deepspeed, json
sys.path.insert(0, "@SRC_TRAIN@")
from train import generate_runtime_deepspeed
os.environ.setdefault("MASTER_ADDR", "localhost"); os.environ.setdefault("MASTER_PORT", "29557")
os.environ.setdefault("RANK", "0"); os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
path = "/tmp/ds-probe.json"
generate_runtime_deepspeed(path, cpu_checkpointing=True, offload_optimizer=True,
                           offload_param=True, param_persistence_threshold=0)
cfg = json.load(open(path))
# The generated file uses "auto" for batch sizes; DeepSpeed needs real numbers standalone.
cfg["train_batch_size"] = 1; cfg["train_micro_batch_size_per_gpu"] = 1
cfg["gradient_accumulation_steps"] = 1; cfg.pop("gradient_clipping", None)
deepspeed.init_distributed(dist_backend="nccl")
model = torch.nn.Linear(8, 8)
engine, *_ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=cfg)
x = torch.ones(2, 2, device="cuda", dtype=torch.float32)
print("result", (x @ x).sum().item())
"""),

    ("after_train_patched", "train_patched.py's monkeypatches do not break it", """
import os, sys
sys.path.insert(0, "@SRC_TRAIN@")
os.environ.setdefault("MASTER_ADDR", "localhost"); os.environ.setdefault("MASTER_PORT", "29558")
os.environ.setdefault("RANK", "0"); os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
import runpy, torch
# Import for its side effects only; __main__ guard keeps fire.Fire from running.
runpy.run_path("@SRC_TRAIN@/train_patched.py", run_name="probe")
x = torch.ones(2, 2, device=torch.device(0))
print("result", x.matmul(x).sum().item())
"""),
]


def run_stage(index: int, verbose: bool) -> bool:
    name, proves, source = STAGES[index]
    print(f"\n[{index}] {name}\n    proves: {proves}", flush=True)
    source = source.replace("@SRC_TRAIN@", str(Path(__file__).resolve().parent))
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
        # Which cuBLAS is actually mapped in, which is what the whole 2026-07-31 hunt
        # turned out to hinge on. Needs a real allocation first to force CUDA init.
        try:
            torch.ones(1, device="cuda")
            for path in sorted({ln.split()[-1] for ln in open("/proc/self/maps")
                                if "cublas" in ln}):
                print(f"  cublas {path}")
        except Exception:
            pass
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
                print(">>> A failure this early is the mismatched cuBLAS pair: torch's wheel")
                print(">>> libcublas against the image's system libcublasLt. Check the paths")
                print(">>> printed above -- both must live under site-packages/nvidia/.")
                print(">>> scripts/lib/platform.sh puts them first on LD_LIBRARY_PATH.")
            sys.exit(1)

    print("\n>>> All stages passed: cuBLAS is healthy here, including under ZeRO-3 init.")
    print(">>> The fault therefore needs the model or the data to reproduce.")


if __name__ == "__main__":
    main()
