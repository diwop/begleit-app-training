"""Does an inference engine actually serve a LoRA adapter on the Gemma 4 MoE base?

Deliberately minimal: no monkeypatches, no S3, no readability metrics, no exception
handling that turns a crash into an empty string. One engine is loaded with the adapter
enabled, then the same prompts are generated twice -- once routed through the adapter and
once bypassing it -- so any difference is attributable to the adapter alone.

Sampling is greedy (temperature 0) so that "the outputs differ" means the adapter was
applied, not that we sampled twice.

    python src-eval/smoke_adapter.py                 # engine auto-detected
    SMOKE_ENGINE=vllm python src-eval/smoke_adapter.py
    SMOKE_BASE=RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic python src-eval/smoke_adapter.py
"""
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from transformers import AutoTokenizer

BASE_MODEL = os.environ.get("SMOKE_BASE", "google/gemma-4-26b-a4b-it")
ADAPTER_DIR = os.environ.get("SMOKE_ADAPTER", "/app/output/adapter/train-gemma4")
ADAPTER_NAME = "ft"
ENGINE = os.environ.get("SMOKE_ENGINE", "auto")
TP_SIZE = int(os.environ.get("TP_SIZE", "2"))
MAX_NEW_TOKENS = int(os.environ.get("SMOKE_MAX_TOKENS", "512"))
MAX_MODEL_LEN = int(os.environ.get("SMOKE_MAX_MODEL_LEN", "8192"))
# vLLM reserves this share of the card up front and refuses to start if it is not free.
GPU_MEMORY_UTILIZATION = float(os.environ.get("SMOKE_GPU_MEM_UTIL", "0.90"))
# Eager skips torch.compile and CUDA graph capture: ~20 min saved, irrelevant for 3 prompts.
EAGER = os.environ.get("SMOKE_EAGER", "1") == "1"

# Iteration 2 in failures-and-fixes.md: FlashInfer cannot handle Gemma 4's mixed head
# dimensions (256 sliding / 512 global). That is a property of the base model, not the
# adapter, so it stays. CUDA graphs are left ON deliberately -- the previous garbled
# output was blamed on them, but that run used an adapter contaminated with router and
# vision weights. Set SMOKE_DISABLE_CUDA_GRAPH=1 to rule them out.
ATTENTION_BACKEND = os.environ.get("SMOKE_ATTENTION_BACKEND", "triton")
DISABLE_CUDA_GRAPH = os.environ.get("SMOKE_DISABLE_CUDA_GRAPH", "0") == "1"


@dataclass
class Case:
    label: str
    text: str
    reference: Optional[str] = None


def engine_loadable(config_path: Path) -> bool:
    """SGLang accepts a list of module names, or the literals 'all' / 'all-linear'.

    A regex string is valid for PEFT but fails at load with:
        Only 'all' or 'all-linear' can be used as the string for target module
    """
    targets = json.loads(config_path.read_text(encoding="utf-8"))["target_modules"]
    return isinstance(targets, list) or targets in ("all", "all-linear")


def resolve_adapter(local_dir: Path) -> Path:
    """Fetch the adapter named by SMOKE_ADAPTER_S3, else the newest '*_run/' prefix.

    Set SMOKE_ADAPTER_S3 whenever the newest prefix is not the one under test -- relying
    on "newest wins" has repeatedly meant a run silently exercised the wrong artifact.
    A local copy is reused only when no prefix was requested explicitly.

    Only the top-level adapter files are fetched; the per-step checkpoint directories
    hold DeepSpeed optimizer states worth several GB and are useless for inference.
    """
    requested = os.environ.get("SMOKE_ADAPTER_S3", "").strip().strip("/")
    local_config = local_dir / "adapter_config.json"

    if local_config.exists() and not requested:
        if engine_loadable(local_config):
            print(f"✅ Using local adapter: {local_dir}", flush=True)
            return local_dir
        print(f"⚠️  Local adapter at {local_dir} has a regex target_modules no engine "
              f"can load. Re-fetching from S3.", flush=True)

    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        sys.exit(f"❌ No adapter at {local_dir} and S3_BUCKET is unset")

    import boto3

    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")

    if requested:
        newest = f"{requested}/"
        print(f"📥 Adapter pinned by SMOKE_ADAPTER_S3: {newest}", flush=True)
    else:
        runs = set()
        for page in paginator.paginate(Bucket=bucket, Delimiter="/"):
            for prefix in page.get("CommonPrefixes", []):
                name = prefix["Prefix"]
                if name.endswith("_run/"):
                    runs.add(name)
        if not runs:
            sys.exit(f"❌ No '*_run/' prefixes in s3://{bucket}")
        newest = sorted(runs)[-1]
        print(f"📥 No local adapter. Newest run in S3: {newest}", flush=True)

    # A stale local copy must not survive a pinned request.
    if requested and local_dir.exists():
        shutil.rmtree(local_dir, ignore_errors=True)

    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=newest):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(newest):]
            # Skip checkpoint-N/ subdirectories: optimizer states, not adapter weights.
            if "/checkpoint-" in f"/{relative}" or relative.count("/") > 1:
                continue
            target = local_dir / Path(relative).name
            print(f"   {key} -> {target}", flush=True)
            client.download_file(bucket, key, str(target))
            downloaded += 1

    if not local_config.exists():
        sys.exit(f"❌ Downloaded {downloaded} objects from {newest} but found no adapter_config.json")
    if not engine_loadable(local_config):
        sys.exit(f"❌ Adapter in {newest} still has a regex target_modules. Run "
                 f"src-train/expand_targets.py on it before serving.")
    print(f"   resolved from s3://{bucket}/{newest}", flush=True)
    return local_dir


def load_cases(repo: Path) -> List[Case]:
    """Two ad-hoc inputs plus one training sample, which should show the strongest shift."""
    cases = [
        Case("greeting", "Guten Tag! Wie geht es Ihnen?"),
        Case(
            "medical",
            "Das hier sind Ihre Blutdruckwerte aus der letzten Woche. Da können Sie sehen, "
            "dass der Blutdruck immer noch zu hoch ist. Sie sollten versuchen, Ihren Blutdruck "
            "zu senken, indem Sie weniger Salz essen und mehr Sport treiben.",
        ),
    ]

    dataset = repo / "data" / "train" / "dataset.jsonl"
    if dataset.exists():
        with open(dataset, "r", encoding="utf-8") as f:
            entry = json.loads(f.readline())
        user = next(m["content"] for m in entry["messages"] if m["role"] == "user")
        assistant = next(m["content"] for m in entry["messages"] if m["role"] == "assistant")
        cases.append(Case(f"train-sample-{entry['id']}", user, assistant))
    else:
        print(f"⚠️  {dataset} missing -- skipping the training sample, which is the "
              f"highest-signal case. Run `dvc pull` first.", flush=True)
    return cases


def build_prompts(cases: List[Case], repo: Path) -> List[str]:
    system_prompt = (repo / "data" / "system-prompt.md").read_text(encoding="utf-8").strip()
    template = (repo / "data" / "prompt-template.md").read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    prompts = []
    for case in cases:
        # The training sample already carries the template; the ad-hoc ones do not.
        user = case.text if "%INPUT%" not in template or case.reference else template.replace("%INPUT%", case.text)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return prompts


def pick_engine() -> str:
    """`SMOKE_ENGINE` wins; otherwise use whichever backend the image provides."""
    if ENGINE != "auto":
        return ENGINE
    import importlib.util
    for name in ("sglang", "vllm"):
        if importlib.util.find_spec(name):
            return name
    sys.exit("❌ Neither sglang nor vllm is installed in this image")


class SGLangBackend:
    """SGLang fuses Q/K/V, so it demands a v_proj for every adapted attention layer --
    which Gemma 4's five global layers do not have (attention_k_eq_v)."""

    def __init__(self, adapter: Path) -> None:
        import sglang as sgl

        kwargs = {
            "model_path": BASE_MODEL,
            "tp_size": TP_SIZE,
            "context_length": 8192,
            "trust_remote_code": True,
            "enable_lora": True,
            "lora_paths": [f"{ADAPTER_NAME}={adapter}"],
            "max_loras_per_batch": 1,
            "attention_backend": ATTENTION_BACKEND,
        }
        if DISABLE_CUDA_GRAPH:
            kwargs["disable_cuda_graph"] = True
        print(f"⚙️  sgl.Engine({kwargs})", flush=True)
        self.engine = sgl.Engine(**kwargs)

    def generate(self, prompts: List[str], use_adapter: bool) -> List[str]:
        params = {"temperature": 0.0, "max_new_tokens": MAX_NEW_TOKENS}
        extra = {"lora_path": [ADAPTER_NAME] * len(prompts)} if use_adapter else {}
        return [o["text"].strip() for o in self.engine.generate(prompts, params, **extra)]


def lora_target_suffixes(config_path: Path) -> Optional[List[str]]:
    """CPU only: restrict which modules vLLM wraps for LoRA. None everywhere else.

    At engine start -- before any adapter is read -- vLLM walks the model and wraps every
    supported module for LoRA (`LoRAModelManager._create_lora_modules`). That includes the
    FusedMoE layers, which on a CPU backend use a monolithic kernel and assert:

        AssertionError: Monolithic kernels are not supported for Fused MoE LoRA

    The decision is made from vLLM's own `lora_config.target_modules`, never from
    adapter_config.json, so no training-side setting can avoid it. Naming the projections
    the adapter actually targets keeps the MoE runner unwrapped.

    Restricted to CPU deliberately. On CUDA the fused path works -- it logs "Using TRITON
    Unquantized MoE LoRA backend" -- and is what would apply expert LoRA weights if an
    adapter ever carried them. PEFT cannot produce those today (failures-and-fixes.md,
    Training Iteration 6: the experts are packed 3D nn.Parameters it cannot see), but
    suppressing the path globally would silently skip them if that ever changes.
    """
    from vllm.platforms import current_platform

    if not current_platform.is_cpu():
        return None

    targets = json.loads(config_path.read_text(encoding="utf-8"))["target_modules"]
    if isinstance(targets, str):
        sys.exit(f"❌ {config_path} still has a regex target_modules. Run "
                 f"src-train/expand_targets.py on it before serving.")
    suffixes = sorted({name.split(".")[-1] for name in targets})
    print(f"ℹ️  CPU backend: restricting LoRA wrapping to {suffixes}", flush=True)
    return suffixes


def diagnose_no_effect(adapter: Path) -> str:
    """Tell "the engine ignored it" apart from "there is nothing in it to apply".

    PEFT initialises every lora_B to exactly zero, so a barely-trained adapter is
    mathematically the identity: it loads and is applied, but changes no argmax over a
    262144-token vocabulary. A 2-step run reached max|B| = 3e-4 and produced identical
    output; 30 steps reached 0.14 and changed every prompt. Without this check the failure
    reads as "the engine is ignoring the adapter", which sends you after the wrong bug.
    """
    weights = adapter / "adapter_model.safetensors"
    if not weights.exists():
        return f"No {weights.name} found -- the adapter directory is incomplete."
    try:
        from safetensors import safe_open

        with safe_open(str(weights), "pt") as f:
            b_max = max(
                (f.get_tensor(k).float().abs().max().item()
                 for k in f.keys() if "lora_B" in k),
                default=0.0,
            )
    except Exception as e:  # diagnostics must never mask the real failure
        return f"Could not inspect {weights.name} ({e})."

    if b_max < 1e-3:
        return (f"max|lora_B| = {b_max:.2e}, i.e. still ~zero. The adapter is applied but "
                f"is numerically the identity -- it is undertrained, not unloaded.")
    return (f"max|lora_B| = {b_max:.2e}, so the weights are non-trivial. The engine "
            f"loaded the adapter but is not applying it.")


class VLLMBackend:
    """max_lora_rank must be raised explicitly: vLLM defaults to 16 and the adapter is r=32."""

    def __init__(self, adapter: Path) -> None:
        from vllm import LLM

        config_path = adapter / "adapter_config.json"
        rank = json.loads(config_path.read_text(encoding="utf-8"))["r"]
        kwargs = {
            "model": BASE_MODEL,
            "tensor_parallel_size": TP_SIZE,
            "max_model_len": MAX_MODEL_LEN,
            "trust_remote_code": True,
            "enable_lora": True,
            "max_loras": 1,
            "max_lora_rank": rank,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "enforce_eager": EAGER,
        }
        # Only set on CPU, so the GPU kwargs stay byte-identical to the RunPod run that
        # was verified working.
        lora_targets = lora_target_suffixes(config_path)
        if lora_targets is not None:
            kwargs["lora_target_modules"] = lora_targets

        print(f"⚙️  vllm.LLM({kwargs})", flush=True)
        self.engine = LLM(**kwargs)
        self.adapter = adapter

    def generate(self, prompts: List[str], use_adapter: bool) -> List[str]:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
        extra = {"lora_request": LoRARequest(ADAPTER_NAME, 1, str(self.adapter))} if use_adapter else {}
        return [o.outputs[0].text.strip() for o in self.engine.generate(prompts, params, **extra)]


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    adapter = resolve_adapter(Path(ADAPTER_DIR))

    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    targets = config["target_modules"]
    print("=" * 78)
    print(f"  base    : {BASE_MODEL}")
    print(f"  adapter : {adapter}")
    print(f"  targets : {targets if isinstance(targets, str) else sorted(targets)}")
    engine_name = pick_engine()
    print(f"  engine  : {engine_name} tp={TP_SIZE}")
    print("=" * 78, flush=True)

    cases = load_cases(repo)
    prompts = build_prompts(cases, repo)

    backend = SGLangBackend(adapter) if engine_name == "sglang" else VLLMBackend(adapter)

    print("\n▶ generating WITHOUT adapter (base)...", flush=True)
    base_out = backend.generate(prompts, use_adapter=False)
    print("▶ generating WITH adapter...", flush=True)
    lora_out = backend.generate(prompts, use_adapter=True)

    changed = 0
    for case, base_text, lora_text in zip(cases, base_out, lora_out):
        differs = base_text != lora_text
        changed += differs
        print("\n" + "=" * 78)
        print(f"[{case.label}] adapter changed output: {differs}")
        print("-" * 78)
        print(f"INPUT:\n{case.text[:300]}")
        print(f"\nBASE:\n{base_text[:600]}")
        print(f"\nADAPTER:\n{lora_text[:600]}")
        if case.reference:
            print(f"\nREFERENCE (training target):\n{case.reference[:600]}")

    print("\n" + "=" * 78)
    print(f"VERDICT: adapter altered {changed}/{len(cases)} outputs under greedy decoding.")
    if changed == 0:
        sys.exit(f"❌ Adapter had no effect. {diagnose_no_effect(adapter)}")
    print("✅ Adapter is being applied at inference time.")


if __name__ == "__main__":
    main()
