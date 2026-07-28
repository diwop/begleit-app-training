"""Does SGLang actually serve a LoRA adapter on the Gemma 4 MoE base?

Deliberately minimal: no monkeypatches, no S3, no readability metrics, no exception
handling that turns a crash into an empty string. One engine is loaded with the adapter
enabled, then the same prompts are generated twice -- once routed through the adapter and
once bypassing it -- so any difference is attributable to the adapter alone.

Sampling is greedy (temperature 0) so that "the outputs differ" means the adapter was
applied, not that we sampled twice.

    python src-eval/smoke_adapter.py
    SMOKE_BASE=RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic python src-eval/smoke_adapter.py
"""
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import sglang as sgl
from transformers import AutoTokenizer

BASE_MODEL = os.environ.get("SMOKE_BASE", "google/gemma-4-26b-a4b-it")
ADAPTER_DIR = os.environ.get("SMOKE_ADAPTER", "/app/output/adapter/train-gemma4")
ADAPTER_NAME = "ft"
TP_SIZE = int(os.environ.get("TP_SIZE", "2"))
MAX_NEW_TOKENS = int(os.environ.get("SMOKE_MAX_TOKENS", "512"))

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


def resolve_adapter(local_dir: Path) -> Path:
    """Use the adapter on the volume, else pull the newest one from S3.

    Only the top-level adapter files are fetched; the per-step checkpoint directories
    hold DeepSpeed optimizer states worth several GB and are useless for inference.
    """
    if (local_dir / "adapter_config.json").exists():
        print(f"✅ Using local adapter: {local_dir}", flush=True)
        return local_dir

    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        sys.exit(f"❌ No adapter at {local_dir} and S3_BUCKET is unset")

    import boto3

    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")
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

    if not (local_dir / "adapter_config.json").exists():
        sys.exit(f"❌ Downloaded {downloaded} objects from {newest} but found no adapter_config.json")
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


def generate(engine: "sgl.Engine", prompts: List[str], use_adapter: bool) -> List[str]:
    sampling_params = {"temperature": 0.0, "max_new_tokens": MAX_NEW_TOKENS}
    kwargs = {}
    if use_adapter:
        kwargs["lora_path"] = [ADAPTER_NAME] * len(prompts)
    outputs = engine.generate(prompts, sampling_params, **kwargs)
    return [o["text"].strip() for o in outputs]


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    adapter = resolve_adapter(Path(ADAPTER_DIR))

    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    targets = config["target_modules"]
    print("=" * 78)
    print(f"  base    : {BASE_MODEL}")
    print(f"  adapter : {adapter}")
    print(f"  targets : {targets if isinstance(targets, str) else sorted(targets)}")
    print(f"  engine  : tp={TP_SIZE} attention={ATTENTION_BACKEND} cuda_graph={not DISABLE_CUDA_GRAPH}")
    print("=" * 78, flush=True)

    cases = load_cases(repo)
    prompts = build_prompts(cases, repo)

    engine_kwargs = {
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
        engine_kwargs["disable_cuda_graph"] = True

    print(f"⚙️  sgl.Engine({engine_kwargs})", flush=True)
    engine = sgl.Engine(**engine_kwargs)

    print("\n▶ generating WITHOUT adapter (base)...", flush=True)
    base_out = generate(engine, prompts, use_adapter=False)
    print("▶ generating WITH adapter...", flush=True)
    lora_out = generate(engine, prompts, use_adapter=True)

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
        sys.exit("❌ Adapter had no effect -- it loaded but is not being applied.")
    print("✅ Adapter is being applied at inference time.")


if __name__ == "__main__":
    main()
