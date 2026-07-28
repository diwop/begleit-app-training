"""Audits which base-model modules a LoRA adapter targets, before it reaches the inference engine.

Runs on the meta device: the base model is instantiated from its config only, so no
weights are downloaded and no GPU is required.

    python src-train/audit_adapter.py                     # inventory the base model only
    python src-train/audit_adapter.py /app/output/adapter # audit a trained adapter
"""
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

DEFAULT_BASE_MODEL = "google/gemma-4-26B-A4B-it"

# Buckets an adapter may touch, and whether an inference engine can serve them.
SERVABLE = ("language attention", "language shared MLP")
UNSERVABLE = ("vision tower", "multimodal projector", "MoE router", "routed experts")


@dataclass
class AdapterAudit:
    targeted: Dict[str, List[str]] = field(default_factory=dict)
    params: Dict[str, int] = field(default_factory=dict)
    missing_v_proj: List[int] = field(default_factory=list)
    saved_keys: int = 0

    @property
    def offending(self) -> List[str]:
        return [b for b in self.targeted if b in UNSERVABLE]


def classify(name: str) -> str:
    if "vision_tower" in name:
        return "vision tower"
    if "embed_vision" in name or "multi_modal" in name:
        return "multimodal projector"
    if "router" in name:
        return "MoE router"
    if "experts" in name:
        return "routed experts"
    if "self_attn" in name:
        return "language attention"
    if ".mlp." in name:
        return "language shared MLP"
    if name.endswith("lm_head") or "embed_tokens" in name:
        return "embeddings"
    return "other"


def load_base_modules(model_id: str) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, Tuple[int, ...]]]:
    """Returns ({linear_name: (out, in)}, {packed_param_name: shape}) for the base model."""
    config = AutoConfig.from_pretrained(model_id)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    linears = {
        name: (module.out_features, module.in_features)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    packed = {
        name: tuple(param.shape)
        for name, param in model.named_parameters()
        if param.dim() == 3
    }
    return linears, packed


def matches(name: str, target_modules) -> bool:
    """Reimplements PEFT target resolution: regex for a string, suffix match for a list."""
    if isinstance(target_modules, str):
        return re.fullmatch(target_modules, name) is not None
    return any(name == t or name.endswith(f".{t}") for t in target_modules)


def find_missing_v_proj(linears: Dict[str, Tuple[int, int]]) -> List[int]:
    """Gemma 4 shares K and V on global-attention layers, so those have no v_proj."""
    pattern = re.compile(r"layers\.(\d+)\.self_attn\.([qv])_proj$")
    seen: Dict[str, set] = {"q": set(), "v": set()}
    for name in linears:
        match = pattern.search(name)
        if match:
            seen[match.group(2)].add(int(match.group(1)))
    return sorted(seen["q"] - seen["v"])


def audit(adapter_dir: Path, linears: Dict[str, Tuple[int, int]], rank: int) -> AdapterAudit:
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    target_modules = config["target_modules"]
    rank = int(config.get("r", rank))

    result = AdapterAudit(missing_v_proj=find_missing_v_proj(linears))
    for name, (out_features, in_features) in linears.items():
        if not matches(name, target_modules):
            continue
        bucket = classify(name)
        result.targeted.setdefault(bucket, []).append(name)
        result.params[bucket] = result.params.get(bucket, 0) + rank * (out_features + in_features)

    weights = adapter_dir / "adapter_model.safetensors"
    if weights.exists():
        from safetensors import safe_open

        with safe_open(str(weights), framework="pt") as handle:
            result.saved_keys = len(handle.keys())
    return result


def report_base(linears: Dict[str, Tuple[int, int]], packed: Dict[str, Tuple[int, ...]]) -> None:
    print("=" * 72)
    print("            BASE MODEL MODULE INVENTORY            ")
    print("=" * 72)

    counts: Dict[str, int] = {}
    for name in linears:
        counts[classify(name)] = counts.get(classify(name), 0) + 1
    for bucket, count in sorted(counts.items(), key=lambda item: -item[1]):
        marker = "  <-- not servable with LoRA" if bucket in UNSERVABLE else ""
        print(f"  {count:5} nn.Linear   {bucket}{marker}")

    print(f"\n  {len(packed):5} packed 3D parameters (invisible to LoRA):")
    for name, shape in list(packed.items())[:2]:
        print(f"        {name}  {shape}")
    print("=" * 72)


def report_audit(result: AdapterAudit, adapter_dir: Path) -> None:
    print("\n" + "=" * 72)
    print(f"            ADAPTER AUDIT: {adapter_dir}            ")
    print("=" * 72)

    total = sum(result.params.values()) or 1
    for bucket, count in sorted(result.params.items(), key=lambda item: -item[1]):
        modules = len(result.targeted[bucket])
        marker = "  <-- ENGINE WILL REJECT" if bucket in UNSERVABLE else ""
        print(f"  {modules:5} modules  {count / 1e6:8.2f}M  {100 * count / total:5.1f}%  {bucket}{marker}")
    print(f"  {'':5}           {total / 1e6:8.2f}M  total (~{total * 2 / 1e6:.0f} MB bf16)")

    if result.saved_keys:
        print(f"\n  {result.saved_keys} tensors in adapter_model.safetensors")

    servable = sum(result.params.get(b, 0) for b in SERVABLE)
    print(f"\n  servable : {servable / 1e6:8.2f}M  ({100 * servable / total:.1f}%)")
    print(f"  wasted   : {(total - servable) / 1e6:8.2f}M  ({100 * (total - servable) / total:.1f}%)")

    if result.missing_v_proj:
        print(f"\n  NOTE: layers {result.missing_v_proj} have no v_proj (K and V are shared).")
        print("        A uniform target list silently skips them; this is expected, not a bug.")

    print("=" * 72)
    if result.offending:
        print(f"\n[FATAL] Adapter targets non-servable modules: {', '.join(result.offending)}")
        print("        Restrict lora_target_modules to the language model in config/base.yml.")
    else:
        print("\n[OK] Adapter targets only modules an inference engine can serve.")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    adapter_dir = Path(args[0]) if args else None
    base_model = args[1] if len(args) > 1 else DEFAULT_BASE_MODEL

    print(f"Instantiating '{base_model}' on the meta device (no weights)...")
    linears, packed = load_base_modules(base_model)
    report_base(linears, packed)

    if adapter_dir is None:
        print("\nNo adapter given. Pass a directory containing adapter_config.json to audit one.")
        return

    if not (adapter_dir / "adapter_config.json").exists():
        print(f"\n[FATAL] No adapter_config.json found in {adapter_dir}")
        sys.exit(1)

    result = audit(adapter_dir, linears, rank=32)
    report_audit(result, adapter_dir)
    sys.exit(1 if result.offending else 0)


if __name__ == "__main__":
    main()
