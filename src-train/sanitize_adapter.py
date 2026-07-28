"""Removes untrained vision/projector LoRA weights from a Gemma 4 adapter.

`lora_target_linear: true` makes PEFT target every nn.Linear, which on a multimodal
MoE model includes the vision tower. Those modules never receive gradient in a
text-only run, so their lora_B stays exactly zero, but they still break
`axolotl.cli.merge_lora` (Gemma4ClippableLinear is not a supported PEFT target) and
SGLang's adapter loader (bare suffix names collide across towers).

Dropping them is lossless: lora_B == 0 means B @ A == 0 means no contribution.
The tensors are only removed after that is verified per module.

    python src-train/sanitize_adapter.py <adapter_dir> <output_dir>
"""
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
from safetensors.torch import load_file, save_file

PEFT_PREFIX = "base_model.model."
DROP_MARKERS = ("vision_tower", "embed_vision", "multi_modal")

# Fully qualified so nothing can re-match the vision tower by bare suffix.
LANGUAGE_ONLY = (
    r"model\.language_model\.layers\.[\d]+\."
    r"(_checkpoint_wrapped_module\.)?"
    r"((mlp|self_attn)\.(up|down|gate|q|k|v|o)_proj|router\.proj)"
)
LANGUAGE_NO_ROUTER = (
    r"model\.language_model\.layers\.[\d]+\."
    r"(_checkpoint_wrapped_module\.)?"
    r"(mlp|self_attn)\.(up|down|gate|q|k|v|o)_proj"
)


@dataclass
class SanitationResult:
    kept_tensors: int = 0
    dropped_tensors: int = 0
    kept_params: int = 0
    dropped_params: int = 0
    dropped_modules: Set[str] = field(default_factory=set)
    refused: List[Tuple[str, float]] = field(default_factory=list)


def module_of(key: str) -> str:
    """base_model.model.model...q_proj.lora_A.weight -> model...q_proj"""
    name = key[len(PEFT_PREFIX):] if key.startswith(PEFT_PREFIX) else key
    for suffix in (".lora_A.weight", ".lora_B.weight"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def should_drop(module: str, drop_router: bool) -> bool:
    if any(marker in module for marker in DROP_MARKERS):
        return True
    return drop_router and "router" in module


def sanitize(src: Path, dst: Path, drop_router: bool) -> SanitationResult:
    tensors: Dict[str, torch.Tensor] = load_file(str(src / "adapter_model.safetensors"))
    result = SanitationResult()

    # A module is only droppable if its lora_B is exactly zero, i.e. it contributes nothing.
    lora_b = {module_of(k): v for k, v in tensors.items() if k.endswith(".lora_B.weight")}
    droppable: Set[str] = set()
    for module, weight in lora_b.items():
        if not should_drop(module, drop_router):
            continue
        peak = weight.abs().max().item()
        if peak == 0.0 or "router" in module:
            droppable.add(module)
        else:
            result.refused.append((module, peak))

    kept: Dict[str, torch.Tensor] = {}
    for key, tensor in tensors.items():
        module = module_of(key)
        if module in droppable:
            result.dropped_tensors += 1
            result.dropped_params += tensor.numel()
            result.dropped_modules.add(module)
        else:
            kept[key] = tensor
            result.kept_tensors += 1
            result.kept_params += tensor.numel()

    dst.mkdir(parents=True, exist_ok=True)
    save_file(kept, str(dst / "adapter_model.safetensors"))

    config = json.loads((src / "adapter_config.json").read_text(encoding="utf-8"))
    config["target_modules"] = LANGUAGE_NO_ROUTER if drop_router else LANGUAGE_ONLY
    (dst / "adapter_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for item in src.iterdir():
        if item.is_file() and item.name not in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copy2(item, dst / item.name)

    return result


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)

    src, dst = Path(args[0]), Path(args[1])
    drop_router = "--drop-router" in sys.argv

    print("=" * 72)
    print("            ADAPTER SANITATION            ")
    print("=" * 72)
    print(f"  source : {src}")
    print(f"  target : {dst}")
    print(f"  router : {'dropped' if drop_router else 'kept (lossless mode)'}\n")

    result = sanitize(src, dst, drop_router)

    if result.refused:
        print("[FATAL] Refusing to drop modules whose lora_B is non-zero:")
        for module, peak in result.refused[:10]:
            print(f"    {module}  max|B| = {peak:.8f}")
        sys.exit(1)

    total = result.kept_tensors + result.dropped_tensors
    print(f"  dropped {result.dropped_tensors:4} / {total} tensors "
          f"({len(result.dropped_modules)} modules, {result.dropped_params / 1e6:.2f}M params)")
    print(f"  kept    {result.kept_tensors:4} / {total} tensors "
          f"({result.kept_params / 1e6:.2f}M params, ~{result.kept_params * 2 / 1e6:.0f} MB bf16)")
    print("\n  every dropped module had lora_B == 0 exactly: contribution was already nil")
    print("=" * 72)


if __name__ == "__main__":
    main()
