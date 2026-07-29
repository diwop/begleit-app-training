"""Removes one projection type from a trained adapter, config and weights together.

Gemma 4 sets `attention_k_eq_v: true`, so its five global-attention layers (5, 11, 17,
23, 29) share one tensor for K and V and have no `v_proj` module at all. An adapter that
correctly omits those layers is rejected by SGLang, whose LoRA loader assumes every layer
carries every targeted projection:

    RuntimeError: Failed to load LoRA adapter ft:
    'base_model.model.model.language_model.layers.5.self_attn.v_proj.lora_A.weight'

Dropping the projection entirely makes the adapter homogeneous. Unlike removing the
untrained vision weights this is lossy, so the discarded magnitude is reported.

    python src-train/drop_projection.py <adapter_dir> <output_dir> v_proj
"""
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from safetensors.torch import load_file, save_file

PEFT_PREFIX = "base_model.model."


@dataclass
class DropResult:
    kept_modules: int
    dropped_modules: int
    kept_params: int
    dropped_params: int
    dropped_rms: float


def module_of(key: str) -> str:
    name = key[len(PEFT_PREFIX):] if key.startswith(PEFT_PREFIX) else key
    for suffix in (".lora_A.weight", ".lora_B.weight"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def drop(src: Path, dst: Path, projection: str) -> DropResult:
    tensors: Dict[str, torch.Tensor] = load_file(str(src / "adapter_model.safetensors"))
    config = json.loads((src / "adapter_config.json").read_text(encoding="utf-8"))
    targets = config["target_modules"]
    if not isinstance(targets, list):
        sys.exit("❌ target_modules must be a list; run src-train/expand_targets.py first")

    kept_targets: List[str] = [t for t in targets if not t.endswith(f".{projection}")]
    dropped_targets = len(targets) - len(kept_targets)
    if not dropped_targets:
        sys.exit(f"❌ No target module ends with '.{projection}'")

    kept: Dict[str, torch.Tensor] = {}
    dropped_params = 0
    squares, count = 0.0, 0
    for key, tensor in tensors.items():
        if module_of(key).endswith(f".{projection}"):
            dropped_params += tensor.numel()
            if key.endswith(".lora_B.weight"):
                squares += tensor.float().pow(2).sum().item()
                count += tensor.numel()
        else:
            kept[key] = tensor

    dst.mkdir(parents=True, exist_ok=True)
    save_file(kept, str(dst / "adapter_model.safetensors"))
    config["target_modules"] = kept_targets
    (dst / "adapter_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for item in src.iterdir():
        if item.is_file() and item.name not in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copy2(item, dst / item.name)

    return DropResult(
        kept_modules=len(kept_targets),
        dropped_modules=dropped_targets,
        kept_params=sum(t.numel() for t in kept.values()),
        dropped_params=dropped_params,
        dropped_rms=(squares / count) ** 0.5 if count else 0.0,
    )


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if len(args) < 3:
        print(__doc__)
        sys.exit(1)

    src, dst, projection = Path(args[0]), Path(args[1]), args[2]
    result = drop(src, dst, projection)

    print("=" * 72)
    print(f"            DROPPED '{projection}' FROM ADAPTER            ")
    print("=" * 72)
    print(f"  modules : {result.dropped_modules} dropped, {result.kept_modules} kept")
    print(f"  params  : {result.dropped_params / 1e6:.2f}M dropped, {result.kept_params / 1e6:.2f}M kept "
          f"({100 * result.dropped_params / (result.dropped_params + result.kept_params):.1f}% lost)")
    print(f"  lora_B rms of discarded weights: {result.dropped_rms:.8f}")
    print("=" * 72)
    print("\nThis is lossy: the discarded weights were trained, unlike the vision tower's zeros.")


if __name__ == "__main__":
    main()
