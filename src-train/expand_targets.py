"""Rewrites an adapter's regex `target_modules` into an explicit list of module names.

PEFT accepts `target_modules` as either a regex string or a list, so a regex is the
maintainable way to express "the language model only" at training time. SGLang does not:

    RuntimeError: Failed to load LoRA adapter ft:
    Only 'all' or 'all-linear' can be used as the string for target module

Expanding the regex against the base model's module tree satisfies both. The names are
fully qualified, so unlike bare suffixes they cannot also match the vision tower.

Only adapter_config.json changes; the weights are untouched.

    python src-train/expand_targets.py <adapter_dir> [base_model_id]
"""
import json
import re
import sys
from pathlib import Path
from typing import List

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

DEFAULT_BASE_MODEL = "google/gemma-4-26B-A4B-it"


def matching_modules(pattern: str, base_model_id: str) -> List[str]:
    """Every nn.Linear in the base model whose fully qualified name matches `pattern`."""
    config = AutoConfig.from_pretrained(base_model_id)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    compiled = re.compile(pattern)
    return sorted(
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and compiled.fullmatch(name)
    )


def expand(adapter_dir: Path, base_model_id: str) -> int:
    config_path = adapter_dir / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    targets = config["target_modules"]

    if not isinstance(targets, str):
        print(f"target_modules is already a list of {len(targets)} names; nothing to do.")
        return len(targets)
    if targets in ("all", "all-linear"):
        print(f"target_modules is '{targets}', which engines accept verbatim; nothing to do.")
        return 0

    print(f"expanding regex: {targets}")
    names = matching_modules(targets, base_model_id)
    if not names:
        sys.exit(f"❌ Regex matched no nn.Linear modules in {base_model_id}")

    config["target_modules"] = names
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return len(names)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        print(__doc__)
        sys.exit(1)

    adapter_dir = Path(args[0])
    base_model_id = args[1] if len(args) > 1 else DEFAULT_BASE_MODEL
    if not (adapter_dir / "adapter_config.json").exists():
        sys.exit(f"❌ No adapter_config.json in {adapter_dir}")

    count = expand(adapter_dir, base_model_id)
    print(f"✅ target_modules now lists {count} fully qualified module names")


if __name__ == "__main__":
    main()
