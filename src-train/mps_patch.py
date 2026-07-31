"""Make Axolotl run on Apple Silicon, so the pipeline can be exercised on a laptop.

Both patches are no-ops off Metal, so the RunPod path is byte-for-byte unchanged. See
docs/local-pipeline.md for how to run the whole thing locally.

1. `install_triton_stub()` -- must run BEFORE `import axolotl`.
   `axolotl/monkeypatch/trainer/__init__.py` eagerly imports a module whose top level does
   `import triton`, and Triton publishes no macOS wheel (checked: triton, triton-cpu,
   pytorch-triton are all Linux-only). The two functions it exports, `entropy_from_logits`
   and `selective_log_softmax`, are referenced only by `core/trainers/grpo/async_trainer.py`
   -- reinforcement learning. Supervised fine-tuning never calls them. Axolotl already
   wraps its *other* import of that module in try/except ImportError; the package
   `__init__` is the one place it does not, which looks like an upstream oversight.

   Stubbing `triton` itself does not work: torch probes for `triton.backends`, the fake
   module is not a package, and the failure cascades into unrelated transformers import
   errors. Replacing the single leaf module is narrower and leaves torch alone.

2. `apply_mps_device_map_patch()` -- must run AFTER `import axolotl`.
   `ModelLoader._set_device_map_config` (axolotl/loaders/model.py) unconditionally
   overwrites device_map with the literal "mps:0" whenever the device is Metal, ignoring
   whatever the config asked for. Measured on an M1 Max loading tiny-random/gemma-4-moe:

       no device_map, then .to("mps")   OK
       device_map="auto"                hangs (>90 s)
       device_map="mps"                 SIGSEGV
       device_map="mps:0"               hangs (>90 s)   <- what Axolotl forces
       device_map={"": "mps"}           hangs (>90 s)
       device_map={"": 0}               OK
       device_map="cpu"                 OK

   {"": 0} is the one accelerate spelling that places the whole model on the Metal device
   without going through the big-model dispatch path that hangs. The same line is why
   plain `from_pretrained(..., device_map="auto")` dies on Metal in any framework.
"""
import sys
from types import ModuleType

import torch

TRITON_MODULE = "axolotl.monkeypatch.trainer.utils"


def on_metal() -> bool:
    """True only on Apple Silicon with a working Metal backend."""
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def install_triton_stub() -> bool:
    """Pre-register the one Axolotl module whose top-level `import triton` fails."""
    if not on_metal() or TRITON_MODULE in sys.modules:
        return False

    def unavailable(*args, **kwargs):
        raise NotImplementedError(
            "Triton kernel unavailable on Apple Silicon. This is a reinforcement-learning "
            "code path (cfg.rl); supervised fine-tuning never reaches it."
        )

    stub = ModuleType(TRITON_MODULE)
    setattr(stub, "entropy_from_logits", unavailable)
    setattr(stub, "selective_log_softmax", unavailable)
    sys.modules[TRITON_MODULE] = stub
    print(f"🔧 MONKEYPATCH: stubbed {TRITON_MODULE} (Triton has no macOS wheel)", flush=True)
    return True


def apply_mps_device_map_patch() -> bool:
    """Rewrite Axolotl's forced "mps:0" device_map to the {"": 0} spelling that works."""
    if not on_metal():
        return False

    from axolotl.loaders.model import ModelLoader

    original = ModelLoader._set_device_map_config

    def patched(self) -> None:
        original(self)
        if str(self.model_kwargs.get("device_map")) == "mps:0":
            self.model_kwargs["device_map"] = {"": 0}
            print("🔧 MONKEYPATCH: device_map 'mps:0' -> {'': 0} (Metal)", flush=True)

    ModelLoader._set_device_map_config = patched
    return True
