# --- src/train_patched.py ---
import sys
import fire

# Import target modules
import axolotl.train
import axolotl.cli.train

original_train = axolotl.train.train

def patched_train(cfg, *args, **kwargs):
    print("\n" + "="*60)
    print("🔧 MONKEYPATCH: Overriding gradient_checkpointing_kwargs to use_reentrant=True")
    print("This bypasses the DeepSpeed ZeRO-3 parameter sharding metadata mismatch.")
    print("="*60 + "\n", flush=True)
    
    if hasattr(cfg, "gradient_checkpointing_kwargs") and cfg.gradient_checkpointing_kwargs:
        cfg.gradient_checkpointing_kwargs["use_reentrant"] = True
    else:
        cfg.gradient_checkpointing_kwargs = {"use_reentrant": True}
        
    return original_train(cfg, *args, **kwargs)

# Apply monkeypatch globally
axolotl.train.train = patched_train
if hasattr(axolotl.cli.train, "train"):
    axolotl.cli.train.train = patched_train

if __name__ == "__main__":
    fire.Fire(axolotl.cli.train.do_cli)
