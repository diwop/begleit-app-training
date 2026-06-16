# --- src/train_patched.py ---
import sys
import fire

# Import target modules
import axolotl.train
import axolotl.cli.train

# --- Apply tokenizer patch to avoid mistral-common validation errors ---
try:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    
    # Patch MistralCommonTokenizer if present
    try:
        from transformers.tokenization_mistral_common import MistralCommonTokenizer
        MistralCommonTokenizer.apply_chat_template = PreTrainedTokenizerBase.apply_chat_template
        print("🔧 MONKEYPATCH: Successfully patched MistralCommonTokenizer.apply_chat_template")
    except ImportError:
        pass

    # Patch TokenizersBackend if present
    try:
        from transformers.tokenization_utils_tokenizers import TokenizersBackend
        TokenizersBackend.apply_chat_template = PreTrainedTokenizerBase.apply_chat_template
        print("🔧 MONKEYPATCH: Successfully patched TokenizersBackend.apply_chat_template")
    except ImportError:
        pass
except Exception as e:
    print(f"⚠️ Warning: Failed to apply tokenizer apply_chat_template monkeypatch: {e}")

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
