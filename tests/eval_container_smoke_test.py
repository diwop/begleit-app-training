import sys

def run_smoke_test():
    print("Running SGLang evaluation dependency sanity check...")

    try:
        import torch
        import transformers
        import sglang

        print(f"PyTorch Version: {torch.__version__}")
        print(f"Transformers Version: {transformers.__version__}")
        print(f"SGLang Version: {sglang.__version__}")

        assert transformers.is_torch_available(), "CRITICAL: Transformers cannot find PyTorch!"

        # Fail here rather than after a pod has booted and pulled 50 GB of weights.
        from sglang import Engine
        assert "enable_lora" in Engine.__init__.__code__.co_varnames, \
            "CRITICAL: sglang.Engine has no enable_lora parameter; adapter serving is unsupported."

        print("Success: Core evaluation dependencies imported perfectly!")

    except Exception as e:
        print(f"\n[FATAL ERROR] SGLang smoke test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_smoke_test()
