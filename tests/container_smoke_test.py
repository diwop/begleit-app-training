import sys

def run_smoke_test():
    print("Running dependency sanity check...")

    try:
        import torch
        import transformers
        
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Transformers Version: {transformers.__version__}")
        
        assert transformers.is_torch_available(), "CRITICAL: Transformers cannot find PyTorch!"
        
        print("\nTesting Axolotl deep-import tree...")
        import axolotl.cli.train
        print("Success: Axolotl imported perfectly!")

        print("\nTesting DeepSpeed import...")
        import deepspeed
        print(f"Success: DeepSpeed ({deepspeed.__version__}) imported perfectly!")

        print("\nTesting Liger Kernel import...")
        import liger_kernel
        print("Success: Liger Kernel imported perfectly!")

    except Exception as e:
        print(f"\n[FATAL ERROR] Smoke test failed: {e}")
        sys.exit(1)

# This prevents the code from running when pytest imports the file!
if __name__ == "__main__":
    run_smoke_test()