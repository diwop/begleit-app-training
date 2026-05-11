import os
import pytest

def test_unsloth_import():
    if os.environ.get("IN_DOCKER") != "true":
        pytest.skip("Skipping unsloth import test outside of Docker environment.")
    
    try:
        from unsloth import FastLanguageModel
    except ImportError as e:
        pytest.fail(f"Failed to import unsloth. FastLanguageModel could not be loaded: {e}")
    except AttributeError as e:
        pytest.fail(f"AttributeError during unsloth import (likely an incompatible PyTorch/Triton version): {e}")
    except NotImplementedError as e:
        if "only works on NVIDIA, AMD and Intel GPUs" in str(e):
            pytest.skip("Unsloth installed correctly, but CI runner lacks GPU. Skipping further checks.")
        else:
            pytest.fail(f"Unexpected NotImplementedError during unsloth import: {e}")

def test_train_script_import():
    if os.environ.get("IN_DOCKER") != "true":
        pytest.skip("Skipping train script import test outside of Docker environment.")
        
    try:
        import src.train
    except Exception as e:
        if "only works on NVIDIA, AMD and Intel GPUs" in str(e):
            pytest.skip("Unsloth installed correctly, but CI runner lacks GPU. Skipping src.train test.")
        else:
            pytest.fail(f"Failed to import src.train: {e}")
