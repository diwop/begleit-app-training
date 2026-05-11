def test_unsloth_import():
    try:
        from unsloth import FastLanguageModel
    except ImportError as e:
        import pytest
        pytest.fail(f"Failed to import unsloth. FastLanguageModel could not be loaded: {e}")
    except AttributeError as e:
        import pytest
        pytest.fail(f"AttributeError during unsloth import (likely an incompatible PyTorch/Triton version): {e}")

def test_train_script_import():
    try:
        import src.train
    except Exception as e:
        import pytest
        pytest.fail(f"Failed to import src.train: {e}")
