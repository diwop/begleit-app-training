# --- tests/test_launcher.py ---
import sys
import os
from unittest.mock import patch, MagicMock
import pytest

# 1. Mock torch cleanly before importing launcher to ensure environment-agnostic CI testing
orig_torch = sys.modules.get('torch')
mock_torch_obj = MagicMock()
sys.modules['torch'] = mock_torch_obj

from src.launcher import merge_configs, main

# Cleanup sys.modules state immediately so other tool layers aren't polluted
if orig_torch is not None:
    sys.modules['torch'] = orig_torch
else:
    del sys.modules['torch']

@pytest.fixture
def mock_cuda():
    # Mock device properties with a real byte integer to prevent MagicMock formatting crashes (:.1f)
    mock_props = MagicMock()
    mock_props.total_memory = 48 * (1024**3)  # Default to a safe 48 GB profile
    mock_torch_obj.cuda.get_device_properties.return_value = mock_props
    yield mock_torch_obj.cuda

@pytest.fixture
def mock_subprocess():
    with patch("src.launcher.subprocess.run") as mock:
        yield mock

@pytest.fixture(autouse=True)
def mock_makedirs():
    with patch("src.launcher.os.makedirs") as mock:
        yield mock

# --- NEW FIXTURE ---
@pytest.fixture(autouse=True)
def mock_hf_env_and_download(monkeypatch):
    """
    Automatically injects a dummy HF_TOKEN to bypass the critical security check 
    and mocks snapshot_download to prevent massive network fetches during CI runs.
    """
    monkeypatch.setenv("HF_TOKEN", "mock_hf_token_for_ci_pipeline")
    with patch("src.launcher.snapshot_download") as mock_download:
        yield mock_download


def test_no_cuda_exits(mock_cuda):
    """Verifies that the launcher terminates immediately if no execution GPUs are found."""
    mock_cuda.is_available.return_value = False
    
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


@patch("src.launcher.merge_configs")
@patch("src.launcher.generate_runtime_deepspeed")
@patch("src.launcher.OmegaConf.save")
@patch("src.launcher.os.path.exists")
def test_pipeline_execution_without_s3(mock_exists, mock_conf_save, mock_gen_ds, mock_merge, mock_cuda, mock_subprocess, monkeypatch):
    """
    Verifies the complete sequential pipeline loop when S3 backups are disabled.
    Should run 2 training subprocesses and exactly 1 final global evaluation call.
    """
    monkeypatch.delenv("S3_BUCKET", raising=False)
    
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 2  # Simulate a 2x L40S cluster configuration
    
    # Mock configuration object attributes returned by merge_configs
    mock_cfg = MagicMock()
    mock_cfg.get.side_effect = lambda key, default=None: "/app/output/adapter/mock_job" if key == "output_dir" else default
    mock_merge.return_value = mock_cfg
    
    # FIXED: Return True for YAML configurations AND the final evaluation script path
    mock_exists.side_effect = lambda path: True if (path.endswith(".yml") or "evaluation.py" in path) else False

    main()