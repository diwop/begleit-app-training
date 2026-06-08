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
    # FIXED: Automatically mock device properties with a real byte integer 
    # to prevent MagicMock object format string precision crashes (:.1f)
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
    # Force S3 bucket variable to be absent
    monkeypatch.delenv("S3_BUCKET", raising=False)
    
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 2  # Simulate a 2x L40S cluster configuration
    
    # Mock configuration object attributes returned by merge_configs
    mock_cfg = MagicMock()
    mock_cfg.get.side_effect = lambda key, default=None: "/app/output/adapter/mock_job" if key == "output_dir" else default
    mock_merge.return_value = mock_cfg
    mock_exists.return_value = False  # Assume EVAL=false or no adapter exists yet

    main()

    # Total expected calls: 2 training jobs + 1 global evaluation = 3 subprocess invocations
    assert mock_subprocess.call_count == 3

    # Verify the first training call parameters
    train_cmd_1 = mock_subprocess.call_args_list[0][0][0]
    assert train_cmd_1[0] == "accelerate"
    assert train_cmd_1[1] == "launch"
    assert "--multi_gpu" in train_cmd_1
    assert train_cmd_1[train_cmd_1.index("--num_processes") + 1] == "2"
    assert "-m" in train_cmd_1
    assert "axolotl.cli.train" in train_cmd_1

    # Verify the second training call parameters
    train_cmd_2 = mock_subprocess.call_args_list[1][0][0]
    assert train_cmd_2[train_cmd_2.index("--num_processes") + 1] == "2"

    # Verify that the final call cleanly invokes the global evaluation script without flags
    eval_cmd = mock_subprocess.call_args_list[2][0][0]
    assert eval_cmd == ["python", "src/evaluation.py"]


@patch("src.launcher.merge_configs")
@patch("src.launcher.generate_runtime_deepspeed")
@patch("src.launcher.OmegaConf.save")
@patch("src.launcher.os.path.exists")
def test_pipeline_execution_with_s3(mock_exists, mock_conf_save, mock_gen_ds, mock_merge, mock_cuda, mock_subprocess, monkeypatch):
    """
    Verifies that the launcher triggers AWS S3 backup synchronizations 
    for completed directories when S3_BUCKET is active.
    """
    monkeypatch.setenv("S3_BUCKET", "production-evaluation-bucket")
    
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 4  # Simulate a 4x GPU cluster setup
    
    mock_cfg = MagicMock()
    mock_cfg.get.side_effect = lambda key, default=None: "/app/output/adapter/mock_job" if key == "output_dir" else default
    mock_merge.return_value = mock_cfg
    
    # Ensure os.path.exists confirms that training artifact records were generated successfully
    mock_exists.side_effect = lambda path: True if "adapter_config.json" in path else False

    main()

    # Total expected calls: 2 training jobs + 2 s3 synchronization pushes + 1 evaluation = 5 invocations
    assert mock_subprocess.call_count == 5

    # Scan subprocess arguments to verify that an AWS S3 command layer was executed
    s3_sync_triggered = False
    for call in mock_subprocess.call_args_list:
        cmd = call[0][0]
        if "aws" in cmd and "s3" in cmd and "sync" in cmd:
            s3_sync_triggered = True
            assert "production-evaluation-bucket" in cmd[4]
            
    assert s3_sync_triggered, "❌ Pipeline failed to trigger aws s3 sync sub-processes."


# Updated to reflect our clean, production-targeted base model definitions
CONFIG_TEST_CASES = [
    (
        "config/train-gemma4.yml", 
        "google/gemma-4-26b-a4b-it"
    ),
    (
        "config/train-mistral4small.yml", 
        "mistralai/Mistral-Small-4-119B-2603"
    ),
]

@pytest.mark.parametrize("override_file, expected_model", CONFIG_TEST_CASES)
@patch("src.launcher.OmegaConf.load")
@patch("src.launcher.OmegaConf.merge")
def test_merge_configs(mock_merge, mock_load, override_file, expected_model):
    """
    Tests that the configuration engine fluidly maps base properties
    and assigns proper base model tags.
    """
    mock_base = MagicMock()
    mock_base.adapter = "qlora"
    
    mock_override = MagicMock()
    mock_override.base_model = expected_model
    
    # Configure mock loaders to mimic configuration structural tracking maps
    mock_load.side_effect = [mock_base, mock_override]
    mock_merge.return_value = mock_override

    merged_cfg = merge_configs("config/base.yml", override_file)
    
    assert merged_cfg.base_model == expected_model