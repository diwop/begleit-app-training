# --- tests/test_launcher.py ---
import sys
import os
from unittest.mock import patch, MagicMock
import pytest

# 1. Mock torch cleanly before importing launcher to ensure environment-agnostic CI testing
orig_torch = sys.modules.get('torch')
mock_torch_obj = MagicMock()
sys.modules['torch'] = mock_torch_obj

from src.launcher import merge_configs, main, run_training_job

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
    mock_cuda.device_count.return_value = 8  # Simulate an 8x L40S cluster configuration
    
    # Mock configuration object attributes returned by merge_configs
    mock_cfg = MagicMock()
    mock_cfg.get.side_effect = lambda key, default=None: "/app/output/adapter/mock_job" if key == "output_dir" else default
    mock_merge.return_value = mock_cfg
    
    # FIXED: Return True for YAML configurations AND the final evaluation script path
    mock_exists.side_effect = lambda path: True if (path.endswith(".yml") or "evaluation.py" in path) else False

    main()


@patch("src.launcher.merge_configs")
@patch("src.launcher.generate_runtime_deepspeed")
@patch("src.launcher.OmegaConf.save")
@patch("src.launcher.os.path.exists")
def test_run_training_job_respects_attn_implementation(mock_exists, mock_conf_save, mock_gen_ds, mock_merge, mock_subprocess):
    """Verifies that the launcher uses flash_attention_2 by default, but respects custom configurations."""
    mock_exists.return_value = False
    
    # Test case 1: Default behavior when no attn_implementation is specified
    mock_cfg_1 = {"base_model": "some-model"}
    mock_merge.return_value = mock_cfg_1
    run_training_job("config/dummy.yml", num_gpus=1, run_id="test_run")
    assert mock_cfg_1.get("attn_implementation") == "flash_attention_2"

    # Test case 2: Overridden behavior when attn_implementation is specified in configuration
    mock_cfg_2 = {"base_model": "some-model", "attn_implementation": "sdpa"}
    mock_merge.return_value = mock_cfg_2
    run_training_job("config/dummy.yml", num_gpus=1, run_id="test_run")
    assert mock_cfg_2.get("attn_implementation") == "sdpa"


@patch("src.launcher.merge_configs")
@patch("src.launcher.generate_runtime_deepspeed")
@patch("src.launcher.OmegaConf.save")
@patch("src.launcher.os.path.exists")
@patch("src.launcher.pre_download_models")
@patch("src.launcher.run_training_job")
def test_launcher_gpu_filtering(mock_run_job, mock_pre_download, mock_exists, mock_conf_save, mock_gen_ds, mock_merge, mock_cuda):
    """
    Verifies that Mistral configurations are only run on exactly 8 GPUs,
    whereas Gemma configurations can run on other GPU counts.
    """
    mock_cuda.is_available.return_value = True
    mock_exists.return_value = True
    
    # We patch TRAINING_PIPELINE in launcher module to have both Gemma and Mistral configs
    import src.launcher
    original_pipeline = src.launcher.TRAINING_PIPELINE
    src.launcher.TRAINING_PIPELINE = [
        "config/train-mistral4small.yml",
        "config/train-gemma4.yml"
    ]
    
    try:
        # Configure return value for run_training_job mock to support unpacking
        mock_run_job.return_value = ("/app/output/adapter/mock", {})
        
        # Case 1: 2 GPUs -> Mistral should be skipped, Gemma should run
        mock_cuda.device_count.return_value = 2
        mock_run_job.reset_mock()
        mock_pre_download.reset_mock()
        
        src.launcher.main()
        
        # Verify only Gemma is passed to download and run
        mock_pre_download.assert_called_once_with(["config/train-gemma4.yml"])
        mock_run_job.assert_called_once()
        assert mock_run_job.call_args[0][0] == "config/train-gemma4.yml"
        
        # Case 2: 8 GPUs -> Both Mistral and Gemma should run
        mock_cuda.device_count.return_value = 8
        mock_run_job.reset_mock()
        mock_pre_download.reset_mock()
        
        src.launcher.main()
        
        # Verify both are downloaded and run
        mock_pre_download.assert_called_once_with(["config/train-mistral4small.yml", "config/train-gemma4.yml"])
        assert mock_run_job.call_count == 2
        called_configs = [args[0][0] for args in mock_run_job.call_args_list]
        assert "config/train-mistral4small.yml" in called_configs
        assert "config/train-gemma4.yml" in called_configs
        
    finally:
        src.launcher.TRAINING_PIPELINE = original_pipeline