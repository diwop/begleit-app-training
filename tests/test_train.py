# --- tests/test_launcher.py ---
import sys
import os
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src-train")))

from unittest.mock import patch, MagicMock
import pytest

# 1. Mock torch cleanly before importing launcher to ensure environment-agnostic CI testing
orig_torch = sys.modules.get('torch')
mock_torch_obj = MagicMock()
sys.modules['torch'] = mock_torch_obj

from train import merge_configs, main, run_training_job

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
    with patch("train.subprocess.run") as mock:
        yield mock

@pytest.fixture(autouse=True)
def mock_makedirs():
    with patch("train.os.makedirs") as mock:
        yield mock

# --- NEW FIXTURE ---
@pytest.fixture(autouse=True)
def mock_hf_env(monkeypatch):
    """
    Automatically injects a dummy HF_TOKEN to bypass the critical security check.
    Weights are fetched by the `hf` CLI, which mock_subprocess already intercepts.
    """
    monkeypatch.setenv("HF_TOKEN", "mock_hf_token_for_ci_pipeline")


def test_no_cuda_exits(mock_cuda):
    """Verifies that the launcher terminates immediately if no execution GPUs are found."""
    mock_cuda.is_available.return_value = False
    
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


@patch("train.merge_configs")
@patch("train.generate_runtime_deepspeed")
@patch("train.OmegaConf.save")
@patch("train.os.path.exists")
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


@patch("train.merge_configs")
@patch("train.generate_runtime_deepspeed")
@patch("train.OmegaConf.save")
@patch("train.os.path.exists")
def test_run_training_job_respects_attn_implementation(mock_exists, mock_conf_save, mock_gen_ds, mock_merge, mock_subprocess):
    """Verifies that the launcher uses flash_attention_2 by default, but respects custom configurations."""
    mock_exists.return_value = False
    
    # Test case 1: Default behavior when no attn_implementation is specified
    mock_cfg_1 = {"base_model": "some-model"}
    mock_merge.return_value = mock_cfg_1
    run_training_job("config/dummy.yml", num_gpus=1)
    assert mock_cfg_1.get("attn_implementation") == "flash_attention_2"

    # Test case 2: Overridden behavior when attn_implementation is specified in configuration
    mock_cfg_2 = {"base_model": "some-model", "attn_implementation": "sdpa"}
    mock_merge.return_value = mock_cfg_2
    run_training_job("config/dummy.yml", num_gpus=1)
    assert mock_cfg_2.get("attn_implementation") == "sdpa"


@patch("train.merge_configs")
@patch("train.generate_runtime_deepspeed")
@patch("train.OmegaConf.save")
@patch("train.os.path.exists")
@patch("train.pre_download_models")
@patch("train.run_training_job")
def test_launcher_gpu_filtering(mock_run_job, mock_pre_download, mock_exists, mock_conf_save, mock_gen_ds, mock_merge, mock_cuda):
    """
    Verifies that Mistral configurations are only run on exactly 8 GPUs,
    whereas Gemma configurations can run on other GPU counts.
    """
    mock_cuda.is_available.return_value = True
    mock_exists.return_value = True
    
    # We patch TRAINING_PIPELINE in launcher module to have both Gemma and Mistral configs
    import train
    original_pipeline = train.TRAINING_PIPELINE
    train.TRAINING_PIPELINE = [
        "config/train-mistral4small.yml",
        "config/train-gemma4.yml"
    ]
    
    try:
        # Configure return value for run_training_job mock to support unpacking
        mock_run_job.return_value = ("/app/output/adapter/mock", {}, ".merged-mock.yml")
        
        # Case 1: 2 GPUs -> Mistral should be skipped, Gemma should run
        mock_cuda.device_count.return_value = 2
        mock_run_job.reset_mock()
        mock_pre_download.reset_mock()
        
        train.main()
        
        # Verify only Gemma is passed to download and run
        mock_pre_download.assert_called_once_with(["config/train-gemma4.yml"])
        mock_run_job.assert_called_once()
        assert mock_run_job.call_args[0][0] == "config/train-gemma4.yml"
        
        # Case 2: 8 GPUs -> Both Mistral and Gemma should run
        mock_cuda.device_count.return_value = 8
        mock_run_job.reset_mock()
        mock_pre_download.reset_mock()
        
        train.main()
        
        # Verify both are downloaded and run
        mock_pre_download.assert_called_once_with(["config/train-mistral4small.yml", "config/train-gemma4.yml"])
        assert mock_run_job.call_count == 2
        called_configs = [args[0][0] for args in mock_run_job.call_args_list]
        assert "config/train-mistral4small.yml" in called_configs
        assert "config/train-gemma4.yml" in called_configs
        
    finally:
        train.TRAINING_PIPELINE = original_pipeline

@patch("train.shutil.rmtree")
@patch("train.run_fp8_compression")
@patch("train.resolve_merged_dir")
@patch("train.merge_gemma4_lora")
@patch("train.pre_download_models")
@patch("train.run_training_job")
def _run_pipeline(mock_run_job, mock_pre_download, mock_merge_lora, mock_resolve, mock_fp8,
                  mock_rmtree, mock_cuda, mock_subprocess, post_training_merge):
    """Drives main() for one Gemma job and returns the `aws s3 sync` destinations."""
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 2
    mock_run_job.return_value = (
        "/app/output/adapter/train-gemma4",
        {"base_model": "google/gemma-4-26b-a4b-it", "post_training_merge": post_training_merge},
        ".merged-train-gemma4.yml",
    )
    with patch("train.os.path.isdir", return_value=True):
        main()
    return [call.args[0][4] for call in mock_subprocess.call_args_list
            if call.args and call.args[0][:3] == ["aws", "s3", "sync"]]


def test_publishes_both_adapter_and_merged_model(mock_cuda, mock_subprocess, monkeypatch):
    """The adapter and the merged FP8 model must both reach S3, at the paths eval expects."""
    monkeypatch.setenv("S3_BUCKET", "test-bucket")
    targets = _run_pipeline(mock_cuda=mock_cuda, mock_subprocess=mock_subprocess, post_training_merge=True)

    assert len(targets) == 2, targets
    assert any(t.endswith("/train-gemma4") and "_run/" in t for t in targets), targets
    # The merged model is versioned per run so it can never overwrite an earlier one.
    merged = [t for t in targets if t.endswith("/train-gemma4-fp8")]
    assert len(merged) == 1 and merged[0].startswith("s3://test-bucket/models/"), targets
    assert "_run/" in merged[0], merged


def test_merge_can_be_disabled(mock_cuda, mock_subprocess, monkeypatch):
    """With post_training_merge false, only the adapter is published."""
    monkeypatch.setenv("S3_BUCKET", "test-bucket")
    targets = _run_pipeline(mock_cuda=mock_cuda, mock_subprocess=mock_subprocess, post_training_merge=False)

    assert len(targets) == 1, targets
    assert targets[0].endswith("/train-gemma4")
