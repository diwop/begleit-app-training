import sys
from unittest.mock import patch, MagicMock
import pytest
import torch
from src.launcher import merge_configs, main

@pytest.fixture
def mock_cuda():
    with patch("torch.cuda.is_available") as mock_is_available, \
         patch("torch.cuda.device_count") as mock_device_count, \
         patch("torch.cuda.get_device_properties") as mock_get_device_properties:
        
        mock_obj = MagicMock()
        mock_obj.is_available = mock_is_available
        mock_obj.device_count = mock_device_count
        mock_obj.get_device_properties = mock_get_device_properties
        yield mock_obj

@pytest.fixture
def mock_subprocess():
    with patch("src.launcher.subprocess.run") as mock:
        yield mock

def test_no_cuda_exits(mock_cuda):
    mock_cuda.is_available.return_value = False
    with patch.object(sys, 'argv', ['launcher.py', '--config', 'config/train.yml']):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

def test_single_gpu_low_vram(mock_cuda, mock_subprocess):
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 1
    # Mock VRAM to 24GB
    mock_props = MagicMock()
    mock_props.total_memory = 24 * (1024**3)
    mock_cuda.get_device_properties.return_value = mock_props

    with patch.object(sys, 'argv', ['launcher.py', '--config', 'config/train.yml']):
        main()

    # Verify subprocess call
    mock_subprocess.assert_called_once()
    cmd = mock_subprocess.call_args[0][0]
    
    assert "accelerate" in cmd
    assert cmd[cmd.index("--num_processes") + 1] == "1"
    assert cmd[cmd.index("--micro_batch_size") + 1] == "1"
    assert cmd[cmd.index("--gradient_accumulation_steps") + 1] == "8"
    assert "--deepspeed" not in cmd

def test_multi_gpu_high_vram(mock_cuda, mock_subprocess):
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 4
    # Mock VRAM to 80GB
    mock_props = MagicMock()
    mock_props.total_memory = 80 * (1024**3)
    mock_cuda.get_device_properties.return_value = mock_props

    with patch.object(sys, 'argv', ['launcher.py', '--config', 'config/train.yml']):
        main()

    mock_subprocess.assert_called_once()
    cmd = mock_subprocess.call_args[0][0]
    
    assert "accelerate" in cmd
    assert cmd[cmd.index("--num_processes") + 1] == "4"
    assert cmd[cmd.index("--micro_batch_size") + 1] == "4"
    assert cmd[cmd.index("--gradient_accumulation_steps") + 1] == "2"
    assert "--deepspeed" in cmd
    assert cmd[cmd.index("--deepspeed") + 1] == "config/zero3.json"


# Define the files and their expected base_model outcomes
CONFIG_TEST_CASES = [
    (
        "config/train.yml", 
        "unsloth/Mixtral-8x7B-Instruct-v0.1-bnb-4bit"
    ),
    (
        "config/train-gemma4.yml", 
        "google/gemma-4-26B-A4B-it"
    ),
    (
        "config/train-mistral4small.yml", 
        "cyankiwi/Mistral-Small-4-119B-2603-AWQ-4bit"
    ),
]

@pytest.mark.parametrize("override_file, expected_model", CONFIG_TEST_CASES)
def test_merge_configs(override_file, expected_model):
    """
    Tests that the override file correctly inherits from base.yml 
    and successfully applies its specific model name.
    """
    base_file = "config/base.yml"
    
    merged_cfg = merge_configs(base_file, override_file)
    
    # Check stable values inherited from base.yml
    assert merged_cfg.sequence_len == 4096, f"Inheritance failed for {override_file}"
    assert merged_cfg.adapter == "qlora", f"Inheritance failed for {override_file}"
    assert len(merged_cfg.datasets) == 1, "Dataset configuration was lost"
    assert merged_cfg.datasets[0].path == "data/train/dataset.jsonl", "Dataset path inherited incorrectly"
    
    # Check override values unique to the specific model
    assert merged_cfg.base_model == expected_model, f"Model override failed for {override_file}"
