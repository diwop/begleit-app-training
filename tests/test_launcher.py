import sys
import pytest
from unittest.mock import patch, MagicMock
from src.launcher import main

@pytest.fixture
def mock_cuda():
    with patch("src.launcher.torch.cuda") as mock:
        yield mock

@pytest.fixture
def mock_subprocess():
    with patch("src.launcher.subprocess.run") as mock:
        yield mock

def test_no_cuda_exits(mock_cuda):
    mock_cuda.is_available.return_value = False
    with patch.object(sys, 'argv', ['launcher.py', '--config', 'config.yml']):
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

    with patch.object(sys, 'argv', ['launcher.py', '--config', 'test.yml']):
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

    with patch.object(sys, 'argv', ['launcher.py', '--config', 'test.yml']):
        main()

    mock_subprocess.assert_called_once()
    cmd = mock_subprocess.call_args[0][0]
    
    assert "accelerate" in cmd
    assert cmd[cmd.index("--num_processes") + 1] == "4"
    assert cmd[cmd.index("--micro_batch_size") + 1] == "4"
    assert cmd[cmd.index("--gradient_accumulation_steps") + 1] == "2"
    assert "--deepspeed" in cmd
    assert cmd[cmd.index("--deepspeed") + 1] == "config/zero3.json"
