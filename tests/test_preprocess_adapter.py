import json
import shutil
import sys
import os
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

# Mock heavy external libraries before importing evaluation
sys.modules['sglang'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['boto3'] = MagicMock()
sys.modules['textstat'] = MagicMock()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src-eval")))
from evaluation import preprocess_adapter

def test_preprocess_adapter_no_dir():
    assert preprocess_adapter("") == ""
    assert preprocess_adapter("nonexistent_path_xyz") == "nonexistent_path_xyz"

def test_preprocess_adapter_no_config(tmp_path):
    # Directory exists but lacks adapter_config.json
    assert preprocess_adapter(str(tmp_path)) == str(tmp_path)

def test_preprocess_adapter_config_only(tmp_path):
    config_file = tmp_path / "adapter_config.json"
    config_data = {
        "target_modules": [
            "language_model.model.layers.0.self_attn.q_proj",
            "base_model.model.language_model.model.layers.1.self_attn.v_proj",
            "language_model.layers.2.self_attn.o_proj",
            "q_proj"
        ],
        "base_model_name_or_path": "google/gemma-4-26b-a4b-it"
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f)
        
    patched_dir_str = preprocess_adapter(str(tmp_path))
    patched_dir = Path(patched_dir_str)
    
    assert patched_dir_str == str(tmp_path.parent / f"{tmp_path.name}-patched")
    assert (patched_dir / "adapter_config.json").exists()
    
    with open(patched_dir / "adapter_config.json", "r", encoding="utf-8") as f:
        patched_config = json.load(f)
        
    # Check that language_model references are stripped from target_modules
    expected_modules = [
        "model.layers.0.self_attn.q_proj",
        "base_model.model.layers.1.self_attn.v_proj",
        "model.layers.2.self_attn.o_proj",
        "q_proj"
    ]
    assert patched_config["target_modules"] == expected_modules

def test_preprocess_adapter_safetensors(tmp_path):
    config_file = tmp_path / "adapter_config.json"
    config_data = {"target_modules": ["q_proj"]}
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f)
        
    safetensors_file = tmp_path / "adapter_model.safetensors"
    # Create empty/mock safetensors file
    safetensors_file.write_text("dummy_safetensors_content")
    
    # Mock safetensors.torch functions
    mock_tensors = {
        "base_model.model.model.language_model.model.layers.0.self_attn.q_proj.weight": "tensor0",
        "base_model.model.model.language_model.layers.1.self_attn.v_proj.weight": "tensor1",
        "base_model.model.model.layers.2.self_attn.o_proj.weight": "tensor2"
    }
    
    with patch("safetensors.torch.load_file", return_value=mock_tensors) as mock_load, \
         patch("safetensors.torch.save_file") as mock_save:
         
        mock_save.side_effect = lambda tensors, path, *args, **kwargs: Path(path).write_text("mocked_safetensors")
        patched_dir_str = preprocess_adapter(str(tmp_path))
        patched_dir = Path(patched_dir_str)
        
        assert (patched_dir / "adapter_model.safetensors").exists()
        mock_load.assert_called_once_with(str(safetensors_file))
        
        # Verify saved keys
        saved_dict = mock_save.call_args[0][0]
        assert "base_model.model.model.model.layers.0.self_attn.q_proj.weight" in saved_dict
        assert "base_model.model.model.model.layers.1.self_attn.v_proj.weight" in saved_dict
        assert "base_model.model.model.layers.2.self_attn.o_proj.weight" in saved_dict
        assert saved_dict["base_model.model.model.model.layers.0.self_attn.q_proj.weight"] == "tensor0"
        assert saved_dict["base_model.model.model.model.layers.1.self_attn.v_proj.weight"] == "tensor1"
        assert saved_dict["base_model.model.model.layers.2.self_attn.o_proj.weight"] == "tensor2"

def test_preprocess_adapter_bin(tmp_path):
    config_file = tmp_path / "adapter_config.json"
    config_data = {"target_modules": ["q_proj"]}
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f)
        
    bin_file = tmp_path / "adapter_model.bin"
    # Create empty/mock bin file
    bin_file.write_text("dummy_bin_content")
    
    mock_state_dict = {
        "base_model.model.model.language_model.model.layers.0.self_attn.q_proj.weight": "tensor0",
        "base_model.model.model.language_model.layers.1.self_attn.v_proj.weight": "tensor1",
        "base_model.model.model.layers.2.self_attn.o_proj.weight": "tensor2"
    }
    
    with patch("torch.load", return_value=mock_state_dict) as mock_load, \
         patch("torch.save") as mock_save:
         
        mock_save.side_effect = lambda obj, path, *args, **kwargs: Path(path).write_text("mocked_bin")
        patched_dir_str = preprocess_adapter(str(tmp_path))
        patched_dir = Path(patched_dir_str)
        
        assert (patched_dir / "adapter_model.bin").exists()
        mock_load.assert_called_once()
        
        # Verify saved dict
        saved_dict = mock_save.call_args[0][0]
        assert "base_model.model.model.model.layers.0.self_attn.q_proj.weight" in saved_dict
        assert "base_model.model.model.model.layers.1.self_attn.v_proj.weight" in saved_dict
        assert "base_model.model.model.layers.2.self_attn.o_proj.weight" in saved_dict
