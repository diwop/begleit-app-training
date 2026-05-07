import os
import tempfile
import json
import pytest
from src.prepare_data import load_and_format_data

@pytest.fixture
def sample_jsonl():
    data = [
        {"system": "Sys1", "user": "User1", "assistant": "Asst1"},
        {"system": "Sys2", "user": "User2", "assistant": "Asst2"}
    ]
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        path = f.name
    
    yield path
    os.remove(path)

def test_load_and_format_data(sample_jsonl):
    dataset = load_and_format_data(sample_jsonl)
    
    assert len(dataset) == 2
    assert "text" in dataset.features
    
    # Check formatting
    text = dataset[0]["text"]
    assert "<|im_start|>system\nSys1<|im_end|>" in text
    assert "<|im_start|>user\nUser1<|im_end|>" in text
    assert "<|im_start|>assistant\nAsst1<|im_end|>" in text

def test_no_empty_strings(sample_jsonl):
    dataset = load_and_format_data(sample_jsonl)
    for row in dataset:
        assert isinstance(row["text"], str)
        assert len(row["text"].strip()) > 0
