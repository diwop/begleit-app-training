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

@pytest.fixture
def output_jsonl():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as f:
        path = f.name
    yield path
    os.remove(path)

def test_load_and_format_data(sample_jsonl, output_jsonl):
    dataset = load_and_format_data(sample_jsonl, output_jsonl)
    
    assert len(dataset) == 2
    assert "conversations" in dataset[0]
    
    # Check formatting
    convs = dataset[0]["conversations"]
    assert len(convs) == 3
    assert convs[0] == {"from": "system", "value": "Sys1"}
    assert convs[1] == {"from": "human", "value": "User1"}
    assert convs[2] == {"from": "gpt", "value": "Asst1"}
    
    # Verify file was written
    with open(output_jsonl, "r") as f:
        lines = f.readlines()
        assert len(lines) == 2
        loaded = json.loads(lines[0])
        assert "conversations" in loaded

def test_no_empty_strings(sample_jsonl, output_jsonl):
    dataset = load_and_format_data(sample_jsonl, output_jsonl)
    for row in dataset:
        for msg in row["conversations"]:
            assert isinstance(msg["value"], str)
            assert len(msg["value"].strip()) > 0
