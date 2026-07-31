from unittest.mock import MagicMock
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src-train")))
from prepare_dataset import (SplitRatios, assign_split, calculate_token_count,
                             read_exclusions)

def test_calculate_token_count_list():
    # Case 1: apply_chat_template with return_dict=False successfully returns a list of tokens
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5]
    
    messages = [{"role": "user", "content": "hello"}]
    count = calculate_token_count(messages, mock_tokenizer)
    
    assert count == 5
    mock_tokenizer.apply_chat_template.assert_called_once_with(messages, tokenize=True, return_dict=False)

def test_calculate_token_count_dict():
    # Case 2: apply_chat_template with return_dict=False raises an error, 
    # but apply_chat_template without return_dict returns a dict containing input_ids
    mock_tokenizer = MagicMock()
    
    def side_effect(*args, **kwargs):
        if "return_dict" in kwargs:
            raise TypeError("Unexpected keyword argument 'return_dict'")
        return {"input_ids": [10, 20, 30], "attention_mask": [1, 1, 1]}
        
    mock_tokenizer.apply_chat_template.side_effect = side_effect
    
    messages = [{"role": "user", "content": "hello"}]
    count = calculate_token_count(messages, mock_tokenizer)
    
    assert count == 3

def test_calculate_token_count_fallback_encode():
    # Case 3: apply_chat_template fails completely, fallback to encode
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.side_effect = RuntimeError("Template error")
    mock_tokenizer.encode.return_value = [1, 2, 3, 4]
    
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"}
    ]
    count = calculate_token_count(messages, mock_tokenizer)
    
    assert count == 4
    mock_tokenizer.encode.assert_called_once_with("sys\nusr")

def test_split_assignment_is_deterministic():
    assert assign_split("0042", SplitRatios()) == assign_split("0042", SplitRatios())


def test_split_assignment_does_not_depend_on_the_rest_of_the_corpus():
    """The property the holdout rests on.

    A seeded shuffle reassigns every document when one is added, which quietly moves last
    month's holdout into this month's training set. Hashing the id alone cannot.
    """
    ratios = SplitRatios()
    before = {f"{i:04d}": assign_split(f"{i:04d}", ratios) for i in range(200)}
    # Ten new documents arrive; the existing 200 must not move.
    after = {f"{i:04d}": assign_split(f"{i:04d}", ratios) for i in range(210)}
    assert all(after[pair_id] == split for pair_id, split in before.items())


def test_split_proportions_track_the_ratios():
    ratios = SplitRatios()
    assigned = [assign_split(f"{i:04d}", ratios) for i in range(4000)]
    for name, want in (("train", 0.70), ("validation", 0.10), ("holdout", 0.20)):
        assert abs(assigned.count(name) / len(assigned) - want) < 0.02, name


def test_splits_are_disjoint():
    ratios = SplitRatios()
    assert len({assign_split(f"{i:04d}", ratios) for i in range(500)}) == 3


def test_ratio_boundaries_are_half_open():
    ratios = SplitRatios()
    assert ratios.name_for(0.0) == "train"
    assert ratios.name_for(0.6999) == "train"
    assert ratios.name_for(0.70) == "validation"
    assert ratios.name_for(0.7999) == "validation"
    assert ratios.name_for(0.80) == "holdout"
    assert ratios.name_for(0.9999) == "holdout"


def test_changing_the_salt_reshuffles():
    """Guards the warning in prepare_dataset.py: the salt is not a free parameter."""
    ratios = SplitRatios()
    ids = [f"{i:04d}" for i in range(300)]
    default = [assign_split(i, ratios) for i in ids]
    other = [assign_split(i, ratios, salt="something-else") for i in ids]
    assert default != other


def test_exclusions_ignore_the_comment_key(tmp_path):
    path = tmp_path / "excluded.json"
    path.write_text('{"_comment": ["why this file exists"], "0013": "not a translation"}',
                    encoding="utf-8")
    assert read_exclusions(path) == {"0013": "not a translation"}


def test_missing_exclusion_file_excludes_nothing(tmp_path):
    assert read_exclusions(tmp_path / "nope.json") == {}


def test_excluding_a_pair_does_not_move_the_others():
    """Why the exclusion list is safe to edit at any time.

    Assignment is sha256(salt:id) per document, so dropping one pair cannot reshuffle the
    rest -- and therefore cannot leak a holdout document into the training set.
    """
    ratios = SplitRatios()
    ids = [f"{i:04d}" for i in range(300)]
    before = {i: assign_split(i, ratios) for i in ids}
    survivors = [i for i in ids if i not in ("0013", "0224")]
    assert all(assign_split(i, ratios) == before[i] for i in survivors)
